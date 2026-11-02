"""Model comparison, overfitting check, tuning, final evaluation (Units I, II, V)."""
import json, time, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (cross_val_score, GridSearchCV, RandomizedSearchCV,
                                     validation_curve, StratifiedKFold)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from .config import *
from .features import make_pipe, load_split

Xtr, Xte, ytr, yte = load_split()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def evaluate(name, pipe, fit=True, fit_time=None):
    t0 = time.time()
    if fit:
        pipe.fit(Xtr, ytr)
    fit_time = fit_time if fit_time is not None else time.time() - t0
    ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pte, average="macro", zero_division=0)
    cvs = cross_val_score(clone(pipe), Xtr, ytr, cv=cv, scoring="f1_macro")
    train_f1 = f1_score(ytr, ptr, average="macro")
    return dict(model=name, train_f1=train_f1, overfit_gap_f1=train_f1 - cvs.mean(),
                train_acc=accuracy_score(ytr, ptr), test_acc=accuracy_score(yte, pte),
                gap=accuracy_score(ytr, ptr) - accuracy_score(yte, pte),
                precision=p, recall=r, f1_macro=f, cv_f1_mean=cvs.mean(), cv_f1_std=cvs.std(),
                fit_seconds=fit_time), pipe


# 1. baseline models --------------------------------------------------------------
base = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree (unpruned)": DecisionTreeClassifier(random_state=SEED),
    "Random Forest (default)": RandomForestClassifier(random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "MLP Neural Net (default)": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                              solver="adam", alpha=1e-4, early_stopping=True,
                                              max_iter=800, random_state=SEED),
}
rows, fitted = [], {}
for n, m in base.items():
    r, p = evaluate(n, make_pipe(m)); rows.append(r); fitted[n] = p
    print(f"{n:28s} train={r['train_acc']:.3f} test={r['test_acc']:.3f} cvF1={r['cv_f1_mean']:.3f}")

# 2. overfitting diagnosis: validation curve over RF max_depth --------------------
depths = [2, 3, 4, 6, 8, 12, 20]
tr_s, va_s = validation_curve(make_pipe(RandomForestClassifier(n_estimators=150, random_state=SEED)),
                              Xtr, ytr, param_name="clf__max_depth", param_range=depths,
                              cv=cv, scoring="f1_macro")
plt.figure(figsize=(6, 4))
plt.plot(depths, tr_s.mean(1), "o-", label="train"); plt.plot(depths, va_s.mean(1), "s-", label="validation (CV)")
plt.xlabel("Random Forest max_depth"); plt.ylabel("macro F1"); plt.legend(); plt.title("Overfitting check")
plt.tight_layout(); plt.savefig(RESULTS / "overfitting_validation_curve.png", dpi=120); plt.close()

# 3. hyperparameter tuning (CV on TRAIN only) --------------------------------------
rf_grid = {"clf__n_estimators": [150, 300], "clf__max_depth": [4, 6, 8, None],
           "clf__min_samples_leaf": [1, 3, 5, 10]}
gs = GridSearchCV(make_pipe(RandomForestClassifier(random_state=SEED)), rf_grid, cv=cv,
                  scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
mlp_dist = {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],        # L2 regularisation
            "clf__learning_rate_init": [1e-3, 3e-3, 1e-2]}
rs = RandomizedSearchCV(make_pipe(MLPClassifier(activation="relu", solver="adam", early_stopping=True,
                                                max_iter=800, random_state=SEED)),
                        mlp_dist, n_iter=12, cv=cv, scoring="f1_macro", n_jobs=-1,
                        random_state=SEED).fit(Xtr, ytr)
print("RF best:", gs.best_params_, round(gs.best_score_, 3))
print("MLP best:", rs.best_params_, round(rs.best_score_, 3))

for n, s in [("Random Forest (tuned)", gs), ("MLP Neural Net (tuned)", rs)]:
    r, p = evaluate(n, s.best_estimator_, fit=False, fit_time=s.refit_time_); rows.append(r); fitted[n] = p

res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "model_comparison.csv", index=False)
print("\n", res.to_string(index=False))

# 4. model choice WITHOUT looking at the test set -----------------------------------
# Rule: keep models whose CV macro-F1 is within one std of the best, then pick the one that
# overfits least (smallest train-F1 minus CV-F1), then the fastest. Accuracy alone is not the criterion.
cand = res[res.model != "Decision Tree (unpruned)"]
top = cand.loc[cand.cv_f1_mean.idxmax()]
cand = cand[cand.cv_f1_mean >= top.cv_f1_mean - top.cv_f1_std]
best_name = cand.sort_values(["overfit_gap_f1", "fit_seconds"]).iloc[0].model
print("Shortlist (within 1 std of best CV F1):", list(cand.model))
best = fitted[best_name]
print("\nSelected model:", best_name)

# 5. final evaluation on the held-out test set ------------------------------------
pred = best.predict(Xte)
LAB = list(range(len(CLASSES)))
rep = classification_report(yte, pred, labels=LAB, target_names=CLASSES, output_dict=True, zero_division=0)
(RESULTS / "classification_report.txt").write_text(classification_report(yte, pred, labels=LAB, target_names=CLASSES))
ConfusionMatrixDisplay.from_predictions(yte, pred, labels=LAB, display_labels=CLASSES, cmap="Blues")
plt.title(best_name); plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=120); plt.close()

pi = permutation_importance(best, Xte, yte, n_repeats=10, random_state=SEED, scoring="f1_macro")
imp = pd.Series(pi.importances_mean, index=RAW).sort_values()
imp.plot.barh(figsize=(6, 5)); plt.title("Permutation importance (test)"); plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=120); plt.close()

mlp = fitted["MLP Neural Net (default)"].named_steps["clf"]
plt.figure(figsize=(6, 4)); plt.plot(mlp.loss_curve_, label="training loss")
plt.plot(1 - np.array(mlp.validation_scores_), label="validation error")
plt.xlabel("epoch"); plt.legend(); plt.title("MLP learning curve (Adam, ReLU)"); plt.tight_layout()
plt.savefig(RESULTS / "mlp_loss_curve.png", dpi=120); plt.close()

joblib.dump(best, MODELS / "best_model.joblib")
json.dump({"selected": best_name, "test_macro_f1": float(f1_score(yte, pred, average="macro")),
           "test_accuracy": float(accuracy_score(yte, pred)),
           "rf_best_params": {k: (v if v is None else int(v)) for k, v in gs.best_params_.items()},
           "mlp_best_params": {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in rs.best_params_.items()},
           "per_class": {c: {k: round(v, 3) for k, v in rep[c].items()} for c in CLASSES}},
          open(RESULTS / "metrics.json", "w"), indent=2)
print(open(RESULTS / "classification_report.txt").read())
