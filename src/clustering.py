"""Unit III: K-Means learner types + PCA visualisation.
WHY: the dataset has no 'learner type' label. Clustering discovers groups of students with
similar HABITS so advice can be tailored per group. PCA compresses 7 habit features to 2-D
so the clusters can be seen and checked."""
import json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import *
from .features import load_split

Xtr, Xte, ytr, yte = load_split()
B = Xtr[BEHAVIOUR]
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(B)   # train only
Z = prep.transform(B)

ks = range(2, 8); sil, inertia = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    sil.append(silhouette_score(Z, km.labels_)); inertia.append(km.inertia_)
# k=2 scores highest but only separates 'good' from 'weak' habits, which is too coarse for
# tailored advice, so we search k>=3 (documented design decision; all scores are saved/plotted).
K = 3 + int(np.argmax(sil[1:]))
km = KMeans(K, n_init=10, random_state=SEED).fit(Z)
print("silhouette by k:", dict(zip(ks, np.round(sil, 3))), "-> chosen k =", K)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "s-"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(RESULTS / "kmeans_k_selection.png", dpi=120); plt.close()

pca = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca.transform(Z)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
plt.figure(figsize=(6, 5)); plt.scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="tab10", s=10)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
plt.title(f"K-Means clusters (k={K}) in PCA space"); plt.tight_layout()
plt.savefig(RESULTS / "clusters_pca.png", dpi=120); plt.close()

# describe each cluster by its biggest z-score deviations (explainable, not hand-labelled)
TIPS = {"attendance": "Attend more classes (target 85%+).",
        "study_hours": "Build a daily 2-3 hour study routine.",
        "sleep_hours": "Sleep 7+ hours; it improves recall.",
        "screen_hours": "Cut non-study screen time; use app limits.",
        "participation": "Ask or answer at least one question per class.",
        "assignment_score": "Submit assignments early and review feedback.",
        "quiz_score": "Do a short self-quiz every week."}
GOOD_HIGH = {"attendance", "study_hours", "sleep_hours", "participation", "assignment_score", "quiz_score"}
profile = {}
cent = km.cluster_centers_
for c in range(K):
    z = dict(zip(BEHAVIOUR, cent[c]))
    harm = {f: (-v if f in GOOD_HIGH else v) for f, v in z.items()}     # bigger = worse habit
    worst = sorted(harm, key=harm.get, reverse=True)[:2]
    best = sorted(harm, key=harm.get)[:1]
    profile[c] = {"label": f"Group {c + 1}: strongest in {best[0].replace('_', ' ')}, "
                           f"weakest in {worst[0].replace('_', ' ')}",
                  "size": int((km.labels_ == c).sum()),
                  "tips": [TIPS[w] for w in worst],
                  "centroid": dict(zip(BEHAVIOUR, np.round(prep.named_steps["standardscaler"]
                                      .inverse_transform(cent[c:c + 1])[0], 2).tolist()))}
tab = pd.crosstab(km.labels_, ytr.map(dict(enumerate(CLASSES))).values)
tab.to_csv(RESULTS / "cluster_vs_risk.csv")
json.dump({"k": K, "silhouette": float(sil[K - 2]), "silhouette_by_k": dict(zip(map(str, ks), map(float, sil))), "pca_var": pca.explained_variance_ratio_.tolist(),
           "profiles": profile}, open(RESULTS / "cluster_summary.json", "w"), indent=2)
joblib.dump({"prep": prep, "kmeans": km, "profiles": profile}, MODELS / "cluster_model.joblib")
print(tab)
