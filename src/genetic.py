"""Unit IV: Genetic Algorithm for the weekly study-time plan.
PROBLEM: a student has H study hours per week and 5 topics. How should hours be split
to give the best predicted outcome?
CHROMOSOME: 5 real numbers (hours per topic), repaired so they sum to H and each <= max.
FITNESS: ML model's expected success = 0.5*P(Average) + 1.0*P(High), computed after
the hours improve topic mastery (diminishing returns: m + (100-m)*(1-exp(-k*h)))."""
import numpy as np, pandas as pd
from .config import *

K_LEARN = 0.12      # learning-rate constant (assumption, documented in README)


def apply_hours(row, hours):
    """Updated mastery vector after studying `hours[i]` hours on topic i."""
    m = row[MASTERY].to_numpy(float)
    return m + (100 - m) * (1 - np.exp(-K_LEARN * hours))


def success(model, X):
    P = model.predict_proba(X)
    cls = list(model.classes_)          # 0=At risk, 1=Average, 2=High
    return 0.5 * P[:, cls.index(1)] + 1.0 * P[:, cls.index(2)]


def _fitness(model, row, pop):
    X = pd.concat([row.to_frame().T] * len(pop), ignore_index=True).astype(float)
    new = np.array([apply_hours(row, h) for h in pop])
    for j, c in enumerate(MASTERY):
        X[c] = new[:, j]
    return success(model, X[RAW])


def _repair(h, H, cap):
    h = np.clip(h, 0, cap)
    for _ in range(10):
        s = h.sum()
        if abs(s - H) < 1e-6: break
        room = (h < cap) if s < H else (h > 0)
        if not room.any(): break
        h = np.clip(h + (H - s) * room / room.sum(), 0, cap)
    return h


def run_ga(model, row, H=14, cap=8, pop_size=40, gens=40, seed=SEED, mut=0.25, elite=2):
    rng = np.random.default_rng(seed)
    pop = np.array([_repair(rng.random(5) * H, H, cap) for _ in range(pop_size)])
    hist = []
    for g in range(gens):
        fit = _fitness(model, row, pop)
        hist.append(float(fit.max()))
        order = np.argsort(-fit)
        nxt = [pop[i].copy() for i in order[:elite]]                    # elitism
        while len(nxt) < pop_size:
            a = max(rng.choice(pop_size, 3, replace=False), key=lambda i: fit[i])   # tournament
            b = max(rng.choice(pop_size, 3, replace=False), key=lambda i: fit[i])
            w = rng.random()
            child = w * pop[a] + (1 - w) * pop[b]                       # arithmetic crossover
            if rng.random() < mut:
                child = child + rng.normal(0, 1.0, 5)                   # gaussian mutation
            nxt.append(_repair(child, H, cap))
        pop = np.array(nxt)
    fit = _fitness(model, row, pop)
    best = pop[int(np.argmax(fit))]
    equal = np.full(5, H / 5)
    weakest = np.zeros(5); m = row[MASTERY].to_numpy(float)
    weakest[np.argsort(m)[:2]] = H / 2
    base = _fitness(model, row, np.zeros((1, 5)))[0]
    return {"hours": dict(zip(TOPICS, best.round(1).tolist())),
            "new_mastery": dict(zip(TOPICS, apply_hours(row, best).round(1).tolist())),
            "success_now": float(base), "success_ga": float(fit.max()),
            "success_equal_split": float(_fitness(model, row, equal[None])[0]),
            "success_weakest_two": float(_fitness(model, row, weakest[None])[0]),
            "history": hist}


if __name__ == "__main__":
    import joblib, json, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .features import load_split
    model = joblib.load(MODELS / "best_model.joblib")
    _, Xte, _, yte = load_split()
    out = []
    for i in np.where(yte.values == 0)[0][:30]:          # 30 at-risk test students
        r = run_ga(model, Xte.iloc[i], seed=int(i))
        out.append([r["success_now"], r["success_equal_split"], r["success_weakest_two"], r["success_ga"]])
    df = pd.DataFrame(out, columns=["no_extra_study", "equal_split", "weakest_two_topics", "GA"])
    df.mean().round(4).to_csv(RESULTS / "ga_comparison.csv", header=["mean_expected_success"])
    print("Mean expected success over 30 at-risk students:\n", df.mean().round(4))
    sample = run_ga(model, Xte.iloc[int(np.where(yte.values == 0)[0][0])])
    plt.figure(figsize=(6, 4)); plt.plot(sample["history"]); plt.xlabel("generation")
    plt.ylabel("best fitness"); plt.title("GA convergence"); plt.tight_layout()
    plt.savefig(RESULTS / "ga_convergence.png", dpi=120)
    json.dump({k: v for k, v in sample.items() if k != "history"}, open(RESULTS / "ga_example.json", "w"), indent=2)
