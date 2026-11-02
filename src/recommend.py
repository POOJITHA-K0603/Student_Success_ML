"""Glue: ML prediction + learner group + GA study plan -> one response."""
import joblib, pandas as pd
from .config import *
from .genetic import run_ga

_model = _clu = None


def _load():
    global _model, _clu
    if _model is None:
        _model = joblib.load(MODELS / "best_model.joblib")
        _clu = joblib.load(MODELS / "cluster_model.joblib")
    return _model, _clu


def analyse(student, weekly_hours=14):
    model, clu = _load()
    row = pd.Series({c: float(student[c]) for c in RAW})
    X = row.to_frame().T
    probs = {CLASSES[int(k)]: v for k, v in zip(model.classes_, model.predict_proba(X)[0].round(3).tolist())}
    label = CLASSES[int(model.predict(X)[0])]
    z = clu["prep"].transform(X[BEHAVIOUR]); c = int(clu["kmeans"].predict(z)[0])
    ga = run_ga(model, row, H=weekly_hours)
    weak = sorted(TOPICS, key=lambda t: row["mastery_" + t])[:2]
    return {"risk": label, "probabilities": probs, "group": clu["profiles"][c]["label"],
            "habit_tips": clu["profiles"][c]["tips"], "weakest_topics": weak,
            "current_mastery": {t: round(row["mastery_" + t], 1) for t in TOPICS},
            "plan": ga["hours"], "new_mastery": ga["new_mastery"],
            "success_now": round(ga["success_now"], 3), "success_plan": round(ga["success_ga"], 3)}
