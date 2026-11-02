"""Synthetic student dataset (documented generator).

Four hidden learner archetypes are mixed; features are drawn around each
archetype's mean. The final score is a NON-linear function of the features
plus unobserved 'motivation' and noise, so the target cannot be recovered
perfectly. The risk class is made from score quantiles.
REPLACE WITH REAL DATA (class survey / UCI Student Performance) using the same
column names if you can: the rest of the project runs unchanged.
"""
import numpy as np, pandas as pd
from .config import *

# order: attendance, study_hours, prev_gpa, assignment, quiz, sleep, screen, participation
ARCH = {
    "achiever":   [90, 4.5, 8.5, 85, 80, 7.2, 2.5, 4.0],
    "crammer":    [72, 2.0, 6.8, 65, 62, 5.8, 4.5, 3.0],
    "disengaged": [55, 1.2, 5.5, 48, 45, 6.5, 6.0, 1.8],
    "steady":     [82, 3.0, 7.3, 72, 68, 7.0, 3.5, 3.3],
}
PROBS = [0.25, 0.20, 0.20, 0.35]
SD = np.array([8, 1.0, 0.7, 8, 8, 0.8, 1.0, 0.6])
LO = np.array([30, 0, 4, 20, 15, 4, 0.5, 1])
HI = np.array([100, 9, 10, 100, 100, 9.5, 10, 5])


def generate(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array(list(ARCH.values()))[rng.choice(4, n, p=PROBS)]
    x = np.clip(mu + rng.normal(size=mu.shape) * SD, LO, HI)
    df = pd.DataFrame(x, columns=BASE)

    # topic mastery depends on quiz, GPA and a topic-specific personal offset
    for t in MASTERY:
        df[t] = np.clip(0.45 * df.quiz_score + 0.35 * df.prev_gpa * 10 + 10
                        + rng.normal(0, 12, n), 5, 100)

    motivation = rng.normal(0, 1, n)               # hidden, never given to model
    score = (0.20 * df.attendance
             + 2.5 * df.study_hours * (1 - 0.05 * df.study_hours)   # diminishing returns
             + 2.6 * df.prev_gpa + 0.10 * df.assignment_score + 0.12 * df.quiz_score
             + 0.22 * df[MASTERY].mean(axis=1)
             + 1.8 * np.clip(df.sleep_hours - 5.5, 0, 2)
             - 1.2 * np.maximum(df.screen_hours - 4, 0)
             + 0.6 * df.study_hours * (df.participation - 3)          # interaction
             + rng.normal(0, 5, n) + 3 * motivation)
    df["final_score"] = (score - score.min()) / (score.max() - score.min()) * 100
    q25, q75 = df.final_score.quantile([.25, .75])
    df[TARGET] = np.where(df.final_score < q25, "At risk",
                  np.where(df.final_score > q75, "High", "Average"))

    # realistic missingness (so the imputation step is genuinely needed)
    for c in ["sleep_hours", "screen_hours", "participation"]:
        df.loc[rng.random(n) < 0.02, c] = np.nan
    return df.round(2)


if __name__ == "__main__":
    d = generate()
    d.to_csv(DATA / "students.csv", index=False)
    print(d.shape, d[TARGET].value_counts().to_dict())
