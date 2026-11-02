from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
for _p in (DATA, MODELS, RESULTS):
    _p.mkdir(exist_ok=True)

SEED = 42
TOPICS = ["maths", "programming", "databases", "ml_basics", "communication"]
MASTERY = [f"mastery_{t}" for t in TOPICS]          # topic mastery, 0-100
BASE = ["attendance", "study_hours", "prev_gpa", "assignment_score",
        "quiz_score", "sleep_hours", "screen_hours", "participation"]
RAW = BASE + MASTERY                                 # inputs the user provides
TARGET = "risk"
CLASSES = ["At risk", "Average", "High"]
# Habit features used for clustering student "learner types" (Unit III)
BEHAVIOUR = ["attendance", "study_hours", "sleep_hours", "screen_hours",
             "participation", "assignment_score", "quiz_score"]
