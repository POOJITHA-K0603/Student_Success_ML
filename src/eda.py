import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from .config import *

df = pd.read_csv(DATA / "students.csv")
out = []
out.append(f"Shape: {df.shape}\n\nMissing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
out.append(f"\nClass balance:\n{df[TARGET].value_counts()}")
out.append(f"\nDescribe:\n{df[RAW].describe().round(2).T}")
(RESULTS / "eda_summary.txt").write_text("\n".join(map(str, out)))

df[BASE].hist(bins=25, figsize=(11, 7)); plt.tight_layout()
plt.savefig(RESULTS / "eda_histograms.png", dpi=120); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[RAW + ["final_score"]].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig(RESULTS / "eda_correlation.png", dpi=120); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for a, c in zip(ax.ravel(), ["attendance", "study_hours", "prev_gpa", "quiz_score", "screen_hours", "sleep_hours"]):
    sns.boxplot(data=df, x=TARGET, y=c, order=CLASSES, ax=a)
plt.tight_layout(); plt.savefig(RESULTS / "eda_boxplots_by_risk.png", dpi=120); plt.close()
print("EDA saved to results/")
