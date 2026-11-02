"""Runs the whole pipeline in order. Each step saves real outputs to results/ and models/."""
import subprocess, sys
for m in ["src.data_gen", "src.eda", "src.train_models", "src.clustering", "src.genetic"]:
    print(f"\n===== {m} =====")
    subprocess.run([sys.executable, "-m", m], check=True)
print("\nDone. Start the app with:  python app/app.py")
