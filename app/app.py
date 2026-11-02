import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, render_template, request, jsonify
from src.config import RAW
from src.recommend import analyse

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyse", methods=["POST"])
def api():
    d = request.get_json()
    missing = [c for c in RAW if c not in d]
    if missing:
        return jsonify(error=f"Missing fields: {missing}"), 400
    return jsonify(analyse(d, float(d.get("weekly_hours", 14))))


if __name__ == "__main__":
    app.run(debug=True)
