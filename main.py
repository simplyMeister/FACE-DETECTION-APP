from flask import Flask, render_template, jsonify, request

# If you already have a Flask app in another file (e.g. app.py, server.py),
# replace this file with: from app import app  (or adjust gunicorn command)
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

# Minimal safe stubs so the app boots under gunicorn. Replace with your real logic.
@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"error": "predict endpoint not implemented in main.py stub"}), 501

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify([])

@app.route("/history", methods=["GET"])
def history():
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)