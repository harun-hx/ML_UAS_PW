import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all domains
CORS(app)

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"
# Use the Standard Inference API
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Setup Headers (Safe Token Handling)
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
} if HF_TOKEN else {
    "Content-Type": "application/json"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed Relay is running (Safe Production Version)"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # 1. Clean Base64 String
        b64_string = data["image"]
        if "base64," in b64_string:
            b64_string = b64_string.split("base64,")[1]

        payload = {
            "inputs": b64_string,
            "options": {"wait_for_model": True}
        }

        # 2. Send to Hugging Face with TIMEOUT (Crucial Fix)
        # 60s timeout handles the "Cold Start" case where model loads
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=60 
        )

        hf_predictions = response.json()

        # 3. Handle HF Specific Errors (Model Loading / Auth Error)
        if response.status_code != 200:
             return jsonify({
                "error": "HF API Error",
                "details": hf_predictions
            }), response.status_code

        # If HF returns a dict with error (even with 200 OK sometimes)
        if isinstance(hf_predictions, dict) and "error" in hf_predictions:
            return jsonify({
                "error": "HF Inference Error",
                "details": hf_predictions["error"]
            }), 503

        # 4. Handle List Format (HF sometimes returns [[...]] or [...])
        if isinstance(hf_predictions, list) and len(hf_predictions) > 0 and isinstance(hf_predictions[0], list):
            hf_predictions = hf_predictions[0]

        # 5. Format & Clean Results
        formatted_results = []
        for i, pred in enumerate(hf_predictions):
            raw_label = pred.get("label", "Unknown")
            
            # Clean Label: "n0210-husky" -> "Husky"
            clean_label = re.sub(r'^n\d+-', '', raw_label).replace('_', ' ').title()
            
            formatted_results.append({
                "label": clean_label,
                "confidence": round(pred.get("score", 0), 4),
                "is_best_match": (i == 0)
            })

        return jsonify({
            "status": "success",
            "predictions": formatted_results
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "Model timed out (Is it waking up? Try again in 30s)"}), 504
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)