import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all domains
CORS(app)

# -----------------------------
# 4️⃣ Correct & FINAL Flask Configuration
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"
# Using the Router endpoint as requested
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# Token is MANDATORY for this endpoint configuration
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN is missing! The Router endpoint might fail.")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed Relay is running (Router Endpoint)"
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

        # 2. Send to Hugging Face Router
        # We use a 60s timeout to allow for model cold-start
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=60 
        )

        hf_predictions = response.json()

        # 3. Handle HF Specific Errors
        if response.status_code != 200:
             # Pass the exact error from HF back to frontend
             return jsonify({
                "error": "HF API Error",
                "details": hf_predictions
            }), response.status_code

        # Check for specific dictionary errors (e.g., "Model is loading")
        if isinstance(hf_predictions, dict) and "error" in hf_predictions:
            return jsonify({
                "error": "HF Inference Error",
                "details": hf_predictions["error"]
            }), 503

        # 4. Handle List Format (Standardize output)
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
        return jsonify({"error": "Model timed out (Cold Start). Please try again in 30s."}), 504
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)