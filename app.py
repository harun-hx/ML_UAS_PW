import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow CORS for all domains
CORS(app)

# CONFIGURATION
MODEL_ID = "harun-767/dog-breed-classifier"
# We send the image to Hugging Face's Public API
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Optional: Add your HF Token if you have one in Railway Variables (HF_TOKEN)
# If not, it will try anonymously (might be rate limited)
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "🐶 Dog Breed Relay is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # 1. Prepare Image Data
        b64_string = data["image"]
        # Remove the header "data:image/jpeg;base64," if it exists
        if "base64," in b64_string:
            b64_string = b64_string.split("base64,")[1]

        # 2. Send to Hugging Face (The "Router" logic)
        payload = {
            "inputs": b64_string,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # 3. Check for Errors from HF
        if response.status_code != 200:
            return jsonify({"error": "HF API Error", "details": response.text}), response.status_code

        # 4. Format the Output
        hf_predictions = response.json()
        
        # Handle different response formats (sometimes list of lists)
        if isinstance(hf_predictions, list) and len(hf_predictions) > 0 and isinstance(hf_predictions[0], list):
            hf_predictions = hf_predictions[0]

        formatted_results = []
        for i, pred in enumerate(hf_predictions):
            raw_label = pred.get("label", "Unknown")
            # Clean: "n0210-husky" -> "Husky"
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

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)