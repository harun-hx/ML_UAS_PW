import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"

# ✅ NEW ROUTER ENDPOINT (Required since api-inference is deprecated)
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

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
        "message": "🐶 Dog Breed Relay is running (Router API)"
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

        # ---------------------------------------------------------
        # ✅ YOUR EXACT FINAL CODE BLOCK
        # ---------------------------------------------------------
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        raw_text = response.text.strip()
        
        if not raw_text:
            return jsonify({
                "error": "HF API Error",
                "details": "Empty response from Hugging Face (cold start or rate limit)"
            }), 502

        try:
            hf_predictions = response.json()
        except ValueError:
            print("HF NON-JSON RESPONSE:", raw_text[:500])
            return jsonify({
                "error": "HF API Error",
                "details": "Non-JSON response from Hugging Face"
            }), 502

        if response.status_code != 200:
            return jsonify({
                "error": "HF API Error",
                "details": hf_predictions
            }), response.status_code
        # ---------------------------------------------------------
        # END OF YOUR BLOCK
        # ---------------------------------------------------------

        # 3. Handle "Soft" Errors (200 OK but contains error message)
        if isinstance(hf_predictions, dict) and "error" in hf_predictions:
            return jsonify({
                "error": "HF Inference Error",
                "details": hf_predictions["error"]
            }), 503

        # 4. Standardize List Format (Handle [[...]] vs [...])
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
        return jsonify({"error": "Model timed out (Cold Start). Try again."}), 504
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)