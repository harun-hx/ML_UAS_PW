import os
import re
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from PIL import Image

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the Official Client
# This handles the URL routing automatically (Router vs Legacy)
client = InferenceClient(token=HF_TOKEN)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok", 
        "message": "🐶 Dog Breed Relay is running (Official Client)"
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

        # 2. Convert to Image Object (PIL)
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes))

        # 3. AI Prediction via Official Client
        # The client automatically handles the URL and Model Loading
        predictions = client.image_classification(image, model=MODEL_ID)

        # 4. Format Results (The client returns a clean list of objects)
        formatted_results = []
        for i, pred in enumerate(predictions):
            raw_label = pred.label # Access object attribute directly
            
            # Clean Label: "n0210-husky" -> "Husky"
            clean_label = re.sub(r'^n\d+-', '', raw_label).replace('_', ' ').title()
            
            formatted_results.append({
                "label": clean_label,
                "confidence": round(pred.score, 4), # Access score attribute
                "is_best_match": (i == 0)
            })

        return jsonify({
            "status": "success",
            "predictions": formatted_results
        })

    except Exception as e:
        # Check for specific HF errors (Like model loading)
        error_msg = str(e)
        print(f"Server Error: {error_msg}")
        
        if "503" in error_msg or "loading" in error_msg.lower():
             return jsonify({
                "error": "Model Loading",
                "details": "The AI is waking up. Please try again in 30 seconds."
            }), 503
            
        if "404" in error_msg:
             return jsonify({
                "error": "Model Not Found",
                "details": "Check if your model is Private or if the 'Inference Widget' is disabled on Hugging Face."
            }), 404

        return jsonify({"error": "Prediction Failed", "details": error_msg}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)