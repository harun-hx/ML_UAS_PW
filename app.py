import os
import re
import base64
import io
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

app = Flask(__name__)
CORS(app)

# -----------------------------
# 1. Load Model (Internal Loading)
# -----------------------------
# This works AFTER you complete Step 1 (Upload)
MODEL_ID = "harun-767/dog-breed-classifier"

print(f"Loading model: {MODEL_ID}...")
try:
    # We load directly from the Hub
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    print("✅ Dog Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you run the upload script to push './vit-dog-model' to Hugging Face?")
    model = None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed AI is running (Internal Inference)"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # ---- 2. Process Base64 Image ----
        b64_string = data["image"]
        
        # Clean header
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)

        # Fix padding
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        # Decode
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ---- 3. AI Prediction ----
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        # Calculate probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

        # Get Top 3 results
        top_k = torch.topk(probs, 3)
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            results.append({
                "label": model.config.id2label[idx.item()],
                "confidence": round(score.item(), 4)
            })

        # Return format matching your frontend expectations
        return jsonify({
            "status": "success",
            "top1": results[0],    # Frontend likely expects 'top1'
            "top5": results,       # Sending top 3 as top5 list is fine
            "predictions": results # Backup key
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)