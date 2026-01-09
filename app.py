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
# 1. Load Model
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"

print(f"Loading model: {MODEL_ID}...")
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    print("✅ Dog Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed AI is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # ---- 2. Process Base64 Image ----
        b64_string = data["image"]
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)

        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

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
        
        # Enumerate gives us the index 'i' to find the top 1
        for i, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
            raw_label = model.config.id2label[idx.item()]
            
            # --- CLEANING LOGIC ---
            # 1. Remove the "n12345-" prefix using Regex
            #    ^n\d+- matches "n" followed by digits and a dash at the start
            clean_label = re.sub(r'^n\d+-', '', raw_label)
            
            # 2. Replace underscores with spaces and Title Case
            #    "Labrador_retriever" -> "Labrador Retriever"
            clean_label = clean_label.replace('_', ' ').title()

            results.append({
                "label": clean_label,
                "confidence": round(score.item(), 4),
                # Add a flag for the frontend to know this is the winner
                "is_best_match": (i == 0) 
            })

        # Return SIMPLIFIED format
        return jsonify({
            "status": "success",
            "predictions": results 
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)