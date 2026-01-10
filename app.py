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
# Enable CORS for all domains (Crucial for Vercel -> Render communication)
CORS(app)

# --- 1. Load Model ---
# We use a relative path so it works both locally and on the cloud
MODEL_PATH = "harun-767/dog-breed-classifier"

print(f"Loading model from {MODEL_PATH}...")
try:
    # Removed subfolder="vit-horse-model" assuming standard Hugging Face structure.
    # If your model is in a subfolder, add subfolder="your-folder-name" back here.
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except OSError:
    print("❌ Critical Error: Model files not found. Check your MODEL_PATH or subfolder configuration.")
    # We don't exit here so the server still starts and you can see the error logs online
    model = None

@app.route("/", methods=["GET"])
def home():
    return "🐶 Dog Breed AI is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # --- 2. Process Base64 Image ---
        b64_string = data["image"]
        
        # Clean up the string (remove "data:image/png;base64," prefix)
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)

        # Fix padding errors
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        # Decode
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- 3. AI Prediction ---
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

        return jsonify({
            "status": "success",
            "predictions": results
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use the PORT environment variable for Render/Heroku, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)