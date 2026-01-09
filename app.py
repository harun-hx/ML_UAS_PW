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
# CONFIGURATION
# -----------------------------
MODEL_PATH = "harun-767/dog-breed-classifier"
DEVICE = torch.device("cpu")  # Force CPU to save memory

# Load processor and model
print(f"Loading model from {MODEL_PATH} on CPU...")
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "🐶 Dog Breed AI is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # --- Decode Base64 image ---
        b64_string = data["image"]
        if "base64," in b64_string:
            b64_string = b64_string.split("base64,")[1]

        # Fix padding
        b64_string += "=" * ((4 - len(b64_string) % 4) % 4)

        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- Preprocess and predict ---
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        # --- Top 3 predictions ---
        top_k = torch.topk(probs, 5)
        results = []
        for i, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
            raw_label = model.config.id2label[idx.item()]
            clean_label = re.sub(r'^n\d+-', '', raw_label).replace('_', ' ').title()
            results.append({
                "label": clean_label,
                "confidence": round(score.item(), 4),
                "is_best_match": (i == 0)  # Only the first one is marked as best match
            })

        return jsonify({"status": "success", "predictions": results})

    except (base64.binascii.Error, IOError) as img_err:
        return jsonify({"error": "Invalid image", "details": str(img_err)}), 400
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "Prediction Failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
