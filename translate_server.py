import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://trelix-livid.vercel.app"], supports_credentials=True)

# Load the smaller French translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
print(f"Loading model '{model_name}'...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "English to French Translation API is running."}), 200

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400
        
        text = data.get("text")

        if not text:
            return jsonify({"error": "Field 'text' is required"}), 400

        # Encode input and generate translation
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated_tokens = model.generate(**encoded)
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        return jsonify({"translated_text": translated_text})

    except Exception as e: 
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
