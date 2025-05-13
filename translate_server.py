import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://trelix-livid.vercel.app"], supports_credentials=True)

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
print(f"Loading model '{model_name}'...")
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Translation API is running."}), 200

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400
        
        text = data.get("text")
        source_lang = data.get("source_lang")
        target_lang = data.get("target_lang")

        if not all([text, source_lang, target_lang]):
            return jsonify({"error": "Fields 'text', 'source_lang', and 'target_lang' are required"}), 400

        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang)
            )
        
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return jsonify({"translated_text": translated_text})

    except Exception as e: 
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use PORT env var or default to 8000
    app.run(host="0.0.0.0", port=port)
