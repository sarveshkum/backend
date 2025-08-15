import os
import json
import whisperx
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import from utils.py
from utils import detect_phishing_sentences, hash_audio_file

# ==== SETTINGS ====
DEVICE = "cpu"  # change to "cuda" if you have GPU
BATCH_SIZE = 4
COMPUTE_TYPE = "int8"  # float32 for better accuracy but slower on CPU
UPLOAD_FOLDER = "uploads"

# ==== FLASK APP ====
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== LOAD MODELS ====
print("Loading WhisperX model...")
whisper_model = whisperx.load_model("large-v3", device=DEVICE, compute_type=COMPUTE_TYPE)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load T5 model for reason generation
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ==== LOAD PHISHING WORDS ====
def load_phishing_words():
    path = os.path.join(os.path.dirname(__file__), "phishing_words.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

phishing_words = load_phishing_words()
phishing_embeddings = embedder.encode(phishing_words, convert_to_tensor=True)

# ==== SEMANTIC DETECTION ====
def detect_phishing_semantic(sentences, threshold=0.6):
    phishing_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        s_emb = embedder.encode(sentence, convert_to_tensor=True)
        scores = util.cos_sim(s_emb, phishing_embeddings)[0]
        if float(scores.max()) >= threshold:
            phishing_sentences.append(sentence.strip())
    pct = (len(phishing_sentences) / len(sentences) * 100) if sentences else 0
    return phishing_sentences, pct

# ==== REASON GENERATION ====
def generate_reason(phishing_sentences):
    if not phishing_sentences:
        return "No phishing indicators detected."
    prompt = "Summarize why the following sentences indicate phishing:\n" + "\n".join(phishing_sentences)
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    reason = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reason

# ==== MAIN PROCESS FUNCTION ====
def process_audio(audio_path):
    audio = whisperx.load_audio(audio_path)
    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
    language = result["language"]

    translated_text = " ".join([
        GoogleTranslator(source='auto', target='en').translate(seg["text"])
        for seg in result["segments"]
    ])

    sentences = translated_text.split(".")

    # Method 1: Semantic similarity detection
    phishing_sentences_sem, phishing_pct_sem = detect_phishing_semantic(sentences)

    # Method 2: Weighted keyword detection from utils.py
    phishing_weighted, total_weight = detect_phishing_sentences(sentences)
    weighted_pct = min((total_weight / max(len(sentences), 1)) * 10, 100)

    # Combine results — taking the higher percentage
    final_pct = max(phishing_pct_sem, weighted_pct)

    # Merge phishing sentences from both methods
    phishing_all = list(set(phishing_sentences_sem + [p[0] for p in phishing_weighted]))

    # Generate reason using T5
    reason = generate_reason(phishing_all)

    return {
        "Detected Language": language,
        "Transcribed and Translated Sentences": translated_text,
        "Total Sentences": len(sentences),
        "Phishing Count": len(phishing_all),
        "Phishing Percentage": final_pct,
        "Phishing Sentences": phishing_all,
        "Reason": reason
    }

# ==== API ENDPOINT ====
@app.route("/upload_call", methods=["POST"])
def upload_call():
    try:
        number = request.form.get("number")
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({"error": "No audio file uploaded"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        result = process_audio(file_path)

        return jsonify({
            "number": number,
            "phishing_percent": result["Phishing Percentage"],
            "analysis": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== RUN MODE ====
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        audio_file = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_file):
            print("Error: File not found!")
        else:
            result = process_audio(audio_file)
            print("\n=== Analysis Report ===")
            print(json.dumps(result, indent=2))
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
