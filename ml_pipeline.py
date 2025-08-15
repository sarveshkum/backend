import os
import json
import whisperx
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from flask import Flask, request, jsonify

# ==== SETTINGS ====
DEVICE = "cpu"  # change to "cuda" for GPU
BATCH_SIZE = 4
COMPUTE_TYPE = "int8"
CHUNK_SECONDS = 30
CHUNK_STRIDE_SECONDS = 5
BEAM_SIZE = 5
TEMPERATURE = [0.0]
BEST_OF = 5
SR = 16000  # WhisperX default
UPLOAD_FOLDER = "uploads"

# ==== FLASK APP ====
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== LOAD MODELS ====
print("Loading WhisperX model...")
whisper_model = whisperx.load_model(
    "large-v3",
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    multilingual=True
)

print("Loading Sentence Transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading T5 model for reason generation...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ==== LOAD PHISHING WORDS ====
def load_phishing_words():
    path = os.path.join(os.path.dirname(__file__), "phishing_words.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

phishing_words = load_phishing_words()
phishing_embeddings = embedder.encode(phishing_words, convert_to_tensor=True)

# ==== UTILS ====
def chunk_audio_array(audio: np.ndarray, sr: int, chunk_s=CHUNK_SECONDS, stride_s=CHUNK_STRIDE_SECONDS):
    chunk_len = int(chunk_s * sr)
    stride = int((chunk_s - stride_s) * sr)
    total_len = audio.shape[0]
    chunks = []
    start = 0
    while start < total_len:
        end = start + chunk_len
        chunks.append(audio[start:end])
        if end >= total_len:
            break
        start += stride
    return chunks

def detect_phishing(sentences, threshold=0.6):
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

def generate_reason_with_t5(phishing_sentences):
    if not phishing_sentences:
        return "No phishing indicators detected."
    prompt = "Explain in one short sentence why the following may be phishing:\n" + "\n".join(phishing_sentences)
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ==== MAIN PIPELINE ====
def process_audio(audio_path):
    audio = whisperx.load_audio(audio_path)
    sr = SR
    audio_chunks = chunk_audio_array(audio, sr)

    all_segments = []
    detected_language = None

    for chunk in audio_chunks:
        try:
            result = whisper_model.transcribe(
                chunk,
                batch_size=BATCH_SIZE,
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                best_of=BEST_OF,
            )
        except TypeError:
            result = whisper_model.transcribe(chunk, batch_size=BATCH_SIZE)

        if not detected_language:
            lang = result.get("language")
            detected_language = lang if isinstance(lang, str) else lang.get("language")

        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                all_segments.append({"text": text})

    translated_segments = []
    for seg in all_segments:
        src_text = seg["text"]
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(src_text)
        except Exception:
            translated = src_text
        translated_segments.append(translated)

    translated_text = ". ".join(ts.strip().rstrip(".") for ts in translated_segments).strip() + "."
    sentences = [s.strip() for s in translated_text.split(".") if s.strip()]

    phishing_sentences, phishing_pct = detect_phishing(sentences)
    reason = generate_reason_with_t5(phishing_sentences)

    return {
        "Detected Language": detected_language,
        "Transcribed and Translated Sentences": translated_text,
        "Total Sentences": len(sentences),
        "Phishing Count": len(phishing_sentences),
        "Phishing Percentage": phishing_pct,
        "Phishing Sentences": phishing_sentences,
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

# ==== RUN ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
