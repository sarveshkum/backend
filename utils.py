import re
import json
import os
import hashlib
from sentence_transformers import SentenceTransformer, util

def load_phishing_keywords():
    file_path = os.path.join(os.path.dirname(__file__), "phishing_words.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []

phishing_keywords = load_phishing_keywords()
model = SentenceTransformer('all-MiniLM-L6-v2')
phishing_embeddings = model.encode(phishing_keywords, convert_to_tensor=True)

def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def detect_phishing_sentences(sentences, threshold=0.6):
    phishing = []
    total_weight = 0.0
    for sentence in sentences:
        sent_embedding = model.encode(sentence, convert_to_tensor=True)
        scores = util.cos_sim(sent_embedding, phishing_embeddings)[0]
        max_score = float(scores.max())
        if max_score >= threshold:
            phishing.append((sentence, max_score))
            total_weight += max_score
    return phishing, total_weight

def calculate_rating(old_rating, phishing_percent):
    alpha = 0.3
    correcting_factor = 2.0
    if old_rating is None:
        old_rating = 8.0
    corr_phishing = (phishing_percent / 10.0)
    new_rating = alpha * (10.0 - (corr_phishing * correcting_factor)) + (1 - alpha) * old_rating
    return round(max(0.0, min(10.0, new_rating)), 2)

def hash_audio_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
