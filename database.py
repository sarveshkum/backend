from pymongo import MongoClient
import gridfs
from bson import ObjectId
import hashlib

# ================== CONFIG ==================
# MongoDB Atlas connection (password encoded: security@990 -> security%40990)
MONGO_URL = "mongodb+srv://sarveshkumarsasikumar:security%40990@audio.gqmptyv.mongodb.net/?retryWrites=true&w=majority&appName=audio"

# Connect to Atlas
client = MongoClient(MONGO_URL)
db = client["phishingDB"]           # database name
col = db["phishing_calls"]          # collection for transcripts/ratings
fs = gridfs.GridFS(db)              # GridFS for storing audio files
hash_col = db["audio_hashes"]       # collection to prevent duplicates

# ================== FUNCTIONS ==================
def get_existing_rating(number):
    """
    Get the last known rating for a phone number.
    Returns rating or None if not found.
    """
    record = col.find_one({"number": number})
    return record.get("rating") if record else None


def hash_audio_file(file_path):
    """
    Generate a SHA256 hash of an audio file to detect duplicates.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_audio_file(file_path, filename):
    """
    Save an audio file to GridFS if not already present.
    Returns (audio_id, upload_count).
    Raises ValueError if duplicate detected.
    """
    file_hash = hash_audio_file(file_path)

    # Check for duplicate by hash
    existing = hash_col.find_one({"hash": file_hash})
    if existing:
        raise ValueError("Duplicate audio file detected. Illegal entry.")

    with open(file_path, "rb") as f:
        audio_id = fs.put(f, filename=filename)

    hash_col.insert_one({
        "hash": file_hash,
        "audio_id": audio_id,
        "upload_count": 1
    })
    return str(audio_id), 1


def save_rating(number, transcript, new_rating, audio_id, phishing_percent):
    """
    Save phishing detection results with rating and transcript.
    """
    update_doc = {
        "$set": {
            "rating": new_rating
        },
        "$push": {
            "transcript": {
                "$each": [transcript]
            },
            "audio_records": {
                "phishing_percent": phishing_percent,
                "audio_id": audio_id
            }
        }
    }
    col.update_one({"number": number}, update_doc, upsert=True)


def get_full_record(number):
    """
    Get full record (transcripts, ratings, audio references) for a number.
    """
    return col.find_one({"number": number})


def get_audio_file(audio_id):
    """
    Retrieve audio file bytes from GridFS.
    """
    return fs.get(ObjectId(audio_id)).read()
