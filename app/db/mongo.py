import os
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import pymongo
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client["talent_track_ai"]
    # Test connection silently
    client.admin.command('ping')
    mongo_available = True
except Exception as e:
    print(f"MongoDB connection notice: Operating with local memory fallback ({e})")
    mongo_available = False
    db = None

# In-memory storage fallback if MongoDB is not running locally
_memory_store: List[Dict[str, Any]] = []

class MongoDatabase:
    """Manages database operations for candidate ATS analyses and job postings."""

    @staticmethod
    def is_connected() -> bool:
        return mongo_available

    @classmethod
    def save_analysis(cls, analysis_data: Dict[str, Any]) -> str:
        """Stores candidate ATS analysis result in MongoDB or in-memory fallback."""
        record = {
            **analysis_data,
            "created_at": datetime.utcnow().isoformat()
        }
        if mongo_available and db is not None:
            try:
                res = db.candidate_analyses.insert_one(record)
                return str(res.inserted_id)
            except Exception as err:
                print(f"MongoDB insert error: {err}")
        
        # Fallback to in-memory store
        _memory_store.insert(0, record)
        return f"mem_{len(_memory_store)}"

    @classmethod
    def get_recent_analyses(cls, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent candidate analysis records."""
        if mongo_available and db is not None:
            try:
                records = list(db.candidate_analyses.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
                return records
            except Exception as err:
                print(f"MongoDB find error: {err}")

        return _memory_store[:limit]
