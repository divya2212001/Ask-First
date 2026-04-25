"""
Text preprocessing utilities for health conversation data.
"""

import re
from datetime import datetime
from dateutil import parser
from typing import Optional, Dict, Any, List


def clean_text(text: str) -> str:
    """Normalize noisy user text: lowercase, trim, normalize whitespace."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text


def normalize_timestamp(value: Any) -> Optional[str]:
    """Convert mixed timestamp formats into ISO string."""
    if not value:
        return None
    try:
        dt = parser.parse(str(value))
        return dt.isoformat()
    except Exception:
        return None


def parse_datetime(value: Any) -> Optional[datetime]:
    """Return datetime object or None."""
    if not value:
        return None
    try:
        if isinstance(value, datetime):
            return value
        return parser.parse(str(value))
    except Exception:
        return None


def compute_day_gap(ts1: Any, ts2: Any) -> Optional[int]:
    """Difference in days between timestamps. Positive if ts2 after ts1."""
    d1 = parse_datetime(ts1)
    d2 = parse_datetime(ts2)
    if not d1 or not d2:
        return None
    return (d2 - d1).days


def extract_keywords(text: str) -> Dict[str, List[str]]:
    """Lightweight keyword extraction for symptoms and lifestyle factors."""
    text = clean_text(text)
    
    vocabulary = {
        "symptoms": [
            "headache", "migraine", "pain", "stomach pain",
            "burning", "acidity", "bloating", "hair fall",
            "hair loss", "fatigue", "tired", "exhausted",
            "weakness", "nausea", "insomnia", "stress",
            "anxiety", "fever", "cough", "cold", "dizziness",
            "brain fog", "acne", "breakout", "cramps", "cramp",
            "mood", "sleepy", "drowsy"
        ],
        "lifestyle": [
            "late night", "late eating", "junk food",
            "fasting", "diet", "800 calories", "calorie",
            "sleep late", "poor sleep", "gym", "coffee",
            "caffeine", "tea", "alcohol", "smoking",
            "stressful work", "travel", "screen time",
            "screen", "phone", "reels", "water", "hydration",
            "dehydration", "dairy", "paneer", "yogurt",
            "milk", "protein", "carbohydrate", "rice",
            "sugar", "workout", "exercise", "deadline"
        ]
    }
    
    found = {"symptoms": [], "lifestyle": []}
    
    for category, words in vocabulary.items():
        for word in words:
            if word in text:
                found[category].append(word)
    
    return found


def session_to_text(session: Dict[str, Any]) -> str:
    """Merge all useful text fields from a session into one block."""
    chunks = []
    for key in [
        "user_message", "user_followup", "clary_response",
        "clary_questions", "message", "assistant_message",
        "follow_up", "notes", "summary"
    ]:
        val = session.get(key)
        if val:
            if isinstance(val, list):
                chunks.extend(str(v) for v in val if v)
            else:
                chunks.append(str(val))
    return clean_text(" ".join(chunks))


def compute_session_stats(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from a timeline."""
    if not timeline:
        return {}
    
    all_symptoms = set()
    all_triggers = set()
    for session in timeline:
        all_symptoms.update(session.get("symptoms", []))
        all_triggers.update(session.get("lifestyle", []))
    
    timestamps = [s.get("timestamp") for s in timeline if s.get("timestamp")]
    date_range = None
    if timestamps:
        dts = [parse_datetime(t) for t in timestamps]
        dts = [d for d in dts if d]
        if dts:
            date_range = {
                "start": min(dts).isoformat(),
                "end": max(dts).isoformat(),
                "days_span": (max(dts) - min(dts)).days
            }
    
    return {
        "total_sessions": len(timeline),
        "unique_symptoms": list(all_symptoms),
        "unique_triggers": list(all_triggers),
        "date_range": date_range
    }
