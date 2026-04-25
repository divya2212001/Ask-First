"""
Timeline builder for Ask First conversations.
Creates chronological, normalized timeline from raw conversation data.
"""

from typing import List, Dict, Any

from core.preprocess import (
    normalize_timestamp,
    session_to_text,
    extract_keywords,
    parse_datetime
)


def build_user_timeline(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build normalized chronological timeline from user conversations.
    
    Args:
        user: User dict with 'conversations' or 'sessions' key
    
    Returns:
        List of normalized session dicts sorted chronologically
    """
    conversations = user.get("conversations", user.get("sessions", []))
    
    timeline = []
    
    for convo in conversations:
        # Extract all text parts
        text_parts = [
            convo.get("user_message", ""),
            convo.get("user_followup", ""),
            convo.get("clary_response", ""),
        ]
        
        # Include questions if present
        questions = convo.get("clary_questions", [])
        if questions:
            text_parts.extend(questions)
        
        merged_text = " ".join(x for x in text_parts if x)
        clean = session_to_text({"message": merged_text})
        
        # Extract keywords
        keywords = extract_keywords(clean)
        
        # Parse timestamp
        ts = normalize_timestamp(convo.get("timestamp"))
        
        # Compute week number for temporal reasoning
        week_num = None
        dt = parse_datetime(ts)
        if dt:
            # Week of year
            week_num = dt.isocalendar()[1]
        
        timeline.append({
            "session_id": convo.get("session_id"),
            "timestamp": ts,
            "week_number": week_num,
            "text": clean,
            "severity": convo.get("severity", "unknown"),
            "tags": [t.lower() for t in convo.get("tags", [])],
            "symptoms": keywords["symptoms"],
            "lifestyle": keywords["lifestyle"],
            "raw": convo  # Keep raw for reference
        })
    
    # Sort chronologically
    timeline.sort(key=lambda x: x["timestamp"] or "")
    
    return timeline


def get_sessions_by_tag(timeline: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    """Return sessions containing a specific tag."""
    tag = tag.lower()
    return [
        item for item in timeline
        if tag in [t.lower() for t in item.get("tags", [])]
    ]


def get_sessions_by_keyword(timeline: List[Dict[str, Any]], word: str) -> List[Dict[str, Any]]:
    """Search inside session text for a keyword."""
    word = word.lower()
    return [
        item for item in timeline
        if word in item.get("text", "")
    ]


def get_sessions_by_symptom(timeline: List[Dict[str, Any]], symptom: str) -> List[Dict[str, Any]]:
    """Find sessions reporting a specific symptom."""
    symptom = symptom.lower()
    return [
        item for item in timeline
        if symptom in [s.lower() for s in item.get("symptoms", [])]
    ]


def get_sessions_by_trigger(timeline: List[Dict[str, Any]], trigger: str) -> List[Dict[str, Any]]:
    """Find sessions mentioning a specific lifestyle trigger."""
    trigger = trigger.lower()
    return [
        item for item in timeline
        if trigger in [t.lower() for t in item.get("lifestyle", [])]
    ]


def get_session_gaps(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute gaps between consecutive sessions.
    Useful for detecting missed sessions or irregular reporting.
    """
    gaps = []
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        
        prev_dt = parse_datetime(prev.get("timestamp"))
        curr_dt = parse_datetime(curr.get("timestamp"))
        
        if prev_dt and curr_dt:
            gap_days = (curr_dt - prev_dt).days
            gaps.append({
                "from_session": prev["session_id"],
                "to_session": curr["session_id"],
                "gap_days": gap_days
            })
    
    return gaps
