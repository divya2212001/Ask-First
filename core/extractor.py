"""
Dynamic health entity extraction with negation detection.
Extracts from user_message, user_followup, and tags only.
Suppresses entities in negated contexts (e.g., "don't exercise").
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.preprocess import clean_text, parse_datetime
from core.models import ExtractedEvent


# Compound entities that should be matched as whole phrases
COMPOUND_SYMPTOMS = [
    "stomach pain", "hair fall", "hair loss", "brain fog", "back pain",
    "period cramps", "post-lunch", "late night"
]

# Simple entities
SEED_SYMPTOMS = [
    "headache", "migraine", "pain", "burning", "acidity",
    "bloating", "fatigue", "tired", "exhausted",
    "weakness", "nausea", "insomnia", "stress", "anxiety", "fever",
    "cough", "cold", "dizziness", "acne", "breakout",
    "cramps", "cramp", "mood", "sleepy", "drowsy"
]

SEED_TRIGGERS = [
    "late night", "late eating", "junk food", "fasting", "diet",
    "calorie", "sleep late", "poor sleep", "gym", "coffee", "caffeine",
    "tea", "alcohol", "smoking", "stressful work", "deadline",
    "screen", "phone", "reels", "water", "hydration", "dehydration",
    "dairy", "paneer", "yogurt", "milk", "protein", "carbohydrate",
    "rice", "sugar", "workout", "exercise"
]

SEED_INTERVENTIONS = [
    "reduce", "cut", "added", "increase", "stopped", "started",
    "try", "avoid", "eliminate", "take", "medication", "therapy"
]

# Negation patterns that suppress entity extraction
NEGATION_PREFIXES = [
    r"\bno\s+", r"\bnot\s+", r"\bnever\s+", r"\bdon'?t\s+", r"\bdoesn'?t\s+",
    r"\bdidn'?t\s+", r"\bcan'?t\s+", r"\bwon'?t\s+", r"\baren'?t\s+",
    r"\bisn'?t\s+", r"\bwasn'?t\s+", r"\bhaven'?t\s+", r"\bhasn'?t\s+",
    r"\bhadn'?t\s+", r"\bwithout\s+", r"\blacks?\s+", r"\babsent\s+",
    r"\bnone\s+", r"\bzero\s+", r"\bnot\s+really\s+", r"\bnot\s+much\s+",
    r"\bnot\s+any\s+", r"\bno\s+new\s+", r"\bno\s+extra\s+",
]


def is_negated(text: str, position: int, window: int = 25) -> bool:
    """Check if the word at position is in a negated context."""
    start = max(0, position - window)
    context = text[start:position]
    for pattern in NEGATION_PREFIXES:
        if re.search(pattern + r"\w*$", context):
            return True
    return False


def extract_compound_entities(text: str, compounds: List[str]) -> Dict[str, List[int]]:
    """Find compound entity matches."""
    hits = {}
    for compound in compounds:
        pattern = r'\b' + re.escape(compound) + r'\b'
        matches = list(re.finditer(pattern, text))
        if matches and not any(is_negated(text, m.start()) for m in matches):
            hits[compound] = [m.start() for m in matches]
    return hits


def extract_simple_entities(text: str, entities: List[str], 
                           exclude_positions: List[tuple] = None) -> Dict[str, List[int]]:
    """Find simple entity matches, excluding positions covered by compounds."""
    hits = {}
    exclude_positions = exclude_positions or []
    
    for entity in entities:
        pattern = r'\b' + re.escape(entity) + r'\b'
        for match in re.finditer(pattern, text):
            pos = match.start()
            # Skip if within a compound's range
            if any(start <= pos < start + len(comp) for start, comp in exclude_positions):
                continue
            if not is_negated(text, pos):
                if entity not in hits:
                    hits[entity] = []
                hits[entity].append(pos)
    return hits


def extract_from_session(session: Dict[str, Any]) -> List[ExtractedEvent]:
    """Extract events from user text and tags only (not clary_response)."""
    events = []
    session_id = session.get("session_id", "unknown")
    timestamp = parse_datetime(session.get("timestamp"))
    tags = [t.lower() for t in session.get("tags", [])]

    # Build user text only (exclude clary_response/questions)
    user_text_parts = []
    for key in ["user_message", "user_followup"]:
        val = session.get(key, "")
        if val:
            user_text_parts.append(str(val))
    user_text = clean_text(" ".join(user_text_parts))

    # Extract compound entities first
    compound_hits = extract_compound_entities(user_text, COMPOUND_SYMPTOMS)
    compound_positions = [(pos, comp) for comp, positions in compound_hits.items() for pos in positions]
    
    for compound, positions in compound_hits.items():
        # Map compound to appropriate type
        if compound in ["stomach pain", "hair fall", "hair loss", "brain fog", 
                       "back pain", "period cramps", "post-lunch"]:
            event_type = "symptom"
        elif compound in ["late night", "late eating"]:
            event_type = "trigger"
        else:
            event_type = "symptom"
            
        events.append(ExtractedEvent(
            event_id=f"{session_id}_{event_type[:3]}_{compound.replace(' ', '_')}",
            session_id=session_id,
            timestamp=timestamp,
            event_type=event_type,
            entity=compound,
            description=user_text[max(0, positions[0]-40):positions[0]+len(compound)+40],
            severity=_infer_severity(user_text, positions[0]),
            attributes={"source": "user_text", "compound": True}
        ))

    # Extract simple symptoms (excluding compound-covered positions)
    symptom_hits = extract_simple_entities(user_text, SEED_SYMPTOMS, compound_positions)
    for symptom, positions in symptom_hits.items():
        events.append(ExtractedEvent(
            event_id=f"{session_id}_sym_{symptom.replace(' ', '_')}",
            session_id=session_id,
            timestamp=timestamp,
            event_type="symptom",
            entity=symptom,
            description=user_text[max(0, positions[0]-40):positions[0]+len(symptom)+40],
            severity=_infer_severity(user_text, positions[0]),
            attributes={"source": "user_text"}
        ))

    # Extract triggers
    trigger_hits = extract_simple_entities(user_text, SEED_TRIGGERS, compound_positions)
    for trigger, positions in trigger_hits.items():
        events.append(ExtractedEvent(
            event_id=f"{session_id}_trg_{trigger.replace(' ', '_')}",
            session_id=session_id,
            timestamp=timestamp,
            event_type="trigger",
            entity=trigger,
            description=user_text[max(0, positions[0]-40):positions[0]+len(trigger)+40],
            attributes={"source": "user_text"}
        ))

    # Extract from tags (high-confidence signals, only if not already extracted)
    tag_map = {
        # symptoms
        "stomach": "symptom", "acidity": "symptom", "headache": "symptom",
        "back pain": "symptom", "fatigue": "symptom", "dizziness": "symptom",
        "skin": "symptom", "acne": "symptom", "hair fall": "symptom",
        "brain fog": "symptom", "cramps": "symptom", "period": "symptom",
        "mood": "symptom", "anxiety": "symptom", "burning": "symptom",
        "pain": "symptom", "breakout": "symptom",
        # triggers
        "late eating": "trigger", "dehydration": "trigger", "screen time": "trigger",
        "caffeine": "trigger", "busy work day": "trigger", "work pressure": "trigger",
        "diet": "trigger", "intermittent fasting": "trigger", "calorie restriction": "trigger",
        "dairy": "trigger", "dairy increase": "trigger", "dairy reintroduction": "trigger",
        "stress": "trigger", "deadline": "trigger", "sleep deprivation": "trigger",
        "poor sleep": "trigger", "late night screen use": "trigger", "screens": "trigger",
        "protein": "trigger", "carbohydrate": "trigger", "rice": "trigger",
        "sugar": "trigger", "high carb lunch": "trigger",
        # interventions
        "dairy reduction": "intervention", "lunch protein": "intervention",
    }

    for tag in tags:
        if tag in tag_map:
            etype = tag_map[tag]
            already = any(e.entity == tag and e.event_type == etype for e in events)
            if not already:
                events.append(ExtractedEvent(
                    event_id=f"{session_id}_tag_{tag.replace(' ', '_')}",
                    session_id=session_id,
                    timestamp=timestamp,
                    event_type=etype,
                    entity=tag,
                    description=f"Tagged: {tag}",
                    attributes={"source": "tag"}
                ))

    return events


def _infer_severity(text: str, position: int) -> Optional[str]:
    context = text[max(0, position-30):position+30]
    severity_map = {
        "mild": ["mild", "slight", "little", "a bit", "somewhat", "okay"],
        "moderate": ["moderate", "pretty", "quite", "fairly", "bad", "worse", "harder"],
        "severe": ["severe", "extreme", "unbearable", "worst", "really bad", "terrible", "dreading"]
    }
    for severity, indicators in severity_map.items():
        for indicator in indicators:
            if indicator in context:
                return severity
    return None


def extract_all_events(user: Dict[str, Any], use_llm: bool = True) -> List[ExtractedEvent]:
    """Extract all events from a user's conversations."""
    conversations = user.get("conversations", user.get("sessions", []))
    all_events = []
    for session in conversations:
        events = extract_from_session(session)
        all_events.extend(events)
    return all_events
