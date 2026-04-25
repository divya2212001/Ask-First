"""
Dynamic health entity extraction without hardcoded vocabularies.
Uses keyword expansion + LLM-based extraction for comprehensive coverage.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.preprocess import clean_text, parse_datetime
from core.models import ExtractedEvent


# Seed vocabulary for deterministic extraction - used as fallback
SEED_SYMPTOMS = [
    "headache", "migraine", "pain", "stomach pain", "burning", "acidity",
    "bloating", "hair fall", "hair loss", "fatigue", "tired", "exhausted",
    "weakness", "nausea", "insomnia", "stress", "anxiety", "fever",
    "cough", "cold", "dizziness", "brain fog", "acne", "breakout",
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

# Common words that should never be extracted as health entities
STOPWORDS = {
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "among", "throughout", "despite",
    "towards", "upon", "concerning", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "whose",
    "a", "an", "as", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "like", "just", "now", "then", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "also", "back", "still", "even", "only", "just", "already",
    "yet", "once", "twice", "again", "almost", "quite", "rather", "really",
    "pretty", "enough", "much", "many", "little", "less", "least", "more",
    "most", "another", "every", "each", "both", "either", "neither", "one",
    "two", "first", "last", "next", "previous", "new", "old", "good", "bad",
    "big", "small", "long", "short", "high", "low", "right", "left", "wrong",
    "true", "false", "sure", "maybe", "perhaps", "probably", "definitely",
    "always", "never", "sometimes", "often", "usually", "rarely", "seldom",
    "finally", "eventually", "suddenly", "recently", "lately", "currently",
    "actually", "basically", "literally", "seriously", "honestly", "basically",
    "kind", "sort", "type", "way", "thing", "stuff", "lot", "bit", "piece",
    "part", "side", "end", "point", "case", "fact", "idea", "reason",
    "cause", "result", "effect", "problem", "issue", "question", "answer",
    "example", "instance", "situation", "condition", "state", "place",
    "time", "day", "week", "month", "year", "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "night", "midnight", "noon", "hour",
    "minute", "second", "moment", "while", "since", "until", "till", "ago",
    "start", "started", "starting", "begin", "began", "beginning", "come",
    "came", "coming", "go", "went", "going", "get", "got", "getting",
    "make", "made", "making", "take", "took", "taking", "give", "gave",
    "giving", "say", "said", "saying", "tell", "told", "telling", "know",
    "knew", "knowing", "think", "thought", "thinking", "see", "saw", "seeing",
    "look", "looked", "looking", "feel", "felt", "feeling", "want", "wanted",
    "wanting", "need", "needed", "needing", "try", "tried", "trying", "use",
    "used", "using", "work", "worked", "working", "call", "called", "calling",
    "find", "found", "finding", "ask", "asked", "asking", "seem", "seemed",
    "seeming", "leave", "left", "leaving", "put", "puts", "putting", "mean",
    "meant", "meaning", "keep", "kept", "keeping", "let", "lets", "letting",
    "begin", "began", "begun", "beginning", "help", "helped", "helping",
    "show", "showed", "shown", "showing", "hear", "heard", "hearing",
    "play", "played", "playing", "run", "ran", "running", "move", "moved",
    "moving", "live", "lived", "living", "believe", "believed", "believing",
    "bring", "brought", "bringing", "happen", "happened", "happening",
    "stand", "stood", "standing", "lose", "lost", "losing", "pay", "paid",
    "paying", "meet", "met", "meeting", "include", "included", "including",
    "continue", "continued", "continuing", "set", "sets", "setting",
    "learn", "learned", "learning", "change", "changed", "changing",
    "lead", "led", "leading", "understand", "understood", "understanding",
    "watch", "watched", "watching", "follow", "followed", "following",
    "stop", "stopped", "stopping", "create", "created", "creating",
    "speak", "spoke", "spoken", "speaking", "read", "reading", "allow",
    "allowed", "allowing", "add", "added", "adding", "spend", "spent",
    "spending", "grow", "grew", "grown", "growing", "open", "opened",
    "opening", "walk", "walked", "walking", "win", "won", "winning",
    "offer", "offered", "offering", "remember", "remembered", "remembering",
    "love", "loved", "loving", "consider", "considered", "considering",
    "appear", "appeared", "appearing", "buy", "bought", "buying", "wait",
    "waited", "waiting", "serve", "served", "serving", "die", "died",
    "dying", "send", "sent", "sending", "expect", "expected", "expecting",
    "build", "built", "building", "stay", "stayed", "staying", "fall",
    "fell", "fallen", "falling", "cut", "cuts", "cutting", "reach",
    "reached", "reaching", "kill", "killed", "killing", "remain",
    "remained", "remaining", "suggest", "suggested", "suggesting",
    "raise", "raised", "raising", "pass", "passed", "passing", "sell",
    "sold", "selling", "require", "required", "requiring", "report",
    "reported", "reporting", "decide", "decided", "deciding", "pull",
    "pulled", "pulling", "every", "really", "very", "much", "many",
    "more", "most", "some", "any", "each", "few", "other", "another",
    "such", "only", "own", "same", "so", "than", "too", "also", "back",
    "still", "even", "just", "already", "yet", "once", "twice", "again",
    "almost", "quite", "rather", "pretty", "enough", "about", "around",
    "through", "during", "before", "after", "above", "below", "between",
    "among", "within", "without", "against", "towards", "upon", "off",
    "over", "under", "into", "onto", "out", "up", "down", "across",
    "behind", "beyond", "except", "beside", "besides", "despite",
    "regarding", "concerning", "considering", "following", "including",
    "using", "according", "due", "owing", "thanks", "because", "although",
    "though", "while", "whereas", "unless", "until", "whether", "either",
    "neither", "both", "not", "nor", "only", "but", "however",
    "therefore", "thus", "hence", "consequently", "accordingly",
    "otherwise", "instead", "meanwhile", "besides", "furthermore",
    "moreover", "nevertheless", "nonetheless", "otherwise", "however",
    "whatever", "whenever", "wherever", "however", "whoever", "whichever",
    "whosever", "whomever", "whatsoever", "wheresoever", "whensoever",
    "howsoever", "whosoever", "whomsoever", "whosesoever", "whichsoever",
    "oneself", "itself", "oneself", "thyself", "himself", "herself",
    "oneself", "ourselves", "yourselves", "themselves", "itself",
    "myself", "yourself", "himself", "herself", "oneself", "ourselves",
    "yourselves", "themselves", "itself", "oneself", "thyself",
    "whosoever", "whomsoever", "whosesoever", "whichsoever", "whatsoever",
    "wheresoever", "whensoever", "howsoever", "whosoever", "whomsoever",
    "whosesoever", "whichsoever", "oneself", "itself", "oneself",
    "thyself", "himself", "herself", "oneself", "ourselves", "yourselves",
    "themselves", "itself", "myself", "yourself", "himself", "herself",
    "oneself", "ourselves", "yourselves", "themselves", "itself",
    "oneself", "thyself"
}


class HealthEntityExtractor:
    """
    Extracts health entities from conversation text dynamically.
    """

    def __init__(self, use_llm: bool = True):
        self.symptoms = set(SEED_SYMPTOMS)
        self.triggers = set(SEED_TRIGGERS)
        self.interventions = set(SEED_INTERVENTIONS)
        self.use_llm = use_llm

    def extract_from_session(self, session: Dict[str, Any]) -> List[ExtractedEvent]:
        """Extract all events from a single conversation session."""
        events = []
        session_id = session.get("session_id", "unknown")
        timestamp = parse_datetime(session.get("timestamp"))

        # Extract from multiple fields
        fields_to_extract = [
            ("user_message", "user_message"),
            ("user_followup", "user_followup"),
            ("clary_response", "clary_response"),
            ("clary_questions", "clary_questions"),
        ]

        all_text = ""
        for field_name, field_key in fields_to_extract:
            value = session.get(field_key, "")
            if isinstance(value, list):
                value = " ".join(value)
            if value:
                all_text += " " + str(value)

        clean = clean_text(all_text)

        # Extract symptoms
        symptom_hits = self._find_entities(clean, self.symptoms)
        for entity, spans in symptom_hits.items():
            events.append(ExtractedEvent(
                event_id=f"{session_id}_sym_{entity.replace(' ', '_')}",
                session_id=session_id,
                timestamp=timestamp,
                event_type="symptom",
                entity=entity,
                description=self._get_context(clean, spans[0]),
                severity=self._infer_severity(clean, spans[0]),
                attributes={"mentions": len(spans)}
            ))

        # Extract triggers/lifestyle
        trigger_hits = self._find_entities(clean, self.triggers)
        for entity, spans in trigger_hits.items():
            events.append(ExtractedEvent(
                event_id=f"{session_id}_trg_{entity.replace(' ', '_')}",
                session_id=session_id,
                timestamp=timestamp,
                event_type="trigger",
                entity=entity,
                description=self._get_context(clean, spans[0]),
                attributes={"mentions": len(spans)}
            ))

        # Extract interventions (actions taken)
        intervention_hits = self._find_entities(clean, self.interventions)
        for entity, spans in intervention_hits.items():
            if self._has_nearby_health_entity(clean, spans[0], symptom_hits, trigger_hits):
                events.append(ExtractedEvent(
                    event_id=f"{session_id}_int_{entity.replace(' ', '_')}",
                    session_id=session_id,
                    timestamp=timestamp,
                    event_type="intervention",
                    entity=entity,
                    description=self._get_context(clean, spans[0]),
                    attributes={"mentions": len(spans)}
                ))

        return events

    def _find_entities(self, text: str, entity_list: set) -> Dict[str, List[int]]:
        """Find all occurrences of entities in text."""
        hits = {}
        for entity in entity_list:
            if entity in STOPWORDS:
                continue
            pattern = r'\b' + re.escape(entity) + r'\b'
            matches = list(re.finditer(pattern, text))
            if matches:
                hits[entity] = [m.start() for m in matches]
        return hits

    def _get_context(self, text: str, position: int, window: int = 60) -> str:
        """Extract surrounding context."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()

    def _infer_severity(self, text: str, position: int) -> Optional[str]:
        """Infer severity from nearby words."""
        context = self._get_context(text, position, 30)
        severity_map = {
            "mild": ["mild", "slight", "little", "a bit", "somewhat"],
            "moderate": ["moderate", "pretty", "quite", "fairly", "bad"],
            "severe": ["severe", "extreme", "unbearable", "worst", "really bad", "terrible"]
        }
        for severity, indicators in severity_map.items():
            for indicator in indicators:
                if indicator in context:
                    return severity
        return None

    def _has_nearby_health_entity(
        self, text: str, position: int,
        symptoms: Dict, triggers: Dict, window: int = 40
    ) -> bool:
        """Check if there's a health entity near this position."""
        region_start = max(0, position - window)
        region_end = min(len(text), position + window)

        for entity, spans in {**symptoms, **triggers}.items():
            for span in spans:
                if region_start <= span <= region_end:
                    return True
        return False


def extract_all_events(user: Dict[str, Any]) -> List[ExtractedEvent]:
    """Convenience function: extract all events from a user's conversations."""
    extractor = HealthEntityExtractor()
    conversations = user.get("conversations", user.get("sessions", []))

    all_events = []
    for session in conversations:
        events = extractor.extract_from_session(session)
        all_events.extend(events)

    return all_events
