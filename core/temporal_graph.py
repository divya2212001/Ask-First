"""
Temporal Event Graph for cross-conversation reasoning.
Builds a directed graph where nodes are health events and edges represent
temporal relationships with medical latency awareness.
Only links triggers to symptoms when both appear in the SAME session's user text.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from core.models import ExtractedEvent, TemporalLink
from core.preprocess import compute_day_gap


class MedicalLatencyRules:
    """
    Evidence-based medical latency windows for common health phenomena.
    """
    
    RULES = {
        "nutritional_deficiency_hair_fall": {
            "min_days": 42,
            "max_days": 84,
            "description": "Telogen effluvium from severe calorie restriction",
            "symptoms": ["hair fall", "hair loss"],
            "triggers": ["calorie", "fasting", "diet", "restriction"]
        },
        "dietary_acne": {
            "min_days": 1,
            "max_days": 4,
            "description": "Dairy/high-glycemic food triggers acne",
            "symptoms": ["acne", "breakout"],
            "triggers": ["dairy", "milk", "paneer", "yogurt", "sugar"]
        },
        "late_eating_acidity": {
            "min_days": 0,
            "max_days": 1,
            "description": "Late night eating causes next-day acidity",
            "symptoms": ["acidity", "stomach pain", "burning", "bloating"],
            "triggers": ["late eating", "late night", "midnight"]
        },
        "dehydration_headache": {
            "min_days": 0,
            "max_days": 2,
            "description": "Chronic low water intake causes headaches",
            "symptoms": ["headache", "migraine"],
            "triggers": ["water", "hydration", "dehydration"]
        },
        "sleep_debt_cumulative": {
            "min_days": 14,
            "max_days": 56,
            "description": "Chronic sleep deprivation builds symptoms over weeks",
            "symptoms": ["fatigue", "anxiety", "mood", "brain fog", "cramps"],
            "triggers": ["sleep", "late night", "screen", "phone"]
        },
        "post_meal_blood_sugar": {
            "min_days": 0,
            "max_days": 1,
            "description": "High carb meal causes same-day energy crash",
            "symptoms": ["fatigue", "tired", "sleepy", "drowsy"],
            "triggers": ["carbohydrate", "rice", "sugar", "protein"]
        },
        "stress_menstrual": {
            "min_days": 7,
            "max_days": 21,
            "description": "Elevated cortisol affects menstrual symptoms",
            "symptoms": ["cramps", "cramp", "period"],
            "triggers": ["stress", "deadline", "work", "sleep"]
        }
    }
    
    @classmethod
    def find_matching_rule(cls, trigger_entity: str, symptom_entity: str) -> Optional[Dict[str, Any]]:
        for rule_name, rule in cls.RULES.items():
            trigger_match = any(t in trigger_entity.lower() for t in rule["triggers"])
            symptom_match = any(s in symptom_entity.lower() for s in rule["symptoms"])
            if trigger_match and symptom_match:
                return rule
        return None
    
    @classmethod
    def get_latency_description(cls, rule_name: str) -> str:
        rule = cls.RULES.get(rule_name)
        if not rule:
            return ""
        return f"{rule['description']}. Typical onset: {rule['min_days']}-{rule['max_days']} days."


class TemporalEventGraph:
    """
    Graph structure for temporal health reasoning.
    Only links triggers to symptoms when BOTH appear in the same session.
    """
    
    def __init__(self):
        self.events: Dict[str, ExtractedEvent] = {}
        self.links: List[TemporalLink] = []
        self.event_index_by_type: Dict[str, List[str]] = defaultdict(list)
        self.event_index_by_session: Dict[str, List[str]] = defaultdict(list)
        self.session_events: Dict[str, List[ExtractedEvent]] = defaultdict(list)
    
    def add_events(self, events: List[ExtractedEvent]) -> None:
        for event in events:
            self.events[event.event_id] = event
            self.event_index_by_type[event.event_type].append(event.event_id)
            self.event_index_by_session[event.session_id].append(event.event_id)
            self.session_events[event.session_id].append(event)
    
    def build_temporal_links(self) -> List[TemporalLink]:
        """
        Build temporal links between trigger and symptom events.
        Only link when both appear in the SAME session (co-occurrence).
        Cross-session links are handled by pattern miner, not here.
        """
        links = []
        
        for session_id, session_events in self.session_events.items():
            triggers = [e for e in session_events if e.event_type == "trigger"]
            symptoms = [e for e in session_events if e.event_type == "symptom"]
            interventions = [e for e in session_events if e.event_type == "intervention"]
            
            # Link triggers -> symptoms within same session
            for trigger in triggers:
                for symptom in symptoms:
                    rule = MedicalLatencyRules.find_matching_rule(trigger.entity, symptom.entity)
                    if rule:
                        # Same-session = 0 day gap, always within latency window
                        confidence = 0.80
                        links.append(TemporalLink(
                            source_id=trigger.event_id,
                            target_id=symptom.event_id,
                            link_type="causes",
                            day_gap=0,
                            confidence=min(confidence, 0.95),
                            reasoning=f"{trigger.entity} -> {symptom.entity} in same session (latency rule: {rule['description']})"
                        ))
            
            # Link interventions -> symptoms within same session
            for intervention in interventions:
                for symptom in symptoms:
                    links.append(TemporalLink(
                        source_id=intervention.event_id,
                        target_id=symptom.event_id,
                        link_type="improves",
                        day_gap=0,
                        confidence=0.55,
                        reasoning=f"{intervention.entity} in same session as {symptom.entity}"
                    ))
        
        self.links = links
        return links
    
    def get_events_for_session(self, session_id: str) -> List[ExtractedEvent]:
        return self.session_events.get(session_id, [])
    
    def get_counterfactual_sessions(self, trigger_entity: str, symptom_entity: str) -> List[str]:
        """
        Find sessions where trigger was absent but we have data.
        If symptom was also absent, that's counterfactual support.
        """
        sessions_with_trigger = set()
        for eid in self.event_index_by_type["trigger"]:
            event = self.events[eid]
            if trigger_entity.lower() in event.entity.lower():
                sessions_with_trigger.add(event.session_id)
        
        sessions_with_symptom = set()
        for eid in self.event_index_by_type["symptom"]:
            event = self.events[eid]
            if symptom_entity.lower() in event.entity.lower():
                sessions_with_symptom.add(event.session_id)
        
        all_sessions = set(self.event_index_by_session.keys())
        no_trigger = all_sessions - sessions_with_trigger
        no_symptom = all_sessions - sessions_with_symptom
        
        return list(no_trigger & no_symptom)
    
    def _compute_gap(self, event1: ExtractedEvent, event2: ExtractedEvent) -> Optional[int]:
        if not event1.timestamp or not event2.timestamp:
            if event1.session_id == event2.session_id:
                return 0
            return None
        delta = event2.timestamp - event1.timestamp
        return delta.days

