"""
Temporal Event Graph for cross-conversation reasoning.
Builds a directed graph where nodes are health events and edges represent
temporal relationships with medical latency awareness.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

from core.models import ExtractedEvent, TemporalLink
from core.preprocess import compute_day_gap


class MedicalLatencyRules:
    """
    Evidence-based medical latency windows for common health phenomena.
    These are rules of thumb, not hardcoded patterns.
    """
    
    RULES = {
        "nutritional_deficiency_hair_fall": {
            "min_days": 42,   # 6 weeks
            "max_days": 84,   # 12 weeks
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
        """Find latency rule that matches a trigger-symptom pair."""
        for rule_name, rule in cls.RULES.items():
            trigger_match = any(t in trigger_entity.lower() for t in rule["triggers"])
            symptom_match = any(s in symptom_entity.lower() for s in rule["symptoms"])
            if trigger_match and symptom_match:
                return rule
        return None
    
    @classmethod
    def get_latency_description(cls, rule_name: str) -> str:
        """Get human-readable latency description."""
        rule = cls.RULES.get(rule_name)
        if not rule:
            return ""
        return f"{rule['description']}. Typical onset: {rule['min_days']}-{rule['max_days']} days."


class TemporalEventGraph:
    """
    Graph structure for temporal health reasoning.
    
    Nodes: ExtractedEvent instances
    Edges: TemporalLink instances with direction and confidence
    """
    
    def __init__(self):
        self.events: Dict[str, ExtractedEvent] = {}
        self.links: List[TemporalLink] = []
        self.event_index_by_type: Dict[str, List[str]] = defaultdict(list)
        self.event_index_by_session: Dict[str, List[str]] = defaultdict(list)
    
    def add_events(self, events: List[ExtractedEvent]) -> None:
        """Add events to the graph and build indexes."""
        for event in events:
            self.events[event.event_id] = event
            self.event_index_by_type[event.event_type].append(event.event_id)
            self.event_index_by_session[event.session_id].append(event.event_id)
    
    def build_temporal_links(self) -> List[TemporalLink]:
        """
        Build all plausible temporal links between events.
        Uses medical latency rules to validate timing.
        """
        links = []
        
        # Get all trigger and symptom events
        trigger_events = [self.events[eid] for eid in self.event_index_by_type["trigger"]]
        symptom_events = [self.events[eid] for eid in self.event_index_by_type["symptom"]]
        intervention_events = [self.events[eid] for eid in self.event_index_by_type["intervention"]]
        
        # Link triggers -> symptoms (forward in time)
        for trigger in trigger_events:
            for symptom in symptom_events:
                # Only link if symptom comes after trigger (or same session)
                gap = self._compute_gap(trigger, symptom)
                if gap is None or gap < -1:  # Allow same-day
                    continue
                
                # Check medical latency rules
                rule = MedicalLatencyRules.find_matching_rule(trigger.entity, symptom.entity)
                
                if rule:
                    # Validate against expected latency window
                    in_window = rule["min_days"] <= gap <= rule["max_days"]
                    if in_window:
                        confidence = 0.75 + min(gap / rule["max_days"], 0.15)
                        links.append(TemporalLink(
                            source_id=trigger.event_id,
                            target_id=symptom.event_id,
                            link_type="causes",
                            day_gap=gap,
                            confidence=min(confidence, 0.95),
                            reasoning=f"{trigger.entity} -> {symptom.entity} "
                                     f"({gap} days, within {rule['min_days']}-{rule['max_days']} day window "
                                     f"for {rule['description']})"
                        ))
                else:
                    # No specific rule - use generic correlation for nearby events
                    if 0 <= gap <= 14:
                        confidence = max(0.3, 0.7 - gap * 0.03)
                        links.append(TemporalLink(
                            source_id=trigger.event_id,
                            target_id=symptom.event_id,
                            link_type="correlates_with",
                            day_gap=gap,
                            confidence=confidence,
                            reasoning=f"{trigger.entity} precedes {symptom.entity} by {gap} days"
                        ))
        
        # Link interventions -> symptom changes (improvement/worsening)
        for intervention in intervention_events:
            for symptom in symptom_events:
                gap = self._compute_gap(intervention, symptom)
                if gap is None:
                    continue
                
                # Intervention can be before symptom (prevented/worsened) or after (resolved)
                if gap >= -1:  # Intervention before or same day as symptom
                    links.append(TemporalLink(
                        source_id=intervention.event_id,
                        target_id=symptom.event_id,
                        link_type="follows",
                        day_gap=gap,
                        confidence=0.5,
                        reasoning=f"{intervention.entity} followed by {symptom.entity}"
                    ))
                else:  # Intervention after symptom (potential resolution)
                    links.append(TemporalLink(
                        source_id=symptom.event_id,
                        target_id=intervention.event_id,
                        link_type="improves",
                        day_gap=abs(gap),
                        confidence=0.55,
                        reasoning=f"{intervention.entity} after {symptom.entity} may indicate intervention"
                    ))
        
        self.links = links
        return links
    
    def find_sequences(self, min_length: int = 3) -> List[List[str]]:
        """
        Find event sequences where the same entities appear repeatedly
        with consistent temporal relationships.
        """
        sequences = []
        
        # Group links by (source_entity, target_entity, link_type)
        link_groups: Dict[Tuple[str, str, str], List[TemporalLink]] = defaultdict(list)
        for link in self.links:
            src = self.events[link.source_id]
            tgt = self.events[link.target_id]
            key = (src.entity, tgt.entity, link.link_type)
            link_groups[key].append(link)
        
        # Find groups with enough repetitions
        for (src_ent, tgt_ent, ltype), links in link_groups.items():
            if len(links) >= min_length:
                # Sort by time
                sorted_links = sorted(links, key=lambda l: l.day_gap if l.day_gap is not None else 0)
                sequences.append([l.source_id for l in sorted_links] + [sorted_links[-1].target_id])
        
        return sequences
    
    def find_dose_response_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect dose-response: more trigger -> more/worse symptom.
        """
        patterns = []
        
        # Group symptom events by entity
        symptom_by_entity: Dict[str, List[ExtractedEvent]] = defaultdict(list)
        for eid in self.event_index_by_type["symptom"]:
            event = self.events[eid]
            symptom_by_entity[event.entity].append(event)
        
        for symptom_entity, symptom_events in symptom_by_entity.items():
            if len(symptom_events) < 3:
                continue
            
            # Find associated triggers for each symptom occurrence
            trigger_counts = []
            for symptom in symptom_events:
                # Find triggers within relevant window before this symptom
                session_id = symptom.session_id
                preceding_events = [
                    self.events[eid] for eid in self.event_index_by_session.get(session_id, [])
                    if self.events[eid].event_type == "trigger"
                ]
                trigger_counts.append({
                    "symptom": symptom,
                    "triggers": preceding_events,
                    "trigger_count": len(preceding_events)
                })
            
            # Check if more triggers correlate with symptom presence
            # This is simplified - in production would use regression
            if len(trigger_counts) >= 3:
                patterns.append({
                    "symptom_entity": symptom_entity,
                    "observations": len(trigger_counts),
                    "type": "dose_response_candidate"
                })
        
        return patterns
    
    def get_counterfactual_sessions(self, trigger_entity: str, symptom_entity: str) -> List[str]:
        """
        Find sessions where trigger was absent but we have data.
        If symptom was also absent, that's counterfactual support.
        """
        # Sessions with trigger
        sessions_with_trigger: Set[str] = set()
        for eid in self.event_index_by_type["trigger"]:
            event = self.events[eid]
            if trigger_entity.lower() in event.entity.lower():
                sessions_with_trigger.add(event.session_id)
        
        # Sessions with symptom
        sessions_with_symptom: Set[str] = set()
        for eid in self.event_index_by_type["symptom"]:
            event = self.events[eid]
            if symptom_entity.lower() in event.entity.lower():
                sessions_with_symptom.add(event.session_id)
        
        # All sessions
        all_sessions = set(self.event_index_by_session.keys())
        
        # Counterfactual: sessions without trigger AND without symptom
        no_trigger = all_sessions - sessions_with_trigger
        no_symptom = all_sessions - sessions_with_symptom
        
        counterfactual = list(no_trigger & no_symptom)
        return counterfactual
    
    def _compute_gap(self, event1: ExtractedEvent, event2: ExtractedEvent) -> Optional[int]:
        """Compute day gap between two events. Positive if event2 after event1."""
        if not event1.timestamp or not event2.timestamp:
            # Same session = 0 gap
            if event1.session_id == event2.session_id:
                return 0
            return None
        delta = event2.timestamp - event1.timestamp
        return delta.days
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph for debugging."""
        return {
            "events": {eid: {
                "type": e.event_type,
                "entity": e.entity,
                "session": e.session_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None
            } for eid, e in self.events.items()},
            "links": [{
                "source": l.source_id,
                "target": l.target_id,
                "type": l.link_type,
                "gap_days": l.day_gap,
                "confidence": l.confidence
            } for l in self.links]
        }