"""
Production Grade Pattern Miner
Cleaned version focused on credible temporal reasoning.
Removes hallucinated cascade patterns.
"""

import uuid
from typing import List, Dict, Tuple
from collections import defaultdict

from core.models import (
    ExtractedEvent,
    TemporalLink,
    DetectedPattern,
    PatternEvidence
)
from core.temporal_graph import TemporalEventGraph, MedicalLatencyRules


class PatternMiner:
    """
    Reliable pattern mining engine.

    Detects:
    1. Repeated trigger -> symptom temporal patterns
    2. Intervention -> symptom improvement
    3. Strong repeated trends only
    """

    def __init__(self, graph: TemporalEventGraph):
        self.graph = graph
        self.patterns: List[DetectedPattern] = []

    # ----------------------------------------------------
    # Main Entry
    # ----------------------------------------------------
    def mine_all_patterns(self, user_id: str) -> List[DetectedPattern]:
        self.patterns = []

        self._mine_temporal_sequences(user_id)
        self._mine_intervention_responses(user_id)

        self._deduplicate_patterns()
        self._sort_patterns()

        return self.patterns

    # ----------------------------------------------------
    # 1. Trigger -> Symptom Repeated
    # ----------------------------------------------------
    def _mine_temporal_sequences(self, user_id: str):

        pair_links: Dict[Tuple[str, str], List[TemporalLink]] = defaultdict(list)

        for link in self.graph.links:
            if link.link_type in ("causes", "correlates_with"):
                src = self.graph.events[link.source_id]
                tgt = self.graph.events[link.target_id]

                pair_links[(src.entity, tgt.entity)].append(link)

        for (trigger, symptom), links in pair_links.items():

            # stricter threshold
            if len(links) < 3:
                continue

            sorted_links = sorted(links, key=lambda x: x.day_gap or 0)

            session_ids = list(set(
                self.graph.events[l.source_id].session_id
                for l in sorted_links
            ) | set(
                self.graph.events[l.target_id].session_id
                for l in sorted_links
            ))

            counterfactual = self.graph.get_counterfactual_sessions(
                trigger, symptom
            )

            gaps = [
                l.day_gap for l in sorted_links
                if l.day_gap is not None
            ]

            consistency = self._compute_temporal_consistency(gaps)

            # require some consistency
            if consistency < 0.60:
                continue

            confidence = self._score_pattern(
                occurrences=len(links),
                consistency=consistency,
                counterfactual=len(counterfactual)
            )

            # medical rule support
            latency_note = self._get_latency_note(trigger, symptom)

            reasoning = self._build_reasoning(
                trigger, symptom, len(links), gaps, counterfactual
            )

            self.patterns.append(
                DetectedPattern(
                    pattern_id=str(uuid.uuid4())[:8],
                    pattern_title=f"{trigger.title()} precedes {symptom}",
                    pattern_type="temporal_sequence",
                    user_id=user_id,
                    confidence_label=self._label(confidence),
                    confidence_score=round(confidence, 2),
                    confidence_justification=(
                        f"{len(links)} repeated episodes with "
                        f"{len(counterfactual)} clean counterfactual sessions."
                    ),
                    reasoning_trace=reasoning,
                    medical_latency_note=latency_note,
                    evidence=PatternEvidence(
                        event_ids=[l.source_id for l in sorted_links],
                        session_ids=session_ids,
                        counterfactual_sessions=counterfactual,
                        temporal_consistency=round(consistency, 2)
                    ),
                    root_cause=trigger,
                    downstream_effects=[symptom]
                )
            )

    # ----------------------------------------------------
    # 2. Intervention Improvement
    # ----------------------------------------------------
    def _mine_intervention_responses(self, user_id: str):

        improve_links = [
            l for l in self.graph.links
            if l.link_type == "improves"
        ]

        grouped = defaultdict(list)

        for link in improve_links:
            intervention = self.graph.events[link.target_id]
            grouped[intervention.entity].append(link)

        for intervention, links in grouped.items():

            if len(links) < 2:
                continue

            symptoms = set()
            sessions = []

            for link in links:
                symptom = self.graph.events[link.source_id]
                symptoms.add(symptom.entity)
                sessions.append(symptom.session_id)

            confidence = min(0.65 + len(links) * 0.05, 0.90)

            self.patterns.append(
                DetectedPattern(
                    pattern_id=str(uuid.uuid4())[:8],
                    pattern_title=f"{intervention.title()} improves symptoms",
                    pattern_type="intervention_response",
                    user_id=user_id,
                    confidence_label=self._label(confidence),
                    confidence_score=round(confidence, 2),
                    confidence_justification=(
                        f"Improvement observed after {intervention} "
                        f"in {len(links)} sessions."
                    ),
                    reasoning_trace=(
                        f"After {intervention}, symptoms reduced: "
                        f"{', '.join(symptoms)}."
                    ),
                    evidence=PatternEvidence(
                        event_ids=[l.target_id for l in links],
                        session_ids=list(set(sessions)),
                        counterfactual_sessions=[],
                        temporal_consistency=0.70
                    ),
                    root_cause=None,
                    downstream_effects=list(symptoms)
                )
            )

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _score_pattern(self, occurrences, consistency, counterfactual):

        score = 0.45
        score += min(occurrences * 0.10, 0.25)
        score += consistency * 0.20
        score += min(counterfactual * 0.03, 0.10)

        return min(score, 0.95)

    def _label(self, score):

        if score >= 0.85:
            return "high"
        elif score >= 0.65:
            return "medium"
        return "low"

    def _compute_temporal_consistency(self, gaps):

        if len(gaps) < 2:
            return 0.65

        mean_gap = sum(gaps) / len(gaps)

        variance = sum(
            (g - mean_gap) ** 2 for g in gaps
        ) / len(gaps)

        std = variance ** 0.5

        consistency = max(0.50, 1 - (std / 14))

        return consistency

    def _get_latency_note(self, trigger, symptom):

        rule = MedicalLatencyRules.find_matching_rule(
            trigger, symptom
        )

        if not rule:
            return None

        rule_name = next(
            (
                k for k, v
                in MedicalLatencyRules.RULES.items()
                if v == rule
            ),
            None
        )

        if rule_name:
            return MedicalLatencyRules.get_latency_description(
                rule_name
            )

        return None

    def _build_reasoning(
        self,
        trigger,
        symptom,
        count,
        gaps,
        counterfactual
    ):

        gap_text = (
            f"{min(gaps)}-{max(gaps)} days"
            if gaps else "same session"
        )

        text = (
            f"{trigger.title()} appeared in {count} sessions "
            f"and was followed by {symptom} within {gap_text}. "
        )

        if counterfactual:
            text += (
                f"In {len(counterfactual)} sessions without "
                f"{trigger}, {symptom} was absent. "
            )

        text += "Repeated timing pattern supports association."

        return text

    # ----------------------------------------------------
    # Cleanup
    # ----------------------------------------------------
    def _deduplicate_patterns(self):

        unique = []
        seen = set()

        for p in self.patterns:

            key = (
                p.root_cause,
                tuple(sorted(p.downstream_effects))
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(p)

        self.patterns = unique

    def _sort_patterns(self):
        self.patterns.sort(
            key=lambda x: x.confidence_score,
            reverse=True
        )