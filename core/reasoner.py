"""
Main Reasoning Orchestrator for Ask First.
Updated version with pattern cleanup, deduplication, filtering,
and top-quality output selection.
"""

import json
from typing import List, Dict, Any, Generator

from core.models import AnalysisResult, DetectedPattern, PatternEvidence
from core.extractor import extract_all_events
from core.temporal_graph import TemporalEventGraph
from core.pattern_miner import PatternMiner
from core.chunker import TimelineChunker, StreamingContextManager
from core.llm_engine import get_reasoner
from core.confidence import ConfidenceScorer


class HealthReasoningPipeline:
    """
    Production pipeline for cross-conversation health pattern detection.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.chunker = TimelineChunker()
        self.context_manager = StreamingContextManager(self.chunker)
        self.confidence_scorer = ConfidenceScorer()
        self.llm = get_reasoner() if use_llm else None

    def analyze_user(self, user: Dict[str, Any]) -> AnalysisResult:
        user_id = user.get("user_id", "unknown")
        user_name = user.get("name", "Unknown")
        conversations = user.get("conversations", user.get("sessions", []))

        # Stage 1: Extract events
        events = extract_all_events(user)

        # Stage 2: Build graph
        graph = TemporalEventGraph()
        graph.add_events(events)
        graph.build_temporal_links()

        # Stage 3: Deterministic mining
        miner = PatternMiner(graph)
        patterns = miner.mine_all_patterns(user_id)

        # Stage 4: LLM enhancement
        if self.llm and self.llm.available():
            llm_patterns = self._llm_enhancement(user, conversations, patterns)

            # only useful llm patterns
            llm_patterns = [
                p for p in llm_patterns
                if p.confidence_score >= 0.60
            ]

            patterns.extend(llm_patterns)

        # Stage 5: Re-score patterns
        patterns = self._rescore_patterns(patterns, len(conversations))

        # Stage 6: Final cleanup
        patterns = self._final_cleanup(patterns)

        # Sort final
        patterns.sort(key=lambda p: p.confidence_score, reverse=True)

        # Keep best only
        patterns = patterns[:8]

        return AnalysisResult(
            user_id=user_id,
            user_name=user_name,
            total_sessions=len(conversations),
            patterns_found=len(patterns),
            patterns=patterns,
            reasoning_strategy=(
                "Hybrid deterministic + LLM reasoning. "
                "Temporal graph mining detects repeated patterns. "
                "LLM used only for missed cross-session insights."
            ),
            chunking_strategy=(
                "Recent sessions kept verbatim, older sessions compressed "
                "into summaries preserving timeline anchors."
            )
        )

    def analyze_user_streaming(self, user: Dict[str, Any]) -> Generator[str, None, None]:
        result = self.analyze_user(user)

        yield json.dumps({
            "meta": {
                "user_id": result.user_id,
                "user_name": result.user_name,
                "total_sessions": result.total_sessions,
                "patterns_found": result.patterns_found,
                "reasoning_strategy": result.reasoning_strategy,
                "chunking_strategy": result.chunking_strategy
            }
        }) + "\n"

        for pattern in result.patterns:
            yield json.dumps({"pattern": pattern.to_dict()}) + "\n"

        yield json.dumps({"done": True}) + "\n"

    # ----------------------------------------------------
    # LLM Enhancement
    # ----------------------------------------------------
    def _llm_enhancement(
        self,
        user: Dict[str, Any],
        conversations: List[Dict],
        existing_patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:

        from core.timeline import build_user_timeline

        timeline = build_user_timeline(user)
        context = self.context_manager.prepare_reasoning_context(timeline)

        existing_dicts = [p.to_dict() for p in existing_patterns]

        llm_result = self.llm.reason_patterns(
            user_profile=user,
            context=context,
            existing_patterns=existing_dicts
        )

        llm_patterns = []

        for p in llm_result.get("patterns", []):
            try:
                pattern = DetectedPattern(
                    pattern_id=f"llm_{hash(p['pattern_title']) % 10000}",
                    pattern_title=p["pattern_title"],
                    pattern_type=p.get("pattern_type", "temporal"),
                    user_id=user.get("user_id", "unknown"),
                    confidence_label=p.get("confidence_label", "medium"),
                    confidence_score=p.get("confidence_score", 0.60),
                    confidence_justification=p.get(
                        "confidence_justification",
                        "LLM-supported pattern"
                    ),
                    reasoning_trace=p.get("reasoning_trace", ""),
                    medical_latency_note=p.get("medical_latency_note"),
                    evidence=PatternEvidence(
                        session_ids=p.get("sessions_involved", []),
                        counterfactual_sessions=[],
                        temporal_consistency=0.5
                    ),
                    root_cause=p.get("root_cause"),
                    downstream_effects=p.get("downstream_effects", [])
                )

                llm_patterns.append(pattern)

            except Exception:
                continue

        return llm_patterns

    # ----------------------------------------------------
    # Rescore
    # ----------------------------------------------------
    def _rescore_patterns(
        self,
        patterns: List[DetectedPattern],
        total_sessions: int
    ) -> List[DetectedPattern]:

        rescored = []

        for pattern in patterns:
            evidence = pattern.evidence

            occurrence_count = len(evidence.session_ids)
            counterfactual_count = len(evidence.counterfactual_sessions)

            has_resolution = (
                "improv" in pattern.reasoning_trace.lower()
                or "resolv" in pattern.reasoning_trace.lower()
            )

            medical_match = pattern.medical_latency_note is not None

            breakdown = self.confidence_scorer.score(
                occurrence_count=occurrence_count,
                temporal_consistency=evidence.temporal_consistency,
                counterfactual_sessions=counterfactual_count,
                total_sessions=total_sessions,
                has_resolution=has_resolution,
                medical_rule_match=medical_match,
                dose_response=evidence.dose_response_score
            )

            pattern.confidence_score = breakdown.final_score
            pattern.confidence_label = breakdown.final_label.value
            pattern.confidence_justification = breakdown.justification

            rescored.append(pattern)

        return rescored

    # ----------------------------------------------------
    # Final Cleanup
    # ----------------------------------------------------
    def _final_cleanup(
        self,
        patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:

        cleaned = []
        seen = set()

        banned_terms = [
            "progressive symptom cascade",
            "causes cascade",
            "decline over",
            "cascade:"
        ]

        for p in patterns:

            title = p.pattern_title.lower().strip()

            # Remove weak confidence
            if p.confidence_score < 0.55:
                continue

            # Remove weird hallucinated wording
            if any(term in title for term in banned_terms):
                continue

            # Fix wording
            title = title.replace("water", "low hydration")
            title = title.replace("consistently", "")
            title = " ".join(title.split())

            # Duplicate key
            key = title.replace("precedes", "").replace("causes", "").strip()

            if key in seen:
                continue

            seen.add(key)

            p.pattern_title = title.title()
            cleaned.append(p)

        return cleaned


# ----------------------------------------------------
# Public Entry Points
# ----------------------------------------------------
def run_analysis(user: Dict[str, Any], use_llm: bool = True):
    pipeline = HealthReasoningPipeline(use_llm=use_llm)
    return pipeline.analyze_user(user)


def run_analysis_streaming(user: Dict[str, Any], use_llm: bool = True):
    pipeline = HealthReasoningPipeline(use_llm=use_llm)
    yield from pipeline.analyze_user_streaming(user)