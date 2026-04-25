"""
Pydantic models for the Ask First reasoning pipeline.
Ensures type safety and structured output across the system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ExtractedEvent(BaseModel):
    """A single health event extracted from a conversation session."""
    event_id: str = Field(..., description="Unique identifier for this event")
    session_id: str = Field(..., description="Source session ID")
    timestamp: Optional[datetime] = Field(None, description="When the event occurred")
    event_type: Literal["symptom", "trigger", "intervention", "lifestyle", "context"] = Field(
        ..., description="Category of the event"
    )
    entity: str = Field(..., description="The specific health entity (e.g., 'hair fall', 'dairy')")
    description: str = Field(..., description="Full context of the event")
    severity: Optional[str] = Field(None, description="Severity if applicable")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional structured data")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            from dateutil import parser
            return parser.parse(v)
        return v


class TemporalLink(BaseModel):
    """A directional temporal relationship between two events."""
    source_id: str = Field(..., description="ID of the causing/preceding event")
    target_id: str = Field(..., description="ID of the resulting/subsequent event")
    link_type: Literal[
        "causes", "correlates_with", "precedes", "follows",
        "worsens", "improves", "dose_response", "intervention_response"
    ] = Field(..., description="Nature of the temporal relationship")
    day_gap: Optional[int] = Field(None, description="Days between source and target")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Strength of the link")
    reasoning: str = Field(..., description="Why this link exists")


class PatternEvidence(BaseModel):
    """Evidence supporting a detected pattern."""
    event_ids: List[str] = Field(default_factory=list, description="Events that support this")
    session_ids: List[str] = Field(default_factory=list, description="Sessions involved")
    counterfactual_sessions: List[str] = Field(
        default_factory=list,
        description="Sessions where trigger was absent and effect did not occur"
    )
    temporal_consistency: float = Field(..., ge=0.0, le=1.0, description="How consistent timing is")
    dose_response_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Strength of dose-response relationship"
    )


class DetectedPattern(BaseModel):
    """A cross-conversation health pattern with full reasoning."""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_title: str = Field(..., description="Human-readable pattern name")
    pattern_type: Literal[
        "temporal_sequence", "dose_response", "intervention_response",
        "compound_cause", "cyclic_pattern", "progressive_decline"
    ] = Field(..., description="Classification of pattern type")
    user_id: str = Field(..., description="User this pattern belongs to")
    
    confidence_label: Literal["high", "medium", "low"] = Field(..., description="Qualitative confidence")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Quantitative confidence 0-1")
    confidence_justification: str = Field(
        ..., description="One-line justification for the confidence score"
    )
    
    reasoning_trace: str = Field(
        ..., description="Full chain-of-thought reasoning with temporal awareness"
    )
    medical_latency_note: Optional[str] = Field(
        None, description="Medical explanation of any time delay (e.g., telogen effluvium 8-12 weeks)"
    )
    
    evidence: PatternEvidence = Field(..., description="Structured evidence")
    root_cause: Optional[str] = Field(None, description="Identified root cause if applicable")
    downstream_effects: List[str] = Field(default_factory=list, description="Symptoms caused by root")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "pattern_id": self.pattern_id,
            "pattern": self.pattern_title,
            "pattern_type": self.pattern_type,
            "user_id": self.user_id,
            "confidence": self.confidence_label,
            "confidence_score": round(self.confidence_score, 2),
            "confidence_justification": self.confidence_justification,
            "reasoning": self.reasoning_trace,
            "medical_latency_note": self.medical_latency_note,
            "evidence": {
                "sessions": self.evidence.session_ids,
                "counterfactual_sessions": self.evidence.counterfactual_sessions,
                "temporal_consistency": round(self.evidence.temporal_consistency, 2),
                "dose_response_score": round(self.evidence.dose_response_score, 2) 
                    if self.evidence.dose_response_score is not None else None
            },
            "root_cause": self.root_cause,
            "downstream_effects": self.downstream_effects
        }


class AnalysisResult(BaseModel):
    """Top-level output of the pattern detection pipeline."""
    user_id: str
    user_name: Optional[str] = None
    total_sessions: int = 0
    patterns_found: int = 0
    patterns: List[DetectedPattern] = Field(default_factory=list)
    reasoning_strategy: str = Field(..., description="Description of the reasoning approach used")
    chunking_strategy: str = Field(..., description="How context was managed")
    
    def to_streaming_json(self):
        """Yield patterns one by one for streaming output."""
        import json
        yield json.dumps({
            "meta": {
                "user_id": self.user_id,
                "user_name": self.user_name,
                "total_sessions": self.total_sessions,
                "patterns_found": self.patterns_found,
                "reasoning_strategy": self.reasoning_strategy,
                "chunking_strategy": self.chunking_strategy
            }
        }) + "\n"
        
        for pattern in self.patterns:
            yield json.dumps({"pattern": pattern.to_dict()}) + "\n"
        
        yield json.dumps({"done": True}) + "\n"

