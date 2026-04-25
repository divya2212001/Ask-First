"""
Advanced confidence scoring with counterfactual analysis,
temporal consistency, and medical plausibility checks.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence decomposition for transparency."""
    repeated_evidence_score: float
    temporal_consistency_score: float
    counterfactual_support_score: float
    medical_plausibility_score: float
    resolution_evidence_score: float
    dose_response_score: float
    final_score: float
    final_label: ConfidenceLevel
    justification: str


class ConfidenceScorer:
    """
    Multi-factor confidence scoring system.
    
    Factors:
    1. Repeated Evidence: How many times was the pattern observed?
    2. Temporal Consistency: Are the time gaps between trigger and effect consistent?
    3. Counterfactual Support: Does the effect disappear when the trigger is absent?
    4. Medical Plausibility: Does the latency match known medical phenomena?
    5. Resolution Evidence: Does the effect improve when trigger is removed?
    6. Dose Response: Does more trigger = more effect?
    """
    
    def __init__(self):
        self.weights = {
            "repeated_evidence": 0.20,
            "temporal_consistency": 0.20,
            "counterfactual": 0.20,
            "medical_plausibility": 0.15,
            "resolution": 0.15,
            "dose_response": 0.10
        }
    
    def score(
        self,
        occurrence_count: int,
        temporal_consistency: float,
        counterfactual_sessions: int,
        total_sessions: int,
        has_resolution: bool,
        medical_rule_match: bool,
        dose_response: Optional[float] = None
    ) -> ConfidenceBreakdown:
        """
        Compute comprehensive confidence score.
        """
        # 1. Repeated Evidence
        repeated_score = min(occurrence_count * 0.22, 0.90)
        
        # 2. Temporal Consistency
        temporal_score = temporal_consistency
        
        # 3. Counterfactual Support
        if total_sessions > 0:
            counterfactual_ratio = counterfactual_sessions / total_sessions
            counterfactual_score = min(counterfactual_ratio * 2, 0.95)
        else:
            counterfactual_score = 0.0
        
        # 4. Medical Plausibility
        medical_score = 0.85 if medical_rule_match else 0.55
        
        # 5. Resolution Evidence
        resolution_score = 0.90 if has_resolution else 0.40
        
        # 6. Dose Response
        dose_score = dose_response if dose_response is not None else 0.50
        
        # Weighted sum
        final = (
            repeated_score * self.weights["repeated_evidence"] +
            temporal_score * self.weights["temporal_consistency"] +
            counterfactual_score * self.weights["counterfactual"] +
            medical_score * self.weights["medical_plausibility"] +
            resolution_score * self.weights["resolution"] +
            dose_score * self.weights["dose_response"]
        )
        
        # Bonus for multiple strong signals
        strong_signals = sum([
            repeated_score > 0.7,
            temporal_score > 0.7,
            counterfactual_score > 0.6,
            medical_score > 0.8,
            resolution_score > 0.8,
            dose_score > 0.7
        ])
        final += min(strong_signals * 0.03, 0.1)
        final = min(final, 0.98)
        
        # Determine label
        if final >= 0.82:
            label = ConfidenceLevel.HIGH
        elif final >= 0.60:
            label = ConfidenceLevel.MEDIUM
        else:
            label = ConfidenceLevel.LOW
        
        # Build justification
        parts = []
        if occurrence_count >= 4:
            parts.append(f"{occurrence_count} repeated episodes")
        elif occurrence_count >= 2:
            parts.append(f"{occurrence_count} episodes observed")
        
        if temporal_consistency > 0.8:
            parts.append("highly consistent timing")
        elif temporal_consistency > 0.6:
            parts.append("moderately consistent timing")
        
        if counterfactual_sessions >= 2:
            parts.append(f"{counterfactual_sessions} counterfactual clean sessions")
        
        if medical_rule_match:
            parts.append("matches known medical latency")
        
        if has_resolution:
            parts.append("symptom resolved after intervention")
        
        if dose_response and dose_response > 0.6:
            parts.append("dose-response relationship evident")
        
        justification = "; ".join(parts) + "." if parts else "Limited evidence, low confidence."
        
        return ConfidenceBreakdown(
            repeated_evidence_score=round(repeated_score, 2),
            temporal_consistency_score=round(temporal_score, 2),
            counterfactual_support_score=round(counterfactual_score, 2),
            medical_plausibility_score=round(medical_score, 2),
            resolution_evidence_score=round(resolution_score, 2),
            dose_response_score=round(dose_score, 2),
            final_score=round(final, 2),
            final_label=label,
            justification=justification
        )
