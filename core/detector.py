"""
Pattern Detector - Production Entry Point.
All hardcoded user-specific logic has been removed.
Uses the dynamic HealthReasoningPipeline for pattern detection.
"""

from typing import Dict, Any, List

from core.reasoner import run_analysis, run_analysis_streaming
from core.models import AnalysisResult


def detect_patterns(user: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect cross-conversation health patterns dynamically.
    
    This is the main entry point called by the UI.
    All detection is now handled by the HealthReasoningPipeline
    which uses temporal graph mining + LLM enhancement.
    
    Args:
        user: Full user profile including conversations
        timeline: Pre-built timeline (optional, kept for API compatibility)
    
    Returns:
        Dict with patterns and metadata in JSON-serializable format
    """
    # Run the full reasoning pipeline
    result: AnalysisResult = run_analysis(user, use_llm=True)
    
    # Convert to dict for JSON serialization
    return {
        "user_id": result.user_id,
        "name": result.user_name,
        "total_sessions": result.total_sessions,
        "patterns_found": result.patterns_found,
        "patterns": [p.to_dict() for p in result.patterns],
        "reasoning_strategy": result.reasoning_strategy,
        "chunking_strategy": result.chunking_strategy
    }


def detect_patterns_streaming(user: Dict[str, Any]) -> str:
    """
    Stream pattern detection results as JSON lines.
    
    Yields one JSON object per pattern for real-time display.
    """
    for line in run_analysis_streaming(user, use_llm=True):
        yield line
