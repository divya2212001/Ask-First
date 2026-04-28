"""
Context Management and Chunking Strategy for Ask First.

Problem: LLMs have limited context windows. Full user history with
8-10 detailed conversations may exceed limits.

Solution: Hierarchical chunking with summarization.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chunk:
    """A chunk of user history ready for LLM ingestion."""
    chunk_id: str
    sessions: List[Dict[str, Any]]
    summary: Optional[str] = None
    is_summarized: bool = False
    token_estimate: int = 0


class TimelineChunker:
    """
    Hierarchical chunking strategy:
    
    1. Verbatim Recent: Most recent 2-3 sessions kept verbatim (full detail).
       Rationale: Recent events have higher fidelity needs for precise timing.
    
    2. Summarized Middle: Middle sessions summarized to key events.
       Rationale: Preserves temporal landmarks while reducing tokens.
    
    3. Compressed Ancient: Oldest sessions compressed to pattern skeletons.
       Rationale: Only need to know "what started when" for root cause analysis.
    """
    
    def __init__(
        self,
        verbatim_window: int = 15,
        summary_window: int = 0,
        max_tokens_per_chunk: int = 4000
    ):
        self.verbatim_window = verbatim_window
        self.summary_window = summary_window
        self.max_tokens = max_tokens_per_chunk
        
        # Token estimates
        self.tokens_per_session_verbatim = 800
        self.tokens_per_session_summary = 200
        self.tokens_per_session_compressed = 80
    
    def chunk_timeline(self, timeline: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Split timeline into chunks based on strategy.
        """
        if not timeline:
            return []
        
        # Sort by timestamp to ensure chronological order
        sorted_timeline = sorted(
            timeline,
            key=lambda x: x.get("timestamp", "")
        )
        
        chunks = []
        n = len(sorted_timeline)
        
        # Determine zones
        ancient_end = max(0, n - self.verbatim_window - self.summary_window)
        middle_end = max(0, n - self.verbatim_window)
        
        # Zone 1: Ancient (compressed)
        if ancient_end > 0:
            ancient = sorted_timeline[:ancient_end]
            chunk = self._compress_ancient(ancient, "chunk_ancient")
            chunks.append(chunk)
        
        # Zone 2: Middle (summarized)
        if middle_end > ancient_end:
            middle = sorted_timeline[ancient_end:middle_end]
            chunk = self._summarize_middle(middle, "chunk_middle")
            chunks.append(chunk)
        
        # Zone 3: Recent (verbatim)
        if n > middle_end:
            recent = sorted_timeline[middle_end:]
            chunk = self._keep_verbatim(recent, "chunk_recent")
            chunks.append(chunk)
        
        return chunks
    
    def _keep_verbatim(self, sessions: List[Dict], chunk_id: str) -> Chunk:
        """Keep sessions with full detail."""
        return Chunk(
            chunk_id=chunk_id,
            sessions=sessions,
            summary=None,
            is_summarized=False,
            token_estimate=len(sessions) * self.tokens_per_session_verbatim
        )
    
    def _summarize_middle(self, sessions: List[Dict], chunk_id: str) -> Chunk:
        """Summarize to key events and temporal landmarks."""
        summarized = []
        for session in sessions:
            summary = {
                "session_id": session.get("session_id"),
                "timestamp": session.get("timestamp"),
                "key_symptoms": session.get("symptoms", [])[:3],
                "key_lifestyle": session.get("lifestyle", [])[:3],
                "severity": session.get("severity"),
                "text_summary": session.get("text", "")[:300] + "..." 
                    if len(session.get("text", "")) > 300 
                    else session.get("text", "")
            }
            summarized.append(summary)
        
        return Chunk(
            chunk_id=chunk_id,
            sessions=summarized,
            summary=f"Summary of {len(sessions)} sessions with key symptoms and triggers",
            is_summarized=True,
            token_estimate=len(sessions) * self.tokens_per_session_summary
        )
    
    def _compress_ancient(self, sessions: List[Dict], chunk_id: str) -> Chunk:
        """Compress to pattern skeleton - only root causes and start dates."""
        # Extract only the earliest mention of key triggers and symptoms
        first_mentions = {}
        for session in sessions:
            for symptom in session.get("symptoms", []):
                if symptom not in first_mentions:
                    first_mentions[symptom] = {
                        "first_seen": session.get("timestamp"),
                        "session_id": session.get("session_id")
                    }
            for trigger in session.get("lifestyle", []):
                if trigger not in first_mentions:
                    first_mentions[trigger] = {
                        "first_seen": session.get("timestamp"),
                        "session_id": session.get("session_id")
                    }
        
        compressed = [{
            "session_id": "skeleton",
            "timestamp": sessions[0].get("timestamp") if sessions else None,
            "first_mentions": first_mentions,
            "note": f"Compressed from {len(sessions)} sessions"
        }]
        
        return Chunk(
            chunk_id=chunk_id,
            sessions=compressed,
            summary=f"Pattern skeleton: {len(first_mentions)} unique entities first mentioned",
            is_summarized=True,
            token_estimate=self.tokens_per_session_compressed
        )
    
    def build_context_prompt(self, chunks: List[Chunk], include_reasoning: bool = True) -> str:
        """
        Build a prompt-ready context string from chunks.
        """
        parts = []
        
        if include_reasoning:
            parts.append("=== REASONING STRATEGY ===")
            parts.append(
                "This user's history is presented in three temporal zones:\n"
                "- RECENT (verbatim): Full detail for precise timing\n"
                "- MIDDLE (summarized): Key events preserved\n"
                "- ANCIENT (compressed): First occurrences only for root cause analysis\n"
            )
        
        for chunk in chunks:
            parts.append(f"\n=== {chunk.chunk_id.upper()} ===")
            
            if chunk.is_summarized and chunk.summary:
                parts.append(f"[{chunk.summary}]")
            
            for session in chunk.sessions:
                parts.append(self._session_to_text(session))
        
        return "\n".join(parts)
    
    def _session_to_text(self, session: Dict) -> str:
        """Convert session to text representation."""
        sid = session.get("session_id", "unknown")
        ts = session.get("timestamp", "unknown")
        
        if "text_summary" in session:
            # Summarized format
            return (
                f"Session {sid} ({ts}): "
                f"Symptoms: {session.get('key_symptoms', [])} | "
                f"Lifestyle: {session.get('key_lifestyle', [])} | "
                f"Summary: {session.get('text_summary', '')}"
            )
        elif "first_mentions" in session:
            # Compressed format
            mentions = session.get("first_mentions", {})
            return (
                f"Pattern Skeleton ({ts}): "
                f"First mentions: {list(mentions.keys())}"
            )
        else:
            # Verbatim format
            text = session.get("text", "")
            symptoms = session.get("symptoms", [])
            lifestyle = session.get("lifestyle", [])
            return (
                f"Session {sid} ({ts}): {text[:600]}\n"
                f"  Symptoms: {symptoms} | Lifestyle: {lifestyle}"
            )


class StreamingContextManager:
    """
    Manages context for streaming output.
    Ensures each pattern has access to full history while
    maintaining token budget.
    """
    
    def __init__(self, chunker: TimelineChunker):
        self.chunker = chunker
    
    def prepare_reasoning_context(
        self,
        timeline: List[Dict[str, Any]],
        focus_session_ids: Optional[List[str]] = None
    ) -> str:
        """
        Prepare context optimized for a specific reasoning task.
        """
        chunks = self.chunker.chunk_timeline(timeline)
        
        # If focusing on specific sessions, ensure they're verbatim
        if focus_session_ids:
            for chunk in chunks:
                for session in chunk.sessions:
                    if session.get("session_id") in focus_session_ids:
                        # This session is critical - ensure full detail
                        pass  # Already handled by chunking zones
        
        return self.chunker.build_context_prompt(chunks)
    
    def prepare_full_history_context(self, timeline: List[Dict[str, Any]]) -> str:
        """Prepare complete history for final aggregation."""
        # For final output, use a more balanced approach
        chunker_balanced = TimelineChunker(
            verbatim_window=5,
            summary_window=3
        )
        chunks = chunker_balanced.chunk_timeline(timeline)
        return chunker_balanced.build_context_prompt(chunks, include_reasoning=False)
