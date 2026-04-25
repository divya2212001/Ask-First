"""
Enhanced LLM Engine for Ask First.
Supports multiple providers (Groq, OpenAI) with structured output
and streaming capabilities.
"""

import os
import json
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60


class BaseReasoner:
    """Base class for LLM reasoners."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.enabled = False
    
    def available(self) -> bool:
        return self.enabled
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Yield chunks of the response for streaming."""
        raise NotImplementedError
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON output matching a schema."""
        raise NotImplementedError


class GroqReasoner(BaseReasoner):
    """Groq LLM implementation using llama3-70b-8192."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config or LLMConfig())
        self.api_key = os.getenv("GROQ_API_KEY")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                self.enabled = False
                self.client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.enabled:
            return ""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error: {e}")
            return ""
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        if not self.enabled:
            yield ""
            return
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"Groq streaming error: {e}")
            yield ""
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON with validation."""
        system = system_prompt or "You are a structured data extraction engine. Return only valid JSON."
        system += f"\n\nYour response must match this JSON schema: {json.dumps(schema, indent=2)}"
        system += "\nReturn ONLY JSON, no markdown, no explanations."
        
        content = self.generate(prompt, system)
        
        # Clean markdown fences
        content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw content: {content[:500]}")
            return {}


class OpenAIReasoner(BaseReasoner):
    """OpenAI GPT implementation as fallback."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config or LLMConfig(provider="openai", model="gpt-4-turbo-preview"))
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.enabled = False
                self.client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.enabled:
            return ""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        system = system_prompt or "You are a structured data extraction engine. Return only valid JSON."
        system += f"\n\nYour response must match this JSON schema: {json.dumps(schema, indent=2)}"
        
        content = self.generate(prompt, system)
        content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}


class LLMReasoningEngine:
    """
    Unified reasoning engine with provider fallback.
    Implements two-stage reasoning:
    1. Extract structured events from raw text
    2. Reason about cross-session temporal patterns
    """
    
    SYSTEM_EXTRACTOR = """
    You are a medical entity extraction system. Extract health events from conversation text.
    Focus on: symptoms, triggers/lifestyle factors, interventions, and temporal markers.
    """
    
    SYSTEM_REASONER = """
    You are an expert temporal health reasoning engine. Analyze cross-conversation patterns.
    
    RULES:
    - Use explicit temporal reasoning (e.g., "X occurred 6 weeks after Y")
    - Consider medical latency windows (e.g., hair fall 8-12 weeks after diet change)
    - Check counterfactuals: when X absent, is Y also absent?
    - Look for dose-response: more trigger = worse symptom
    - Identify root causes vs downstream effects
    
    Output structured JSON with confidence scores and reasoning traces.
    """
    
    def __init__(self):
        self.groq = GroqReasoner()
        self.openai = OpenAIReasoner()
        self.primary = self.groq if self.groq.available() else self.openai
    
    def available(self) -> bool:
        return self.primary.available()
    
    def extract_events(self, session_text: str) -> List[Dict[str, Any]]:
        """Stage 1: Extract structured events from session text."""
        schema = {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "event_type": {"type": "string", "enum": ["symptom", "trigger", "intervention"]},
                            "description": {"type": "string"},
                            "severity": {"type": "string", "enum": ["mild", "moderate", "severe", "none"]},
                            "temporal_marker": {"type": "string"}
                        },
                        "required": ["entity", "event_type", "description"]
                    }
                }
            },
            "required": ["events"]
        }
        
        prompt = f"""
        Extract health events from this conversation:
        
        {session_text}
        
        Identify:
        - Symptoms (what the user is experiencing)
        - Triggers (diet, habits, stressors)
        - Interventions (things they tried)
        - Temporal markers (when things started, duration)
        """
        
        result = self.primary.generate_structured(prompt, schema, self.SYSTEM_EXTRACTOR)
        return result.get("events", [])
    
    def reason_patterns(
        self, 
        user_profile: Dict[str, Any],
        context: str,
        existing_patterns: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Stage 2: Reason about temporal patterns from chunked context."""
        schema = {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern_title": {"type": "string"},
                            "pattern_type": {"type": "string", "enum": ["temporal_sequence", "dose_response", "intervention_response", "compound_cause", "progressive_decline"]},
                            "confidence_label": {"type": "string", "enum": ["high", "medium", "low"]},
                            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "confidence_justification": {"type": "string"},
                            "reasoning_trace": {"type": "string"},
                            "medical_latency_note": {"type": "string"},
                            "root_cause": {"type": "string"},
                            "downstream_effects": {"type": "array", "items": {"type": "string"}},
                            "sessions_involved": {"type": "array", "items": {"type": "string"}},
                            "counterfactual_evidence": {"type": "string"}
                        },
                        "required": ["pattern_title", "pattern_type", "confidence_label", "confidence_score", "confidence_justification", "reasoning_trace"]
                    }
                },
                "reasoning_strategy_used": {"type": "string"}
            },
            "required": ["patterns"]
        }
        
        existing = ""
        if existing_patterns:
            existing = f"\n\nAlready detected patterns (avoid duplicates):\n{json.dumps(existing_patterns, indent=2)}"
        
        prompt = f"""
        Analyze this user's health history for hidden cross-conversation patterns.
        
        USER PROFILE:
        Name: {user_profile.get('name')}
        Age: {user_profile.get('age')}
        Gender: {user_profile.get('gender')}
        Occupation: {user_profile.get('occupation')}
        
        CONVERSATION HISTORY:
        {context}
        
        {existing}
        
        TASK:
        Find temporal patterns with causal reasoning. For each pattern:
        1. State the pattern with explicit time references
        2. Explain the temporal logic (why the timing matters)
        3. Provide confidence with justification
        4. Note any medical latency (time between cause and effect)
        5. Check counterfactuals (when cause absent, is effect absent?)
        
        Be specific: "Session X (Week Y)" not vague "earlier".
        """
        
        return self.primary.generate_structured(prompt, schema, self.SYSTEM_REASONER)
    
    def streaming_reason(
        self,
        user_profile: Dict[str, Any],
        context: str
    ) -> Generator[str, None, None]:
        """Stream reasoning output for real-time display."""
        prompt = f"""
        Analyze patterns for {user_profile.get('name')}.
        
        History:
        {context}
        
        Output each pattern as a JSON object, one per line.
        """
        
        yield from self.primary.generate_streaming(prompt, self.SYSTEM_REASONER)


def get_reasoner() -> LLMReasoningEngine:
    """Factory function to get configured reasoner."""
    return LLMReasoningEngine()
