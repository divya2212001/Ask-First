# Health Ask First - AI Health Pattern Detection

**Cross-conversation temporal reasoning for longitudinal health history.**

## Overview

Ask First is a production-grade health reasoning system that detects hidden patterns across multiple conversations over time. Unlike simple keyword matching, this system uses temporal graph analysis, medical latency rules, and counterfactual scoring to surface causally connected health events.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Conversation   │────▶│  Event Extractor │────▶│ Temporal Graph  │
│     Data        │     │  (Dynamic NER)   │     │  Builder        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                              ┌───────────────────────────┼───────────┐
                              ▼                           ▼           ▼
                    ┌─────────────────┐          ┌──────────────┐  ┌──────────┐
                    │ Pattern Miner   │          │ Confidence   │  │ LLM      │
                    │ (Deterministic) │◄─────────│ Scorer       │  │ Enhancer │
                    └────────┬────────┘          └──────────────┘  └────┬─────┘
                             │                                         │
                             └─────────────────┬───────────────────────┘
                                               ▼
                                        ┌──────────────┐
                                        │ JSON Stream  │
                                        │ Output       │
                                        └──────────────┘
```

## Key Features

- **Dynamic Pattern Mining**: No hardcoded user-specific logic. All patterns emerge from temporal graph analysis.
- **Medical Latency Awareness**: Knows that hair fall manifests 8-12 weeks after nutritional deficiency (telogen effluvium).
- **Counterfactual Scoring**: Boosts confidence when effect disappears in absence of trigger.
- **Dose-Response Detection**: Identifies when more trigger = worse symptom.
- **Hierarchical Chunking**: Recent sessions verbatim, middle summarized, oldest compressed for token efficiency.
- **Streaming JSON Output**: Real-time pattern delivery for downstream integration.
- **Multi-Factor Confidence**: Repeated evidence × temporal consistency × counterfactuals × medical plausibility × resolution × dose-response.

## Setup

### Prerequisites

- Python 3.10+
- Groq API key (optional but recommended for LLM enhancement)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Ask-First

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional for LLM)
export GROQ_API_KEY="your-key-here"
# Or create .env file:
echo "GROQ_API_KEY=your-key-here" > .env
```

### Running the Application

```bash
# Streamlit UI
streamlit run app.py

# Or run from any directory
streamlit run /path/to/app.py
```

The app will open at `http://localhost:8501`.

### Dataset Format

The system expects JSON with this structure:

```json
{
  "users": [
    {
      "user_id": "USR001",
      "name": "Arjun",
      "age": 26,
      "conversations": [
        {
          "session_id": "USR001_S01",
          "timestamp": "2026-01-05T23:14:00",
          "user_message": "...",
          "user_followup": "...",
          "clary_response": "...",
          "severity": "mild",
          "tags": ["stomach", "acidity", "late eating"]
        }
      ]
    }
  ]
}
```

## How It Works

### 1. Event Extraction

Every conversation is parsed for:

- **Symptoms**: "hair fall", "headache", "cramps"
- **Triggers**: "late eating", "dairy", "stress"
- **Interventions**: "cut dairy", "added protein"
- **Temporal markers**: "started 3 weeks ago", "since January"

### 2. Temporal Graph Building

Events become nodes in a directed graph. Edges are created when:

- A trigger precedes a symptom within a medically plausible window
- Multiple occurrences of same trigger→sympair reinforce the link

### 3. Pattern Mining

The `PatternMiner` searches for:

- **Temporal Sequences**: Trigger consistently precedes symptom (≥2 times)
- **Dose-Response**: Higher trigger exposure correlates with worse symptoms
- **Intervention Response**: Symptom improves after intervention
- **Progressive Decline**: Same cause produces worsening symptoms over time
- **Compound Causes**: Multiple triggers required together

### 4. Confidence Scoring

| Factor               | Weight | Description                               |
| -------------------- | ------ | ----------------------------------------- |
| Repeated Evidence    | 20%    | How many times observed                   |
| Temporal Consistency | 20%    | Regular timing between trigger and effect |
| Counterfactual       | 20%    | Effect absent when trigger absent         |
| Medical Plausibility | 15%    | Matches known medical latency             |
| Resolution           | 15%    | Symptom improves after removing trigger   |
| Dose-Response        | 10%    | More trigger = more effect                |

### 5. LLM Enhancement (Optional)

If `GROQ_API_KEY` is set, a two-stage LLM pipeline:

1. **Extract**: Structured events from raw text
2. **Reason**: Cross-session causal analysis with medical knowledge

### 6. Chunking Strategy

To manage token limits with long histories:

| Zone    | Sessions        | Detail Level                                |
| ------- | --------------- | ------------------------------------------- |
| Recent  | Last 3          | Verbatim (full text)                        |
| Middle  | 4 before recent | Summarized (key events + 300 char snippets) |
| Ancient | Remaining       | Compressed (first mention dates only)       |

This preserves temporal landmarks for root cause analysis while keeping within token budgets.

## Output Format

### Streaming JSON

Each pattern is streamed as a JSON line:

```json
{"meta": {"user_id": "USR002", "patterns_found": 4, ...}}
{"pattern": {"pattern": "Hair fall from calorie restriction", "confidence": "high", ...}}
{"pattern": {"pattern": "Dairy triggers acne", "confidence": "high", ...}}
{"done": true}
```

### Pattern Object

```json
{
  "pattern_id": "abc123",
  "pattern": "Hair fall caused by severe calorie restriction",
  "pattern_type": "progressive_decline",
  "confidence": "high",
  "confidence_score": 0.92,
  "confidence_justification": "4 repeated episodes; matches known medical latency; symptom resolved after intervention.",
  "reasoning": "Calorie restriction to 700-800 cal began Jan 8. Hair fall first noticed Feb 19 (6 weeks later). Telogen effluvium typically manifests 6-12 weeks after deficiency onset.",
  "medical_latency_note": "Telogen effluvium from severe calorie restriction. Typical onset: 42-84 days.",
  "evidence": {
    "sessions": ["S01", "S05", "S06", "S09"],
    "counterfactual_sessions": ["S04", "S07"],
    "temporal_consistency": 0.85,
    "dose_response_score": null
  },
  "root_cause": "calorie restriction",
  "downstream_effects": ["hair fall", "fatigue", "brain fog"]
}
```

## Tech Stack

| Component    | Choice                 | Why                                                 |
| ------------ | ---------------------- | --------------------------------------------------- |
| LLM          | Groq (llama3-70b-8192) | Fast inference, 70B parameter model, cost-effective |
| Fallback     | OpenAI GPT-4           | If Groq unavailable                                 |
| Framework    | Streamlit              | Rapid UI prototyping with wide layout               |
| Validation   | Pydantic v2            | Type-safe structured output                         |
| Date Parsing | python-dateutil        | Handles mixed timestamp formats                     |
| Retry Logic  | tenacity               | Resilient API calls                                 |

## Testing Against Hidden Patterns

The system is designed to find these pattern types:

| Pattern                         | User   | Key Signal                                               |
| ------------------------------- | ------ | -------------------------------------------------------- |
| Late eating → stomach pain      | USR001 | 4 episodes, all after 11pm                               |
| Dehydration → headaches         | USR001 | Week 2 of each month, busy work days                     |
| Calorie restriction → hair fall | USR002 | 6-week medical latency (telogen effluvium)               |
| Dairy → acne                    | USR002 | Dose-response: high dairy = breakout                     |
| High-carb lunch → 2pm crash     | USR003 | Resolved only with protein addition                      |
| Sleep deprivation → cramps      | USR003 | Isolated variable: low stress + bad sleep = still cramps |
| Screen use → multi-symptoms     | USR003 | One root cause, 3 downstream effects                     |
| Progressive symptom cascade     | USR002 | Dizziness → fatigue → hair fall sequence                 |

## License

Internal use for Ask First assignment.
