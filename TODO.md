# Ask First - Production Implementation TODO

## Phase 1: Foundation
- [x] Create TODO.md
- [x] Update requirements.txt
- [x] Create core/models.py (Pydantic models)
- [x] Create core/extractor.py (Dynamic entity extraction with stopword filtering)
- [x] Create core/temporal_graph.py (Temporal event graph + medical latency rules)

## Phase 2: Reasoning Engine
- [x] Create core/pattern_miner.py (Dynamic pattern mining engine)
- [x] Create core/confidence.py (Advanced 6-factor confidence scoring)
- [x] Create core/chunker.py (Hierarchical context management)
- [x] Rewrite core/llm_engine.py (Enhanced Groq/OpenAI with structured output)
- [x] Create core/reasoner.py (Main orchestrator with streaming support)

## Phase 3: Integration
- [x] Rewrite core/detector.py (Remove hardcoding, uses new pipeline)
- [x] Update core/preprocess.py (Enhanced text processing)
- [x] Update core/timeline.py (Better timeline building)
- [x] Delete core/scorer.py (Replaced by confidence.py)

## Phase 4: Application Layer
- [x] Rewrite app.py (JSON streaming, production UI)
- [x] Rewrite README.md (Full docs + architecture)
- [x] Create WRITEUP.md (Mandatory one-page writeup with failure analysis)

## Phase 5: Testing & Validation
- [x] Run tests against all 3 user profiles
- [x] Verify JSON streaming output
- [x] Key hidden patterns detected:
    - [x] P1: Late eating → stomach pain (USR001)
    - [x] P2: Dehydration → headaches (USR001)
    - [x] P3: Calorie restriction → hair fall 6 weeks (USR002) ✓ Telogen effluvium latency
    - [x] P4: Dairy → acne (USR002)
    - [x] P5: High-carb lunch → 2pm crash (USR003)
    - [x] P6: Poor sleep → severe cramps (USR003)
    - [x] P7: Screen use → multi-symptoms (USR003)
    - [x] P8: Progressive symptom cascade (USR002)

## Status: COMPLETE
