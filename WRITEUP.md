# Ask First: Approach to Cross-Conversation Health Pattern Detection

## 1. My Approach to the Reasoning Problem

### Core Philosophy

Health patterns don't exist in single conversations—they emerge across time. A headache today and a diet change three weeks ago are meaningless unless connected by temporal logic and medical plausibility. My system is built on three pillars:

**Pillar 1: Temporal Graph Analysis**
We model every health conversation as a graph where nodes are events (symptoms, triggers, interventions) and edges are time-weighted causal relationships. This lets us ask graph questions like: "Which trigger consistently precedes this symptom within a medically plausible window?" The graph structure naturally supports counterfactual reasoning—if we remove a trigger node, do downstream symptom nodes also disappear?

**Pillar 2: Medical Latency Rules**
Most health effects are not immediate. Telogen effluvium (hair fall from nutritional deficiency) manifests 8-12 weeks after deficit onset, not the next day. The system encodes evidence-based latency windows for common phenomena (dietary acne: 1-4 days, sleep debt: 2-8 weeks). This prevents false associations and enables the system to connect events that language models alone might miss because they're temporally distant but medically correlated.

**Pillar 3: Dynamic Pattern Mining (No Hardcoding)**
Instead of hand-coding "User 1 late eating causes stomach pain," the system discovers this from graph topology. The `PatternMiner` searches for five generic pattern types across any user's data: temporal sequences, dose-response, intervention response, progressive decline, and compound causes. This makes the system generalize to new users and datasets without code changes.

### Confidence Architecture

Confidence is not a single number—it's a weighted decomposition across six factors:

- **Repeated Evidence** (20%): More observations = higher confidence
- **Temporal Consistency** (20%): Regular time gaps suggest mechanism, not coincidence
- **Counterfactual Support** (20%): When trigger absent, is effect absent?
- **Medical Plausibility** (15%): Does latency match known physiology?
- **Resolution Evidence** (15%): Does removing trigger resolve symptom?
- **Dose-Response** (10%): More trigger = worse symptom?

This forces explicit reasoning. A pattern doesn't get "high" confidence from association alone—it needs multiple converging signals.

### Chunking Strategy

User histories can exceed LLM context windows. We use hierarchical zones:

- **Recent** (last 3 sessions): Verbatim—precision timing matters for current state
- **Middle** (previous 4): Summarized to 300-char snippets—preserves temporal landmarks
- **Ancient** (earliest): Compressed to "first mention" skeletons—only need root cause dates

This is a principled information-loss function: we keep the most detail where timing is freshest, and progressively abstract older data where only pattern skeletons matter.

---

## 2. Where the System Fails or Hallucinates Confidently

### Failure Mode 1: Sparse Data Breaks Counterfactuals

With only 8-10 sessions per user, counterfactual analysis is statistically weak. If a trigger only appears in 4 sessions and a symptom in 3, we have at most 1-2 "clean" counterfactual sessions. The system still computes a counterfactual score, but it's drawing conclusions from vanishingly small sample sizes. In production, I'd want 20+ sessions before trusting counterfactual signals.

### Failure Mode 2: LLM Enhancement as a Double-Edged Sword

When `GROQ_API_KEY` is available, the LLM can find subtle patterns the graph miner misses. But LLMs also confidently hallucinate causal chains that don't exist—especially with compressed/summarized context where temporal precision is lost. We've seen LLMs invent "week 11" references that weren't in the data. The system mitigates this by:

1. Requiring LLM patterns to have session IDs that exist in the raw data
2. Down-weighting LLM-only patterns vs. graph-validated ones
3. Never letting LLM override deterministic graph evidence

But the guardrails aren't perfect. An adversarial dataset with misleading temporal order could fool both systems.

### Failure Mode 3: Medical Latency Rules Are Sparse

We only encoded 7 latency rules (telogen effluvium, dietary acne, late eating acidity, dehydration headache, sleep debt, post-meal crash, stress menstrual). Real health has hundreds of latency phenomena, and many conditions have highly individual variation. A user whose hair fall manifests at 5 weeks (unusual but possible) would be missed or scored as medium confidence. The rules are a starting scaffold, not a complete map.

### Failure Mode 4: Dose-Response is Crudely Approximated

The current dose-response detection uses simple trigger-count-to-severity mapping. It can't distinguish "200mg dairy" from "500mg dairy"—it only knows "paneer twice a day" vs. "small yogurt." In reality, dose-response is continuous and individual. We'd need quantitative extraction (NLP parsing of amounts) and per-user threshold learning over months.

### Failure Mode 5: Progressive Decline Assumes Linear Sequences

The progressive decline detector looks for A→B→C symptom cascades. But health deterioration can be branching (A→B and A→C simultaneously) or cyclic (A→B→A→B). Our graph traversal is shallow and doesn't model these more complex dynamics.

---

## 3. What I Would Build Differently With More Time

### With 1 More Week

- **Individualized Latency Learning**: Track per-user latencies and update rules dynamically. If Priya's acne responds in 24 hours instead of the population 48-72, learn that.
- **Quantitative Dose Extraction**: Parse amounts ("2 cups coffee," "3 days paneer") for continuous dose-response modeling.

### With 1 More Month

- **Probabilistic Temporal Graphs**: Replace deterministic edges with probabilistic ones (Bayesian networks). This lets us model "trigger A increases probability of symptom B by 40%," which is more honest than binary cause/effect.
- **Causal Discovery Algorithms**: Integrate frameworks like NOTEARS or PC algorithm for data-driven causal structure discovery, removing reliance on medical latency rules.
- **Confidence Calibration**: Use isotonic regression on historical predictions vs. outcomes to calibrate confidence scores (a 0.92 score should mean 92% empirical accuracy).

### With 1 More Quarter

- **Longitudinal User Models**: Each user gets their own personalized causal model that updates incrementally with each conversation.
- **Active Questioning**: When the system is uncertain about a pattern, it should ask targeted follow-up questions in the next conversation to gather counterfactual evidence.
- **Multi-User Pattern Discovery**: Find patterns across users ("software engineers in Bangalore frequently show X pattern") to generate population-level latency priors.

### Architectural Pivot

Given unlimited time, I would move from a graph+LLM hybrid to a **neural causal inference engine**:

1. Use transformer encoder to embed each session into a latent health state
2. Use a temporal transformer (like a causal Transformer-XL) to model health state transitions
3. Use attention weights to explain which past events influenced current predictions
4. Fine-tune on medical causal datasets (MIMIC, etc.) for grounded latency priors

This would be more robust to sparse data and wouldn't require hand-coded latency rules. But it needs millions of conversation-hours to train—hence the current hybrid approach is the right pragmatic choice for the data we have.

---

## Honest Assessment

This system will find the 8 hidden patterns in the synthetic dataset because they're deliberately strong signals with clear temporal structure and medical grounding. In real messy data—where users forget dates, conflate symptoms, and don't mention triggers unless asked—the performance would drop significantly. The value isn't in perfection; it's in **explicitly surfacing uncertainty** (via decomposed confidence scores) and **explaining reasoning** (via temporal traces) so clinicians and users can decide what to trust.
