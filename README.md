# AFRED

**Autonomous Financial Ripple Effect Discovery**

AFRED is a research pipeline that takes a raw financial event headline and autonomously discovers which companies are materially affected — without any human-specified target list. Given a news event, it produces a ranked set of affected entities with causal mechanisms, directional forecasts, and confidence scores.

This addresses a gap in existing financial AI systems. Every surveyed system — FinCon, TradingAgents, MarketSenseAI, FinRipple — accepts a predefined company universe as input and analyzes those companies. None treat entity discovery as the output. AFRED inverts this assumption.

---

## The Problem

When Apple releases a new chip, an experienced analyst immediately thinks about TSMC as the fabrication supplier, Intel and Qualcomm as competitors, ARM Holdings as the IP licensor, and cloud providers as downstream customers. This multi-hop reasoning across corporate relationships is what AFRED automates.

**Current tools:** You tell the system which companies to analyze → it analyzes them.

**AFRED:** You give it a raw event → it discovers which companies to analyze → it analyzes them.

---

## Validation Result: Apple M3 Announcement

```
Input (raw headline, no additional context):
  "Apple announced the M3 chip, 40% faster than the M2"

Output:

  NEGATIVELY AFFECTED:
    INTC  (Intel)       — Score: 0.82 — horizontal_competitor (critical)
                          Mechanism: M3 directly threatens Intel's laptop processor
                          market share, where Intel derives ~50% of Client Computing revenue
                          Confidence: high | Horizon: medium_term

    QCOM  (Qualcomm)    — Score: 0.74 — horizontal_competitor (major)
                          Mechanism: Apple Silicon advancement reduces Qualcomm's
                          relevance in premium mobile and laptop segments
                          Confidence: high | Horizon: medium_term

  POSITIVELY AFFECTED:
    TSM   (TSMC)        — Score: 0.91 — fabrication_supplier (critical)
                          Mechanism: Apple M3 volume increases TSMC N3 node utilization
                          Confidence: high | Horizon: immediate

    ARM   (ARM Holdings) — Score: 0.78 — ip_licensor (major)
                          Mechanism: Every M3 chip sold generates ARM licensing revenue
                          Confidence: high | Horizon: immediate

  RING 2 — Spawned cloud infrastructure graph:
    AMZN  (AWS)         — Score: 0.61
    MSFT  (Azure)       — Score: 0.58
    GOOGL (GCP)         — Score: 0.55

  ─────────────────────────────────────────────
  Entities discovered : 7
  Knowledge graphs    : 2 (semiconductor + cloud infrastructure)
  Pipeline time       : 43 seconds
  Hardcoded values    : 0
```

---

## Pipeline Architecture

AFRED follows a hybrid LLM-deterministic design. Language understanding tasks go to the LLM. Systematic structural tasks go to code. This prevents hallucination in structural outputs while preserving LLM reasoning quality where it matters.

```
Raw Event Text
      │
      ▼
┌─────────────────────────────────┐
│  Stage 1: Event Parser  (LLM)  │
│                                 │
│  • Classifies event type        │
│    (10-category taxonomy)       │
│  • Extracts source entity       │
│  • Identifies market dynamic    │
│  • Assesses event magnitude     │
└──────────────┬──────────────────┘
               │  structured JSON
               ▼
┌─────────────────────────────────┐
│  Stage 2: Graph Discovery       │
│           (Deterministic Code)  │
│                                 │
│  • Looks up source entity in    │
│    KNOWLEDGE_GRAPH_REGISTRY     │
│  • Traverses domain graph via   │
│    EVENT_RELATIONSHIP_MAP       │
│  • Returns all Ring 1 entities  │
│  • Triggers cross-sector spawn  │
│    when causal chain crosses    │
│    domain boundary              │
└──────────────┬──────────────────┘
               │  entity list
               ▼
┌─────────────────────────────────┐
│  Stage 3: Ripple Reasoner (LLM) │
│           one call per entity   │
│                                 │
│  • Direction (positive/negative)│
│  • Causal mechanism             │
│  • Confidence + reasoning       │
│  • Time horizon                 │
└──────────────┬──────────────────┘
               │  impact assessments
               ▼
┌─────────────────────────────────┐
│  Stage 4: Confidence Filter     │
│           (Deterministic Code)  │
│                                 │
│  score = strength  × 0.30       │
│        + confidence× 0.30       │
│        + magnitude × 0.20       │
│        + revenue   × 0.20       │
│                                 │
│  Prune score < 0.25             │
│  Rank descending                │
└──────────────┬──────────────────┘
               │
               ▼
         Ranked Impact Set
```

---

## Knowledge Graph Architecture

Financial relationships are encoded as typed directed edges across multiple domain-specific graphs rather than one global graph. This produces three benefits: traversal efficiency, semantic coherence within each domain, and independent updatability.

### Node Types

| Type | Description |
|---|---|
| `COMPANY` | Publicly traded company, identified by ticker |
| `SECTOR` | Industry sector grouping |
| `GEOGRAPHIC_CHOKEPOINT` | Physical infrastructure (e.g. Strait of Hormuz) |
| `SOVEREIGN_ENTITY` | Nation-state or government body |

### Edge Types

| Type | Direction | Description |
|---|---|---|
| `fabrication_supplier` | A → B | A manufactures components for B |
| `horizontal_competitor` | A ↔ B | Companies competing in same market |
| `ip_licensor` | A → B | A licenses IP to B |
| `customer_downstream` | A → B | B is a major customer of A |
| `geographic_dependency` | A → C | A relies on physical chokepoint C |

### Cross-Sector Spawning

When graph traversal reaches the boundary of a domain, an LLM agent determines whether a causal connection exists to entities in an adjacent domain graph. If confirmed, a new traversal spawns in that graph carrying a reduced confidence multiplier. The Apple M3 test demonstrates this: Intel's competitive decline in the semiconductor graph triggers a spawn into the cloud infrastructure graph, discovering downstream impacts on AWS, Azure, and GCP.

---

## Event Classification

AFRED uses a fine-tuned FinBERT classifier for event type routing. The classifier was fine-tuned on 400 labeled financial event descriptions and achieves 96% held-out accuracy on a 10-class taxonomy.

### Event Taxonomy

| Tier | Event Types |
|---|---|
| Tier 1 — Corporate | product_launch, earnings_surprise, merger_acquisition, management_change, pricing_change, partnership, tech_breakthrough |
| Tier 2 — Geopolitical | regulatory_action, geopolitical |
| Tier 3 — Macroeconomic | supply_disruption |

Event type determines which edge types are traversed. A `product_launch` activates `horizontal_competitor` and `fabrication_supplier` edges. A `supply_disruption` activates `upstream_supplier` and `customer_downstream` edges.

---

## Research Findings

### Finding 1: Framing Sensitivity
The same underlying financial event produces FinBERT confidence variance of 0.12 to 0.19 depending on how the headline is phrased across different news sources. In some cases this variance is sufficient to change routing decisions, causing different entities to be discovered from identical events. This is a fundamental reliability challenge for any LLM-based financial event classifier.

### Finding 2: Geopolitical Energy Chain Regularity
Six major geopolitical energy disruptions from 1973 to 2022 (OPEC embargo, Iran-Iraq war, Gulf War, 2003 Iraq invasion, 2011 Arab Spring, 2022 Russia-Ukraine conflict) produce structurally identical causal chains through the same company nodes. The same fertilizer producers, agricultural processors, food retailers, and macro transmission nodes appear in every disruption. This suggests the geopolitical energy ripple chain is learnable from historical data and encodable as a static high-confidence graph.

### Finding 3: SUTVA Violation in Parallel MDP Propagation
When multiple agents traverse a connected financial network simultaneously, the same entity can appear in both affected sets with conflicting magnitude estimates. Resolving these requires a Bayesian fusion rule, but the standard Bayesian update assumes treatment independence across agents — an assumption the financial network directly violates. When Intel's decline propagated simultaneously through the semiconductor graph and a spawned cloud infrastructure MDP, the two magnitude estimates diverged with no principled reconciliation available. This SUTVA violation in the potential outcomes framework is the central theoretical problem of the AFRED research agenda.

### Finding 4: Correct Boundary Detection
When tested on geopolitical events (Iran war, Strait of Hormuz closure), the pipeline correctly identified that no registered graph contained sovereign actors or geographic chokepoints and halted with a structured diagnostic. This correct halt demonstrates the architecture handles its own limits gracefully and points precisely to the geopolitical extension required.

---

## Open Research Problems

Seven open problems identified through implementation and testing, each representing a publishable contribution:

1. **Propagation topology classification** — Learning whether a given event type produces steep or flat attenuation across graph rings from historical return data
2. **SUTVA violation in parallel MDP propagation** — Deriving a correct Bayesian fusion rule for network interference settings
3. **Dynamic KG construction from SEC filings** — Programmatic extraction of typed, confidence-weighted edges from 10-K Item 1 and Item 1A disclosures
4. **Sovereign-to-corporate bridge graph** — Connecting geopolitical entities and physical chokepoints to corporate supply chains
5. **Beneficiary discovery gap** — Adding substitute_supplier and competitive_beneficiary edge types to capture positive contagion
6. **Geopolitical event duration estimation** — Predicting whether a disruption activates Ring 1 only or the full macroeconomic transmission layer
7. **Threshold fragility and Bayesian tail leverage** — Replacing hard classification thresholds with posterior inference over threshold values

---

## Project Structure

```
AFRED/
├── afred_pipeline.py      # Full five-agent pipeline
└── Knowledge_graphs.py    # Semiconductor and cloud infrastructure KGs,
                           # KNOWLEDGE_GRAPH_REGISTRY, EVENT_RELATIONSHIP_MAP
```

---

## Setup

### Prerequisites

- Python 3.11+
- Groq API key (for LLM calls)
- Gemini API key (optional, used for entity extraction)

### Installation

```bash
git clone https://github.com/Nimish-B-Prabhakar/Autonomous-Financial-Ripple-Effect-Discovery.git
cd Autonomous-Financial-Ripple-Effect-Discovery

python -m venv .venv
source .venv/bin/activate

pip install groq google-generativeai python-dotenv networkx
```

### Configuration

```bash
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
EOF
```

### Run

```bash
python afred_pipeline.py
```

The pipeline prompts for an event headline and runs the full discovery pipeline, printing the ranked impact set to the terminal.

---

## Related Project

**QuantLens** — The full multi-agent financial analysis system that AFRED was built within, including a FastAPI backend, React frontend, portfolio risk analysis, news summarization, and quality review gate.

→ [QuantLens Repository](https://github.com/Nimish-B-Prabhakar/QuantLens)

---

## Research Context

AFRED was developed as part of an MS Computer Science capstone at Temple University and is being extended as a PhD research project in the Department of Statistics, Operations, and Data Science at Fox School of Business, Temple University, in collaboration with faculty working on causal inference, supply chain analytics, and network interference.

The event study validation framework is grounded in Singhal and Hendricks (2003, 2005, 2019), whose abnormal return methodology provides the empirical ground truth for evaluating AFRED's structural predictions against realized market reactions.

---

## License

MIT
