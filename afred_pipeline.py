import json
from dotenv import load_dotenv
load_dotenv()
import time
import os
from google import genai
from Knowledge_graphs import (
    SEMICONDUCTOR_KG,
    CLOUD_INFRASTRUCTURE_KG,
    KNOWLEDGE_GRAPH_REGISTRY,
)
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Rate limit — free tier is 10 requests per minute
RATE_LIMIT_DELAY = 2

# Pipeline thresholds
MIN_SPAWN_MAGNITUDE = 0.05
MAX_RING_DEPTH = 2
EVENT_SIGNIFICANCE = {
    "product_launch": 0.7,
    "earnings_miss": 0.9,
    "supply_chain_disruption": 0.85,
    "merger_acquisition": 0.8,
}

# ─────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────


def build_entity_extraction_prompt(event_text):
    prompt = f"""You are a financial entity extractor operating within 
a ripple effect discovery system.

Extract all entities from this financial event text.

EVENT TEXT:
"{event_text}"

DEFINITIONS:
- source_entity: The PRIMARY company or actor that is the SUBJECT 
  of the event — the one taking the action or being directly named
- sub_entities: ALL other companies, organizations, or significant 
  entities mentioned — including suppliers, competitors, regulators,
  geographic locations with economic significance, and any other
  publicly traded companies referenced

For each entity provide:
- name: full company name as mentioned
- ticker: stock ticker if publicly traded, null if not
- sector: industry sector
- role: their role in this specific event
  (choose from: event_initiator, fabrication_supplier, 
   competitor, customer, regulator, geographic_chokepoint,
   technology_provider, financial_backer, mentioned_entity)

Return ONLY valid JSON, no other text:
{{
    "source_entity": {{
        "name": "company name",
        "ticker": "TICKER or null",
        "sector": "sector name",
        "role": "event_initiator"
    }},
    "sub_entities": [
        {{
            "name": "company name",
            "ticker": "TICKER or null", 
            "sector": "sector name",
            "role": "their role"
        }}
    ],
    "event_summary": "one sentence describing what happened"
}}"""
    return prompt


# ─────────────────────────────────────────────
# FINBERT CLASSIFICATION
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertConfig

FINBERT_MODEL_PATH = "../models/finbert-event-classifier"
CONFIDENCE_THRESHOLD = 0.85

EVENT_TYPE_LABELS = [
    "product_launch",
    "earnings_surprise",
    "regulatory_action",
    "supply_disruption",
    "management_change",
    "merger_acquisition",
    "pricing_change",
    "tech_breakthrough",
    "partnership",
    "geopolitical",
]


class FinBERTEventClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        config = BertConfig.from_pretrained("ProsusAI/finbert")
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


def load_finbert():
    print("Loading FinBERT event classifier...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = FinBERTEventClassifier(num_labels=len(EVENT_TYPE_LABELS))
    state_dict = torch.load(
        f"{FINBERT_MODEL_PATH}/model.pt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("FinBERT loaded.\n")
    return tokenizer, model


def classify_event(event_text, tokenizer, model):
    inputs = tokenizer(
        event_text, return_tensors="pt", max_length=128, truncation=True, padding=True
    )
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        probs = F.softmax(logits, dim=-1).squeeze(0)

    scores = {
        label: round(probs[i].item(), 4) for i, label in enumerate(EVENT_TYPE_LABELS)
    }
    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    needs_validation = best_conf < CONFIDENCE_THRESHOLD

    return {
        "event_type": best_label,
        "confidence": best_conf,
        "top3": top3,
        "needs_validation": needs_validation,
    }


def validate_classification_with_llm(event_text, finbert_result):
    prompt = f"""You are validating a financial event classification.

EVENT TEXT: "{event_text}"

FINBERT CLASSIFICATION:
- Primary type: {finbert_result['event_type']}
- Confidence: {finbert_result['confidence']}
- Top 3: {finbert_result['top3']}

FinBERT confidence is below 0.85 — validate and correct if needed.

Available event types:
product_launch, earnings_surprise, regulatory_action,
supply_disruption, management_change, merger_acquisition,
pricing_change, tech_breakthrough, partnership, geopolitical

Return ONLY valid JSON:
{{
    "primary_type": "confirmed or corrected event type",
    "secondary_type": "second type if event spans two categories, else null",
    "finbert_correct": true or false,
    "confidence": "high / medium / low",
    "reasoning": "one sentence explanation",
    "relationship_types": ["list", "of", "relevant", "KG", "relationship", "types"]
}}"""
    return call_llm(prompt)


# ─────────────────────────────────────────────
# DYNAMIC STATE BUILDER
# ─────────────────────────────────────────────

# Maps event types to relevant KG relationship types
EVENT_RELATIONSHIP_MAP = {
    "product_launch": [
        "fabrication_supplier",
        "component_supplier",
        "horizontal_competitor",
        "ip_licensor",
        "shared_resource_competitor",
    ],
    "earnings_surprise": [
        "fabrication_supplier",
        "component_supplier",
        "horizontal_competitor",
        "customer_downstream",
        "debt_holder",
    ],
    "supply_disruption": [
        "fabrication_supplier",
        "component_supplier",
        "customer_downstream",
        "logistics_partner",
    ],
    "merger_acquisition": [
        "horizontal_competitor",
        "vertical_integrator",
        "customer_downstream",
        "shared_resource_competitor",
    ],
    "regulatory_action": [
        "horizontal_competitor",
        "customer_downstream",
        "regulatory_dependent",
        "debt_holder",
    ],
    "geopolitical": [
        "geographic_dependency",
        "physical_chokepoint",
        "sovereign_resource",
        "logistics_partner",
        "customer_downstream",
    ],
    "management_change": [
        "horizontal_competitor",
        "customer_downstream",
        "fabrication_supplier",
    ],
    "pricing_change": [
        "horizontal_competitor",
        "customer_downstream",
        "fabrication_supplier",
        "shared_resource_competitor",
    ],
    "tech_breakthrough": [
        "ip_licensor",
        "horizontal_competitor",
        "fabrication_supplier",
        "customer_downstream",
    ],
    "partnership": [
        "horizontal_competitor",
        "fabrication_supplier",
        "customer_downstream",
        "ip_licensor",
    ],
}


def find_kg_and_node(source_entity_name, event_type):
    """
    Look up which KG contains the source entity
    and find the right parent node to query.
    Returns (graph_name, node_key) or (None, None) if not found.
    """
    for graph_name, graph_data in KNOWLEDGE_GRAPH_REGISTRY.items():
        kg = graph_data["graph"]
        if source_entity_name in kg:
            return graph_name, source_entity_name
    return None, None


def build_dynamic_s0(event_text, finbert_tokenizer, finbert_model):
    """
    Full dynamic state builder:
    1. Extract entities via LLM
    2. Classify event via FinBERT (+ LLM validation if needed)
    3. Look up KG and parent node
    4. Build complete s0 state
    """
    print(f"\n{'='*60}")
    print("DYNAMIC STATE BUILDER")
    print(f"{'='*60}")
    print(f"Event: {event_text[:80]}...")

    # ── Step 1: Entity Extraction ──────────────────────────
    print("\n[ENTITY EXTRACTION] Running LLM entity extractor...")
    entity_result = extract_entities(event_text)

    source = entity_result["source_entity"]
    sub_entities = entity_result.get("sub_entities", [])

    print(f"Source entity: {source['name']} ({source['ticker']}) — {source['role']}")
    print(f"Sub entities found: {len(sub_entities)}")
    for e in sub_entities:
        print(f"  {e['name']} ({e['ticker']}) — {e['role']}")

    # ── Step 2: FinBERT Classification ────────────────────
    print("\n[FINBERT] Classifying event type...")
    finbert_result = classify_event(event_text, finbert_tokenizer, finbert_model)

    print(f"Classification: {finbert_result['event_type'].upper()}")
    print(f"Confidence: {finbert_result['confidence']:.4f}", end="  ")

    if finbert_result["needs_validation"]:
        print("⚠  triggering LLM validation...")
        validation = validate_classification_with_llm(event_text, finbert_result)
        event_type = validation["primary_type"]
        secondary_type = validation.get("secondary_type")
        relationship_types = validation.get(
            "relationship_types", EVENT_RELATIONSHIP_MAP.get(event_type, [])
        )
        print(f"LLM validated as: {event_type.upper()}", end="")
        if secondary_type:
            print(f" + {secondary_type.upper()} (dual-nature event)", end="")
        print()
    else:
        print("✓  confidence above threshold")
        event_type = finbert_result["event_type"]
        secondary_type = None
        relationship_types = EVENT_RELATIONSHIP_MAP.get(event_type, [])

    # ── Step 3: KG Lookup ─────────────────────────────────
    print(f"\n[KG LOOKUP] Finding graph for: {source['name']}...")
    graph_name, node_key = find_kg_and_node(source["name"], event_type)

    if graph_name:
        print(f"Found in: {graph_name} graph, node: {node_key}")
        kg = KNOWLEDGE_GRAPH_REGISTRY[graph_name]["graph"]
        candidates = kg[node_key]["candidates"]
        excluded = kg[node_key].get("excluded", [])
    else:
        print(f"Not found in any registered KG — using empty candidate set")
        print(f"NOTE: Dynamic graph construction would trigger here")
        candidates = []
        excluded = []

    # ── Step 4: Build s0 ──────────────────────────────────
    s0 = {
        "event": {
            "description": event_text,
            "type": event_type,
            "secondary_type": secondary_type,
            "source_entity": source["name"],
            "source_ticker": source["ticker"],
            "source_sector": source["sector"],
            "ring_depth": 0,
            "causal_direction": f"{source['name']} is the cause. "
            f"Each candidate entity is a potential effect.",
            "relationship_types": relationship_types,
            "finbert_confidence": finbert_result["confidence"],
            "llm_validated": finbert_result["needs_validation"],
        },
        "sub_entities_mentioned": sub_entities,
        "affected_set": [],
        "magnitude_vector": [],
        "graph_snapshot": {"candidates": candidates, "excluded": excluded},
        "active_graph": graph_name or "unknown",
    }

    print(f"\n[STATE BUILT] s0 ready")
    print(f"  Event type:    {event_type}")
    print(f"  Source entity: {source['name']} ({source['ticker']})")
    print(f"  Active graph:  {graph_name or 'none — needs dynamic construction'}")
    print(f"  Candidates:    {len(candidates)}")
    print(f"  Sub-entities:  {len(sub_entities)}")

    return s0


def extract_entities(event_text):
    prompt = build_entity_extraction_prompt(event_text)
    result = call_llm(prompt)
    return result


def get_primary_exposure(candidate):
    exposures = {
        "revenue": candidate.get("revenue_exposure", 0),
        "competitive": candidate.get("competitive_exposure", 0),
        "resource": candidate.get("resource_exposure", 0),
    }
    # return the field name and value of the highest exposure
    primary_field = max(exposures, key=exposures.get)
    primary_value = exposures[primary_field]
    return primary_field, primary_value


def direction_to_sign(direction_prior):
    if direction_prior == "positive":
        return 1
    elif direction_prior == "negative":
        return -1
    else:
        return None  # ambiguous — exclude


def make_ring0_decisions(state):
    # your code here

    candidates = state["graph_snapshot"]["candidates"]
    for candidate in candidates:
        primary_field, primary_value = get_primary_exposure(candidate)
        if (
            (primary_field == "revenue" and primary_value > 0.05)
            or (primary_field == "competitive" and primary_value > 0.15)
            or (primary_field == "resource" and primary_value > 0.08)
        ):
            sign = direction_to_sign(candidate["direction_prior"])
            if sign is None:
                continue
            magnitude = round(primary_value * 0.7 * sign, 4)
            state["affected_set"].append(candidate["entity"])
            state["magnitude_vector"].append(
                {"entity": candidate["entity"], "magnitude": magnitude, "ring": 0}
            )

    return state


def get_parent_data(state, entity_name):
    entity_magnitude_vector = state["magnitude_vector"]
    entity_candidates = state["graph_snapshot"]["candidates"]
    desired_entity = {}

    for entity in entity_magnitude_vector:
        if entity["entity"] == entity_name:
            desired_entity["magnitude"] = entity["magnitude"]
            desired_entity["ring"] = entity["ring"]

    for candidate in entity_candidates:
        if candidate["entity"] == entity_name:
            desired_entity["relationship_type"] = candidate["relationship_type"]
            desired_entity["revenue_exposure"] = candidate.get("revenue_exposure", 0)
            desired_entity["competitive_exposure"] = candidate.get(
                "competitive_exposure", 0
            )
            desired_entity["resource_exposure"] = candidate.get("resource_exposure", 0)
            desired_entity["direction"] = candidate["direction_prior"]
            desired_entity["confidence"] = candidate["confidence"]

    return desired_entity


def build_enrichment_prompt(parent_entity, parent_data, original_event):
    prompt = f"""You are operating within a financial ripple effect 
discovery system. Your task is to generate an enriched event context.

ORIGINAL EVENT:
- Description: {original_event["description"]}
- Type: {original_event["type"]}
- Source: {original_event["source_entity"]}

RING 0 FINDING FOR {parent_entity}:
- Relationship to source: {parent_data["relationship_type"]}
- Magnitude: {parent_data["magnitude"]} (negative = suffering, positive = benefiting)
- Direction: {parent_data["direction"]}
- Competitive exposure: {parent_data.get("competitive_exposure", 0)}
- Revenue exposure: {parent_data.get("revenue_exposure", 0)}

CONTEXT:
{parent_entity} has magnitude {parent_data["magnitude"]}.
{"This is a NEGATIVE outcome — " + parent_entity + " is LOSING competitive ground, revenue, or market position. The downstream consequence for " + parent_entity + "'s suppliers, partners, and dependents is NEGATIVE." if parent_data["magnitude"] < 0 else "This is a POSITIVE outcome — " + parent_entity + " is GAINING competitive ground or revenue."}

The enriched event type MUST reflect {parent_entity}'s negative situation.
demand_increase is ONLY valid if {parent_entity} is gaining orders.
For a negative magnitude entity, valid types are:
market_share_shift / revenue_pressure / capex_reduction

STRICT RULES:
- If magnitude is negative, event type MUST reflect negative consequence
- Choose from: market_share_shift / revenue_pressure / capex_reduction / 
  demand_increase / royalty_increase
- market_share_shift: entity losing competitive positioning
- revenue_pressure: entity facing direct revenue decline
- capex_reduction: entity cutting capital expenditure
- demand_increase: entity gaining orders or demand
- royalty_increase: entity earning more licensing revenue

Return ONLY a JSON object, no other text:
{{
    "description": "one specific sentence describing what {parent_entity}'s neighbours experience",
    "type": "event type from the list above",
    "source_entity": "{parent_entity}",
    "ring_depth": 1,
    "causal_direction": "one sentence — {parent_entity} is the cause, neighbours are the effects",
    "estimated_impact_magnitude": <float between 0 and 1>,
    "confidence": "high / medium / low"
}}"""
    return prompt


def build_segment_discovery_prompt(parent_entity, enriched_event, active_graph):
    prompt = f"""You are operating within a financial ripple effect
discovery system. Identify all market segments affected by this situation.

SITUATION:
- Entity: {parent_entity}
- Event type: {enriched_event["type"]}
- Description: {enriched_event["description"]}
- Causal direction: {enriched_event["causal_direction"]}
- Active knowledge graph: {active_graph}

TASK:
Identify affected market segments in THREE categories:

1. WITHIN-DOMAIN: segments directly in {parent_entity}'s industry
2. CROSS-DOMAIN: segments in adjacent industries that consume 
   {parent_entity}'s products as inputs or compete for same resources
   (e.g. data centers buying {parent_entity}'s chips,
    OEMs building products around {parent_entity}'s technology)
3. MACRO: broader economic segments indirectly exposed

You MUST include at least one CROSS-DOMAIN segment.
Be specific — name actual market segments, not generic categories.

Return ONLY a JSON object, no other text:
{{
    "within_domain": ["segment1", "segment2"],
    "cross_domain": ["segment3", "segment4"],
    "macro": ["segment5"]
}}"""
    return prompt


# ─────────────────────────────────────────────
# INITIAL STATE
# ─────────────────────────────────────────────

s0 = {
    "event": {
        "description": "Apple announces the M3 chip, built on TSMC 3nm process",
        "type": "product_launch",
        "source_entity": "Apple",
        "ring_depth": 0,
        "causal_direction": "Apple is the cause. Each candidate is a potential effect.",
    },
    "affected_set": [],
    "magnitude_vector": [],
    "graph_snapshot": {
        "candidates": SEMICONDUCTOR_KG["Apple"]["candidates"],
        "excluded": SEMICONDUCTOR_KG["Apple"]["excluded"],
    },
    "active_graph": "semiconductor",
}

# ─────────────────────────────────────────────
# LLM FUNCTIONS
# ─────────────────────────────────────────────


def call_llm(prompt):
    time.sleep(RATE_LIMIT_DELAY)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature for structured JSON output
        response_format={"type": "json_object"},  # forces JSON output
    )
    raw_text = response.choices[0].message.content.strip()
    return json.loads(raw_text)


def classify_segments(affected_segments, active_graph, enriched_event):

    # build registry description string for the prompt
    registry_description = ""
    for graph_name, graph_data in KNOWLEDGE_GRAPH_REGISTRY.items():
        registry_description += f"\n[{graph_name}]\n"
        registry_description += f"Description: {graph_data['description']}\n"
        registry_description += f"Known segments: {', '.join(graph_data['segments'])}\n"

    prompt = f"""You are operating within a financial ripple effect 
discovery system that maintains separate knowledge graphs.

REGISTERED KNOWLEDGE GRAPHS:
{registry_description}

CURRENT ACTIVE GRAPH: {active_graph}

AFFECTED SEGMENTS TO CLASSIFY:
{json.dumps(affected_segments)}

TASK:
For each segment return one of three actions:
- CONTINUE: belongs to current active graph
- SPAWN: belongs to a different registered graph  
- UNKNOWN: does not fit any registered graph

CURRENT SITUATION:
- Entity experiencing event: {enriched_event["source_entity"]}
- Event type: {enriched_event["type"]}  
- Description: {enriched_event["description"]}
- Causal direction: {enriched_event["causal_direction"]}

For any SPAWN classification, the seed_event MUST be grounded in 
the situation above — not a generic event. It should describe how
{enriched_event["source_entity"]}'s {enriched_event["type"]} 
specifically affects entities in the target graph.

Return ONLY a JSON array, no other text:
[
  {{
    "segment": "segment_name",
    "action": "CONTINUE or SPAWN or UNKNOWN",
    "target_graph": "graph name if SPAWN, null otherwise",
    "seed_event": "if SPAWN — one sentence seed event for new MDP, null otherwise",
    "reason": "one sentence explaining the classification"
  }}
]"""

    result = call_llm(prompt)

    # Groq wraps array in an object — extract the list
    if isinstance(result, dict):
        # find the first value that is a list
        for value in result.values():
            if isinstance(value, list):
                return value
    return result  # already a list


def compute_ring1_state(parent_entity, parent_magnitude, enriched_event, active_graph):

    # get Ring 1 candidates from the active knowledge graph
    kg = KNOWLEDGE_GRAPH_REGISTRY[active_graph]["graph"]

    # check if parent entity has Ring 1 neighbours in this graph
    if parent_entity not in kg:
        return None  # no neighbours in this graph — dead end

    ring1_candidates = kg[parent_entity]["candidates"]

    # apply attenuation to each candidate's seed magnitude
    ALPHA = {
        "fabrication_supplier": 0.50,
        "horizontal_competitor": -0.40,
        "ip_licensor": 0.35,
        "customer_downstream": 0.45,
    }

    # attach attenuated seed magnitude to each candidate
    enriched_candidates = []
    for candidate in ring1_candidates:
        alpha = ALPHA.get(candidate["relationship_type"], 0.30)

        # for customer_downstream, scale by their resource exposure
        # higher dependency on parent = stronger effect
        if candidate["relationship_type"] == "customer_downstream":
            exposure_scale = candidate.get("resource_exposure", 0.15)
            seed_magnitude = round(parent_magnitude * alpha * exposure_scale * 10, 4)
        else:
            seed_magnitude = round(parent_magnitude * alpha, 4)

        # skip if below minimum spawn threshold
        if abs(seed_magnitude) < MIN_SPAWN_MAGNITUDE:
            continue

        enriched_candidate = candidate.copy()
        enriched_candidate["seed_magnitude"] = seed_magnitude
        enriched_candidates.append(enriched_candidate)

    # build the s1 state
    s1 = {
        "event": {
            "description": enriched_event["description"],
            "type": enriched_event["type"],
            "source_entity": parent_entity,
            "ring_depth": 1,
            "causal_direction": enriched_event["causal_direction"],
            "estimated_impact_magnitude": enriched_event["estimated_impact_magnitude"],
        },
        "affected_set": [],
        "magnitude_vector": [],
        "graph_snapshot": {
            "candidates": enriched_candidates,
            "excluded": kg[parent_entity].get("excluded", []),
        },
        "active_graph": active_graph,
        "spawned_from": parent_entity,
    }

    return s1


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────


def run_pipeline(event_text, finbert_tokenizer, finbert_model):
    # ── DYNAMIC STATE CONSTRUCTION ───────────────────────
    s0 = build_dynamic_s0(event_text, finbert_tokenizer, finbert_model)

    # rest of pipeline continues unchanged from here...
    print("\n[RING 0] Running threshold policy...")
    ring0_state = make_ring0_decisions(s0)

    print(f"Affected set: {ring0_state['affected_set']}")
    for mv in ring0_state["magnitude_vector"]:
        print(f"  {mv['entity']}: {mv['magnitude']}")

    # ── ENRICHMENT ──────────────────────────

    if not ring0_state["affected_set"]:
        print("\n[PIPELINE HALT] No entities in affected set.")
        print("Dynamic graph construction would trigger here.")
        print("Cannot proceed with enrichment — no Ring 0 entities discovered.")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Event type: {ring0_state['event']['type']}")
        print(f"Source: {ring0_state['event']['source_entity']}")
        print(f"Active graph: {ring0_state['active_graph']}")
        print(f"Status: HALTED — requires dynamic KG construction")
        print(f"Sub-entities discovered: {len(s0['sub_entities_mentioned'])}")
        for e in s0["sub_entities_mentioned"]:
            print(f"  {e['name']} ({e['ticker']}) — {e['role']}")
        return

    print("\n[ENRICHMENT] Generating enriched event context for Intel...")
    intel_data = get_parent_data(ring0_state, "Intel")
    enrichment_prompt = build_enrichment_prompt(
        "Intel", intel_data, ring0_state["event"]
    )
    enriched_event = call_llm(enrichment_prompt)

    print(f"Enriched event type: {enriched_event['type']}")
    print(f"Description: {enriched_event['description']}")

    # ── SEGMENT DISCOVERY ───────────────────
    print("\n[SEGMENTS] Discovering affected segments...")
    segment_prompt = build_segment_discovery_prompt(
        "Intel", enriched_event, ring0_state["active_graph"]
    )
    segment_result = call_llm(segment_prompt)

    all_segments = (
        segment_result.get("within_domain", [])
        + segment_result.get("cross_domain", [])
        + segment_result.get("macro", [])
    )
    print(f"Segments discovered: {all_segments}")

    # ── SEGMENT CLASSIFICATION ───────────────
    print("\n[CLASSIFY] Classifying affected segments...")
    classifications = classify_segments(
        all_segments, ring0_state["active_graph"], enriched_event
    )

    within_graph = []
    spawns = []
    unknowns = []

    for c in classifications:
        if c["action"] == "CONTINUE":
            within_graph.append(c)
            print(f"  CONTINUE: {c['segment']}")
        elif c["action"] == "SPAWN":
            spawns.append(c)
            print(f"  SPAWN → {c['target_graph']}: {c['segment']}")
        else:
            unknowns.append(c)
            print(f"  UNKNOWN: {c['segment']} — {c['reason']}")

    # ── RING 1 — semiconductor graph ─────────
    print("\n[RING 1] Computing Ring 1 state for Intel (semiconductor graph)...")
    s1_intel = compute_ring1_state(
        "Intel", intel_data["magnitude"], enriched_event, "semiconductor"
    )

    if s1_intel:
        print(f"Ring 1 candidates in semiconductor graph:")
        for c in s1_intel["graph_snapshot"]["candidates"]:
            print(f"  {c['entity']}: seed_magnitude {c['seed_magnitude']}")
    else:
        print("No Ring 1 neighbours found in semiconductor graph.")

    # ── SPAWN — cloud infrastructure MDP ─────
    spawned_graphs = set()
    for spawn in spawns:
        if (
            spawn["target_graph"] == "cloud_infrastructure"
            and "cloud_infrastructure" not in spawned_graphs
        ):
            spawned_graphs.add("cloud_infrastructure")
            print(f"\n[SPAWN] Spawning new MDP in cloud_infrastructure graph...")
            print(f"Seed event: {spawn['seed_event']}")

            s1_cloud = compute_ring1_state(
                "Intel", intel_data["magnitude"], enriched_event, "cloud_infrastructure"
            )

            if s1_cloud:
                print(f"\nCloud Infrastructure Ring 1 candidates:")
                for c in s1_cloud["graph_snapshot"]["candidates"]:
                    print(
                        f"  {c['entity']} ({c['ticker']}): "
                        f"seed_magnitude {c['seed_magnitude']}"
                    )
                print(f"\nSpawned MDP event context:")
                print(f"  Type: {s1_cloud['event']['type']}")
                print(f"  Description: {s1_cloud['event']['description']}")
                print(f"  Spawned from: {s1_cloud['spawned_from']}")

    # ── SUMMARY ──────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Root MDP: Apple M3 → semiconductor graph")
    print(f"Ring 0 affected: {ring0_state['affected_set']}")
    print(f"Child MDP spawned: {len(spawned_graphs)} cross-graph spawn(s)")
    for graph in spawned_graphs:
        print(f"  → {graph}")
    if unknowns:
        print(f"Unknown segments flagged: {len(unknowns)}")
        for u in unknowns:
            print(f"  → {u['segment']}: {u['reason']}")


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # load FinBERT once
    finbert_tokenizer, finbert_model = load_finbert()

    # test event 1 — known financial event (should find KG)
    apple_event = (
        "Apple announces the M3 chip built on TSMC 3nm process "
        "delivering 60% faster CPU performance than M1."
    )

    # test event 2 — Iran war (will NOT find KG — shows dynamic construction gap)
    iran_event = (
        "US and Israel launch strikes on Iran. Strait of Hormuz "
        "effectively closed disrupting 20% of global oil and LNG supply."
    )

    print("\n" + "=" * 60)
    print("TEST 1: Apple M3 Event")
    print("=" * 60)
    run_pipeline(apple_event, finbert_tokenizer, finbert_model)

    print("\n" + "=" * 60)
    print("TEST 2: Iran War Event")
    print("=" * 60)
    run_pipeline(iran_event, finbert_tokenizer, finbert_model)
