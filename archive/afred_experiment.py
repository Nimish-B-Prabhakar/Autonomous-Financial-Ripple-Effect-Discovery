# AFRED Proof of Concept
# Experiment: Does structured MDP state improve LLM analysis?
# Event: Apple M3 chip announcement
import json

SEMICONDUCTOR_EVENT_RELATIONSHIP_MAP = {
    "product_launch": [
        "fabrication_supplier",
        "component_supplier",
        "horizontal_competitor",
        "ip_licensor",
        "shared_resource_competitor",
    ],
    "earnings_miss": [
        "fabrication_supplier",
        "component_supplier",
        "horizontal_competitor",
        "customer_downstream",
        "debt_holder",
    ],
    "supply_chain_disruption": [
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
}


s0 = {
    "event": {
        "description": "Apple announces the M3 chip, built on TSMC 3nm process",
        "type": "product_launch",
        "source_entity": "Apple",
        "ring_depth": 0,
        "causal_direction": "Analyze how this event affects each candidate entity. Apple is the cause. Each candidate is a potential effect.",
    },
    "affected_set": [],
    "magnitude_vector": [],
    "graph_snapshot": {
        "candidates": [
            {
                "entity": "TSMC",
                "ticker": "TSM",
                "relationship_type": "fabrication_supplier",
                "revenue_exposure": 0.25,
                "competitive_exposure": 0.0,
                "historical_correlation": 0.61,
                "direction_prior": "positive",
                "confidence": "high",
            },
            {
                "entity": "Intel",
                "ticker": "INTC",
                "relationship_type": "horizontal_competitor",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.38,
                "historical_correlation": -0.23,
                "direction_prior": "negative",
                "confidence": "high",
            },
            {
                "entity": "Qualcomm",
                "ticker": "QCOM",
                "relationship_type": "horizontal_competitor",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.42,
                "historical_correlation": -0.18,
                "direction_prior": "negative",
                "confidence": "medium",
            },
            {
                "entity": "ARM Holdings",
                "ticker": "ARM",
                "relationship_type": "ip_licensor",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.14,
                "historical_correlation": 0.48,
                "direction_prior": "positive",
                "confidence": "medium",
            },
            {
                "entity": "Samsung",
                "ticker": "SSNLF",
                "relationship_type": "dual_role",
                "revenue_exposure": 0.04,
                "competitive_exposure": 0.31,
                "historical_correlation": 0.12,
                "direction_prior": "ambiguous",
                "confidence": "low",
            },
        ],
        "excluded": [
            {
                "entity": "NVIDIA",
                "ticker": "NVDA",
                "reason_excluded": "no_documented_supply_or_competitive_relationship_to_M3_product_line",
            }
        ],
    },
}


def serialize_state_to_prompt(state):
    event = state["event"]
    candidates = state["graph_snapshot"]["candidates"]
    excluded = state["graph_snapshot"]["excluded"]

    # --- Role + Event Header ---
    prompt = f"""You are a financial analyst assistant operating within a structured 
financial ripple effect discovery system.

EVENT INFORMATION:
- Event: {event["description"]}
- Event Type: {event["type"]}
- Source Entity: {event["source_entity"]}
- Analysis Ring: {event["ring_depth"]} (direct relationships only)

"""

    # --- Excluded Entities FIRST ---
    prompt += (
        "PRE-EXCLUDED FROM ANALYSIS (these are not candidates - do not analyze them):\n"
    )
    for exc in excluded:
        prompt += f"- {exc['entity']} ({exc['ticker']}): {exc['reason_excluded']}\n"

    # --- Task + Note + Constraints ---
    prompt += """
TASK:
For each candidate entity listed below, analyze whether it is materially 
affected by this event. For each entity provide:
1. INCLUDE or EXCLUDE decision
2. Direction of impact: positive, negative, or ambiguous
3. Magnitude estimate: high, medium, low, or negligible
4. Causal mechanism: one specific sentence explaining exactly why

NOTE: All four points are required for every candidate entity. 
An EXCLUDE decision does not skip points 2, 3, and 4 — 
you must still provide direction, magnitude, and causal mechanism 
explaining specifically why this entity is not materially affected.

CONSTRAINTS:
- Do not add any entities outside the candidate list
- Do not contradict the relationship type without explicit justification
- If direction_prior is ambiguous, you must explain both forces
- Base your reasoning on the structured data provided, not general knowledge

"""

    # --- Candidate Clarifier BEFORE loop ---
    prompt += """CANDIDATE ENTITIES:
These entities have documented relationships with the source entity 
and ARE eligible for analysis. A candidate with zero revenue_exposure 
is not excluded — it may have competitive or resource exposure instead.
Analyze each one fully.\n"""

    # --- Candidate Loop ---
    for candidate in candidates:
        prompt += f"\n[{candidate['entity']} | {candidate['ticker']}]"
        prompt += f"\n  Relationship Type: {candidate['relationship_type']}"

        if candidate.get("revenue_exposure", 0) > 0:
            prompt += f"\n  Revenue Exposure: {candidate['revenue_exposure']:.0%} of revenue tied to source entity"
        if candidate.get("competitive_exposure", 0) > 0:
            prompt += f"\n  Competitive Exposure: {candidate['competitive_exposure']:.0%} of revenue in directly competing segments"
        if candidate.get("resource_exposure", 0) > 0:
            prompt += f"\n  Resource Exposure: {candidate['resource_exposure']:.0%} of revenue from licensing to source entity"

        prompt += f"\n  Historical Correlation: {candidate['historical_correlation']} (1.0 = perfect co-movement, -1.0 = perfect inverse)"
        prompt += f"\n  Direction Prior: {candidate['direction_prior']}"
        prompt += f"\n  Confidence Level: {candidate['confidence']}\n"
        prompt += f"- Causal Direction: {event['causal_direction']}\n"

    # --- Output Format ---
    prompt += """
OUTPUT FORMAT (follow this exactly for every candidate):
[ENTITY NAME]
- Decision: INCLUDE or EXCLUDE
- Direction: positive / negative / ambiguous
- Magnitude: high / medium / low / negligible
- Mechanism: [one specific sentence]

---
Provide your analysis for each candidate entity now:"""

    return prompt


prompt = serialize_state_to_prompt(s0)


# ─────────────────────────────────────────────
# MDP MATH PIPELINE — no LLM involved
# ─────────────────────────────────────────────


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


"""

So for TSMC:
```
revenue: 0.25, competitive: 0.0, resource: 0.0
→ primary = revenue, value = 0.25
```

For Intel:
```
revenue: 0.0, competitive: 0.38, resource: 0.0
→ primary = competitive, value = 0.38
```

For Samsung:
```
revenue: 0.04, competitive: 0.31, resource: 0.0
→ primary = competitive, value = 0.31

"""

"""

The function should:
- Loop through candidates in the graph snapshot
- For each candidate check if any exposure field exceeds its threshold
- Return a list of decisions — each one being a dict with entity, ticker, decision, magnitude, direction

Thresholds to use:

revenue_exposure     > 0.05
competitive_exposure > 0.15
resource_exposure    > 0.08

magnitude_j = primary_exposure_j × event_significance × direction_sign

"""


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


ring_0_state = make_ring0_decisions(s0)


ALPHA = {
    "fabrication_supplier": 0.50,
    "horizontal_competitor": 0.40,
    "ip_licensor": 0.35,
    "customer_downstream": 0.45,
}

RING1_GRAPH = {
    "TSMC": {
        "entity": "ASML",
        "ticker": "ASML",
        "relationship_type": "fabrication_supplier",
        "alpha": 0.50,
        "direction_prior": "positive",
        "confidence": "medium",
    },
    "Intel": {
        "entity": "AMD",
        "ticker": "AMD",
        "relationship_type": "horizontal_competitor",
        "alpha": -0.40,
        "direction_prior": "negative",
        "confidence": "high",
    },
    "Qualcomm": {
        "entity": "MediaTek",
        "ticker": "MTKI",
        "relationship_type": "horizontal_competitor",
        "alpha": -0.40,
        "direction_prior": "negative",
        "confidence": "medium",
    },
    "ARM Holdings": {
        "entity": "NVIDIA",
        "ticker": "NVDA",
        "relationship_type": "ip_licensor",
        "alpha": 0.35,
        "direction_prior": "positive",
        "confidence": "low",
    },
}


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


intel_data = get_parent_data(ring_0_state, "Intel")


def build_enrichment_prompt(parent_entity, parent_data, original_event):
    prompt = f"""You are operating within a financial ripple effect 
discovery system. Your task is to generate an enriched event context.

ORIGINAL EVENT:
- Description: {original_event["description"]}
- Type: {original_event["type"]}
- Source: {original_event["source_entity"]}

RING 0 FINDING FOR {parent_entity}:
- Relationship to source: {parent_data["relationship_type"]}
- Magnitude: {parent_data["magnitude"]}
- Direction: {parent_data["direction"]}
- Competitive exposure: {parent_data.get("competitive_exposure", 0)}
- Revenue exposure: {parent_data.get("revenue_exposure", 0)}

TASK:
Generate the enriched event context that {parent_entity}'s direct 
neighbours would experience. This is NOT Apple's announcement — it 
is the downstream consequence that {parent_entity}'s situation 
creates for its own network.

Return ONLY a JSON object with exactly these fields, no other text:

{{
    "description": "one specific sentence describing what {parent_entity}'s neighbours experience",
    "type": "one of: revenue_pressure / demand_increase / market_share_shift / royalty_increase / capex_reduction",
    "source_entity": "{parent_entity}",
    "ring_depth": 1,
    "causal_direction": "one sentence — who is the cause and who are the potential effects",
    "estimated_impact_magnitude": <float between 0 and 1>,
    "affected_segments": ["list", "of", "segments"],
    "confidence": "high / medium / low"
}}"""
    return prompt


enrichment_prompt = build_enrichment_prompt("Intel", intel_data, s0["event"])
print(enrichment_prompt)
