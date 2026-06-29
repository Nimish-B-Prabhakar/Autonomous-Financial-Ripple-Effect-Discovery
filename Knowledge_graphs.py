# ─────────────────────────────────────────────
# KNOWLEDGE GRAPHS
# ─────────────────────────────────────────────

SEMICONDUCTOR_KG = {
    "Apple": {
        "candidates": [
            {
                "entity": "TSMC",
                "ticker": "TSM",
                "relationship_type": "fabrication_supplier",
                "revenue_exposure": 0.25,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.0,
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
                "resource_exposure": 0.0,
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
                "resource_exposure": 0.0,
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
                "resource_exposure": 0.0,
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
    "Intel": {
        "candidates": [
            {
                "entity": "AMD",
                "ticker": "AMD",
                "relationship_type": "horizontal_competitor",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.45,
                "resource_exposure": 0.0,
                "historical_correlation": -0.31,
                "direction_prior": "negative",
                "confidence": "high",
            },
            {
                "entity": "ASML",
                "ticker": "ASML",
                "relationship_type": "fabrication_supplier",
                "revenue_exposure": 0.12,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.0,
                "historical_correlation": 0.44,
                "direction_prior": "positive",
                "confidence": "medium",
            },
        ],
        "excluded": [],
    },
}

CLOUD_INFRASTRUCTURE_KG = {
    "Intel": {
        "candidates": [
            {
                "entity": "Amazon AWS",
                "ticker": "AMZN",
                "relationship_type": "customer_downstream",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.22,  # highest — most Intel dependent
                "historical_correlation": 0.34,
                "direction_prior": "negative",
                "confidence": "high",
            },
            {
                "entity": "Microsoft Azure",
                "ticker": "MSFT",
                "relationship_type": "customer_downstream",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.18,  # high — significant Intel deployment
                "historical_correlation": 0.29,
                "direction_prior": "negative",
                "confidence": "high",
            },
            {
                "entity": "Google Cloud",
                "ticker": "GOOGL",
                "relationship_type": "customer_downstream",
                "revenue_exposure": 0.0,
                "competitive_exposure": 0.0,
                "resource_exposure": 0.11,  # lower — more aggressive ARM/AMD adoption
                "historical_correlation": 0.22,
                "direction_prior": "negative",
                "confidence": "medium",
            },
        ],
        "excluded": [],
    }
}

KNOWLEDGE_GRAPH_REGISTRY = {
    "semiconductor": {
        "description": "Covers entities involved in chip design, fabrication, equipment, IP licensing, and direct hardware supply chains. Includes foundries, fabless designers, EDA tools, lithography equipment, and OEMs that directly consume chips as primary inputs.",
        "graph": SEMICONDUCTOR_KG,
        "segments": [
            "semiconductor_manufacturing_equipment",
            "foundry_services",
            "chip_design_tools",
            "ip_licensing",
            "mobile_processors",
            "laptop_oems",
            "memory_manufacturers",
        ],
    },
    "cloud_infrastructure": {
        "description": "Covers hyperscalers, data center operators, enterprise compute infrastructure, cloud services, and networking equipment. Entities whose primary business involves deploying, operating, or selling compute infrastructure at scale.",
        "graph": CLOUD_INFRASTRUCTURE_KG,
        "segments": [
            "enterprise_data_centers",
            "hyperscaler_compute",
            "cloud_services",
            "enterprise_software",
            "networking_equipment",
        ],
    },
}
