"""
Test fine-tuned FinBERT event classifier on Iran war event
and compare with Apple M3 (known financial event) as baseline.
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

# ── config ───────────────────────────────────────────────────────
MODEL_PATH = "../models/finbert-event-classifier"

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

CONFIDENCE_THRESHOLD = 0.85  # below this triggers LLM validation

# ── test events ──────────────────────────────────────────────────
TEST_EVENTS = [
    {
        "label": "Known financial event (Apple M3)",
        "text": "Apple announces the M3 chip built on TSMC 3nm process "
        "delivering 60% faster CPU performance than M1.",
    },
    {
        "label": "Iran war — direct headline",
        "text": "US and Israel launch strikes on Iran. "
        "Strait of Hormuz effectively closed, disrupting "
        "20% of global oil and LNG supply.",
    },
    {
        "label": "Iran war — economic framing",
        "text": "Brent crude oil surges above $100 per barrel as "
        "Strait of Hormuz closure halts Gulf energy exports. "
        "Fertilizer prices spike 30% amid supply disruption.",
    },
    {
        "label": "Iran war — geopolitical framing",
        "text": "US-Israel military conflict with Iran triggers "
        "geopolitical supply shock. Trade sanctions and "
        "international tensions disrupt global energy markets.",
    },
    {
        "label": "Classic supply disruption (control)",
        "text": "TSMC factory fire disrupts chip supply. "
        "Production halt expected to last 3 weeks "
        "affecting Apple and Qualcomm orders.",
    },
]


from transformers import BertModel, BertConfig


class FinBERTEventClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        config = BertConfig.from_pretrained("ProsusAI/finbert")
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token — matches fine_tune_bert.py exactly
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


def load_model(model_path):
    print(f"Loading fine-tuned FinBERT from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    # build empty architecture
    model = FinBERTEventClassifier(num_labels=len(EVENT_TYPE_LABELS))

    # load ALL weights from checkpoint — bert + classifier together
    state_dict = torch.load(
        f"{model_path}/model.pt", map_location=torch.device("cpu"), weights_only=True
    )

    # strict=True will error if any key mismatches — good for debugging
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def classify(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", max_length=128, truncation=True, padding=True
    )
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # no token_type_ids
        )
        probs = F.softmax(logits, dim=-1).squeeze(0)

    scores = {
        label: round(probs[i].item(), 4) for i, label in enumerate(EVENT_TYPE_LABELS)
    }
    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "label": best_label,
        "confidence": best_conf,
        "top3": top3,
        "all_scores": scores,
        "needs_validation": best_conf < CONFIDENCE_THRESHOLD,
    }


# ── run ──────────────────────────────────────────────────────────
def run_test():
    tokenizer, model = load_model(MODEL_PATH)

    print("=" * 65)
    print("FINBERT EVENT CLASSIFICATION TEST")
    print("Confidence threshold for LLM validation:", CONFIDENCE_THRESHOLD)
    print("=" * 65)

    for event in TEST_EVENTS:
        result = classify(event["text"], tokenizer, model)

        print(f"\n[{event['label']}]")
        print(f"Text: {event['text'][:80]}...")
        print(f"Classification: {result['label'].upper()}")
        print(f"Confidence:     {result['confidence']:.4f}", end="  ")

        if result["needs_validation"]:
            print("⚠  BELOW THRESHOLD — LLM VALIDATION NEEDED")
        else:
            print("✓  ABOVE THRESHOLD")

        print("Top 3 predictions:")
        for label, score in result["top3"]:
            bar = "█" * int(score * 40)
            print(f"  {label:<22} {score:.4f}  {bar}")

        # run this right after loading the model
    test_input = "Apple quarterly earnings beat expectations by 20 percent"
    result = classify(test_input, tokenizer, model)
    print("\nDIAGNOSTIC — earnings event:")
    print(result)

    test_input2 = "CEO of Microsoft steps down effective immediately"
    result2 = classify(test_input2, tokenizer, model)
    print("\nDIAGNOSTIC — management change:")
    print(result2)

    print("\n" + "=" * 65)
    print("KEY QUESTION: Does Iran war get classified as 'geopolitical'")
    print("or does it bleed into 'supply_disruption'?")
    print("And what is the confidence score — does it trigger validation?")
    print("=" * 65)


if __name__ == "__main__":
    run_test()
