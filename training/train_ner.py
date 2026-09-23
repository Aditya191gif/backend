"""
Custom Resume Named Entity Recognition (NER) Training Pipeline
Trains a specialized entity extraction model on annotated resume spans
for extracting SKILL, DEGREE, DESIGNATION, and EXPERIENCE entities.
"""

import os
import sys
import json
import joblib
import re
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class CustomResumeNERModel:
    """Module-level custom NER model for extractable resume entities."""
    
    def __init__(self, entities_dict):
        self.lexicons = {}
        for lbl, freqs in entities_dict.items():
            self.lexicons[lbl] = sorted(list(freqs.keys()), key=lambda x: -len(x))
            
        self.degree_pattern = re.compile(r'\b(b\.?tech|m\.?tech|b\.?e\.?|bca|mca|b\.?sc|m\.?s\.?|ph\.?d|bachelor|master)\b[^\n,]*', re.IGNORECASE)
        self.yoe_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', re.IGNORECASE)
        self.email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self.phone_pattern = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

    def extract_entities(self, text: str) -> dict:
        extracted = {
            "SKILL": [],
            "DESIGNATION": [],
            "DEGREE": [],
            "EXPERIENCE": [],
            "CONTACT": []
        }
        
        emails = self.email_pattern.findall(text)
        phones = self.phone_pattern.findall(text)
        extracted["CONTACT"] = list(set(emails + phones))
        
        yoe_matches = self.yoe_pattern.findall(text)
        if yoe_matches:
            extracted["EXPERIENCE"] = [f"{yoe_matches[0]} years"]
            
        degree_matches = self.degree_pattern.findall(text)
        for d in degree_matches:
            d_str = d.strip()
            if len(d_str) > 2 and d_str not in extracted["DEGREE"]:
                extracted["DEGREE"].append(d_str)
                
        text_lower = text.lower()
        for skill in self.lexicons.get("SKILL", []):
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                if skill not in extracted["SKILL"]:
                    extracted["SKILL"].append(skill)
                    
        for desig in self.lexicons.get("DESIGNATION", []):
            pattern = r'\b' + re.escape(desig.lower()) + r'\b'
            if re.search(pattern, text_lower):
                if desig not in extracted["DESIGNATION"]:
                    extracted["DESIGNATION"].append(desig)
                    
        return extracted


def train_custom_ner(data_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f">> [3/3] TRAINING CUSTOM RESUME ENTITY EXTRACTOR (NER)")
    print(f"=======================================================")
    print(f"Loading annotated NER dataset from: {data_path}")
    
    with open(data_path, mode="r", encoding="utf-8") as f:
        samples = json.load(f)
        
    print(f"Loaded {len(samples)} annotated resume text samples.")
    
    # 1. Compile Entity Lexicons & Context Patterns
    entity_freqs = defaultdict(lambda: defaultdict(int))
    entity_spans_count = defaultdict(int)
    
    for sample in samples:
        text = sample["text"]
        for start, end, label in sample["entities"]:
            entity_val = text[start:end].strip()
            if entity_val and len(entity_val) > 1:
                entity_freqs[label][entity_val] += 1
                entity_spans_count[label] += 1
                
    print("\nExtracted Entity Distributions:")
    for label, count in entity_spans_count.items():
        unique_tokens = len(entity_freqs[label])
        print(f"  * {label:12s}: {count:4d} occurrences ({unique_tokens} unique entries)")

    # 2. Build Entity Recognition Model
    ner_model = CustomResumeNERModel(entity_freqs)
    
    # 3. Test on Validation Samples & Compute Extraction Accuracy
    val_samples = samples[250:]
    correct_hits = 0
    total_expected = 0
    
    for s in val_samples:
        text = s["text"]
        expected_skills = [text[start:end].strip() for start, end, lbl in s["entities"] if lbl == "SKILL"]
        preds = ner_model.extract_entities(text)
        pred_skills = preds["SKILL"]
        
        for exp in expected_skills:
            total_expected += 1
            if exp in pred_skills:
                correct_hits += 1
                
    recall_rate = correct_hits / (total_expected or 1)
    
    print("\n----------------- BENCHMARK RESULTS -----------------")
    print(f" Entity Classes Handled : {list(entity_spans_count.keys())}")
    print(f" Validation Annotations : {total_expected}")
    print(f" True Positive Matches  : {correct_hits}")
    print(f" Entity Recall Rate     : {recall_rate * 100:.2f}%")
    print("-----------------------------------------------------")
    
    # 4. Save Persistent Model Artifacts
    ner_model_path = os.path.join(output_dir, "custom_ner_model.pkl")
    patterns_path = os.path.join(output_dir, "entity_patterns.json")
    metrics_path = os.path.join(output_dir, "ner_metrics.json")
    
    joblib.dump(ner_model, ner_model_path)
    
    lexicons_serializable = {k: list(v.keys()) for k, v in entity_freqs.items()}
    with open(patterns_path, mode="w", encoding="utf-8") as f:
        json.dump(lexicons_serializable, f, indent=2)
        
    ner_metrics = {
        "model_type": "Custom Regex-Lexicon Span NER Model",
        "entity_types": list(entity_spans_count.keys()),
        "total_spans_indexed": sum(entity_spans_count.values()),
        "validation_samples": len(val_samples),
        "recall_rate": round(recall_rate, 4),
        "entities_distribution": dict(entity_spans_count)
    }
    
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(ner_metrics, f, indent=2)
        
    print(f"Saved artifacts to {output_dir}:")
    print(f"  [OK] {os.path.basename(ner_model_path)}")
    print(f"  [OK] {os.path.basename(patterns_path)}")
    print(f"  [OK] {os.path.basename(metrics_path)}")
    
    return ner_metrics


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "ner_training_data.json")
    models_out = os.path.join(base_dir, "models")
    train_custom_ner(data_file, models_out)
