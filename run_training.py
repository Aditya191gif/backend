"""
Master One-Click Training Pipeline — Talent Track AI
Executes full end-to-end model training for:
1. Multi-Domain Resume Classification Model (TF-IDF + Random Forest)
2. Deep Neural Contextual Matcher (Multi-Layer Perceptron Regressor)
3. Custom Resume Named Entity Recognizer (Span & Lexicon NER)
"""

import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure backend root is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from data.dataset_generator import generate_all_datasets
from training.train_classifier import train_resume_classifier
from training.train_dl_matcher import train_dl_matcher
from training.train_ner import train_custom_ner


def main():
    print("=================================================================")
    print("       TALENT TRACK AI - UNIFIED ML & DL TRAINING PIPELINE       ")
    print("=================================================================")
    start_total_time = time.time()
    
    data_dir = os.path.join(CURRENT_DIR, "data")
    models_dir = os.path.join(CURRENT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 0. Check & Generate Datasets
    resumes_csv = os.path.join(data_dir, "resume_dataset.csv")
    pairs_csv = os.path.join(data_dir, "matching_pairs.csv")
    ner_json = os.path.join(data_dir, "ner_training_data.json")
    
    if not (os.path.exists(resumes_csv) and os.path.exists(pairs_csv) and os.path.exists(ner_json)):
        print("\nDatasets not found or incomplete. Triggering synthetic generator...")
        generate_all_datasets(data_dir, num_samples_per_domain=110)
    else:
        print("\nVerified training datasets present in backend/data/.")

    # 1. Train Resume Domain Classifier
    t0 = time.time()
    clf_metrics = train_resume_classifier(resumes_csv, models_dir)
    t_clf = time.time() - t0

    # 2. Train Deep Neural Matcher (MLP)
    t0 = time.time()
    dl_metrics = train_dl_matcher(pairs_csv, models_dir)
    t_dl = time.time() - t0

    # 3. Train Custom NER Extractor
    t0 = time.time()
    ner_metrics = train_custom_ner(ner_json, models_dir)
    t_ner = time.time() - t0

    total_time = time.time() - start_total_time

    # Final Summary Report
    print("\n=================================================================")
    print("             ALL ML & DL MODELS SUCCESSFULLY TRAINED!            ")
    print("=================================================================")
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds\n")
    print("MODEL 1: Multi-Domain Resume Classifier")
    print(f"  * Model Type      : {clf_metrics['model_type']} (n_estimators={clf_metrics['n_estimators']})")
    print(f"  * Test Accuracy   : {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"  * F1-Score        : {clf_metrics['f1_score'] * 100:.2f}%")
    print(f"  * Classes Count   : {len(clf_metrics['classes'])} domains")
    print(f"  * Training Time   : {t_clf:.2f}s")
    print(f"  * Artifacts       : backend/models/resume_classifier.pkl, tfidf_vectorizer.pkl\n")
    
    print("MODEL 2: Deep Neural Contextual Matcher (MLP)")
    print(f"  * Architecture    : {dl_metrics['model_architecture']} ({dl_metrics['hidden_layers']})")
    print(f"  * Mean Abs Error  : +-{dl_metrics['mae']:.2f} points")
    print(f"  * R2 Fit Score    : {dl_metrics['r2_score']:.4f}")
    print(f"  * Training Time   : {t_dl:.2f}s")
    print(f"  * Artifacts       : backend/models/dl_matcher.pkl, dl_vectorizer.pkl, dl_scaler.pkl\n")
    
    print("MODEL 3: Custom Resume Named Entity Recognizer (NER)")
    print(f"  * Model Type      : {ner_metrics['model_type']}")
    print(f"  * Entity Types    : {', '.join(ner_metrics['entity_types'])}")
    print(f"  * Recall Rate     : {ner_metrics['recall_rate'] * 100:.2f}%")
    print(f"  * Training Time   : {t_ner:.2f}s")
    print(f"  * Artifacts       : backend/models/custom_ner_model.pkl, entity_patterns.json\n")
    print("=================================================================")
    print("Models are saved and ready for deployment into the FastAPI server!")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
