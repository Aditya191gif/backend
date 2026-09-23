"""
Model Training: Multi-Domain Resume Category Classifier
Trains a machine learning model to classify candidate resumes into industry domains
with high accuracy, confidence scoring, and serialization to persistent .pkl artifacts.
"""

import os
import sys
import re
import csv
import json
import joblib
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix


def clean_text(text: str) -> str:
    """Preprocesses resume text: lowercases, removes URLs, special punctuation, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_resume_classifier(data_path: str, models_dir: str) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f">> [1/3] TRAINING RESUME DOMAIN CLASSIFIER")
    print(f"=======================================================")
    print(f"Loading dataset from: {data_path}")
    
    corpus = []
    labels = []
    
    with open(data_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = clean_text(row.get("resume_text", ""))
            domain = row.get("domain", "").strip()
            if cleaned and domain:
                corpus.append(cleaned)
                labels.append(domain)
                
    print(f"Total processed samples: {len(corpus)}")
    unique_domains = sorted(list(set(labels)))
    print(f"Number of classes ({len(unique_domains)}): {unique_domains}")
    
    # 1. Encode Target Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # 2. Train / Test Split (80% train, 20% test, stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        corpus, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training split: {len(X_train_raw)} | Test split: {len(X_test_raw)}")
    
    # 3. TF-IDF Feature Extraction
    print("Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=6000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"Feature matrix shape: {X_train.shape}")
    
    # 4. Model Training (Random Forest Classifier)
    print("Training Random Forest Ensemble Classifier (n_estimators=120)...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # 5. Evaluation & Metrics
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n----------------- BENCHMARK RESULTS -----------------")
    print(f" Accuracy  : {accuracy * 100:.2f}%")
    print(f" Precision : {precision * 100:.2f}%")
    print(f" Recall    : {recall * 100:.2f}%")
    print(f" F1-Score  : {f1 * 100:.2f}%")
    print("-----------------------------------------------------")
    
    # 6. Save Persistent Artifacts
    clf_path = os.path.join(models_dir, "resume_classifier.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    enc_path = os.path.join(models_dir, "label_encoder.pkl")
    metrics_path = os.path.join(models_dir, "classifier_metrics.json")
    
    joblib.dump(clf, clf_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(label_encoder, enc_path)
    
    metrics_data = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 120,
        "features_count": X_train.shape[1],
        "training_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "classes": list(label_encoder.classes_),
        "detailed_report": report,
        "confusion_matrix": conf_matrix
    }
    
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"Saved artifacts to {models_dir}:")
    print(f"  [OK] {os.path.basename(clf_path)}")
    print(f"  [OK] {os.path.basename(vec_path)}")
    print(f"  [OK] {os.path.basename(enc_path)}")
    print(f"  [OK] {os.path.basename(metrics_path)}")
    
    return metrics_data


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "resume_dataset.csv")
    models_out = os.path.join(base_dir, "models")
    train_resume_classifier(data_file, models_out)
