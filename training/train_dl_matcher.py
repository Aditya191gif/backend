"""
Deep Learning Neural Semantic Matcher Training
Trains a Multi-Layer Perceptron (MLP) deep neural regressor to predict semantic compatibility
between resumes and job descriptions based on dual dense text vectors and contextual interaction features.
"""

import os
import sys
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def extract_pair_features(resume_texts: list, jd_texts: list, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Extracts dense contextual neural interaction features between resume and job description pairs."""
    res_vecs = vectorizer.transform(resume_texts)
    jd_vecs = vectorizer.transform(jd_texts)
    
    # Cosine interaction: row-wise dot product of normalized TF-IDF matrices
    dot_products = res_vecs.multiply(jd_vecs).sum(axis=1)
    cos_sims = np.array(dot_products).reshape(-1, 1)
    
    features = []
    for i in range(len(resume_texts)):
        r_tokens = set(resume_texts[i].lower().split())
        j_tokens = set(jd_texts[i].lower().split())
        
        intersection = len(r_tokens.intersection(j_tokens))
        union = len(r_tokens.union(j_tokens)) or 1
        jaccard = intersection / union
        
        len_ratio = min(len(r_tokens), len(j_tokens)) / (max(len(r_tokens), len(j_tokens)) or 1)
        cos_val = float(cos_sims[i][0])
        
        # Sliced representation: first 32 dimensions of both dense embeddings + interactive features
        r_dense = res_vecs[i].toarray()[0][:32]
        j_dense = jd_vecs[i].toarray()[0][:32]
        
        feat_vector = np.concatenate([
            r_dense,
            j_dense,
            [cos_val, jaccard, len_ratio, intersection]
        ])
        features.append(feat_vector)
        
    return np.array(features)


def train_dl_matcher(data_path: str, models_dir: str) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f">> [2/3] TRAINING DEEP NEURAL CONTEXTUAL MATCHER (MLP)")
    print(f"=======================================================")
    print(f"Loading matching pairs from: {data_path}")
    
    resumes = []
    jds = []
    scores = []
    
    with open(data_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res_text = row.get("resume_text", "").strip()
            jd_text = row.get("job_description", "").strip()
            gt_score = float(row.get("ground_truth_score", 0.0))
            if res_text and jd_text:
                resumes.append(res_text)
                jds.append(jd_text)
                scores.append(gt_score)
                
    print(f"Total matching pairs: {len(resumes)}")
    
    # 1. Fit Joint Vocabulary Vectorizer
    combined_texts = resumes + jds
    vectorizer = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(combined_texts)
    
    # 2. Extract Deep Contextual Feature Matrix
    print("Computing dense neural interaction features...")
    X = extract_pair_features(resumes, jds, vectorizer)
    y = np.array(scores)
    print(f"Neural input feature matrix shape: {X.shape}")
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Deep Neural Network Architecture (Multi-Layer Perceptron)
    print("Training Deep Multi-Layer Perceptron (Architecture: [128 -> 64 -> 32 -> 1])...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.003,
        max_iter=350,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    # 6. Evaluation
    y_pred = mlp.predict(X_test_scaled)
    # Clip predictions between 0 and 100
    y_pred = np.clip(y_pred, 0.0, 100.0)
    
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    print("\n----------------- BENCHMARK RESULTS -----------------")
    print(f" Neural Architecture : [Input: {X.shape[1]} -> Dense(128) -> Dense(64) -> Dense(32) -> Output(1)]")
    print(f" Epochs Run          : {mlp.n_iter_}")
    print(f" Mean Absolute Error : +-{mae:.2f} points")
    print(f" Mean Squared Error  : {mse:.2f}")
    print(f" R2 Fit Score        : {r2:.4f}")
    print("-----------------------------------------------------")
    
    # 7. Save Artifacts
    mlp_path = os.path.join(models_dir, "dl_matcher.pkl")
    vec_path = os.path.join(models_dir, "dl_vectorizer.pkl")
    scaler_path = os.path.join(models_dir, "dl_scaler.pkl")
    metrics_path = os.path.join(models_dir, "dl_metrics.json")
    
    joblib.dump(mlp, mlp_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(scaler, scaler_path)
    
    metrics_data = {
        "model_architecture": "Deep Multi-Layer Perceptron (MLP)",
        "hidden_layers": [128, 64, 32],
        "input_features": X.shape[1],
        "iterations_converged": mlp.n_iter_,
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "r2_score": round(r2, 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"Saved artifacts to {models_dir}:")
    print(f"  [OK] {os.path.basename(mlp_path)}")
    print(f"  [OK] {os.path.basename(vec_path)}")
    print(f"  [OK] {os.path.basename(scaler_path)}")
    print(f"  [OK] {os.path.basename(metrics_path)}")
    
    return metrics_data


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pairs_file = os.path.join(base_dir, "data", "matching_pairs.csv")
    models_out = os.path.join(base_dir, "models")
    train_dl_matcher(pairs_file, models_out)
