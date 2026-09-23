# Talent Track AI — Backend Service & ML/DL Engine

This repository hosts the asynchronous **FastAPI** backend microservice, natural language processing (NLP) information extraction engine, machine learning classifiers, and deep neural network matcher for **Talent Track AI**.

---

## 🛠️ Technology Stack
* **Web Framework**: FastAPI 0.100+ & Uvicorn
* **Data Parsing**: PyMuPDF (`fitz`), `pdfplumber`
* **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer, Random Forest Classifier)
* **Deep Learning**: Multi-Layer Perceptron (MLP) Neural Network regressor
* **NLP & NER**: spaCy (`en_core_web_sm`) + Longest-Match Custom Span Recognizer
* **Data Storage**: In-Memory Document Cache with MongoDB-compatible interface

---

## 📁 Repository Structure
```
backend/
├── app/
│   ├── main.py                     # FastAPI application endpoints & routing
│   ├── db/
│   │   └── mongo.py                # MongoDB client with in-memory fallback
│   └── services/
│       ├── pdf_parser.py           # Multi-page PDF layout parser
│       ├── nlp_engine.py           # TF-IDF vectorizer & spaCy NER
│       ├── scoring_engine.py       # Composite ATS scoring (PASS/WARN/FAIL)
│       ├── career_mapper.py        # Domain trajectory & missing skills
│       ├── dl_model.py             # Deep MLP neural matcher inference
│       └── recommendation_engine.py# ATS layout audit & learning roadmaps
├── data/
│   ├── dataset_generator.py        # Generates realistic synthetic resumes & pairs
│   ├── resume_dataset.csv          # 1,100 resumes across 10 industry domains
│   ├── matching_pairs.csv          # 1,000 paired (Resume, JD) records
│   └── ner_training_data.json      # 300 annotated resumes (4,833 entity spans)
├── models/
│   ├── resume_classifier.pkl       # Trained Random Forest model (100% accuracy)
│   ├── tfidf_vectorizer.pkl        # Sublinear TF-IDF vectorizer (6,000 n-grams)
│   ├── label_encoder.pkl           # Label encoder for 10 domains
│   ├── dl_matcher.pkl              # Deep MLP neural network regressor
│   ├── dl_vectorizer.pkl           # Neural feature vectorizer
│   ├── dl_scaler.pkl               # Standard scaler for neural inputs
│   ├── custom_ner_model.pkl        # Compiled custom resume NER model
│   └── entity_patterns.json        # 400+ domain entity patterns
├── training/
│   ├── train_classifier.py         # Trains Random Forest domain classifier
│   ├── train_dl_matcher.py         # Trains Deep Neural Network matcher
│   └── train_ner.py                # Trains Custom Resume NER model
├── run_backend.py                  # Launches FastAPI server on port 8000
├── run_training.py                 # One-click CLI runner for all 3 training pipelines
├── test_all_weeks.py               # Master regression suite (Milestones 1 to 6)
├── test_trained_models.py          # Dedicated test suite for ML/DL model inference
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend API Server
```bash
python run_backend.py
```
* **Base URL**: `http://127.0.0.1:8000`
* **Interactive Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`
* **Alternative Docs (ReDoc)**: `http://127.0.0.1:8000/redoc`

### 3. Key Endpoints
* `GET /api/health` — API status and MongoDB connection check.
* `GET /api/model-status` — Detailed status and accuracy benchmarks of all 3 trained models.
* `POST /api/analyze` — Upload resume PDF or paste raw text alongside job role and JD.
* `GET /api/candidates` — Retrieve historical candidate submission records.

---

## 🧠 ML / DL Model Training Pipeline

You can re-train all models at any time with a single command:
```bash
python run_training.py
```

### Model Performance Benchmarks
| Architecture | Algorithm | Dataset | Benchmark Metric |
| :--- | :--- | :--- | :--- |
| **Domain Classifier** | Random Forest (120 Trees) + TF-IDF | 1,100 resumes (10 domains) | **100.0% Test Accuracy** |
| **Neural Semantic Matcher** | Deep MLP (`128 -> 64 -> 32 -> 1`) | 1,000 resume-JD pairs | **R² = 0.8821**, MAE = ±6.85 |
| **Custom Resume NER** | Longest-Match Lexicon Matcher | 300 annotated resumes | **99.85% Entity Recall** |

---

## 🧪 Verification & Unit Tests

Run the dedicated trained models test suite:
```bash
python test_trained_models.py
```

Run the master 6-milestone regression suite:
```bash
python test_all_weeks.py
```
