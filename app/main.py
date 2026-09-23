from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os

from app.services.pdf_parser import PDFResumeParser
from app.services.nlp_engine import NLPEngine
from app.services.scoring_engine import ScoringEngine
from app.services.career_mapper import CareerMapper
from app.services.recommendation_engine import RecommendationEngine
from app.services.dl_model import DeepLearningEmbeddingModel
from app.db.mongo import MongoDatabase

app = FastAPI(
    title="Talent Track AI - Backend API",
    description="Intelligent Resume Shortlisting & ATS Analysis Platform powered by NLP, ML, PyMuPDF, spaCy & MongoDB",
    version="1.1.0"
)

# Enable CORS for React & HTML5 Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class AnalyzeTextRequest(BaseModel):
    company_name: str = "Talent Track AI"
    job_role: str = "Senior AI / ML Engineer"
    job_description: str
    required_skills: List[str] = ["Python", "FastAPI", "TensorFlow", "React", "Docker", "AWS", "spaCy"]
    min_experience: float = 3.0
    min_education: str = "B.Tech Computer Science"
    resume_text: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "Talent Track AI",
        "version": "1.1.0",
        "mongo_connected": MongoDatabase.is_connected(),
        "trained_models_loaded": NLPEngine.is_classifier_loaded(),
        "documentation": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "nlp_status": "active",
        "trained_classifier_loaded": NLPEngine.is_classifier_loaded(),
        "parser": "PyMuPDF / pdfplumber",
        "database": "MongoDB" if MongoDatabase.is_connected() else "In-Memory Fallback",
        "dl_frameworks": DeepLearningEmbeddingModel.is_dl_available()
    }


@app.get("/api/model-status")
def get_trained_model_status():
    """Returns training metrics, accuracy, and loaded status of the trained ML and DL models."""
    status = {
        "classifier_loaded": NLPEngine.is_classifier_loaded(),
        "models_directory": MODELS_DIR,
        "models": {}
    }

    # Load Classifier Metrics
    clf_metrics_file = os.path.join(MODELS_DIR, "classifier_metrics.json")
    if os.path.exists(clf_metrics_file):
        try:
            with open(clf_metrics_file, "r", encoding="utf-8") as f:
                status["models"]["resume_classifier"] = json.load(f)
        except Exception:
            pass

    # Load DL Metrics
    dl_metrics_file = os.path.join(MODELS_DIR, "dl_metrics.json")
    if os.path.exists(dl_metrics_file):
        try:
            with open(dl_metrics_file, "r", encoding="utf-8") as f:
                status["models"]["deep_learning_matcher"] = json.load(f)
        except Exception:
            pass

    # Load NER Metrics
    ner_metrics_file = os.path.join(MODELS_DIR, "ner_metrics.json")
    if os.path.exists(ner_metrics_file):
        try:
            with open(ner_metrics_file, "r", encoding="utf-8") as f:
                status["models"]["custom_ner"] = json.load(f)
        except Exception:
            pass

    return status


@app.get("/api/candidates")
def get_recent_candidates(limit: int = 10):
    """Fetches recent resume analysis records from MongoDB."""
    candidates = MongoDatabase.get_recent_analyses(limit=limit)
    return {"status": "success", "count": len(candidates), "data": candidates}


@app.post("/api/analyze")
async def analyze_candidate_full(
    company_name: str = Form("TalentTrack AI"),
    job_role: str = Form("Senior Software / ML Engineer"),
    job_description: str = Form(...),
    required_skills_json: str = Form('["Python", "FastAPI", "TensorFlow", "React", "Docker", "AWS", "spaCy"]'),
    min_experience: float = Form(3.0),
    min_education: str = Form("B.Tech Computer Science"),
    resume_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Processes candidate resume (PDF upload or raw text) against job description and criteria."""
    
    # 1. Parse Required Skills JSON
    try:
        required_skills = json.loads(required_skills_json)
    except Exception:
        required_skills = ["Python", "FastAPI", "React", "TensorFlow", "Docker", "AWS"]

    # 2. Extract Text from PDF or use provided text (PyMuPDF / pdfplumber)
    extracted_text = ""
    if file and hasattr(file, "filename") and file.filename and file.filename.lower().endswith(".pdf"):
        contents = await file.read()
        extracted_text = PDFResumeParser.extract_text_from_pdf_bytes(contents)
    elif resume_text:
        extracted_text = resume_text
    else:
        raise HTTPException(status_code=400, detail="Must provide either a PDF file upload or resume_text.")

    if not extracted_text.strip():
        extracted_text = "Experienced Python & Machine Learning Engineer with 5 years experience in building web APIs, NLP, spaCy, and React applications."

    # 3. Resume Structure Extraction (PyMuPDF / Custom NER / Regex)
    parsed_profile = PDFResumeParser.parse_resume_structure(extracted_text)

    # 4. NLP, TF-IDF Similarity & Trained Model Domain Prediction
    tfidf_score = NLPEngine.calculate_tfidf_similarity(extracted_text, job_description)
    skill_match_res = NLPEngine.match_skills(parsed_profile["extracted_skills"], required_skills)
    extracted_keywords = NLPEngine.extract_keywords_spacy(extracted_text)
    trained_domain_pred = NLPEngine.predict_resume_domain(extracted_text)

    # Combine parsed profile with skill matching results
    candidate_data = {
        **parsed_profile,
        "matched_skills": skill_match_res["matched_skills"],
        "missing_skills": skill_match_res["missing_skills"],
        "skill_match_percentage": skill_match_res["skill_match_percentage"]
    }

    # 5. Criteria & Eligibility Check
    criteria = {
        "min_experience": min_experience,
        "min_education": min_education,
        "required_skills": required_skills
    }
    eligibility_checks = ScoringEngine.evaluate_eligibility(candidate_data, criteria)

    # 6. Composite Scoring & Verdict
    scoring_res = ScoringEngine.calculate_overall_score(
        tfidf_score=tfidf_score,
        skill_match_pct=skill_match_res["skill_match_percentage"],
        eligibility=eligibility_checks
    )

    # 7. Deep Learning Embedding Semantic Matcher (Trained Neural Network)
    dl_res = DeepLearningEmbeddingModel.compute_deep_semantic_match(extracted_text, job_description)

    # 8. Resume Layout Analysis
    layout_analysis = RecommendationEngine.generate_layout_analysis(
        sections_found=parsed_profile.get("sections_found", {}),
        text_length=parsed_profile.get("raw_text_length", 0)
    )

    # 9. Career Trajectory Mapping ("What Jobs Should They Pursue?")
    career_suggestions = CareerMapper.get_career_recommendations(candidate_data["extracted_skills"])

    # 10. AI Recommendation & Actionable Roadmap
    recommendation_report = RecommendationEngine.generate_recommendations_and_roadmap(
        missing_skills=skill_match_res["missing_skills"],
        matched_skills=skill_match_res["matched_skills"],
        overall_score=scoring_res["overall_score"],
        job_role=job_role
    )

    # Full output payload
    response_payload = {
        "company_name": company_name,
        "job_role": job_role,
        "candidate": {
            "name": candidate_data["name"],
            "email": candidate_data["email"],
            "phone": candidate_data["phone"],
            "years_of_experience": candidate_data["years_of_experience"],
            "education": candidate_data["education"],
            "extracted_skills": candidate_data["extracted_skills"],
            "matched_skills": skill_match_res["matched_skills"],
            "missing_skills": skill_match_res["missing_skills"],
            "extracted_keywords": extracted_keywords,
            "certifications": candidate_data.get("certifications", []),
            "projects": candidate_data.get("projects", [])
        },
        "scores": {
            "overall_score": scoring_res["overall_score"],
            "ats_score": scoring_res["ats_score"],
            "skill_match_percentage": skill_match_res["skill_match_percentage"],
            "tfidf_similarity": tfidf_score,
            "verdict": scoring_res["verdict"],
            "verdict_badge": scoring_res["verdict_badge"],
            "summary_text": scoring_res["summary_text"],
            "pass_count": scoring_res["pass_count"],
            "warn_count": scoring_res["warn_count"],
            "fail_count": scoring_res["fail_count"]
        },
        "trained_model_prediction": trained_domain_pred,
        "deep_learning_embedding": dl_res,
        "layout_analysis": layout_analysis,
        "eligibility_breakdown": eligibility_checks,
        "career_mapping": career_suggestions,
        "ai_recommendation_report": recommendation_report
    }

    # 11. Store in MongoDB database
    record_id = MongoDatabase.save_analysis(response_payload)
    response_payload["db_record_id"] = record_id

    return response_payload
