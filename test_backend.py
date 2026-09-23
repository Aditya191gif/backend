import sys
import os

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.services.pdf_parser import PDFResumeParser
from app.services.nlp_engine import NLPEngine
from app.services.scoring_engine import ScoringEngine
from app.services.career_mapper import CareerMapper
from app.services.recommendation_engine import RecommendationEngine
from app.services.dl_model import DeepLearningEmbeddingModel
from app.db.mongo import MongoDatabase

def test_full_pipeline():
    print("--- 1. Testing PDF / Text Parsing ---")
    sample_text = """
    Ananya Patel - Senior Software Engineer
    Email: ananya.patel@techmail.com | Phone: +91 98765 43210
    
    Summary: 5.5 years experience in Python, FastAPI, React, PyTorch, Docker, PostgreSQL, AWS, MLOps, spaCy.
    Education: B.Tech Computer Science (8.4 CGPA)
    Experience: 5+ years building scalable cloud APIs and ML pipelines.
    Projects: Built Talent Track AI Resume Shortlisting Engine with PyMuPDF and spaCy.
    Certifications: AWS Certified Solutions Architect.
    """
    
    parsed = PDFResumeParser.parse_resume_structure(sample_text)
    assert parsed['name'] == "Ananya Patel"
    assert parsed['years_of_experience'] >= 5.0
    print(f"Candidate: {parsed['name']} | Experience: {parsed['years_of_experience']} yrs")
    print(f"Extracted Skills: {parsed['extracted_skills']}")
    
    print("\n--- 2. Testing NLP & TF-IDF Matching ---")
    jd = "Seeking Senior AI Engineer with Python, FastAPI, React, Docker, AWS, TensorFlow, spaCy"
    tfidf_score = NLPEngine.calculate_tfidf_similarity(sample_text, jd)
    skill_res = NLPEngine.match_skills(parsed['extracted_skills'], ["Python", "FastAPI", "React", "Docker", "AWS", "spaCy"])
    print(f"TF-IDF Similarity: {tfidf_score}% | Skill Match: {skill_res['skill_match_percentage']}%")
    print(f"Matched Skills: {skill_res['matched_skills']}")
    
    print("\n--- 3. Testing Eligibility Checks & Composite Scoring ---")
    cand_data = {**parsed, **skill_res}
    criteria = {"min_experience": 3.0, "min_education": "B.Tech Computer Science", "required_skills": ["Python", "FastAPI", "React", "Docker", "AWS", "spaCy"]}
    eligibility = ScoringEngine.evaluate_eligibility(cand_data, criteria)
    scores = ScoringEngine.calculate_overall_score(tfidf_score, skill_res['skill_match_percentage'], eligibility)
    
    print(f"Overall Score: {scores['overall_score']}% | ATS Score: {scores['ats_score']}% | Verdict: {scores['verdict']}")
    assert scores['overall_score'] > 60
    
    print("\n--- 4. Testing Layout & Recommendation Engine ---")
    layout = RecommendationEngine.generate_layout_analysis(parsed.get("sections_found", {}), len(sample_text))
    recs = RecommendationEngine.generate_recommendations_and_roadmap(skill_res['missing_skills'], skill_res['matched_skills'], scores['overall_score'], "Senior AI Engineer")
    
    print(f"Layout Score: {layout['layout_score']}% | ATS Friendly: {layout['ats_friendly']}")
    print(f"Learning Roadmap Phases: {len(recs['learning_roadmap'])}")
    
    print("\n--- 5. Testing Deep Learning & Database Integration ---")
    dl_res = DeepLearningEmbeddingModel.compute_deep_semantic_match(sample_text, jd)
    print(f"DL Match Score: {dl_res['dl_similarity_score']}% via {dl_res['dl_framework']}")
    
    rec_id = MongoDatabase.save_analysis({"candidate": parsed['name'], "score": scores['overall_score']})
    print(f"Database Record Saved: {rec_id}")

    print("\n[OK] All Talent Track AI backend services verified successfully!")

if __name__ == "__main__":
    test_full_pipeline()
