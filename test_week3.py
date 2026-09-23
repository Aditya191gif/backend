import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.nlp_engine import NLPEngine
from app.services.scoring_engine import ScoringEngine

def run_week3_tests():
    print("=" * 66)
    print("       TALENT TRACK AI — WEEK 3 SUBMISSION VERIFICATION        ")
    print("=" * 66)

    # 1. Candidate profile mock data
    candidate_profile = {
        "name": "Ananya Patel",
        "email": "ananya.patel@techmail.com",
        "phone": "+91 98765 43210",
        "years_of_experience": 5.5,
        "education": "B.Tech Computer Science",
        "extracted_skills": ["Python", "React", "FastAPI", "PostgreSQL", "PyTorch", "spaCy", "Docker", "AWS", "MLOps"],
        "matched_skills": ["Python", "FastAPI", "React", "Docker", "AWS", "spaCy"],
        "missing_skills": ["TensorFlow"],
        "certifications": ["AWS Certified Solutions Architect"],
        "cgpa": 8.8
    }

    job_description = """
    We are seeking a Senior AI/ML Engineer with 4+ years of hands-on experience building NLP models, 
    spaCy parsing engines, FastAPI microservices, and React.js web apps. Requires proficiency in 
    Python, TensorFlow, PyTorch, Docker, PostgreSQL, MongoDB, and Cloud AWS deployment.
    """

    required_skills = ["Python", "FastAPI", "TensorFlow", "React", "Docker", "AWS", "spaCy"]

    print("\n[STEP 1] Testing Scikit-Learn TF-IDF Cosine Similarity Engine...")
    resume_text = f"Candidate skills: {', '.join(candidate_profile['extracted_skills'])}. Experience: 5.5 years in AI/ML engineering, FastAPI, React, Docker, and AWS."
    tfidf_score = NLPEngine.calculate_tfidf_similarity(resume_text, job_description)
    print(f"  [OK] TF-IDF Similarity Score: {tfidf_score}%")
    assert tfidf_score > 5.0, "TF-IDF similarity calculation failed."

    print("\n[STEP 2] Testing Rule-Based Eligibility Evaluation Engine...")
    criteria = {
        "min_experience": 4.0,
        "min_education": "B.Tech Computer Science",
        "required_skills": required_skills
    }
    eligibility_checks = ScoringEngine.evaluate_eligibility(candidate_profile, criteria)
    
    passes = sum(1 for e in eligibility_checks if e['status'] == 'PASS')
    warns = sum(1 for e in eligibility_checks if e['status'] == 'WARN')
    fails = sum(1 for e in eligibility_checks if e['status'] == 'FAIL')

    print(f"  [OK] Eligibility Checks Total: {len(eligibility_checks)}")
    print(f"  [OK] Rule Results Breakdown  : PASS={passes}, WARN={warns}, FAIL={fails}")
    assert len(eligibility_checks) >= 5, "Eligibility checks count incomplete."
    assert passes >= 4, "Expected high pass count for qualified candidate."

    print("\n[STEP 3] Testing Weighted Composite ATS Scoring Algorithm...")
    skill_match_res = NLPEngine.match_skills(candidate_profile['extracted_skills'], required_skills)
    scoring_result = ScoringEngine.calculate_overall_score(
        tfidf_score=tfidf_score,
        skill_match_pct=skill_match_res['skill_match_percentage'],
        eligibility=eligibility_checks
    )

    print(f"  [OK] Overall Composite Score : {scoring_result['overall_score']}%")
    print(f"  [OK] ATS Technical Match Score: {scoring_result['ats_score']}%")
    print(f"  [OK] Final Decision Verdict   : {scoring_result['verdict']} ({scoring_result['verdict_badge']})")
    print(f"  [OK] Verdict Summary          : {scoring_result['summary_text']}")

    assert scoring_result['overall_score'] >= 70, "Composite score calculation below threshold."
    assert scoring_result['verdict'] in ["SHORTLIST", "CONSIDER"], "Verdict mismatch for candidate."

    print("=" * 66)
    print("  [SUCCESS] WEEK 3 DELIVERABLES FULLY VERIFIED & READY FOR SUBMISSION!")
    print("=" * 66)

if __name__ == "__main__":
    run_week3_tests()
