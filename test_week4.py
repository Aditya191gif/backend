import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.recommendation_engine import RecommendationEngine
from app.services.career_mapper import CareerMapper

def run_week4_tests():
    print("=" * 66)
    print("       TALENT TRACK AI — WEEK 4 SUBMISSION VERIFICATION        ")
    print("=" * 66)

    # 1. Resume Layout Format Analysis
    print("\n[STEP 1] Testing Resume Layout Format Audit...")
    sections_present = {
        "education": True,
        "experience": True,
        "skills": True,
        "projects": True,
        "certifications": True
    }
    layout_good = RecommendationEngine.generate_layout_analysis(sections_present, text_length=1200)
    print(f"  [OK] Standard Resume Layout Score: {layout_good['layout_score']}% (ATS Friendly: {layout_good['ats_friendly']})")
    assert layout_good["layout_score"] >= 80, "Expected high layout score for well-formatted resume."
    assert layout_good["ats_friendly"] is True, "Expected ATS friendly status."

    # Test missing sections penalty
    sections_missing = {"education": False, "experience": False, "skills": True}
    layout_poor = RecommendationEngine.generate_layout_analysis(sections_missing, text_length=300)
    print(f"  [OK] Degraded Layout Score: {layout_poor['layout_score']}% (Suggestions: {len(layout_poor['layout_suggestions'])})")
    assert layout_poor["layout_score"] < layout_good["layout_score"], "Missing sections should decrease score."

    # 2. Career Trajectory Mapping ("What Jobs Should They Pursue?")
    print("\n[STEP 2] Testing Career Trajectory Role Fit Mapping...")
    candidate_skills = ["Python", "TensorFlow", "PyTorch", "Scikit-Learn", "Docker", "FastAPI"]
    career_recs = CareerMapper.get_career_recommendations(candidate_skills)
    print(f"  [OK] Evaluated {len(career_recs)} target roles")
    top_role = career_recs[0]
    print(f"  [OK] Top Recommended Role: '{top_role['role']}' with {top_role['match_score']}% Fit")
    assert len(career_recs) >= 3, "Expected at least 3 career path recommendations."
    assert top_role["is_top_pick"] is True, "Top role must be flagged as top pick."
    assert "salary_range" in top_role and "demand" in top_role, "Role must contain market analytics."

    # 3. Actionable AI Recommendations & 3-Phase Roadmap
    print("\n[STEP 3] Testing Missing Skill Gap Analysis & 3-Phase Learning Roadmap...")
    missing_skills = ["TensorFlow", "AWS", "Kubernetes"]
    matched_skills = ["Python", "FastAPI", "React", "Docker"]
    rec_report = RecommendationEngine.generate_recommendations_and_roadmap(
        missing_skills=missing_skills,
        matched_skills=matched_skills,
        overall_score=72.5,
        job_role="Senior AI Engineer"
    )

    cert_advice = rec_report["certificate_advice"]
    proj_sugg = rec_report["project_suggestions"]
    roadmap = rec_report["learning_roadmap"]

    print(f"  [OK] Certificate Recommendations : {len(cert_advice)} verified ({cert_advice[0]})")
    print(f"  [OK] Portfolio Project Ideas      : {len(proj_sugg)} generated")
    print(f"  [OK] 3-Phase Learning Roadmap     : {len(roadmap)} phases created")
    for phase in roadmap:
        print(f"       -> {phase['phase']}: {phase['focus']}")

    assert len(cert_advice) >= 1, "Certificate advice should be generated for missing cloud/AI skills."
    assert len(proj_sugg) >= 2, "Project suggestions should be provided."
    assert len(roadmap) == 3, "Learning roadmap must contain exactly 3 actionable phases."

    print("=" * 66)
    print("  [SUCCESS] WEEK 4 DELIVERABLES FULLY VERIFIED & READY FOR SUBMISSION!")
    print("=" * 66)

if __name__ == "__main__":
    run_week4_tests()
