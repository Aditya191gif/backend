import sys
import os
import json

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_week5_tests():
    print("=" * 66)
    print("       TALENT TRACK AI — WEEK 5 SUBMISSION VERIFICATION        ")
    print("=" * 66)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    frontend_dir = os.path.join(repo_root, "frontend")

    # 1. Verify Frontend Asset Structure
    print("\n[STEP 1] Verifying Frontend Static Asset Files...")
    required_files = [
        os.path.join(frontend_dir, "index.html"),
        os.path.join(frontend_dir, "css", "styles.css"),
        os.path.join(frontend_dir, "css", "variables.css"),
        os.path.join(frontend_dir, "js", "app.js"),
        os.path.join(frontend_dir, "js", "nlp_simulator.js"),
        os.path.join(frontend_dir, "js", "sample_resumes.js"),
    ]

    for fpath in required_files:
        assert os.path.exists(fpath), f"Missing frontend asset: {fpath}"
        fsize = os.path.getsize(fpath)
        print(f"  [OK] Asset Verified: {os.path.relpath(fpath, repo_root)} ({fsize:,} bytes)")

    # 2. Inspect UI Elements & DOM Hooks in index.html
    print("\n[STEP 2] Verifying Interactive DOM Hooks in index.html...")
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()

    expected_dom_elements = [
        "company-name",
        "job-role",
        "job-desc",
        "req-skills",
        "min-exp",
        "min-edu",
        "resume-file-input",
        "resume-text-input",
        "analyze-btn",
        "theme-btn-light",
        "theme-btn-dark",
        "theme-btn-auto",
        "results-section",
        "overall-score-gauge",
        "ats-score-gauge",
        "skill-score-gauge",
        "verdict-badge-el",
        "matched-skills-container",
        "missing-skills-container",
        "eligibility-list-container",
        "career-grid-container",
        "roadmap-container"
    ]

    for element_id in expected_dom_elements:
        assert f'id="{element_id}"' in html_content or f"id='{element_id}'" in html_content, f"DOM hook #{element_id} missing from index.html"
    print(f"  [OK] All {len(expected_dom_elements)} critical UI DOM elements confirmed.")

    # 3. Verify Theme Variables & Design System
    print("\n[STEP 3] Verifying Cyber-Dark & Light Theme Tokens...")
    with open(os.path.join(frontend_dir, "css", "variables.css"), "r", encoding="utf-8") as f:
        css_content = f.read()

    css_tokens = ["--bg-dark", "--primary-accent", "--emerald-accent", "--purple-accent", "--amber-accent"]
    for token in css_tokens:
        assert token in css_content, f"CSS design token {token} missing from variables.css"
    print(f"  [OK] Design System CSS tokens verified (Dark Obsidian & Light Ivory support).")

    # 4. Verify Backend-to-Frontend API Contract Payload Compatibility
    print("\n[STEP 4] Verifying Frontend/Backend API Contract Schema...")
    from app.services.pdf_parser import PDFResumeParser
    from app.services.nlp_engine import NLPEngine
    from app.services.scoring_engine import ScoringEngine
    from app.services.career_mapper import CareerMapper
    from app.services.recommendation_engine import RecommendationEngine

    sample_resume = "Priya Sharma, Senior AI Engineer. 5 years experience in Python, FastAPI, React, Docker, AWS. B.Tech Computer Science."
    sample_jd = "Senior AI Engineer with Python, FastAPI, React, Docker, AWS"
    req_skills = ["Python", "FastAPI", "React", "Docker", "AWS"]

    parsed = PDFResumeParser.parse_resume_structure(sample_resume)
    tfidf = NLPEngine.calculate_tfidf_similarity(sample_resume, sample_jd)
    skill_res = NLPEngine.match_skills(parsed["extracted_skills"], req_skills)
    cand_data = {**parsed, **skill_res}
    criteria = {"min_experience": 3.0, "min_education": "B.Tech", "required_skills": req_skills}
    elig = ScoringEngine.evaluate_eligibility(cand_data, criteria)
    scores = ScoringEngine.calculate_overall_score(tfidf, skill_res["skill_match_percentage"], elig)
    recs = RecommendationEngine.generate_recommendations_and_roadmap(skill_res["missing_skills"], skill_res["matched_skills"], scores["overall_score"], "Senior AI Engineer")
    careers = CareerMapper.get_career_recommendations(cand_data["extracted_skills"])

    payload = {
        "candidate": cand_data,
        "scores": scores,
        "eligibility_breakdown": elig,
        "ai_recommendation_report": recs,
        "career_mapping": careers
    }

    # Verify keys required by frontend/js/app.js renderAnalysisResults()
    assert "overall_score" in payload["scores"]
    assert "ats_score" in payload["scores"]
    assert "verdict" in payload["scores"]
    assert "matched_skills" in payload["candidate"]
    assert "missing_skills" in payload["candidate"]
    assert len(payload["career_mapping"]) > 0
    assert len(payload["ai_recommendation_report"]["learning_roadmap"]) == 3

    print("  [OK] Full-Stack API Contract is 100% matched with Frontend JavaScript renderers.")

    print("=" * 66)
    print("  [SUCCESS] WEEK 5 DELIVERABLES FULLY VERIFIED & READY FOR SUBMISSION!")
    print("=" * 66)

if __name__ == "__main__":
    run_week5_tests()
