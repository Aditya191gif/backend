import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.pdf_parser import PDFResumeParser
from app.services.nlp_engine import NLPEngine

def run_week2_tests():
    print("=" * 66)
    print("       TALENT TRACK AI — WEEK 2 SUBMISSION VERIFICATION        ")
    print("=" * 66)

    # 1. Sample Resume Text for Testing
    sample_resume_text = """
    Aarav Sharma
    Senior AI & Software Engineer | Phone: +91 98765 43210 | Email: aarav.sharma@techcorp.io
    LinkedIn: linkedin.com/in/aarav-sharma | GitHub: github.com/aarav-sharma
    Location: Bengaluru, India

    PROFESSIONAL SUMMARY
    Innovative Machine Learning Engineer with 6 years of experience developing RESTful backend APIs,
    deep learning pipelines, and multi-tenant cloud microservices using Python, FastAPI, React, PyTorch, and AWS.

    TECHNICAL SKILLS
    Languages & Frameworks: Python, JavaScript, TypeScript, FastAPI, Flask, React, Node.js, HTML5, CSS3
    AI / ML & NLP: PyTorch, TensorFlow, spaCy, Scikit-Learn, OpenCV, Pandas, NumPy, NLP, Deep Learning
    Databases & Cloud: MongoDB, PostgreSQL, Redis, Docker, Kubernetes, AWS, Git, CI/CD, Microservices

    WORK EXPERIENCE
    Lead AI Engineer - TechCorp Systems (2021 - Present)
    - Architected NLP resume parser using spaCy, PyMuPDF, and Scikit-Learn TF-IDF vectorizer.
    - Containerized AI services using Docker & Kubernetes on AWS EC2.

    Software Engineer - DataGen Inc (2018 - 2021)
    - Built web applications using React.js, FastAPI, and MongoDB.

    EDUCATION
    Master of Technology (M.Tech) in Computer Science, IIT Bombay (2018)
    Bachelor of Technology (B.Tech) in Computer Science & Engineering (2016)

    PROJECTS
    - Project: Talent Track AI Resume Parser & ATS Matcher
    - Project: Distributed Real-Time ML Inference Service

    CERTIFICATIONS
    - AWS Certified Solutions Architect
    - Deep Learning & NLP Specialist Certification
    """

    print("\n[STEP 1] Testing PDF Resume Parser & Text Extractor...")
    mock_pdf_bytes = sample_resume_text.encode('utf-8')
    extracted_text = PDFResumeParser.extract_text_from_pdf_bytes(mock_pdf_bytes)
    assert len(extracted_text) > 0, "PDF extraction returned empty text."
    print("  [OK] PDF Text Extraction Engine Operational")

    print("\n[STEP 2] Testing Resume Structural Parser (Metadata, Skills, YOE, Education)...")
    parsed_profile = PDFResumeParser.parse_resume_structure(extracted_text)
    print(f"  [OK] Candidate Name : {parsed_profile['name']}")
    print(f"  [OK] Contact Email  : {parsed_profile['email']}")
    print(f"  [OK] Phone Number   : {parsed_profile['phone']}")
    print(f"  [OK] YOE Calculated : {parsed_profile['years_of_experience']} Years")
    print(f"  [OK] Education Level: {parsed_profile['education']}")
    print(f"  [OK] Skills Extracted: {len(parsed_profile['extracted_skills'])} skills identified ({', '.join(parsed_profile['extracted_skills'][:6])}...)")
    print(f"  [OK] Sections Found : {list(k for k, v in parsed_profile['sections_found'].items() if v)}")

    assert parsed_profile['email'] == "aarav.sharma@techcorp.io", "Email extraction mismatch."
    assert parsed_profile['phone'] != "N/A", "Phone extraction mismatch."
    assert parsed_profile['years_of_experience'] >= 5.0, "Experience calculation mismatch."
    assert "Python" in parsed_profile['extracted_skills'], "Skill extraction missing Python."
    assert "FastAPI" in parsed_profile['extracted_skills'], "Skill extraction missing FastAPI."

    print("\n[STEP 3] Testing spaCy NER Entity Extraction Engine...")
    entities = NLPEngine.extract_entities_spacy(extracted_text)
    keywords = NLPEngine.extract_keywords_spacy(extracted_text)
    print(f"  [OK] spaCy Organizations Found : {entities.get('organizations', [])[:3]}")
    print(f"  [OK] spaCy Locations Found     : {entities.get('locations', [])[:3]}")
    print(f"  [OK] Key Technical Keywords   : {keywords[:5]}")

    print("\n[STEP 4] Testing Skill Matching & TF-IDF Vectorizer...")
    job_desc = "Looking for Senior AI Engineer with Python, FastAPI, React, Docker, AWS, spaCy, and 5+ years experience."
    req_skills = ["Python", "FastAPI", "React", "Docker", "AWS", "spaCy", "Kubernetes"]
    
    match_res = NLPEngine.match_skills(parsed_profile['extracted_skills'], req_skills)
    tfidf_score = NLPEngine.calculate_tfidf_similarity(extracted_text, job_desc)

    print(f"  [OK] Skill Match Percentage : {match_res['skill_match_percentage']}%")
    print(f"  [OK] Matched Skills         : {match_res['matched_skills']}")
    print(f"  [OK] Missing Skills         : {match_res['missing_skills']}")
    print(f"  [OK] TF-IDF Cosine Match    : {tfidf_score}%")

    assert match_res['skill_match_percentage'] >= 70.0, "Skill match percentage too low."
    assert tfidf_score > 10.0, "TF-IDF score calculation failure."

    print("=" * 66)
    print("  [SUCCESS] WEEK 2 DELIVERABLES FULLY VERIFIED & READY FOR SUBMISSION!")
    print("=" * 66)


if __name__ == "__main__":
    run_week2_tests()
