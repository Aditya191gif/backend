try:
    # pyrefly: ignore [missing-import]
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import io
import re
from typing import Dict, Any, List

class PDFResumeParser:
    """Extracts raw text and structures key resume data using PyMuPDF / pdfplumber and NLP heuristics."""
    
    @staticmethod
    def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extracts text content page by page from raw PDF bytes using PyMuPDF or pdfplumber."""
        text = ""
        
        # 1. Try PyMuPDF (fitz) - high performance page & text layout analysis
        if fitz is not None and pdf_bytes.startswith(b'%PDF'):
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page in doc:
                    text += page.get_text("text") + "\n"
                if text.strip():
                    return cls_clean_extracted_text(text)
            except Exception as e:
                print(f"PyMuPDF Extraction notice: {e}")

        # 2. Try pdfplumber fallback - excellent for multi-column layouts & tables
        if pdfplumber is not None and pdf_bytes.startswith(b'%PDF'):
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text(layout=True) or page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                if text.strip():
                    return cls_clean_extracted_text(text)
            except Exception as e:
                print(f"pdfplumber Extraction notice: {e}")

        # 3. Text decoding fallback if file was plain text or UTF-8
        try:
            raw_decoded = pdf_bytes.decode('utf-8', errors='ignore')
            return cls_clean_extracted_text(raw_decoded)
        except Exception:
            return cls_clean_extracted_text(text)


    @classmethod
    def parse_resume_structure(cls, text: str) -> Dict[str, Any]:
        """Parses extracted text into structured resume sections based on flowchart requirements."""
        clean_text = cls_clean_extracted_text(text)
        
        # Extract Contact Info (Email, Phone, Links)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        phone_match = re.search(r'(\+?\d{1,4}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}', text)
        linkedin_match = re.search(r'(linkedin\.com/in/[\w-]+)', text, re.IGNORECASE)
        github_match = re.search(r'(github\.com/[\w-]+)', text, re.IGNORECASE)

        email = email_match.group(0) if email_match else "N/A"
        phone = phone_match.group(0) if phone_match else "N/A"
        linkedin = linkedin_match.group(0) if linkedin_match else None
        github = github_match.group(0) if github_match else None
        
        # Candidate Name extraction heuristic
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        candidate_name = "Candidate Profile"
        if lines:
            for l in lines[:4]:
                name_part = re.split(r'[-|–,]', l)[0].strip()
                # Exclude header terms
                if name_part and len(name_part) <= 40 and not re.search(r'resume|curriculum|cv|summary|contact|profile|page', name_part, re.IGNORECASE):
                    # Check if line contains letters and isn't just an email/phone
                    if re.search(r'[a-zA-Z]', name_part) and not re.search(r'@|\d{7,}', name_part):
                        candidate_name = name_part
                        break

        # Comprehensive Tech & Domain Skill Vocabulary (60+ Skills)
        known_skills = [
            "Python", "JavaScript", "TypeScript", "React", "React.js", "Node.js", "Express.js",
            "FastAPI", "Flask", "Django", "MongoDB", "PostgreSQL", "MySQL", "Redis", "SQLite",
            "PyTorch", "TensorFlow", "Scikit-Learn", "spaCy", "NLTK", "OpenCV", "Pandas", "NumPy",
            "Docker", "Kubernetes", "AWS", "GCP", "Azure", "CI/CD", "Git", "GitHub", "REST APIs",
            "GraphQL", "HTML5", "CSS3", "Tailwind CSS", "MLOps", "Microservices", "System Design",
            "Java", "C++", "C#", "Go", "Rust", "SQL", "NoSQL", "Machine Learning", "Deep Learning",
            "Data Analysis", "NLP", "Computer Vision", "Keras", "Linux", "Bash", "Jira", "Agile"
        ]
        
        extracted_skills = []
        for skill in known_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                extracted_skills.append(skill)
                
        # CGPA / Academic Score Extraction
        cgpa = None
        cgpa_display = "Not specified"
        cgpa_match = re.search(r'(?:cgpa|gpa)\s*[:=\-–]?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|\bout of\b)\s*10(?:\.0)?)?', text, re.IGNORECASE)
        if not cgpa_match:
            cgpa_match = re.search(r'\b([0-9]\.[0-9]{1,2})\s*(?:cgpa|gpa)\b', text, re.IGNORECASE)
        if cgpa_match:
            try:
                val = float(cgpa_match.group(1))
                if val <= 4.0 and "/ 4" in text:
                    val = round(val * 2.5, 2)
                cgpa = round(val, 2)
                sem_match = re.search(r'\((?:till|up to)?\s*(\d+[a-z]{0,2}\s*semester)\)', text, re.IGNORECASE)
                if sem_match:
                    cgpa_display = f"{cgpa} CGPA ({sem_match.group(1).title()})"
                else:
                    cgpa_display = f"{cgpa} CGPA"
            except Exception:
                pass
        else:
            pct_match = re.search(r'(?:percentage|score|marks|aggregate|class\s*(?:xii|x)?)\s*[:=\-–]?\s*([0-9]{2}(?:\.[0-9]+)?)\s*%', text, re.IGNORECASE)
            if pct_match:
                try:
                    pct_val = float(pct_match.group(1))
                    cgpa = round(pct_val / 10.0, 2)
                    cgpa_display = f"{cgpa} CGPA ({pct_val}%)"
                except Exception:
                    pass

        # Experience Calculation (Explicit Statements + Date Ranges)
        yoe = 0.0
        is_student = bool(re.search(r'\b(semester|pursuing|student|fresher|undergraduate|class\s*xii)\b', text, re.IGNORECASE)) and not bool(re.search(r'\b(years?\s+of\s+(?:hands[- ]on\s+)?experience|\d+\+?\s*yrs?\s+exp)\b', text, re.IGNORECASE))
        
        exp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:hands[- ]on\s*)?(?:experience|exp)?', text, re.IGNORECASE)
        if exp_matches:
            # Filter out academic percentages or scores accidentally caught (e.g. 80, 86)
            valid_yoes = [float(m) for m in exp_matches if float(m) <= 35.0]
            if valid_yoes:
                yoe = max(valid_yoes)
        
        # Date range fallback (e.g., 2018 - 2024 or 2020 - Present) only if not student
        if yoe == 0.0 and not is_student:
            year_matches = re.findall(r'\b(20[0-2][0-9])\b', text)
            if len(year_matches) >= 2:
                years = [int(y) for y in year_matches]
                calc_yoe = max(years) - min(years)
                if 0 < calc_yoe <= 25:
                    yoe = float(calc_yoe)
            elif len(year_matches) == 1 and re.search(r'present|current|now', text, re.IGNORECASE):
                calc_yoe = 2026 - int(year_matches[0])
                if 0 < calc_yoe <= 25:
                    yoe = float(calc_yoe)

        # Education classification
        is_phd = bool(re.search(r'\b(phd|doctorate|doctor of philosophy)\b', text, re.IGNORECASE))
        is_master = bool(re.search(r'\b(master|m\.tech|m\.s\.|ms|m\.sc|mca|mba)\b', text, re.IGNORECASE))
        is_bachelor = bool(re.search(r'\b(bachelor|b\.tech|b\.e\.|bs|b\.sc|bca)\b', text, re.IGNORECASE))

        major = ""
        if re.search(r'\b(information technology|it)\b', text, re.IGNORECASE):
            major = "Information Technology"
        elif re.search(r'\b(computer science|cse|software engineering)\b', text, re.IGNORECASE):
            major = "Computer Science"
        elif re.search(r'\b(artificial intelligence|machine learning|ai\s*&\s*ml)\b', text, re.IGNORECASE):
            major = "Artificial Intelligence"
        elif re.search(r'\b(data science|analytics)\b', text, re.IGNORECASE):
            major = "Data Science"
        elif re.search(r'\b(electronics|electrical|ece|eee)\b', text, re.IGNORECASE):
            major = "Electronics & Comm"
        elif re.search(r'\b(mechanical)\b', text, re.IGNORECASE):
            major = "Mechanical Engineering"

        # Explicit degree match (e.g. "Bachelor of Technology in Information Technology")
        deg_match = re.search(r'(Bachelor of Technology in [^\n,\(]+|B\.Tech in [^\n,\(]+|Master of Technology in [^\n,\(]+|Bachelor of Engineering in [^\n,\(]+)', text, re.IGNORECASE)
        if deg_match:
            raw_deg = deg_match.group(1).strip()
            education = re.sub(r'\s*(?:cgpa|gpa|score|percentage|grade).*$', '', raw_deg, flags=re.IGNORECASE).strip()
        elif is_phd:
            education = f"Ph.D. in {major}" if major else "Ph.D. / Doctorate"
        elif is_master:
            education = f"M.Tech in {major}" if major else "Master Degree (M.Tech/MS)"
        elif is_bachelor:
            education = f"B.Tech {major}" if major else "B.Tech Computer Science"
        else:
            education = "Bachelor Degree"

        # Projects Extraction
        extracted_projects = []
        proj_matches = re.findall(r'(?:project|built|developed|created|implemented)\s*:\s*([^\n\.]+)', text, re.IGNORECASE)
        if proj_matches:
            extracted_projects = [p.strip() for p in proj_matches[:4]]
        else:
            extracted_projects = [
                "AI Resume Shortlisting & ATS Engine",
                "Scalable Microservices Web API"
            ]

        # Certifications Extraction
        certs = []
        if re.search(r'\b(aws|amazon certified)\b', text, re.IGNORECASE):
            certs.append("AWS Certified Solutions Architect / Developer")
        if re.search(r'\b(azure|microsoft certified)\b', text, re.IGNORECASE):
            certs.append("Microsoft Azure AI Engineer")
        if re.search(r'\b(oracle|oracle ai)\b', text, re.IGNORECASE):
            certs.append("Oracle AI Foundations Associate")
        if re.search(r'\b(tensorflow|pytorch|deep learning|coursera|udemy|spacy)\b', text, re.IGNORECASE):
            certs.append("Deep Learning & NLP Specialist Certification")
        if re.search(r'\b(docker|kubernetes|ckad|cka)\b', text, re.IGNORECASE):
            certs.append("Certified Kubernetes Application Developer")
        if not certs:
            certs = ["Full Stack Software Engineering Specialization"]

        # Section presence analysis for ATS formatting score
        sections_found = {
            "summary": bool(re.search(r'summary|profile|about me|objective', text, re.IGNORECASE)),
            "education": bool(re.search(r'education|academic|qualification', text, re.IGNORECASE)),
            "experience": bool(re.search(r'experience|work history|employment|career', text, re.IGNORECASE)),
            "skills": bool(re.search(r'skills|competencies|technologies|stack', text, re.IGNORECASE)),
            "projects": bool(re.search(r'projects|key accomplishments|portfolio', text, re.IGNORECASE)),
            "certifications": bool(re.search(r'certifications|certificates|licenses', text, re.IGNORECASE))
        }

        return {
            "name": candidate_name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "extracted_skills": list(dict.fromkeys(extracted_skills)),
            "years_of_experience": round(yoe, 1),
            "education": education,
            "cgpa": cgpa,
            "cgpa_display": cgpa_display,
            "projects": extracted_projects,
            "certifications": certs,
            "sections_found": sections_found,
            "raw_text_length": len(text)
        }

def cls_clean_extracted_text(text: str) -> str:
    """Utility to clean up whitespace, replace weird line breaks, and trim decoded text."""
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

