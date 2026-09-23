from typing import Dict, Any, List

class RecommendationEngine:
    """Generates layout suggestions, missing skill roadmaps, certificate advice, and project recommendations."""

    @classmethod
    def generate_layout_analysis(cls, sections_found: Dict[str, bool], text_length: int) -> Dict[str, Any]:
        """Evaluates resume structure, formatting, and layout for ATS readability."""
        suggestions = []
        score = 100

        if not sections_found.get("education", False):
            suggestions.append("Explicitly add an 'Education' section heading so ATS scanners can index degree details.")
            score -= 15

        if not sections_found.get("experience", False):
            suggestions.append("Add a clear 'Work Experience' or 'Projects' section heading.")
            score -= 20

        if not sections_found.get("skills", False):
            suggestions.append("Include a dedicated 'Technical Skills' bulleted list section for keyphrase parsing.")
            score -= 15

        if text_length < 400:
            suggestions.append("Resume content is concise. Add project details, quantifiable impact metrics, and key technologies used.")
            score -= 10
        elif text_length > 4000:
            suggestions.append("Resume text is lengthy (>2 pages). Condense experience descriptions to focus on core achievements.")
            score -= 5

        if not suggestions:
            suggestions.append("Excellent layout structure! Clean section demarcations, contact info, and clear typography detected.")

        return {
            "layout_score": max(50, score),
            "layout_suggestions": suggestions,
            "ats_friendly": score >= 75
        }

    @classmethod
    def generate_recommendations_and_roadmap(
        cls,
        missing_skills: List[str],
        matched_skills: List[str],
        overall_score: float,
        job_role: str
    ) -> Dict[str, Any]:
        """Creates actionable recommendations, portfolio project ideas, certificate advice, and 3-phase learning roadmap."""
        
        # 1. Certificate Advice
        cert_advice = []
        if any(s.lower() in ["aws", "gcp", "azure", "cloud"] for s in missing_skills):
            cert_advice.append("AWS Certified Solutions Architect / Cloud Practitioner")
        if any(s.lower() in ["tensorflow", "pytorch", "deep learning", "spacy", "scikit-learn"] for s in missing_skills):
            cert_advice.append("DeepLearning.AI Machine Learning & NLP Specialization")
        if any(s.lower() in ["react", "node.js", "fastapi", "docker"] for s in missing_skills):
            cert_advice.append("Meta Full Stack Engineer Professional Certificate")
        if not cert_advice:
            cert_advice.append("Certified Kubernetes Application Developer (CKAD) or Advanced AI Engineer Specialization")

        # 2. Project Suggestions
        project_suggestions = []
        if missing_skills:
            top_missing = missing_skills[:3]
            project_suggestions.append(f"Build a real-world project incorporating: {', '.join(top_missing)}.")
        project_suggestions.append(f"Develop a production-grade full-stack microservice for {job_role} showcasing CI/CD & Docker.")
        project_suggestions.append("Create an open-source GitHub repository with clean documentation, unit tests, and API endpoints.")

        # 3. Resume Content Suggestions
        resume_suggestions = []
        if overall_score < 75:
            resume_suggestions.append("Incorporate quantitative impact metrics in bullet points (e.g., 'Improved API performance by 40%').")
            resume_suggestions.append(f"Align resume terminology with job description keywords: {', '.join(missing_skills[:4])}.")
        else:
            resume_suggestions.append("Highlight leadership roles, architecture decisions, and system optimization achievements.")

        # 4. 3-Phase Actionable Learning Roadmap
        phase1_skills = missing_skills[:2] if missing_skills else ["Advanced System Architecture"]
        phase2_skills = missing_skills[2:4] if len(missing_skills) > 2 else ["Performance Tuning & MLOps"]
        phase3_skills = missing_skills[4:] if len(missing_skills) > 4 else ["Production Cloud Deployment"]

        learning_roadmap = [
            {
                "phase": "Phase 1: Immediate Skill Gap Mastery (Weeks 1-2)",
                "focus": f"Master core fundamentals: {', '.join(phase1_skills)}",
                "action": "Complete hands-on tutorials, build small standalone prototypes, and update resume core competencies."
            },
            {
                "phase": "Phase 2: Portfolio Project Integration (Weeks 3-4)",
                "focus": f"Integrate skills into end-to-end applications: {', '.join(phase2_skills)}",
                "action": "Build a featured project repository, write clean documentation, and add GitHub links to your resume."
            },
            {
                "phase": "Phase 3: Certification & ATS Optimization (Weeks 5-6)",
                "focus": f"Certification & Industry Alignment: {', '.join(phase3_skills)}",
                "action": "Earn target industry certification, tailor resume summary, and re-run Talent Track AI ATS check."
            }
        ]

        return {
            "missing_skills": missing_skills,
            "matched_skills": matched_skills,
            "certificate_advice": cert_advice,
            "project_suggestions": project_suggestions,
            "resume_suggestions": resume_suggestions,
            "learning_roadmap": learning_roadmap
        }
