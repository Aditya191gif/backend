from typing import Dict, Any, List

class CareerMapper:
    """Generates career role trajectory mapping and personalized candidate recommendations."""
    
    PRESET_ROLES = [
        {
            "role": "ML Engineer",
            "is_top_pick": True,
            "match_score": 94,
            "skills": ["Python", "TensorFlow", "MLOps", "PyTorch", "Scikit-Learn"],
            "salary_range": "₹18–28 LPA",
            "demand": "+34% YoY demand"
        },
        {
            "role": "Backend Developer",
            "is_top_pick": False,
            "match_score": 88,
            "skills": ["Node.js", "PostgreSQL", "Docker", "FastAPI", "MongoDB"],
            "salary_range": "₹14–22 LPA",
            "demand": "+21% YoY demand"
        },
        {
            "role": "AI Research Scientist",
            "is_top_pick": False,
            "match_score": 81,
            "skills": ["Research", "PyTorch", "Stats", "spaCy", "Transformers"],
            "salary_range": "₹22–40 LPA",
            "demand": "+41% YoY demand"
        },
        {
            "role": "Full Stack Developer",
            "is_top_pick": False,
            "match_score": 79,
            "skills": ["React", "Node.js", "AWS", "TypeScript", "REST APIs"],
            "salary_range": "₹12–20 LPA",
            "demand": "+19% YoY demand"
        }
    ]

    @classmethod
    def get_career_recommendations(cls, candidate_skills: List[str]) -> List[Dict[str, Any]]:
        """Calculates role match fit based on extracted candidate skills."""
        cand_set = set([s.lower() for s in candidate_skills])
        
        results = []
        for role_def in cls.PRESET_ROLES:
            req_set = set([s.lower() for s in role_def["skills"]])
            overlap = len(cand_set.intersection(req_set))
            score = min(98, max(65, int(60 + (overlap / max(1, len(req_set))) * 38)))
            
            role_item = dict(role_def)
            role_item["match_score"] = score
            results.append(role_item)
            
        # Sort by match score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        # Mark top pick
        for idx, item in enumerate(results):
            item["is_top_pick"] = (idx == 0)
            
        return results

    @staticmethod
    def generate_recommendation_report(missing_skills: List[str], overall_score: int) -> Dict[str, Any]:
        """Generates candidate feedback report with priority badges and step-by-step learning roadmap."""
        
        areas_to_improve = []
        if any(s.lower() in ["aws", "gcp", "azure", "docker", "kubernetes"] for s in missing_skills) or True:
            areas_to_improve.append({
                "priority": "HIGH",
                "text": "Obtain AWS / GCP Cloud Certification or Containerization experience (Docker/K8s)."
            })
            
        if any(s.lower() in ["tensorflow", "pytorch", "deep learning"] for s in missing_skills) or True:
            areas_to_improve.append({
                "priority": "MED",
                "text": "Add open-source GitHub project contributions showcasing ML pipeline deployment."
            })
            
        areas_to_improve.append({
            "priority": "MED",
            "text": "Strengthen system design & microservices architectural patterns."
        })
        
        areas_to_improve.append({
            "priority": "LOW",
            "text": "Include quantified impact metrics in experience section (e.g., 'Improved accuracy by 15%')."
        })

        roadmap = [
            {"step": "Phase 1 (Week 1-2)", "title": "Core Skill Refinement", "action": f"Master key missing stack items: {', '.join(missing_skills[:3]) if missing_skills else 'Advanced NLP & Vector DBs'}."},
            {"step": "Phase 2 (Week 3-4)", "title": "Hands-on Capstone", "action": "Build and deploy an end-to-end fullstack AI application with Docker & AWS."},
            {"step": "Phase 3 (Week 5)", "title": "Resume Optimization", "action": "Update resume layout with ATS-friendly formatting, metrics, and cloud badges."}
        ]

        percentile = min(99, max(5, int(overall_score * 0.95)))

        return {
            "overall_score": overall_score,
            "percentile": f"Top {100 - percentile}% of applicants",
            "areas_to_improve": areas_to_improve,
            "learning_roadmap": roadmap
        }
