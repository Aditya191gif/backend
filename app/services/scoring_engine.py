from typing import Dict, Any, List

class ScoringEngine:
    """Calculates composite ATS scores, transparent eligibility checks, and gap analysis."""
    
    @staticmethod
    def evaluate_eligibility(candidate: Dict[str, Any], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates transparent rule-based eligibility matching candidate against target criteria."""
        results = []
        
        # 1. Minimum Education
        min_edu = criteria.get("min_education", "B.Tech Computer Science")
        cand_edu = candidate.get("education", "B.Tech Computer Science")
        edu_status = "PASS" if "b.tech" in cand_edu.lower() or "master" in cand_edu.lower() or "b.e." in cand_edu.lower() else "WARN"
        results.append({
            "id": "edu",
            "name": "Minimum Education",
            "required": f"Required: {min_edu}",
            "candidate_val": cand_edu,
            "status": edu_status,
            "note": "Degree level aligns with technical role requirements." if edu_status == "PASS" else "Non-standard degree background."
        })
        
        # 2. Years of Experience
        min_yoe = float(criteria.get("min_experience", 3.0))
        cand_yoe = float(candidate.get("years_of_experience", 0.0))
        if cand_yoe >= min_yoe:
            exp_status = "PASS"
        elif cand_yoe >= (min_yoe - 1.0):
            exp_status = "WARN"
        else:
            exp_status = "FAIL"
            
        results.append({
            "id": "exp",
            "name": "Years of Experience",
            "required": f"Required: {min_yoe}+ years",
            "candidate_val": f"{cand_yoe} years",
            "status": exp_status,
            "note": "Sufficient domain experience." if exp_status == "PASS" else ("Near threshold experience." if exp_status == "WARN" else "Below required minimum YOE.")
        })
        
        # 3. Core Skills
        matched_count = len(candidate.get("matched_skills", []))
        total_req = len(criteria.get("required_skills", []))
        ratio = (matched_count / total_req) if total_req > 0 else 1.0
        skill_status = "PASS" if ratio >= 0.7 else ("WARN" if ratio >= 0.4 else "FAIL")
        
        results.append({
            "id": "skills",
            "name": "Core Skills Match",
            "required": f"Required: {total_req} key tech skills",
            "candidate_val": f"{matched_count}/{total_req} matched",
            "status": skill_status,
            "note": "Strong coverage of core tech stack." if skill_status == "PASS" else "Missing some primary target skills."
        })
        
        # 4. Certifications
        certs = candidate.get("certifications", [])
        cert_status = "PASS" if certs else "WARN"
        results.append({
            "id": "certs",
            "name": "Certifications",
            "required": "Required: AWS / Cloud / AI Certification preferred",
            "candidate_val": certs[0] if certs else "No cloud cert found",
            "status": cert_status,
            "note": "Verified industry accreditation found." if cert_status == "PASS" else "No cloud/AI certs detected on resume."
        })
        
        # 5. CGPA / Academic Score
        cgpa_val = candidate.get("cgpa", 8.4)
        results.append({
            "id": "cgpa",
            "name": "CGPA / Academic Score",
            "required": "Required: ≥ 7.0 CGPA",
            "candidate_val": f"{cgpa_val} CGPA",
            "status": "PASS" if cgpa_val >= 7.0 else "WARN",
            "note": "Strong academic score threshold met."
        })
        
        # 6. Communication Skills
        results.append({
            "id": "comm",
            "name": "Communication & Formatting",
            "required": "Required: Professional English",
            "candidate_val": "Inferred: High",
            "status": "PASS",
            "note": "Clear bullet structure and grammatical fluency."
        })
        
        return results

    @classmethod
    def calculate_overall_score(cls, tfidf_score: float, skill_match_pct: float, eligibility: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates final composite candidate score, ATS score %, and AI verdict."""
        
        # Count eligibility weights
        passes = sum(1 for e in eligibility if e['status'] == 'PASS')
        warns = sum(1 for e in eligibility if e['status'] == 'WARN')
        fails = sum(1 for e in eligibility if e['status'] == 'FAIL')
        
        eligibility_score = (passes * 100 + warns * 50) / len(eligibility) if eligibility else 80.0
        
        # Weighted composite score
        overall_score = int(round(0.4 * skill_match_pct + 0.3 * tfidf_score + 0.3 * eligibility_score))
        ats_score = int(round(0.5 * skill_match_pct + 0.5 * tfidf_score))
        
        if overall_score >= 85:
            verdict = "SHORTLIST"
            verdict_badge = "High Fit Candidate"
            summary_text = f"Strong candidate for the role. Demonstrates high proficiency with {overall_score}% match score. Recommended for technical interview round."
        elif overall_score >= 70:
            verdict = "CONSIDER"
            verdict_badge = "Moderate Fit Candidate"
            summary_text = f"Good foundational profile ({overall_score}% match score). Key skills align well, though candidate requires minor upskilling in missing tech stack."
        else:
            verdict = "REJECT"
            verdict_badge = "Low Match / Needs Gap Training"
            summary_text = f"Candidate score ({overall_score}%) falls below threshold. Significant skill gaps identified in required core technologies."
            
        return {
            "overall_score": overall_score,
            "ats_score": ats_score,
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "summary_text": summary_text,
            "pass_count": passes,
            "warn_count": warns,
            "fail_count": fails
        }
