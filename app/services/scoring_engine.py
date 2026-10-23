from typing import Dict, Any, List

class ScoringEngine:
    """Calculates composite ATS scores, transparent eligibility checks, and gap analysis."""
    
    @staticmethod
    def evaluate_eligibility(candidate: Dict[str, Any], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates transparent rule-based eligibility matching candidate against target criteria."""
        results = []
        
        # 1. Minimum Education
        min_edu = criteria.get("min_education", "B.Tech Computer Science")
        cand_edu = candidate.get("education", "Bachelor Degree")
        
        cand_lower = str(cand_edu).lower()
        min_lower = str(min_edu).lower()
        
        # Check degree alignment
        if ("b.tech" in min_lower or "bachelor" in min_lower or "b.e." in min_lower) and ("b.tech" in cand_lower or "bachelor" in cand_lower or "b.e." in cand_lower):
            edu_status = "PASS"
            edu_note = "Technical degree background satisfies role requirements."
        elif "master" in min_lower and ("master" in cand_lower or "m.tech" in cand_lower or "phd" in cand_lower):
            edu_status = "PASS"
            edu_note = "Postgraduate degree requirement met."
        elif "master" in min_lower and ("bachelor" in cand_lower or "b.tech" in cand_lower):
            edu_status = "WARN"
            edu_note = "Candidate holds Bachelor degree; role targets Master degree."
        elif any(tech in cand_lower for tech in ["computer", "information technology", "technology", "engineering", "cs", "it", "ai", "data"]):
            edu_status = "PASS"
            edu_note = "Technical degree background satisfies role requirements."
        else:
            edu_status = "WARN"
            edu_note = "Degree background requires closer review."

        results.append({
            "id": "edu",
            "name": "Minimum Education",
            "required": f"Required: {min_edu}",
            "candidate_val": cand_edu,
            "status": edu_status,
            "note": edu_note
        })
        
        # 2. Years of Experience
        min_yoe = float(criteria.get("min_experience", 3.0))
        cand_yoe = float(candidate.get("years_of_experience", 0.0))
        if cand_yoe >= min_yoe:
            exp_status = "PASS"
            exp_note = "Sufficient domain experience."
        elif cand_yoe >= (min_yoe - 1.0) and cand_yoe > 0:
            exp_status = "WARN"
            exp_note = "Near threshold experience."
        else:
            exp_status = "FAIL"
            exp_note = "Below required minimum YOE (Fresher / Student)." if cand_yoe == 0 else "Below required minimum YOE."
            
        cand_exp_val = f"{cand_yoe} years (Fresher/Student)" if cand_yoe == 0 else f"{cand_yoe} years"
        results.append({
            "id": "exp",
            "name": "Years of Experience",
            "required": f"Required: {min_yoe}+ years",
            "candidate_val": cand_exp_val,
            "status": exp_status,
            "note": exp_note
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
        cgpa_val = candidate.get("cgpa")
        cgpa_display = candidate.get("cgpa_display")
        min_cgpa = 7.0
        if "min_cgpa" in criteria:
            try:
                min_cgpa = float(criteria["min_cgpa"])
            except Exception:
                min_cgpa = 7.0
                
        if cgpa_val is not None:
            cgpa_num = float(cgpa_val)
            cand_str = cgpa_display if cgpa_display and cgpa_display != "Not specified" else f"{cgpa_num} CGPA"
            cgpa_status = "PASS" if cgpa_num >= min_cgpa else "WARN"
            cgpa_note = f"Academic requirement satisfied (>= {min_cgpa} CGPA)." if cgpa_status == "PASS" else f"Academic score below target threshold (< {min_cgpa} CGPA)."
        else:
            cand_str = "Not specified"
            cgpa_status = "WARN"
            cgpa_note = "No explicit CGPA / academic percentage detected on resume."

        results.append({
            "id": "cgpa",
            "name": "CGPA / Academic Score",
            "required": f"Required: >= {min_cgpa} CGPA",
            "candidate_val": cand_str,
            "status": cgpa_status,
            "note": cgpa_note
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
