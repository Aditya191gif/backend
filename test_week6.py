import sys
import os
import json

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.dl_model import DeepLearningEmbeddingModel
from app.main import app

def run_week6_tests():
    print("=" * 66)
    print("       TALENT TRACK AI — WEEK 6 SUBMISSION VERIFICATION        ")
    print("=" * 66)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1. Deep Learning Transformer Semantic Matcher
    print("\n[STEP 1] Testing Deep Learning Embedding Module...")
    resume_text = """
    Senior AI/ML Researcher specializing in PyTorch, Transformer embeddings, 
    FastAPI microservices, and NLP data pipelines. 5 years building scalable AI.
    """
    job_description = """
    Seeking an AI Engineer with proficiency in PyTorch, Transformer embeddings,
    Python, and cloud deployments.
    """

    dl_result = DeepLearningEmbeddingModel.compute_deep_semantic_match(resume_text, job_description)
    print(f"  [OK] DL Similarity Score : {dl_result['dl_similarity_score']}%")
    print(f"  [OK] DL Framework Engine : {dl_result['dl_framework']}")
    print(f"  [OK] Model Architecture  : {dl_result['model_architecture']}")
    print(f"  [OK] Semantic Density    : {dl_result['semantic_density']}")

    assert dl_result["dl_similarity_score"] > 50.0, "DL score below baseline threshold."
    assert "dl_framework" in dl_result, "Missing framework descriptor."

    # 2. Postman Collection Verification
    print("\n[STEP 2] Verifying Postman Collection Specification...")
    postman_path = os.path.join(repo_root, "postman", "Talent_Track_AI.postman_collection.json")
    assert os.path.exists(postman_path), f"Missing Postman collection at {postman_path}"

    with open(postman_path, "r", encoding="utf-8") as f:
        col_data = json.load(f)

    assert "info" in col_data and "item" in col_data, "Invalid Postman collection schema."
    items = col_data["item"]
    print(f"  [OK] Found {len(items)} API endpoints in Postman Collection:")
    for it in items:
        req = it["request"]
        print(f"       -> [{req['method']}] {it['name']} ({req['url']['raw']})")

    assert len(items) >= 3, "Postman collection must contain at least 3 endpoints."

    # 3. FastAPI Route Registry & Schema Introspection
    print("\n[STEP 3] Inspecting FastAPI Registered API Routes...")
    registered_routes = [route.path for route in app.routes]
    print(f"  [OK] Total Registered Routes: {len(registered_routes)}")
    print(f"  [OK] Route List: {registered_routes}")

    expected_routes = ["/", "/api/health", "/api/candidates", "/api/analyze"]
    for er in expected_routes:
        assert er in registered_routes, f"Required route {er} missing from FastAPI app."

    print("=" * 66)
    print("  [SUCCESS] WEEK 6 DELIVERABLES FULLY VERIFIED & READY FOR SUBMISSION!")
    print("=" * 66)

if __name__ == "__main__":
    run_week6_tests()
