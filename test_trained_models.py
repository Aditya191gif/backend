"""
Test Suite: Trained Machine Learning & Deep Learning Models Verification
Validates that model artifacts exist, load correctly, make accurate predictions,
and integrate seamlessly with the FastAPI backend API.
"""

import os
import sys
import unittest
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.services.nlp_engine import NLPEngine
from app.services.dl_model import DeepLearningEmbeddingModel
from app.main import get_trained_model_status, analyze_candidate_full

class TestTrainedModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(CURRENT_DIR, "models")

    def test_01_model_files_exist(self):
        """Verifies that all trained serialized model weights exist on disk."""
        expected_files = [
            "resume_classifier.pkl",
            "tfidf_vectorizer.pkl",
            "label_encoder.pkl",
            "classifier_metrics.json",
            "dl_matcher.pkl",
            "dl_vectorizer.pkl",
            "dl_scaler.pkl",
            "dl_metrics.json",
            "custom_ner_model.pkl",
            "entity_patterns.json",
            "ner_metrics.json"
        ]
        for fname in expected_files:
            fpath = os.path.join(self.models_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing model artifact: {fname}")
            self.assertGreater(os.path.getsize(fpath), 100, f"Artifact file too small: {fname}")
        print("  [PASS] test_01_model_files_exist: All 11 model artifacts verified.")

    def test_02_resume_domain_classification(self):
        """Tests domain classification inference using the trained Random Forest model."""
        sample_ds = """
        Results-driven Data Scientist with 5 years experience in Python, Pandas, Scikit-Learn,
        TensorFlow, and PyTorch. Built predictive machine learning models and end-to-end NLP pipelines.
        """
        pred = NLPEngine.predict_resume_domain(sample_ds)
        self.assertTrue(pred["model_loaded"])
        self.assertEqual(pred["predicted_domain"], "Data Science & Machine Learning")
        self.assertGreaterEqual(pred["confidence"], 10.0)
        self.assertIn("top_predictions", pred)
        self.assertGreater(len(pred["top_predictions"]), 0)
        print(f"  [PASS] test_02_resume_domain_classification: Predicted '{pred['predicted_domain']}' (Confidence: {pred['confidence']}%)")

    def test_03_deep_neural_matcher_inference(self):
        """Tests inference with the trained Multi-Layer Perceptron neural network."""
        resume = "Senior DevOps Engineer with Docker, Kubernetes, AWS, Terraform, CI/CD pipelines, and Linux administration."
        jd = "Looking for Senior DevOps Engineer to manage Kubernetes clusters, Docker containers, and AWS CI/CD pipelines."
        dl_res = DeepLearningEmbeddingModel.compute_deep_semantic_match(resume, jd)
        self.assertTrue(dl_res.get("trained_neural_model"))
        self.assertGreaterEqual(dl_res["dl_similarity_score"], 20.0)
        self.assertLessEqual(dl_res["dl_similarity_score"], 100.0)
        self.assertIn("Deep Multi-Layer Perceptron", dl_res["dl_framework"])
        print(f"  [PASS] test_03_deep_neural_matcher_inference: Score={dl_res['dl_similarity_score']}%, Framework={dl_res['dl_framework']}")

    def test_04_api_model_status_endpoint(self):
        """Tests the new /api/model-status endpoint directly."""
        body = get_trained_model_status()
        self.assertTrue(body["classifier_loaded"])
        self.assertIn("models", body)
        self.assertIn("resume_classifier", body["models"])
        self.assertIn("deep_learning_matcher", body["models"])
        self.assertIn("custom_ner", body["models"])
        accuracy = body["models"]["resume_classifier"]["accuracy"]
        self.assertGreaterEqual(accuracy, 0.90)
        print(f"  [PASS] test_04_api_model_status_endpoint: Verified classifier accuracy={accuracy*100}%")

    def test_05_api_analyze_with_trained_model(self):
        """Tests that analyze_candidate_full returns trained_model_prediction in the JSON payload."""
        async def run_analyze():
            return await analyze_candidate_full(
                company_name="Test Labs",
                job_role="Data Scientist",
                job_description="Seeking Python and Scikit-Learn machine learning engineer with 3+ years experience.",
                required_skills_json='["Python", "Machine Learning", "Scikit-Learn"]',
                min_experience=2.0,
                min_education="B.Tech Computer Science",
                resume_text="Experienced Python Data Scientist with 4 years in machine learning, statistics, and pandas.",
                file=None
            )
        
        data = asyncio.run(run_analyze())
        self.assertIn("trained_model_prediction", data)
        self.assertTrue(data["trained_model_prediction"]["model_loaded"])
        self.assertEqual(data["trained_model_prediction"]["predicted_domain"], "Data Science & Machine Learning")
        self.assertTrue(data["deep_learning_embedding"]["trained_neural_model"])
        print("  [PASS] test_05_api_analyze_with_trained_model: Integration payload validated.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
