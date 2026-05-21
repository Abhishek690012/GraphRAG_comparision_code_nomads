import math
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basic_rag.embedding_engine import EmbeddingEngine

class TestEmbeddingsLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configuration for local BGE-Small-en-v1.5
        cls.config = {
            "mode": "local",
            "model_name_or_path": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "normalize_embeddings": True,
            "batch_size": 16,
            "cache_dir": "./cache/embeddings",
            "trust_remote_code": False
        }
        cls.engine = EmbeddingEngine(cls.config)
        cls.test_strings = [
            "diabetes",
            "BCR::ABL1 fusion",
            "What is chemotherapy?"
        ]

    def test_embedding_shape_and_dimension(self):
        embeddings = self.engine.encode_batch(self.test_strings)
        
        # Check batch size
        self.assertEqual(len(embeddings), 3)
        
        # Check dimensionality
        for emb in embeddings:
            self.assertEqual(len(emb), 384, "Embedding dimension should be 384 for BGE-small-en-v1.5")
            
        self.assertEqual(self.engine.dimension, 384)

    def test_l2_normalization(self):
        embeddings = self.engine.encode_batch(self.test_strings)
        
        for i, emb in enumerate(embeddings):
            norm = math.sqrt(sum(x * x for x in emb))
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Vector {i} is not L2 normalized")

    def test_determinism(self):
        run1 = self.engine.encode_batch(self.test_strings)
        run2 = self.engine.encode_batch(self.test_strings)
        
        for emb1, emb2 in zip(run1, run2):
            for v1, v2 in zip(emb1, emb2):
                self.assertAlmostEqual(v1, v2, places=6, msg="Embeddings are not deterministic across runs")

if __name__ == "__main__":
    unittest.main()
