import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEvaluator:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)

    def embed(self, text: str):
        return np.array(self.embedder.encode(text))

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def evaluate(self, target: str, generated: str):
        # Uses consine similarity to measure similarity between target and generated answer
        target_vec = self.embed(target)
        gen_vec = self.embed(generated)
        return float(self.cosine_similarity(target_vec, gen_vec))