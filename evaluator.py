from schemas import TestCase, AgentResponse, EvaluationResult
from embedder import EmbeddingEvaluator


class RAGEvaluator:
    # Threshold determines pass or fail difficulty based on similarity (higher is harder)
    def __init__(self, threshold: float = 0.8):
        self.embedding_evaluator = EmbeddingEvaluator()
        self.threshold = threshold

    def evaluate_case(self, test_case: TestCase, agent_response: AgentResponse):
        similarity = self.embedding_evaluator.evaluate(
            test_case.target_answer,
            agent_response.generated_answer
        )

        return EvaluationResult(
            test_id=test_case.id,
            similarity_score=similarity,
            pass_status=similarity >= self.threshold
        )