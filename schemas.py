from pydantic import BaseModel
from typing import List, Optional, Dict


class TestCase(BaseModel):
    id: str # Unique identifier for the test case
    query: str # Question to be answered
    target_answer: str # Correct answer to the question (will be compared via vector similarity)
    context_docs: Optional[List[str]] = None # Expected documents to be retrieved
    metadata: Optional[Dict] = None


class AgentResponse(BaseModel):
    generated_answer: str # Generated response by the agent
    retrieved_docs: Optional[List[str]] = None # Documents retrieved by the agent


class EvaluationResult(BaseModel):
    test_id: str # Same identifier as in the original test case
    similarity_score: float # Similarity score between the target answer and the generated response
    pass_status: bool # Pass or fail for the test