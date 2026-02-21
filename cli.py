import json
import typer
from typing import List
from schemas import TestCase, AgentResponse
from evaluator import RAGEvaluator

# Defines commands that can be run from the CLI
app = typer.Typer()

# Load tests from JSON file
def load_tests(path: str) -> List[TestCase]:
    with open(path, "r") as f:
        data = json.load(f)
    # For every dictionary in data, create a TestCase object using contents and return them as a list
    return [TestCase(**item) for item in data]


@app.command()
def run(
    tests_path: str,
    responses_path: str,
    # Change the threshold value here to adjust difficulty
    threshold: float = 0.8,
    output_path: str = "results.json",
):
    tests = load_tests(tests_path)

    with open(responses_path, "r") as f:
        responses_data = json.load(f)

    evaluator = RAGEvaluator(threshold=threshold)

    results = []

    for test in tests:
        response_data = responses_data[test.id]
        response = AgentResponse(**response_data)

        result = evaluator.evaluate_case(test, response)
        results.append(result.model_dump())

        typer.echo(
            f"Test {result.test_id} | "
            f"Similarity: {result.similarity_score:.4f} | "
            f"PASS: {result.pass_status}"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    app()