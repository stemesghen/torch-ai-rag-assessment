import json
import os
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import GeminiModel
from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVAL_PATH = PROJECT_ROOT / "data" / "generation_eval_data.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "generation_eval_results.json"


#load generated responses and retrieved context for DeepEval scoring
with open(EVAL_PATH, "r") as file:
    evaluation_data = json.load(file)


#create test cases
test_cases = []

for sample in evaluation_data:

    test_case = LLMTestCase(
        input=sample["input"],
        actual_output=sample["actual_output"],
        retrieval_context=sample["retrieval_context"]
    )

    test_cases.append(test_case)


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


#gemini model used as the DeepEval judge
evaluation_model = GeminiModel(
    model="gemini-3.8-flash",
    api_key=gemini_api_key
)


#generation evaluation metrics
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=evaluation_model,
    include_reason=True,
    async_mode=False
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=evaluation_model,
    include_reason=True,
    async_mode=False
)


# run evaluation
evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric
    ]
)


results = []

for test_result in evaluation_results.test_results:

    metric_scores = {
        metric.name: metric.score
        for metric in test_result.metrics_data
    }

    results.append({
        "input": test_result.input,
        "faithfulness": metric_scores.get("Faithfulness"),
        "answer_relevancy": metric_scores.get("Answer Relevancy")
    })


with open(OUTPUT_PATH, "w") as file:
    json.dump(
        results,
        file,
        indent=4
    )


print(
    f"\nSaved results to {OUTPUT_PATH}"
)