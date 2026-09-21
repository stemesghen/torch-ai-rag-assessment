import json
from pathlib import Path

from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever
from src.reranker import CrossEncoderModel
from src.llm import LLM_Model
from src.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEGATIVE_EVAL_PATH = (
    PROJECT_ROOT / "data" / "negative_eval.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "negative_eval_results.json"
)

#negative test cases .json for testing answer generation
with open(NEGATIVE_EVAL_PATH, "r") as file:
    negative_tests = json.load(file)

#initialize
embedder = Embedding_Model_QWEN()
retriever = ElasticsearchRetriever()
reranker = CrossEncoderModel()
llm_model = LLM_Model()

pipeline = RAGPipeline(
    embedder=embedder,
    retriever=retriever,
    reranker=reranker,
    llm_model=llm_model
)

#run neg. test case
results = []

for test in negative_tests:

    query_id = test["query_id"]
    query_text = test["query"]
    expected_behavior = test["expected_behavior"]

    print(f"Running {query_id}")

    print(f"Question: {query_text}")
    print(f"Expected behavior: {expected_behavior}")

    #run the negative question through the same RAG pipeline used for normal user questions.
    response, model_used, final_results = pipeline.answer_question(
        query_text
    )

    #retreive the chunk IDs that the LLM chose to cite.
    cited_chunk_ids = [
        source.chunk_id
        for source in response.sources
    ]

    # retrieve the chunks that were retrieved and passed to the generation model.
    retrieved_chunk_ids = [
        result["chunk_id"]
        for result in final_results
    ]

    print(f"\nAnswer:\n{response.answer}")
    print(f"\nRetrieved chunks: {retrieved_chunk_ids}")
    print(f"Cited chunks: {cited_chunk_ids}")
    print(f"Model used: {model_used}")

    # save the result of negative behavior
    results.append({
        "query_id": query_id,
        "query": query_text,
        "expected_behavior": expected_behavior,
        "actual_answer": response.answer,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "cited_chunk_ids": cited_chunk_ids,
        "model_used": model_used
    })


#store results in output path
with open(OUTPUT_PATH, "w") as file:
    json.dump(
        results,
        file,
        indent=2
    )

print(
    f"\nNegative evaluation complete. "
    f"Results saved to {OUTPUT_PATH}"
)