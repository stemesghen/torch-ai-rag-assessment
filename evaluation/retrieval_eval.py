import json
from pathlib import Path

from ranx import Qrels, Run, evaluate

from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever
from src.reranker import CrossEncoderModel


#Load the data & Initialize models

project_path = Path(__file__).resolve().parent.parent
eval_path = project_path / "data" / "retrieval_eval.json"

with open(eval_path, "r") as file:
    evaluation_data = json.load(file)



embedder = Embedding_Model_QWEN()

retriever = ElasticsearchRetriever()

reranker = CrossEncoderModel()



qrels_dict = {} # Ground-truth relevance judgments for each evaluation query.

for sample in evaluation_data["samples"]:

    query_id = sample["query_id"]
    relevance = sample["relevance"]

    qrels_dict[query_id] = relevance


# Convert dictionary to ranx Qrels object for scoring
qrels = Qrels(qrels_dict)


#RUN RETRIEVAL PIPELINE

# Store the ranked retrieval results for each query.
run_dict = {}


for sample in evaluation_data["samples"]:

    query_id = sample["query_id"]
    query_text = sample["query"]

    print(f"Evaluating {query_id}")


    embedded_query = embedder.embed_query(query_text)

    #hybrid search + rrf
    rrf_results = retriever.hybrid_search(
        query_text,
        embedded_query[0],
        k=7
    )

    #cross-encoder rerank
    final_results = reranker.reranker(
        query_text,
        rrf_results,
        k=7
    )


    query_results = {}

    for result in final_results:

        # ranx document IDs should be strings
        chunk_id = str(result["chunk_id"])

        # CrossEncoder score determines the final ranking
        score = float(result["cross_encoder_score"])

        query_results[chunk_id] = score


    # Store this query's results
    run_dict[query_id] = query_results


run = Run(
    run_dict,
    name="RRF + CrossEncoder"
)


#retrieval calculation metrics
scores = evaluate(
    qrels,
    run,
    metrics=[
        # Precision@K 
        "precision@1",
        "precision@3",
        "precision@5",

        # Recall@K 
        "recall@1",
        "recall@3",
        "recall@5",

        # MRR 
        "mrr",

        # nDCG@K 
        "ndcg@1",
        "ndcg@3",
        "ndcg@5"
    ]
)

print("Retrieval Evaluation Results:")


for metric, score in scores.items():
    print(f"{metric}: {score:.4f}")