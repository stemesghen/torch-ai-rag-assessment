# Generates generation_eval_data.json for generation evaluation.

import json
from pathlib import Path

from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever
from src.reranker import CrossEncoderModel
from src.llm import LLM_Model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVAL_PATH = PROJECT_ROOT / "data" / "retrieval_eval.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "generation_eval_data.json"


with open(EVAL_PATH, "r") as file:
    evaluation_data = json.load(file)


# store the generated answers and their retrieved evidence.
generation_evaluation_data = []

# Initialize 
embedder = Embedding_Model_QWEN()
retriever = ElasticsearchRetriever()
reranker = CrossEncoderModel()
llm_model = LLM_Model()


#each eval Q is ran through the full RAG
# only the query is passed through the RAG system; reference
# answers and relevance judgments are not exposed to Gemini.
for sample in evaluation_data["samples"]:

    # Retrieve the query identifier and question.
    query_id = sample["query_id"]

    query_text = sample["query"]

    print(f"\nGenerating answer for {query_id}")

    # embed query
    embedded_query = embedder.embed_query(
        query_text
    )

    # Hybrid search w/
    # retrieval using Reciprocal Rank Fusion (RRF). - keep 7 for reranking
    rrf_results = retriever.hybrid_search(
        query_text,
        embedded_query[0],
        k=7
    )


    #crossencoder reranker - score each query/chunk pair and keep top 3
    final_results = reranker.reranker(
        query_text,
        rrf_results,
        k=3
    )

    # Generate an answer using only the query and retrieved
    # evidence. Ground-truth answers and relevance judgments
    # remain isolated from the generation step.
    #
    # model_used records which Gemini model completed the
    # request when model fallback is required.
    response, model_used = llm_model.generate_answer(
        query_text,
        final_results
    )


    # The retreived text passed into the model is preserved for faithfulness eval
    retrieval_context = []

    for result in final_results:

        retrieval_context.append(
            result["text"]
        )


    #generation results
    generation_evaluation_data.append({

        "query_id": query_id,

        "input": query_text,

        "actual_output": response.answer,

        "retrieval_context": retrieval_context,

        # the generation model 
        "model_used": model_used,

        # the chunks provided to the model
        "retrieved_chunk_ids": [
            result["chunk_id"]
            for result in final_results
        ],

        # chunks identified by the model as directly supporting the generated answer.
        "cited_chunk_ids": [
            source.chunk_id
            for source in response.sources
        ]
    })

    # Save progress after each question so completed responses are preserved if a later generation request fails.

    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            generation_evaluation_data,
            file,
            indent=2
        )


    print(
        f"Saved {query_id} using {model_used}"
    )


print(
    f"\nSaved {len(generation_evaluation_data)} "
    "generation evaluation samples."
)