from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever
from src.reranker import CrossEncoderModel
from src.rag_pipeline import RAGPipeline
from src.llm import LLM_Model


# initialize pipeline 
embedder = Embedding_Model_QWEN()
retriever = ElasticsearchRetriever()
reranker = CrossEncoderModel()
llm_model = LLM_Model()


# RAG pipeline.
pipeline = RAGPipeline(
    embedder=embedder,
    retriever=retriever,
    reranker=reranker,
    llm_model=llm_model
)


query_text = input("Ask a question: ")

response, model_used, final_results = pipeline.answer_question(
    query_text
)


print(f"\nAnswer:\n{response.answer}")

print("\nSources:")

for source in response.sources:

    # Match the cited chunk ID to its retrieved chunk metadata.
    result = next(
        (
            result
            for result in final_results
            if result["chunk_id"] == source.chunk_id
        ),
        None
    )

    if result:

        metadata = result.get(
            "metadata",
            {}
        )

        # Get the original document filename.
        filename = metadata.get(
            "origin",
            {}
        ).get(
            "filename",
            "Unknown source"
        )

        # Get the section heading when available.
        headings = metadata.get(
            "headings",
            []
        )

        section = (
            headings[0]
            if headings
            else "Unknown section"
        )

        # Get the source page from Docling provenance.
        page_number = None

        for item in metadata.get(
            "doc_items",
            []
        ):
            provenance = item.get(
                "prov",
                []
            )

            if provenance:
                page_number = provenance[0].get(
                    "page_no"
                )
                break

        print(
            f"- {filename} | "
            f"Page {page_number} | "
            f"{section} | "
            f"Chunk {source.chunk_id}"
        )

    else:

        print(
            f"- Chunk {source.chunk_id}"
        )

print(f"\nModel used: {model_used}")