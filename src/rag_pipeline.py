class RAGPipeline:
    """Orchestrates retrieval, reranking, and answer generation."""

    def __init__(
        self,
        embedder,
        retriever,
        reranker,
        llm_model
    ):
        # Reuse the initialized components across user questions.
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm_model = llm_model


    def answer_question(self, query_text):

        # Embed the user query for dense retrieval.
        embedded_query = self.embedder.embed_query(
            query_text
        )

        # Combine BM25 and dense retrieval with RRF and
        # keep 7 candidates for second-stage reranking.
        rrf_results = self.retriever.hybrid_search(
            query_text,
            embedded_query[0],
            k=7
        )

        # Rerank the candidates and keep the top 3 chunks
        # as context for answer generation.
        final_results = self.reranker.reranker(
            query_text,
            rrf_results,
            k=3
        )

        # Generate a grounded answer from the retrieved context.
        response, model_used = self.llm_model.generate_answer(
            query_text,
            final_results
        )

        # Return the answer, generation model, and retrieved evidence.
        return response, model_used, final_results