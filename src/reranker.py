from sentence_transformers.cross_encoder import CrossEncoder


class CrossEncoderModel:
    """
    Reranker for hybrid retrieval results.

    Uses a cross-encoder to jointly score each query/chunk pair,
    providing more precise relevance ranking than the initial
    BM25 and dense retrieval stages.
    """

    def __init__(self):
        #lightweight MS MARCO cross-encoder for relevance scoring.
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

    def reranker(self, query_text, rrf_results, k=3):
        """
        Rerank hybrid retrieval candidates and return the top-k chunks.
        """

        # construct query/chunk pairs for cross-encoder scoring.
        pairs = []

        for result in rrf_results:
            pairs.append([
                query_text,
                result["text"]
            ])

        # score each query/chunk pair for relevance.
        scores = self.cross_encoder.predict(pairs)

        # attach the relevance score to its corresponding chunk.
        for result, score in zip(rrf_results, scores):
            result["cross_encoder_score"] = float(score)

        # sort candidates by cross-encoder relevance score.
        reranked_results = sorted(
            rrf_results,
            key=lambda x: x["cross_encoder_score"],
            reverse=True
        )

        # return the highest-ranked chunks for LLM context.
        return reranked_results[:k]


