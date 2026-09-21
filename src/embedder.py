from sentence_transformers import SentenceTransformer


class Embedding_Model_QWEN:

    def __init__(self):

        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B"
        )

        print(
            f"Embedding model device: {self.model.device}"
        )


    def embed_data(self, chunk_text):

        embeddings = self.model.encode(
            chunk_text,
            batch_size=8,
            show_progress_bar=True
        )

        return embeddings


    def embed_query(self, query_text):

        # Qwen3-Embedding is instruction-aware.
        #
        # The "query" prompt tells the embedding model that
        # this text represents a retrieval query rather than
        # document content.
        #
        # Document chunks are embedded normally, while the
        # query receives the retrieval instruction.

        embeddings = self.model.encode(
            [query_text],
            prompt_name="query"
        )

        return embeddings