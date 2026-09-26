import os
from elasticsearch import Elasticsearch


"""
Elasticsearch retrieval layer for the RAG pipeline.

Stores document chunks, their metadata, and dense embeddings in Elasticsearch
and supports:

- BM25 sparse retrieval over chunk text.
- kNN dense retrieval over Qwen embeddings using cosine similarity.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF) to combine
  sparse and dense rankings.

The Elasticsearch index stores the chunk ID, original chunk text,
Docling metadata, and 1024-dimensional embedding vector.
"""



class ElasticsearchRetriever:
    def __init__(self):
        elasticsearch_url = os.getenv(
            "ELASTICSEARCH_URL",
            "http://localhost:9200"
        )

        self.client = Elasticsearch(elasticsearch_url)
        self.index_name = "climate_rag"

    def create_index(self):
        """Create the Elasticsearch index and field mappings."""

        # Keep the existing index if it has already been created.
        if self.client.indices.exists(index=self.index_name):
            return

        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "chunk_id": {
                        "type": "integer"
                    },

                    # Text is indexed for BM25 lexical retrieval.
                    "text": {
                        "type": "text"
                    },

                    # Preserve Docling metadata associated with each
                    # document chunk for provenance and citations.
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    },

                    # Dense vectors support semantic retrieval using
                    # the 1024-dimensional Qwen embedding representation.
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        )


    def index_chunks(self, chunks, embedded_data):
        """
        Store document chunks, metadata, and dense embeddings in Elasticsearch.

        Each chunk retains its chunk ID and Docling metadata so results from
        sparse and dense retrieval can be matched and traced back to their
        original document context.
        """

        for i in range(len(chunks)):

            document = {
                "chunk_id": chunks[i]["chunk_id"],
                "text": chunks[i]["text"],
                "metadata": chunks[i]["metadata"],

                # Elasticsearch expects the embedding as a Python list
                # rather than the NumPy array returned by the embedder.
                "embedding": embedded_data[i].tolist()
            }

            self.client.index(
                index=self.index_name,
                id=chunks[i]["chunk_id"],
                document=document
            )

        return len(chunks)


    def knn_search(self, embedded_query, k=3):
        """
        Retrieve semantically similar chunks using the query embedding.

        Elasticsearch searches the indexed dense vectors using approximate
        nearest-neighbor retrieval and ranks candidates using cosine
        similarity.
        """

        response = self.client.search(
            index=self.index_name,
            knn={
                "field": "embedding",
                "query_vector": embedded_query,
                "k": k,

                # Search a larger candidate pool before selecting
                # the nearest k vectors.
                "num_candidates": 7
            }
        )

        return response


    def BM25_search(self, query_text, k=3):
        """
        Retrieve chunks using BM25 lexical search.

        Sparse retrieval complements dense semantic retrieval by preserving
        exact-term matching for terminology, acronyms, names, and other
        lexical signals that embeddings may not rank as strongly.
        """

        response = self.client.search(
            index=self.index_name,
            query={
                "match": {
                    "text": query_text
                }
            },
            size=k
        )

        return response


    def hybrid_search(self, query_text, embedded_query, k=3):
        """
        Perform hybrid retrieval using BM25 and dense semantic search.

        BM25 captures lexical matches while dense retrieval captures
        semantic similarity. Reciprocal Rank Fusion (RRF) combines the
        two ranked result sets without requiring their raw scores to be
        directly comparable.

        A larger candidate set is retrieved before fusion, and the final
        top-k fused chunks are returned for downstream reranking.
        """

        # Retrieve a larger candidate set from each retrieval strategy
        # before fusion so potentially relevant chunks are not removed
        # too early in the retrieval pipeline.
        candidate_k = 7

        sparse = self.BM25_search(
            query_text,
            candidate_k
        )

        dense = self.knn_search(
            embedded_query,
            candidate_k
        )

        # RRF scores are stored by chunk ID because the same chunk may
        # appear in both dense and sparse retrieval results.
        rrf_scores = {}

        # Store the chunk data separately so it can be included in the
        # final ranked output after the RRF scores are combined.
        chunks = {}


        # Add dense retrieval rankings to the RRF scores.
        for rank, result in enumerate(
            dense["hits"]["hits"],
            start=1
        ):
            chunk_id = result["_source"]["chunk_id"]

            chunks[chunk_id] = {
                "text": result["_source"]["text"],
                "metadata": result["_source"].get("metadata", {})
            }

            # Standard RRF uses rank rather than the original retrieval
            # score, allowing results from different retrieval methods
            # to be combined without score normalization.
            rrf = 1 / (60 + rank)

            rrf_scores[chunk_id] = rrf


        # Add BM25 rankings to the RRF scores.
        # Chunks appearing in both result sets receive contributions
        # from both retrieval strategies.
        for rank, result in enumerate(
            sparse["hits"]["hits"],
            start=1
        ):
            chunk_id = result["_source"]["chunk_id"]

            chunks[chunk_id] = {
                "text": result["_source"]["text"],
                "metadata": result["_source"].get("metadata", {})
            }

            rrf = 1 / (60 + rank)

            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf

            else:
                rrf_scores[chunk_id] = rrf


        # Rank candidates by their combined RRF score.
        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )


        # Construct a consistent result format for downstream reranking.
        final_results = []

        for chunk_id, score in ranked_results:

            final_results.append({
                "chunk_id": chunk_id,
                "rrf_score": score,
                "text": chunks[chunk_id]["text"],
                "metadata": chunks[chunk_id]["metadata"]
            })


        # Return only the highest-ranked fused candidates.
        return final_results[:k]