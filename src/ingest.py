
# """
# One-time document ingestion pipeline.

# 1. Processes and chunks the source document, or loads cached chunks when available.
# 2. Generates dense embeddings for the document chunks.
# 3. Stores the chunk text and embeddings in Elasticsearch.

# Ingestion only needs to be rerun when the source document changes or
# the Elasticsearch index is recreated.
# """

# from pathlib import Path

# from src.document_processing import process_document
# from src.embedder import Embedding_Model_QWEN
# from src.retrieval import ElasticsearchRetriever


# def build_index(chunk_texts):
#     """Embed document chunks and index them in Elasticsearch."""

#     embedder = Embedding_Model_QWEN()

#     embedded_data = embedder.embed_data(
#         chunk_texts
#     )

#     retriever = ElasticsearchRetriever()

#     # Create the index before ingesting document chunks.
#     retriever.create_index()

#     # Store chunk text and dense embeddings for retrieval.
#     stored_data_embed = retriever.index_chunks(
#         chunk_texts,
#         embedded_data
#     )

#     print(
#         f"Stored {stored_data_embed} chunks "
#         "in Elasticsearch."
#     )


# if __name__ == "__main__":

#     project_path = Path(__file__).resolve().parent.parent

#     file_path = (
#         project_path
#         / "data"
#         / "final_data.pdf"
#     )

#     cache_path = (
#         project_path
#         / "data"
#         / "all_chunks.json"
#     )

#     chunk_texts = process_document(
#         file_path,
#         cache_path
#     )

#     build_index(chunk_texts)


"""
One-time document ingestion pipeline.

1. Checks whether the Elasticsearch index already exists.
2. Processes and chunks the source document, or loads cached chunks when available.
3. Generates dense embeddings, or loads cached embeddings when available.
4. Stores the chunk text and embeddings in Elasticsearch.

Ingestion only needs to run when the Elasticsearch index does not exist
or the source document has intentionally been changed.
"""

from pathlib import Path

import numpy as np

from src.document_processing import process_document
from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever


def build_index(chunk_texts, embeddings_cache, retriever):
    """
    Build the Elasticsearch index using cached embeddings when available.
    """

    # Load previously generated embeddings when available.
    if embeddings_cache.exists():

        print("Loading cached embeddings...")

        embedded_data = np.load(
            embeddings_cache
        )

        # Make sure cached embeddings still correspond to the chunks.
        if len(embedded_data) != len(chunk_texts):
            raise ValueError(
                "Cached embedding count does not match chunk count. "
                "Delete embeddings.npy and rerun ingestion."
            )

        print(
            f"Loaded {len(embedded_data)} cached embeddings."
        )

    else:

        print("No embedding cache found.")
        print("Generating embeddings...")

        embedder = Embedding_Model_QWEN()

        embedded_data = embedder.embed_data(
            chunk_texts
        )

        np.save(
            embeddings_cache,
            embedded_data
        )

        print(
            f"Cached {len(embedded_data)} embeddings."
        )

    # Create the Elasticsearch index.
    retriever.create_index()

    # Store chunks and embeddings.
    stored_data_embed = retriever.index_chunks(
        chunk_texts,
        embedded_data
    )

    print(
        f"Stored {stored_data_embed} chunks "
        "in Elasticsearch."
    )


if __name__ == "__main__":

    project_path = Path(__file__).resolve().parent.parent

    file_path = (
        project_path
        / "data"
        / "final_data.pdf"
    )

    cache_path = (
        project_path
        / "data"
        / "all_chunks.json"
    )

    embeddings_cache = (
        project_path
        / "data"
        / "embeddings.npy"
    )

    retriever = ElasticsearchRetriever()

    # ---------------------------------------------------------
    # Check Elasticsearch BEFORE doing any expensive processing.
    # ---------------------------------------------------------

    if retriever.client.indices.exists(
        index=retriever.index_name
    ):

        count = retriever.client.count(
            index=retriever.index_name
        )["count"]

        print(
            f"Existing index '{retriever.index_name}' "
            f"found with {count} chunks."
        )

        print("Skipping ingestion.")

    else:

        print(
            f"Index '{retriever.index_name}' not found."
        )

        print("Starting ingestion...")

        # Process the document or load cached chunks.
        chunk_texts = process_document(
            file_path,
            cache_path
        )

        # Build Elasticsearch index.
        build_index(
            chunk_texts,
            embeddings_cache,
            retriever
        )