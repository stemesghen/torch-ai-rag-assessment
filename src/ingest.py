
"""
One-time document ingestion pipeline.

1. Processes and chunks the source document, or loads cached chunks when available.
2. Generates dense embeddings for the document chunks.
3. Stores the chunk text and embeddings in Elasticsearch.

Ingestion only needs to be rerun when the source document changes or
the Elasticsearch index is recreated.
"""

from pathlib import Path

from src.document_processing import process_document
from src.embedder import Embedding_Model_QWEN
from src.retrieval import ElasticsearchRetriever


def build_index(chunk_texts):
    """Embed document chunks and index them in Elasticsearch."""

    embedder = Embedding_Model_QWEN()

    embedded_data = embedder.embed_data(
        chunk_texts
    )

    retriever = ElasticsearchRetriever()

    # Create the index before ingesting document chunks.
    retriever.create_index()

    # Store chunk text and dense embeddings for retrieval.
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

    chunk_texts = process_document(
        file_path,
        cache_path
    )

    build_index(chunk_texts)