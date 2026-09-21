
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

"""
Parser - Docling & Docling Chunker & Emebedder
Docling

"""

#CHECK FILEPATH

project_path = Path(__file__).resolve().parent.parent

file_path = project_path / "data" / "final_data.pdf"


import json
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import json

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


def process_document(file_path, cache_path=None):

    # Use cached chunks if they already exist.
    if cache_path and cache_path.exists():

        with open(cache_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(f"Loaded {len(chunk_texts)} cached chunks")

        return chunk_texts

    # No cache exists — process the original document.
    converter = DocumentConverter()

    result = converter.convert(file_path)
    docling_doc = result.document

    print("Conversion complete")

    # Structure-aware chunking with token-based splitting.
    hybrid_chunker = HybridChunker()

    chunkings = list(
        hybrid_chunker.chunk(docling_doc)
    )

    print("Number of chunks:", len(chunkings))

    chunk_texts = [
        chunk.text
        for chunk in chunkings
    ]

    # Cache chunks for future runs.
    if cache_path:

        cached_chunks = [
            {
                "chunk_id": i,
                "text": text
            }
            for i, text in enumerate(chunk_texts)
        ]

        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(
                cached_chunks,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(f"Cached {len(chunk_texts)} chunks")

    return chunk_texts