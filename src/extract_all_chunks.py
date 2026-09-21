"""EXPORT the chunks all 175 
"""
import json

from src.retrieval import ElasticsearchRetriever

retriever = ElasticsearchRetriever()

response = retriever.client.search(
    index="climate_rag",
    query={
        "match_all": {}
    },
    size=1000,
    source=["chunk_id", "text"]
)
all_chunks = []

for hit in response["hits"]["hits"]:
    all_chunks.append({
        "chunk_id": hit["_source"]["chunk_id"],
        "text": hit["_source"]["text"]
    })

all_chunks = sorted(
    all_chunks,
    key=lambda x: x["chunk_id"]
)

with open("all_chunks.json", "w", encoding="utf-8") as file:
    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )