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
    print(f"- Chunk {source.chunk_id}")

print(f"\nModel used: {model_used}")