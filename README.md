# Technical Document RAG Assistant

A prototype retrieval-augmented generation (RAG) assistant for
processing technical PDF documents, retrieving relevant evidence, and
generating grounded, structured answers.

The system was designed for the AI/ML engineering programming exercise
with an emphasis on **retrieval design, grounding, evaluation, and
explainable system architecture** rather than production completeness.

## Overview

The assistant separates document ingestion from question answering:

``` text
DOCUMENT INGESTION — run once

PDF
 ↓
Docling
 ↓
Structured document representation
 ↓
HybridChunker
 ↓
Document chunks
 ↓
Qwen3-Embedding-0.6B
 ↓
Elasticsearch
 ├── original chunk text
 ├── chunk ID
 └── dense vector embedding


QUESTION ANSWERING — run per query

User question
 ↓
Qwen query embedding
 ↓
┌──────────────────────┐
│ Hybrid Retrieval     │
│ BM25 + Dense kNN     │
│ HNSW / cosine        │
└──────────┬───────────┘
           ↓
Reciprocal Rank Fusion (RRF)
           ↓
Top 7 candidates
           ↓
CrossEncoder reranking
           ↓
Top 3 evidence chunks
           ↓
Gemini
           ↓
Validated structured response
(question + answer + source chunk IDs)
```

The retrieval pipeline intentionally combines lexical and semantic
retrieval. BM25 helps preserve exact-term matching, while dense
retrieval captures semantic similarity. Reciprocal Rank Fusion combines
the two rankings without requiring their incompatible raw scores to be
normalized. A CrossEncoder then performs a more expensive second-stage
relevance comparison on the smaller fused candidate set.


## Quick Start with Docker

The recommended way to run the prototype is with Docker Compose. This starts the application and its Elasticsearch dependency in a reproducible environment.

### 1. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

```text
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
```

Do not commit `.env` or API keys to source control.

### 2. Build the application

From the project root:

```bash
make build
```

The application image installs the Python dependencies, Docling document-processing dependencies, and the system libraries required for OCR.

### 3. Start Elasticsearch

```bash
make up
```

Elasticsearch must be running before document ingestion or question answering.

### 4. Ingest the document

```bash
make ingest
```

Ingestion processes `data/final_data.pdf` once:

```text
PDF
→ Docling
→ HybridChunker
→ Qwen embeddings
→ Elasticsearch index
```

The resulting chunks and embeddings remain in Elasticsearch and are reused for subsequent questions.

### 5. Run the RAG assistant

```bash
make run
```

The query-time pipeline performs:

```text
Question
→ Qwen query embedding
→ BM25 + dense kNN retrieval
→ Reciprocal Rank Fusion
→ CrossEncoder reranking
→ Top 3 evidence chunks
→ Gemini
→ Pydantic-validated structured response
```

Document ingestion does not need to be repeated for every question. Rerun `make ingest` when the source document changes or when the Elasticsearch index is recreated.

### Local Development

The application can also be run directly in a Python 3.12 virtual environment.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the application dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and provide the required API key:

```text
GEMINI_API_KEY=your_key_here
```

Ensure Elasticsearch is running locally before ingestion or question answering.

Run document ingestion:

```bash
python -m src.ingest
```

Run the assistant:

```bash
python -m src.main
```

The Docker workflow is recommended because it provides Elasticsearch and a consistent application environment.

## Features

-   PDF parsing with Docling
-   Structure-aware and token-aware chunking with `HybridChunker`
-   Local dense embeddings with `Qwen/Qwen3-Embedding-0.6B`
-   Elasticsearch storage and retrieval
-   BM25 lexical retrieval
-   Dense approximate nearest-neighbor retrieval using Elasticsearch kNN
    with cosine similarity
-   Reciprocal Rank Fusion (RRF)
-   CrossEncoder reranking with `cross-encoder/ms-marco-MiniLM-L6-v2`
-   Gemini-based grounded answer generation
-   Pydantic structured output validation
-   Retrieval evaluation with graded relevance judgments
-   Generation evaluation for faithfulness and answer relevancy
-   Negative testing for unsupported and false-premise questions

## Project Structure

``` text
Torch.AI-project/
│
├── data/
│   └── final_data.pdf
│
├── evaluation/
│   ├── __init__.py
│   ├── DeepEval_LLM.py
│   ├── generation_eval_data.py
│   ├── negative_eval.py
│   └── retrieval_eval.py
│
├── src/
│   ├── __init__.py
│   ├── document_processing.py
│   ├── embedder.py
│   ├── extract_all_chunks.py
│   ├── ingest.py
│   ├── llm.py
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   └── retrieval.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── README.md
├── requirements.txt
└── requirements-deepeval.txt
```

The notebook contains experimentation, inspection, and development
notes. The reusable application components live in `src/` so the final
pipeline does not depend on notebook state.

## 1. Document Processing

The ingestion pipeline begins by parsing the source PDF with Docling:

``` text
PDF → Docling → structured document → HybridChunker → chunks
```

### Why Docling?

Technical PDFs can contain headings, sections, tables, layout
information, and other structural signals. I wanted to avoid
unnecessarily flattening the document before retrieval because document
structure can provide useful context for chunk construction and source
attribution.

Docling creates a structured representation of the source document
before chunking.

### Why HybridChunker?

I used Docling's `HybridChunker` rather than a fixed character splitter.
It combines document-aware hierarchical chunking with tokenizer-aware
size constraints.

This preserves meaningful document boundaries where possible while
preventing chunks from becoming too large for downstream embedding and
generation.

The resulting Docling chunk objects are converted to their text
representation before embedding:

``` python
chunk_texts = []

for chunk in chunkings:
    chunk_texts.append(chunk.text)
```

The entire document is **not** represented by one embedding. Each chunk
receives its own embedding so a query can retrieve the specific portions
of the document that are most relevant.

## 2. Embeddings

Document chunks are embedded with:

``` text
Qwen/Qwen3-Embedding-0.6B
```

The model produces 1024-dimensional dense vectors in this
implementation.

### Model selection

Considering retrieval quality together with model size, memory
requirements, embedding dimensionality, deployment complexity, and the
relatively small size of the assessment corpus.

Selecting 0.6B Qwen3 embedding model was as a quality-versus-resource
tradeoff rather than selecting the largest available model. It is small
enough to run locally for the prototype while supporting
instruction-aware retrieval.

Document chunks are embedded normally. Queries use Qwen's
retrieval-specific query prompt:

``` python
self.model.encode(
    [query_text],
    prompt_name="query"
)
```

This explicitly tells the embedding model that the input represents a
retrieval query.

## 3. Elasticsearch Index

Each document chunk is stored in Elasticsearch with three core fields:

``` json
{
  "chunk_id": 0,
  "text": "...",
  "embedding": [0.01, -0.03, "..."]
}
```

The dense vector mapping is configured with:

``` text
dimensions: 1024
similarity: cosine
index: true
```

The text field supports BM25 lexical search, while the dense vector
field supports semantic retrieval.

Document ingestion is intentionally separate from question answering.
Parsing and embedding the source document are relatively expensive
operations and do not need to be repeated for every query.

## 4. Hybrid Retrieval

A user query is searched through two retrieval paths.

### BM25 lexical retrieval

The raw query text is passed directly to Elasticsearch's text search.

BM25 is useful when exact terminology, acronyms, or distinctive
technical phrases are important.

### Dense semantic retrieval

The query is embedded with Qwen and passed to Elasticsearch kNN search.

Elasticsearch uses an approximate nearest-neighbor vector index to
efficiently locate vectors that are close to the query according to the
configured cosine similarity measure.

Conceptually:

``` text
HNSW / ANN = how candidate vectors are searched efficiently
Cosine     = how vector closeness is measured
```

### Reciprocal Rank Fusion

The BM25 and dense rankings are combined with Reciprocal Rank Fusion:

``` text
RRF score = Σ 1 / (60 + rank)
```

RRF operates on ranking positions rather than attempting to directly
compare BM25 scores with vector similarity scores.

This is useful because the raw scores produced by lexical and dense
retrieval are not naturally on the same scale.

## 5. CrossEncoder Reranking

Hybrid retrieval returns seven candidates.

These candidates are reranked with:

``` text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Unlike the embedding stage, which encodes the query and document
independently, the CrossEncoder evaluates each query/chunk pair
together.

``` text
BM25 + Dense
      ↓
     RRF
      ↓
Top 7 candidates
      ↓
CrossEncoder
      ↓
Top 3 chunks
```

Only the final top three chunks are supplied to the generation model. 

This keeps the more computationally expensive CrossEncoder focused on a
small candidate set while improving the ordering of evidence supplied to
the LLM.

## 6. Grounded Question Answering

The final three chunks are passed to Gemini as retrieved evidence.

The generation prompt instructs the model to:

-   answer using only the supplied document context
-   avoid unsupported inference
-   state when the available context is insufficient
-   cite only chunks that directly support the answer
-   synthesize the evidence rather than simply reproducing retrieved
    text

The LLM provider is isolated behind its own component so generation can
be changed without redesigning retrieval.

The implementation also records which Gemini model successfully
generated the response when model fallback is used.

## 7. Structured Output

Responses are validated with Pydantic.

The output schema is conceptually:

``` json
{
  "question": "What are the main barriers to climate change adaptation?",
  "answer": "The document identifies ...",
  "sources": [
    {
      "chunk_id": 42
    },
    {
      "chunk_id": 47
    }
  ]
}
```

Each source corresponds to a retrieved chunk that the generation model
determined directly supports the answer.

Importantly, **retrieved chunks and cited chunks are not treated as the
same thing**. Retrieval always returns the nearest available candidates,
but the generation layer is instructed not to cite a chunk unless it
actually supports the response.

This distinction is important for unsupported questions.

## 8. Evaluation

I evaluated retrieval and generation separately so failures can be
attributed to the appropriate stage of the RAG pipeline.

### Retrieval Evaluation

Because the source document did not include an existing labeled
retrieval benchmark, I constructed a small evaluation set of 25
representative questions.

For each question, relevant source chunks were identified and assigned
graded relevance judgments. The benchmark was then frozen and reused
across retrieval experiments.

Evaluation was performed with `ranx` using:

-   Precision@K
-   Recall@K
-   Mean Reciprocal Rank (MRR)
-   normalized Discounted Cumulative Gain (nDCG)

The final instruction-aware query encoding produced:

  Metric           Score
  ------------- --------
  Precision@1     0.8800
  Precision@3     0.4933
  Precision@5     0.3360
  Recall@1        0.5000
  Recall@3        0.8033
  Recall@5        0.8967
  MRR             0.9300
  nDCG@1          0.8667
  nDCG@3          0.8383
  nDCG@5          0.8707

The high MRR indicates that relevant evidence was generally ranked near
the top. Recall@3 of approximately 0.80 indicates that the three chunks
ultimately supplied to the LLM captured most of the labeled relevant
evidence.

Precision@3 should be interpreted together with the benchmark
construction: several questions contain fewer than three labeled
relevant chunks, which places a natural ceiling on Precision@3 even when
the available relevant evidence is ranked correctly.

### Query Encoding Experiment

I also compared the baseline query embedding against Qwen's
instruction-aware query encoding using the same frozen evaluation set.

Instruction-aware encoding increased:

``` text
Recall@5: 0.8767 → 0.8967
nDCG@3:   0.8319 → 0.8383
nDCG@5:   0.8597 → 0.8707
```

Top-ranked metrics remained stable, so I retained instruction-aware
query encoding.

### Generation Evaluation

Generation quality was evaluated separately using DeepEval across 25
generated responses.

Metrics:

-   **Faithfulness** --- whether the answer is supported by the
    retrieved context
-   **Answer Relevancy** --- whether the answer addresses the user's
    question

Results:

  Metric               Average    Passed
  ------------------ --------- ---------
  Faithfulness            1.00   25 / 25
  Answer Relevancy        0.99   25 / 25

A threshold of `0.7` was used.

Gemini 3.8 Flash was used as the LLM judge. These results should
therefore be interpreted as an **LLM-as-judge evaluation**, not as a
substitute for human evaluation.

The generation samples themselves were produced through the configured
Gemini generation layer with model fallback, so this evaluation
represents the generation system rather than a controlled benchmark of
one single generation model.

### Negative / Groundedness Evaluation

I also tested three unsupported-query cases:

1.  an out-of-domain question
2.  a plausible but unsupported factual question
3.  a question containing an unsupported premise

All three produced the intended grounded behavior.

``` text
Negative tests passed: 3 / 3
```

For unsupported questions, retrieval still returned nearest-neighbor
chunks, as expected. However, the generation layer did not cite those
chunks when they did not support an answer.

For the false-premise test, the model rejected the unsupported premise
and described what the retrieved document actually supported.


## 9. Query-Time Pipeline

After ingestion, each question follows:

``` text
Question
 ↓
Instruction-aware Qwen query embedding
 ↓
BM25 + dense kNN retrieval
 ↓
RRF
 ↓
7 candidates
 ↓
CrossEncoder
 ↓
3 evidence chunks
 ↓
Gemini
 ↓
Structured response
```

The `RAGPipeline` coordinates these reusable components:

``` python
response, model_used, final_results = pipeline.answer_question(
    query_text
)
```

`response` contains the validated answer and citations, `model_used`
records the Gemini model used for generation, and `final_results`
contains the exact chunks supplied to the LLM.

## 10. Evaluation Environments

The main application and retrieval evaluation use the primary
environment:

``` text
.venv
├── document processing
├── embeddings
├── retrieval
├── reranking
├── generation
├── retrieval evaluation
└── negative evaluation
```

DeepEval generation evaluation is isolated in a second environment
because its dependency requirements conflicted with the main prototype
environment:

``` text
.venv-eval
└── DeepEval generation evaluation
```

To run the DeepEval evaluation:

``` bash
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-deepeval.txt
```

Evaluation is kept separate from normal application startup. The
assistant does not rerun the 25-case generation benchmark every time the
application starts.

## 11. Main Dependencies

The primary implementation uses:

``` text
Docling / docling-core     document parsing and chunking
sentence-transformers     Qwen embeddings and CrossEncoder
Elasticsearch             BM25 + vector storage/retrieval
google-genai              Gemini generation
Pydantic                   structured output validation
ranx                       retrieval evaluation
python-dotenv              environment configuration
```

DeepEval is installed separately for generation evaluation.


## Example Query

``` text
What are the main barriers to climate change adaptation?
```

The system embeds the question, retrieves lexical and semantic
candidates, fuses and reranks them, sends the final evidence to Gemini,
and returns a structured response similar to:

``` json
{
  "question": "What are the main barriers to climate change adaptation?",
  "answer": "The retrieved document context identifies several barriers ...",
  "sources": [
    {
      "chunk_id": 123
    }
  ]
}
```

The example above illustrates the response schema; exact answer text and
source IDs depend on the indexed source document and retrieval results.



