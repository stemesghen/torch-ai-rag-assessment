# Technical Document RAG Assistant

A prototype retrieval-augmented generation (RAG) assistant for
processing technical PDF documents, retrieving relevant evidence, and
generating grounded, structured answers.

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

## Document Ingestion

Before running the RAG application for the first time, ingest the source document:

```bash
python -m src.ingest
```

The ingestion pipeline:

1. Processes and chunks the source document using Docling.
2. Caches the processed chunks as JSON so the document does not need to be reparsed on every run.
3. Generates dense embeddings for the chunks and caches them as a NumPy (`.npy`) file.
4. Creates the Elasticsearch index if needed.
5. Stores the chunk text and embeddings in Elasticsearch for retrieval.

After ingestion is complete, run the application:

```bash
python -m src.main
```

The ingestion step only needs to be rerun when the source document changes, the cached chunks or embeddings are invalidated, or the Elasticsearch index is recreated.

## Quick Start with Docker

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

The source PDF is parsed with **Docling** and chunked using `HybridChunker`.

```text
PDF → Docling → structured document → HybridChunker → chunks
```

### Why Docling and HybridChunker?

Docling preserves structural information from technical PDFs rather than immediately flattening the document to plain text. `HybridChunker` combines document-aware hierarchical chunking with tokenizer-aware size constraints, preserving meaningful boundaries while keeping chunks appropriately sized for retrieval and generation.

## 2. Embeddings

Document chunks are embedded with `Qwen/Qwen3-Embedding-0.6B`, producing 1024-dimensional vectors.

The 0.6B model was selected as a quality-versus-resource tradeoff for the relatively small assessment corpus. It can run locally while supporting instruction-aware retrieval.

Queries use Qwen's retrieval-specific query encoding:

```python
self.model.encode(
    [query_text],
    prompt_name="query"
)
```

Instruction-aware query encoding was retained after evaluation showed improvements in Recall@5 and nDCG.

## 3. Elasticsearch Index

Elasticsearch stores each chunk's text, chunk ID, and 1024-dimensional dense embedding.

The text field supports **BM25 lexical retrieval**, while the indexed dense-vector field supports **approximate nearest-neighbor retrieval using cosine similarity**.

Document ingestion is separated from question answering so parsing and document embedding do not need to be repeated for every query.

## 4. Hybrid Retrieval

Each query follows two retrieval paths:

* **BM25** captures exact terminology, acronyms, and distinctive technical phrases.
* **Dense retrieval** uses the Qwen query embedding with Elasticsearch kNN search to capture semantic similarity.

The rankings are combined using **Reciprocal Rank Fusion (RRF)**:

```text
RRF score = Σ 1 / (60 + rank)
```

RRF combines ranking positions rather than directly comparing BM25 and vector similarity scores, which are not naturally on the same scale.

## 5. CrossEncoder Reranking

The top seven hybrid-retrieval candidates are reranked with `cross-encoder/ms-marco-MiniLM-L6-v2`.

Unlike the embedding retriever, the CrossEncoder evaluates each query/chunk pair jointly. The top three reranked chunks are then supplied to the generation model.


```text
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

This provides a broader candidate set during retrieval while limiting the more expensive CrossEncoder and generation stages to a small set of high-quality evidence.

## 6. Evaluation

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

## 7. Evaluation Environments

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

# Retrieval evaluation
python -m evaluation.retrieval_eval

# Negative tests
python -m evaluation.negative_eval

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

# Generation evaluation
source .venv-eval/bin/activate
python -m evaluation.DeepEval_LLM

Evaluation is kept separate from normal application startup. The
assistant does not rerun the 25-case generation benchmark every time the
application starts.

## 8. Main Dependencies

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


## 9. Example Query

### 1. Multi-Chunk Synthesis

**Query**

> How does the IPCC assess the relationship between human-caused warming, observed changes in climate extremes, and the risks expected to increase with additional global warming?

**Answer**

> Based on the provided context, the IPCC assesses these relationships as follows:
>
> - **Human-caused changes:** Observed impacts are driven by and connected to physical climate conditions (climatic impact-drivers), many of which are increasingly attributed to human influence.
> - **Escalation of risks with warming:** Risks, projected adverse impacts, and related losses and damages escalate with every increment of global warming (very high confidence). They are higher at 1.5°C than at present and even higher at 2°C (high confidence).
> - **Long-term and compound risks:** Projected long-term impacts are up to multiple times higher than currently observed (high confidence). Additionally, climatic and non-climatic risks will increasingly interact, creating compound and cascading risks that are more complex and difficult to manage (high confidence).
> - **Lower threshold for high risk:** Compared to AR5, global aggregated risk levels are assessed to reach high to very high levels at lower warming levels due to recent evidence of observed impacts, improved process understanding, and new insights into exposure, vulnerability, and adaptation limits (high confidence).
>
> *(Note: The provided text refers to physical climate changes/conditions and climatic impact-drivers rather than detailing specific "climate extremes" by name.)*

**Sources:**
- `final_data.pdf | Page 13 | Adverse impacts from human-caused climate change will continue to intensify | Chunk 38`
- `final_data.pdf | Page 20 | Climate Change Impacts and Climate-Related Risks | Chunk 73`
- `final_data.pdf | Page 21 | Climate Change Impacts and Climate-Related Risks | Chunk 75`

**Model:** `gemini-3.8-flash`

This example demonstrates retrieval and synthesis across multiple chunks while preserving source-document, page, section, and chunk-level provenance from the document metadata.

---

### 2. Factual Retrieval

**Query**

> How much higher was global surface temperature in 2011–2020 compared with 1850–1900?

**Answer**

> Global surface temperature was 1.09 [0.95 to 1.20]°C higher (also cited as reaching 1.1°C above) in 2011–2020 compared with 1850–1900, with larger increases over land (1.59 [1.34 to 1.83]°C) than over the ocean (0.88 [0.68 to 1.01]°C).

**Sources:**
- `final_data.pdf | Page 10 | Observed Warming and its Causes | Chunk 21`
- `final_data.pdf | Page 10 | Observed Warming and its Causes | Chunk 22`

**Model:** `gemini-3.8-flash`

This example demonstrates retrieval of specific quantitative information with source-document, page, section, and chunk-level attribution.

---

### 3. Unsupported Query / Groundedness

**Query**

> What was the weather in 2018?

**Answer**

> The available context does not contain enough information to determine what the weather was in 2018.

**Sources:** None

**Model:** `gemini-3.8-flash`

This example demonstrates grounded behavior for an unsupported question. Although retrieval can still return semantically related chunks, the generation layer does not treat those chunks as sufficient evidence and returns no source citations.