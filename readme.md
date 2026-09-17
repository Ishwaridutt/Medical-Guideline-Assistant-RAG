# Medical Guideline RAG Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** system designed to answer questions using information retrieved from official medical guidelines.

The project combines **hybrid retrieval, cross-encoder reranking, HyDE-based query improvement, Corrective RAG (CRAG), and input/output guardrails** to improve retrieval quality, answer grounding, and safety.

> **Disclaimer:** This project is for educational and experimental purposes. It is not a diagnostic, treatment, or emergency medical system.

---

## Architecture

The system follows a multi-stage RAG pipeline:

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Input Guardrails  │
                         │                     │
                         │ • Medical intent    │
                         │ • Out-of-scope      │
                         │ • Emergency/crisis  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Query Router     │
                         │                     │
                         │ Query Quality Score │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             Good retrieval query            Weak query
                    │                               │
                    │                               ▼
                    │                       ┌──────────────┐
                    │                       │     HyDE      │
                    │                       │              │
                    │                       │ Generate     │
                    │                       │ hypothetical │
                    │                       │ document     │
                    │                       └──────┬───────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                       ┌──────────────────────┐
                       │   Hybrid Retrieval   │
                       │                      │
                       │ Dense + Sparse       │
                       │ Qdrant RRF           │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Cross-Encoder        │
                       │ Reranking            │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │       CRAG           │
                       │                      │
                       │ Evaluate retrieved   │
                       │ documents            │
                       └──────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Relevant enough              Not relevant
                    │                           │
                    │                           ▼
                    │                  ┌─────────────────┐
                    │                  │ Query Correction│
                    │                  │ + Retry         │
                    │                  └────────┬────────┘
                    │                           │
                    │                    bounded retries
                    │                           │
                    │                           └──────┐
                    │                                  │
                    └──────────────────┬───────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │        LLM          │
                            │                     │
                            │ Grounded generation │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Output Guardrails  │
                            │                     │
                            │ • Citation checks   │
                            │ • Claim grounding   │
                            │ • Safety checks     │
                            │ • Medical advice    │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │    Final Answer     │
                            └─────────────────────┘
```

---

# End-to-End Flow

The complete flow is:

```text
User Query
    ↓
Input Guardrails
    ↓
Query Quality Evaluation
    ↓
┌─────────────────────────────┐
│ Is the query retrieval-ready?│
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
      YES              NO
       │               │
       │              HyDE
       │               ↓
       │       Hypothetical Document
       │               │
       └───────┬───────┘
               ↓
        Hybrid Retrieval
               ↓
        Cross-Encoder
          Reranking
               ↓
             CRAG
               ↓
     Evaluate Retrieved Context
               ↓
       ┌───────┴────────┐
       │                │
    Relevant         Not Relevant
       │                │
       │                ↓
       │          Correct Query
       │                ↓
       │             Retry
       │                │
       │        ┌───────┴───────┐
       │        │               │
       │     Success        Max retries
       │        │               │
       └────────┤               ↓
                │          Safe Failure
                ↓
              LLM
                ↓
        Output Guardrails
                ↓
          Final Answer
```

---

# Key Components

## 1. Input Guardrails

Input guardrails run before retrieval.

They prevent inappropriate queries from entering the RAG pipeline, including:

* Personalized medical advice
* Emergency or crisis requests
* Non-medical/out-of-scope queries

The guardrail can short-circuit the pipeline when the input should not proceed to retrieval.

```text
User Query
    ↓
Input Guardrail
    ↓
Invalid / Unsafe ─────→ Safe Refusal
    │
    └── Valid
         ↓
      RAG Pipeline
```

---

# 2. Query Router

The query router determines whether the original query is suitable for direct retrieval.

A model-based **query quality score** is generated between `0` and `1`.

Conceptually:

```text
High Score
    ↓
Direct Retrieval

Low Score
    ↓
HyDE
    ↓
Hypothetical Document
    ↓
Retrieval
```

The important distinction is that the system keeps the original question separate from the retrieval query.

```python
{
    "user_question": original_query,
    "retrieval_query": retrieval_query,
    "hyde_used": True,
    "query_quality_score": score
}
```

This allows the system to:

* Retrieve using the improved query
* Generate the final answer against the original question
* Evaluate CRAG against the original question

---

# 3. HyDE — Hypothetical Document Embeddings

**HyDE (Hypothetical Document Embeddings)** is used when the query is considered less suitable for direct retrieval.

Instead of immediately searching using the user's question, the system generates a hypothetical answer/document that represents what relevant guideline content might look like.

```text
Original Query
      ↓
Query Quality Evaluation
      ↓
Low Retrieval Suitability
      ↓
Generate Hypothetical Document
      ↓
Use Hypothetical Document
as Retrieval Query
      ↓
Hybrid Retrieval
```

The hypothetical document is used to improve retrieval.

It is **not treated as factual medical evidence** and is not directly passed to the final answer as authoritative context.

The final answer is generated from retrieved guideline documents.

---

# 4. Hybrid Retrieval

The project uses **Qdrant hybrid retrieval** combining:

* Dense vector retrieval
* Sparse/BM25 retrieval
* Reciprocal Rank Fusion (RRF)

### Dense Retrieval

Dense embeddings are generated using:

```text
BAAI/bge-base-en-v1.5
```

This captures semantic similarity between queries and guideline content.

### Sparse Retrieval

Sparse retrieval uses:

```text
Qdrant/bm25
```

This helps preserve lexical matching for important medical terminology.

### Hybrid Retrieval

Qdrant combines the dense and sparse retrieval signals using hybrid retrieval/RRF.

```text
                 Query
                   │
          ┌────────┴────────┐
          │                 │
     Dense Search       Sparse Search
          │                 │
          │            BM25 matching
          │                 │
          └────────┬────────┘
                   ↓
                 RRF
                   ↓
          Combined Candidates
```

---

# 5. Cross-Encoder Reranking

After hybrid retrieval, the retrieved documents are reranked using a cross-encoder.

Instead of relying only on embedding similarity, the reranker evaluates the relationship between:

```text
Query ↔ Document
```

This provides a second-stage relevance check before documents are passed further into the pipeline.

```text
Hybrid Retrieval
      ↓
Candidate Documents
      ↓
Cross-Encoder
      ↓
Reranked Documents
```

---

# 6. CRAG — Corrective RAG

CRAG is used to determine whether the retrieved context is actually relevant enough to answer the user's question.

The system evaluates retrieved documents against the **original user query**.

This is important because HyDE may change the retrieval query, but CRAG should still determine whether the retrieved evidence answers what the user actually asked.

### CRAG Flow

```text
Original User Query
        │
        │
        ▼
Retrieved Documents
        │
        ▼
Document Relevance Evaluation
        │
        ▼
Relevant Documents
        │
        ▼
CRAG Decision
```

If the retrieved context is sufficiently relevant:

```text
CRAG_SUCCESS
     ↓
Generate Answer
```

If the context is insufficient:

```text
CRAG_FAILED
     ↓
Correct / Rewrite Retrieval Query
     ↓
Retrieve Again
     ↓
Rerank Again
     ↓
Evaluate Again
```

---

## CRAG Retry Protection

A corrective loop can potentially continue forever.

To prevent this, the pipeline tracks the retry count.

Conceptually:

```python
retry_count = 0

while retry_count < MAX_RETRIES:
    retrieve()
    rerank()
    evaluate()

    if relevant:
        break

    retry_count += 1
    correct_query()
```

The retry limit ensures that CRAG remains bounded.

Example configuration:

```text
MAX_RETRIES = 2
```

The exact threshold and retry count are configurable.

---

# 7. Grounded LLM Generation

Once CRAG accepts the retrieved context, the documents are formatted and passed to the LLM.

The prompt instructs the model to answer using the retrieved guideline context.

The context contains metadata such as:

```text
Page Content
Page Number
Chunk Id
File Location
```

This allows generated answers to reference the source material.

The LLM should not treat HyDE output or unsupported model knowledge as authoritative evidence.

---

# 8. Output Guardrails

After the LLM generates an answer, the output passes through output guardrails.

The output checks include:

### Citation Validation

Verify that cited `chunk_id`s exist in the retrieved context.

### Claim Grounding

Check whether generated claims are supported by the retrieved guideline content.

### Medical Safety

Detect problematic output such as:

* Personalized medical advice
* Diagnostic language
* Dosing instructions
* Emergency guidance

### Standard Disclaimer

The final assistant response includes:

> This assistant provides information from official guidelines only and does not give personalized medical advice.

---

# Complete Pipeline

At a high level, the project now follows:

```text
                         USER
                          │
                          ▼
                  INPUT GUARDRAILS
                          │
                          ▼
                   QUERY ROUTER
                          │
                Query Quality Score
                          │
             ┌────────────┴────────────┐
             │                         │
          Direct                     HyDE
        Retrieval                     │
             │              Hypothetical Document
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                 HYBRID RETRIEVAL
                 Dense + Sparse
                          │
                          ▼
                 QDRANT + RRF
                          │
                          ▼
              CROSS-ENCODER RERANKING
                          │
                          ▼
                         CRAG
                          │
             ┌────────────┴────────────┐
             │                         │
           PASS                       FAIL
             │                         │
             │                  Correct Query
             │                         │
             │                       Retry
             │                         │
             │                 ┌───────┴───────┐
             │                 │               │
             │              Success       Max Retries
             │                 │               │
             └─────────────────┘               │
                                               ▼
                                         Safe Failure
             │
             ▼
             LLM
             │
             ▼
      OUTPUT GUARDRAILS
             │
             ▼
       FINAL ANSWER
```

---

# LangSmith Observability

The project uses **LangSmith** for tracing and observability.

The goal is to keep the complete request inside a single trace:

```text
RAG Pipeline
│
├── Input Guardrails
├── Query Router
│   └── Query Quality Evaluation
│   └── HyDE (when required)
│
├── CRAG
│   ├── Retrieval
│   ├── Reranking
│   ├── Relevance Evaluation
│   └── Correction / Retry
│
├── LLM Generation
│
└── Output Guardrails
```

The top-level trace is configured with:

```python
config = {
    "run_name": "RAG Pipeline"
}
```

CRAG retry cycles should remain visible inside the same request trace so retrieval corrections and evaluation decisions can be inspected.

---

# Technology Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Language             | Python                  |
| RAG Framework        | LangChain               |
| Vector Database      | Qdrant                  |
| Dense Embeddings     | `BAAI/bge-base-en-v1.5` |
| Sparse Retrieval     | `Qdrant/bm25`           |
| Retrieval            | Hybrid Dense + Sparse   |
| Fusion               | RRF                     |
| Reranking            | Cross-Encoder           |
| LLM                  | Groq                    |
| Query Improvement    | HyDE                    |
| Retrieval Correction | CRAG                    |
| Guardrails           | Guardrails AI           |
| Observability        | LangSmith               |
| Environment          | python-dotenv           |

---

# Project Structure

The project is organized into separate components so that retrieval, query processing, CRAG, and safety logic remain independent.

```text
medical-guideline-rag/
│
├── Documents/
│   └── *.pdf
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   │
│   ├── query/
│   │   ├── query_router.py
│   │   └── hyde.py
│   │
│   ├── crag/
│   │   ├── evaluator.py
│   │   ├── corrector.py
│   │   └── pipeline.py
│   │
│   ├── safety/
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   │
│   ├── prompts/
│   │   ├── rag_prompt.py
│   │   ├── hyde_prompt.py
│   │   └── crag_prompt.py
│   │
│   └── llm/
│       └── client.py
│
├── tests/
│
├── main.py
├── .env
├── requirements.txt
└── README.md
```

---

# Data Ingestion

The ingestion pipeline performs the following operations:

```text
Medical Guideline PDFs
        ↓
Document Loading
        ↓
Text Extraction
        ↓
Chunking
        ↓
Metadata Enrichment
        ↓
Dense Embeddings
        +
Sparse BM25 Representation
        ↓
Qdrant
```

Each chunk retains metadata required for traceability, including page information and a unique `chunk_id`.

---

# Retrieval Strategy

The retrieval system uses a multi-stage strategy:

```text
Query
  ↓
Dense Retrieval
  +
Sparse Retrieval
  ↓
RRF
  ↓
Candidate Documents
  ↓
Cross-Encoder Reranking
  ↓
CRAG Evaluation
  ↓
Accepted Context
```

This separates retrieval into distinct stages:

1. **Recall** — hybrid retrieval finds potentially relevant documents.
2. **Precision** — cross-encoder reranking improves ordering.
3. **Correction** — CRAG verifies whether the retrieved evidence is actually useful.

---

# Why HyDE + CRAG?

The two components solve different problems.

### HyDE

HyDE primarily improves the **retrieval query**.

```text
Weak / ambiguous query
        ↓
Hypothetical document
        ↓
Better retrieval representation
```

### CRAG

CRAG evaluates the **retrieved evidence**.

```text
Retrieved documents
        ↓
Are they actually relevant?
        ↓
YES → Continue
NO  → Correct + Retry
```

Therefore, they complement each other:

```text
HyDE
 ↓
Improve retrieval

Hybrid Retrieval
 ↓
Find candidates

Reranker
 ↓
Improve ordering

CRAG
 ↓
Verify and correct retrieval
```

---

# Safety Architecture

Safety is applied at multiple stages.

```text
              User Query
                  │
                  ▼
          ┌───────────────┐
          │ Input Safety  │
          └───────┬───────┘
                  │
                  ▼
             RAG Pipeline
                  │
                  ▼
          ┌───────────────┐
          │ Output Safety │
          └───────┬───────┘
                  │
                  ▼
            Final Answer
```

This prevents unsafe or unsupported requests from simply flowing through the retrieval and generation pipeline.

---

# Configuration

Environment variables are loaded using `python-dotenv`.

Example:

```env
GROQ_MODEL=openai/gpt-oss-120b

LANGSMITH_PROJECT=Medical-guidance-assistant-RAG-1

QDRANT_URL=http://localhost:6333

TOP_K_RESULTS_VALUE=4

HYDE_THRESHOLD=<configured-value>

CRAG_THRESHOLD=<configured-value>

MAX_RETRIES=2
```

Secrets such as API keys should not be committed to Git.

---

# Running the Project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Start Qdrant

Make sure Qdrant is running locally:

```text
http://localhost:6333
```

## 3. Add medical guidelines

Place the required guideline PDFs inside:

```text
Documents/
```

## 4. Index the documents

Run the ingestion/indexing process to create the Qdrant collection.

The collection contains the dense and sparse representations required for hybrid retrieval.

## 5. Start the application

```bash
python main.py
```

The application runs in a CLI loop and continues accepting questions until:

```text
exit
```

is entered.

---

# Example Request Flow

For a query such as:

```text
What are the recommended diagnostic criteria for malaria?
```

the system performs approximately:

```text
1. Input Guardrails
        ↓
2. Query Quality Evaluation
        ↓
3. Direct Retrieval OR HyDE
        ↓
4. Qdrant Hybrid Retrieval
        ↓
5. Cross-Encoder Reranking
        ↓
6. CRAG evaluates retrieved documents
        ↓
7. If insufficient:
       Correct query
       ↓
       Retrieve again
       ↓
       Rerank again
       ↓
       Re-evaluate
        ↓
8. Accepted context
        ↓
9. LLM generates grounded answer
        ↓
10. Output Guardrails
        ↓
11. Final response with guideline references
```

---

# Design Principles

The project is built around several principles:

### 1. Retrieval before generation

The LLM should rely on retrieved guideline evidence rather than attempting to answer purely from model knowledge.

### 2. Separate user intent from retrieval optimization

The original user query is preserved throughout the pipeline even when HyDE or CRAG modifies the retrieval query.

### 3. Multiple retrieval quality checks

Retrieval quality is improved through:

```text
Query Quality
     ↓
HyDE
     ↓
Hybrid Retrieval
     ↓
Reranking
     ↓
CRAG
```

### 4. Bounded self-correction

CRAG can retry retrieval, but retries are explicitly bounded to prevent infinite loops.

### 5. Traceability

Retrieved chunks retain identifiers and page metadata so generated answers can be traced back to source material.

### 6. Safety at both boundaries

Input guardrails prevent inappropriate requests from entering the pipeline, while output guardrails inspect generated responses before they reach the user.

---

# Current RAG Pipeline

The core architecture can be summarized as:

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Input Guardrails   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Query Router      │
                    └──────────┬───────────┘
                               ↓
                     ┌─────────┴─────────┐
                     │                   │
                  Direct                HyDE
                     │                   │
                     └─────────┬─────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Hybrid Qdrant Search │
                    │   Dense + Sparse     │
                    │       + RRF          │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Cross-Encoder        │
                    │ Reranking            │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │        CRAG          │
                    │ Relevance Evaluation │
                    └──────────┬───────────┘
                               ↓
                      ┌────────┴────────┐
                      │                 │
                    PASS              FAIL
                      │                 │
                      │          Correct + Retry
                      │                 │
                      │          Max retries?
                      │                 │
                      └────────┬────────┘
                               ↓
                    ┌──────────────────────┐
                    │     Prompt + LLM     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Output Guardrails   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Final Answer      │
                    └──────────────────────┘
```

---

# Future Improvements

Potential future improvements include:

* Small transformer-based query-quality classifier to replace LLM-based HyDE routing
* Better CRAG correction strategies
* More granular document/claim-level grounding evaluation
* Evaluation datasets for retrieval and answer quality
* Automated RAG evaluation
* Additional medical guideline sources
* Improved observability of individual CRAG retry cycles
* Retrieval and generation latency monitoring
* Automated regression testing for retrieval quality

---

## Disclaimer

This project is an experimental/educational implementation of a medical guideline RAG system.

It should **not** be used as a substitute for qualified medical professionals, clinical judgment, diagnosis, treatment decisions, or emergency services.

**This assistant provides information from official guidelines only and does not give personalized medical advice.**
