# Medical Guideline RAG

A production-oriented **Retrieval-Augmented Generation (RAG)** system for querying medical guidelines and returning answers grounded in the retrieved source documents.

The project focuses on building a reliable and safety-conscious RAG pipeline rather than simply connecting an LLM to a vector database.

## ⚠️ Medical Disclaimer

This project is an **experimental / educational AI system** and is **not a medical device, diagnostic system, or substitute for professional medical advice**.

The generated responses should not be used to diagnose, treat, or make clinical decisions about a patient.

Always consult a qualified healthcare professional and the original medical guidelines for clinical decisions.

---

## 🎯 Project Goals

The goal of this project is to explore how a production-oriented RAG system can improve retrieval quality, answer grounding, observability, and safety when working with medical guideline documents.

The system focuses on:

* Hybrid information retrieval
* Query routing
* HyDE-based retrieval for difficult queries
* Cross-encoder reranking
* Input safety guardrails
* Output validation
* Citation / source grounding
* Observability with LangSmith
* Modular RAG architecture

---

## 🏗️ Architecture

The high-level pipeline looks like this:

```text
                         User Query
                              │
                              ▼
                    ┌──────────────────┐
                    │  Input Guardrails│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Query Router   │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
          Direct Retrieval            HyDE
                  │                     │
                  │              Generate Hypothetical
                  │                  Document
                  │                     │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Hybrid Retrieval   │
                  │                     │
                  │ Dense + Sparse      │
                  │      +              │
                  │       RRF           │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Cross-Encoder       │
                  │     Reranking       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Context Construction│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │        LLM          │
                  │  Grounded Answer    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Output Guardrails   │
                  └──────────┬──────────┘
                             │
                             ▼
                         Final Answer
```

---

## 🔍 Retrieval Pipeline

### 1. Dense Retrieval

The system uses semantic embeddings to retrieve documents based on the meaning of the query.

Current embedding model:

```text
BAAI/bge-base-en-v1.5
```

Embeddings are normalized before being stored and queried.

---

### 2. Sparse Retrieval

The system also uses sparse retrieval to capture exact terminology, keywords, medical terms, and other lexical matches.

Current sparse model:

```text
Qdrant/bm25
```

---

### 3. Hybrid Retrieval

Dense and sparse retrieval are combined using Qdrant's hybrid retrieval capabilities.

Conceptually:

```text
User Query
    │
    ├── Dense Search ──► Semantic Matches
    │
    └── Sparse Search ─► Keyword Matches
                          │
                          ▼
                         RRF
                          │
                          ▼
                  Combined Candidates
```

This allows the system to benefit from both:

* Semantic similarity
* Exact lexical matching

---

## 🔄 Query Routing

Not every query benefits equally from HyDE.

The system therefore evaluates the query before retrieval.

```text
Query
  │
  ▼
Query Quality Scoring
  │
  ├── High score ──► Direct Retrieval
  │
  └── Low score ───► HyDE
```

The query-quality scoring component currently uses a model-based approach and is designed to be replaceable with a smaller dedicated transformer model in the future.

The intent is:

> Determine whether the original query is already suitable for retrieval or whether generating a hypothetical answer/document could produce a better retrieval representation.

---

## 🧠 HyDE

For queries that are considered difficult for direct retrieval, the system uses **Hypothetical Document Embeddings (HyDE)**.

Instead of embedding only the original query:

```text
"What are the criteria for starting treatment?"
```

the system generates a hypothetical document representing the type of content that would ideally answer the query.

The generated hypothetical document is then used as the retrieval representation.

Conceptually:

```text
User Query
    │
    ▼
Hypothetical Answer / Document
    │
    ▼
Embedding
    │
    ▼
Hybrid Retrieval
```

HyDE is used selectively rather than for every query.

---

## 🎯 Cross-Encoder Reranking

Initial retrieval is optimized for recall.

The resulting candidate documents are then passed through a cross-encoder reranker to improve the ordering of the retrieved context.

```text
Hybrid Retrieval
      │
      ▼
Top-N Candidates
      │
      ▼
Cross Encoder
      │
      ▼
Re-ranked Documents
      │
      ▼
Top-K Context
```

This separates the retrieval and ranking responsibilities:

* **Retriever:** Find potentially relevant documents.
* **Reranker:** Determine which retrieved documents are most relevant to the specific query.

---

## 🛡️ Safety & Guardrails

Because the system operates on medical information, safety is treated as a first-class component.

### Input Guardrails

The input layer evaluates the user's query before retrieval.

The project includes checks for:

* Out-of-scope medical requests
* Requests requiring personalized medical advice
* Emergency / crisis-related requests
* Prompt injection attempts

Unsafe or unsupported queries can be short-circuited before they reach the retrieval pipeline.

```text
User Query
    │
    ▼
Input Guardrails
    │
    ├── Unsafe / Out of Scope
    │          │
    │          ▼
    │       Refusal
    │
    └── Safe
         │
         ▼
      Retrieval
```

### Output Guardrails

The generated response is also validated before being returned.

The output validation layer is intended to ensure that responses:

* Remain grounded in retrieved context
* Contain valid source references
* Do not introduce unsupported claims
* Respect the application's medical-safety constraints

---

## 📚 Document Processing

Medical guideline documents are processed before being added to the retrieval system.

The general indexing pipeline is:

```text
Medical Guidelines
       │
       ▼
Document Loading
       │
       ▼
Chunking
       │
       ▼
Metadata Enrichment
       │
       ▼
Dense Embeddings
       │
       ▼
Sparse Representation
       │
       ▼
Qdrant
```

Each chunk retains metadata such as its source document and page information so that retrieved information can be traced back to the original guideline.

---

## 🗄️ Vector Database

The project uses **Qdrant** as the vector database.

Qdrant stores the representations required for hybrid retrieval:

```text
Qdrant
 ├── Dense vectors
 ├── Sparse vectors
 └── Document metadata
```

This allows the application to perform dense and sparse retrieval without maintaining a separate application-level BM25 index.

---

## 🔬 Observability

The RAG pipeline is instrumented using **LangSmith**.

The objective is to maintain a clean trace for each user query.

A typical trace represents the complete flow:

```text
RAG Pipeline
    │
    ├── Input Guardrails
    ├── Query Router
    ├── HyDE (when required)
    ├── Retrieval
    ├── Reranking
    ├── Prompt
    ├── LLM
    └── Output Guardrails
```

This makes it possible to investigate:

* Which retrieval path was selected
* Whether HyDE was triggered
* Retrieved documents
* Reranking results
* LLM input/output
* Guardrail decisions
* Overall latency

---

## 🧰 Technology Stack

| Component              | Technology                 |
| ---------------------- | -------------------------- |
| Language               | Python                     |
| RAG Framework          | LangChain                  |
| Vector Database        | Qdrant                     |
| Dense Embeddings       | BAAI/bge-base-en-v1.5      |
| Sparse Retrieval       | Qdrant BM25                |
| LLM                    | Groq                       |
| Reranking              | Cross-Encoder              |
| Observability          | LangSmith                  |
| Environment Management | python-dotenv              |
| Local Development      | Python virtual environment |

---

## 📁 Project Structure

The project is intentionally split into separate components so that individual RAG stages can be developed and tested independently.

```text
medical-guideline-rag/
│
├── Documents/
│   └── medical_guidelines.pdf
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── chunking.py
│   │   └── indexing.py
│   │
│   ├── retrieval/
│   │   ├── qdrant_vectordb_client.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── query/
│   │   ├── query_router.py
│   │   ├── hyde.py
│   │   └── query_quality.py
│   │
│   ├── guards/
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   │
│   ├── llm/
│   │   └── groq_client.py
│   │
│   └── prompts/
│       └── prompts.py
│
├── tests/
│
├── .env.example
├── requirements.txt
├── README.md
└── main.py
```

> The exact structure may evolve as the project develops.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <repository-url>

cd medical-guideline-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=
```

Add any additional provider-specific variables required by the application.

**Never commit `.env` to Git.**

---

## 🐳 Running Qdrant

Run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant should then be available at:

```text
http://localhost:6333
```

---

## 📥 Indexing Documents

Place the medical guideline documents inside:

```text
Documents/
```

Run the indexing pipeline:

```bash
python <indexing-script>
```

The indexing process will:

1. Load the documents
2. Split them into chunks
3. Generate dense embeddings
4. Generate sparse representations
5. Store the vectors and metadata in Qdrant

---

## ▶️ Running the Application

Start the application:

```bash
python main.py
```

Example:

```text
Enter your question:

What are the recommended criteria for initiating treatment?

Answer:
...
```

The final response should be grounded in the retrieved medical guideline content and provide source/page information where available.

---

## 🧪 Example Query Flow

Example query:

```text
What are the recommended criteria for initiating treatment?
```

The application may process it as:

```text
User Query
    │
    ▼
Input Guardrails
    │
    ▼
Query Quality Score
    │
    ▼
Direct Retrieval / HyDE
    │
    ▼
Qdrant Hybrid Retrieval
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Relevant Guideline Context
    │
    ▼
LLM
    │
    ▼
Output Validation
    │
    ▼
Grounded Response + Sources
```

---

## 📈 Design Principles

### 1. Retrieval before generation

The LLM should not be treated as the source of truth.

The system retrieves relevant guideline content first and uses that content as the basis for generation.

### 2. Recall first, precision second

The retrieval pipeline is intentionally separated into two stages:

```text
Hybrid Retrieval → Recall
Reranking        → Precision
```

### 3. Query-adaptive retrieval

HyDE is not automatically applied to every query.

The query router determines whether the query may benefit from hypothetical-document retrieval.

### 4. Safety before retrieval

Potentially unsafe or unsupported requests should be handled before they enter the retrieval and generation pipeline.

### 5. Observability by default

Important RAG decisions should be observable through LangSmith rather than hidden inside application code.

---

## 🚧 Current Status

This project is under active development.

Current components include:

* [x] Medical guideline document ingestion
* [x] Qdrant vector database
* [x] Dense retrieval
* [x] Sparse retrieval
* [x] Hybrid retrieval
* [x] RRF-based result fusion
* [x] Cross-encoder reranking
* [x] Query routing
* [x] Model-based query quality scoring
* [x] HyDE retrieval path
* [x] Input guardrails
* [x] Output guardrails
* [x] LangSmith tracing
* [ ] Evaluation dataset
* [ ] Automated retrieval evaluation
* [ ] Automated end-to-end RAG evaluation
* [ ] Production deployment

---

## 🔮 Future Improvements

Potential improvements include:

* Dedicated lightweight query-routing model
* Retrieval evaluation using Recall@K / MRR / NDCG
* RAGAS or equivalent end-to-end evaluation
* Better citation verification
* Query rewriting
* Multi-query retrieval
* Parent-document retrieval
* Context compression
* Automated regression testing for RAG quality
* Latency and cost optimization
* Production deployment
* Continuous evaluation using a medical guideline benchmark

---

## ⚠️ Limitations

This system has several important limitations:

* Retrieval quality depends on the quality and coverage of the indexed guidelines.
* The LLM can still generate unsupported information if retrieval or grounding fails.
* Medical terminology can be ambiguous.
* Guidelines may change over time.
* A retrieved guideline may not represent the most recent clinical recommendation.
* The system does not replace clinical judgment or professional medical advice.

For any real clinical use case, the system would require substantially stronger validation, governance, security, monitoring, and regulatory review.

---

## 📄 License

Add the project's license here.

---

## 👤 Author

Built as a learning and engineering project focused on exploring **production-oriented RAG systems, retrieval optimization, LLM safety, and GenAI engineering**.
