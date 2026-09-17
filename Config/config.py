
DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_EMBEDDING_MODEL_NAME = "Qdrant/bm25"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"
COLLECTION_NAME = "rag_hybrid_collection_1"
QDRANT_URL = "http://localhost:6333"
HYDE_TRIGGER_THRESHOLD = 0.60
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
# smaller model with 8B params
GROQ_LLAMA_MODEL_NAME = "llama-3.1-8b-instant"
CRAG_DOCUMENT_RELEVANCE_THRESHOLD = 0.6
CRAG_MIN_RELEVANCE_RATIO = 0.45
CRAG_MAX_RETRIES = 1
SAFE_CRAG_FAILURE_RESPONSE = (
    "\nI couldn't find sufficiently relevant information "
    "in the available medical guidelines to answer this question."
)