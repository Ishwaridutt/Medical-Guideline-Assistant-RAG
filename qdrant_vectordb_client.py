from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from Config.config import SPARSE_EMBEDDING_MODEL_NAME, QDRANT_URL, DENSE_EMBEDDING_MODEL_NAME, COLLECTION_NAME

TOP_K_RESULTS_VALUE = 10

# Vector Embeddings
dense_embedding_model = HuggingFaceEmbeddings(
    model_name = DENSE_EMBEDDING_MODEL_NAME,
    model_kwargs = {
        "device": "mps"
    },
    encode_kwargs = {
        "normalize_embeddings": True
    }
)

sparse_embedding_model = FastEmbedSparse(
    model_name = SPARSE_EMBEDDING_MODEL_NAME
)

# start connection with Vector Database
vector_db = QdrantVectorStore.from_existing_collection(
    url = QDRANT_URL,
    collection_name = COLLECTION_NAME,
    embedding = dense_embedding_model,
     sparse_embedding = sparse_embedding_model,
     # Qdrant performs hybrid retrieval + RRF
    retrieval_mode =  RetrievalMode.HYBRID,
)

# Create retriever
hybrid_retriever = vector_db.as_retriever(
    search_kwargs={
        "k": TOP_K_RESULTS_VALUE
        }
    ).with_config(
    {
        "run_name": "Hybrid Retrieval"
    }
)

