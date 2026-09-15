# Import of the required libraries
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from Config.config import DENSE_EMBEDDING_MODEL_NAME, SPARSE_EMBEDDING_MODEL_NAME, COLLECTION_NAME, QDRANT_URL
from pathlib import Path

# To load the .env file
# load_dotenv()

# File path
# PDF_FOLDER_PATH = Path(__file__).parent/"/Documents"
# PDF_FOLDER_PATH = "./Documents/MoHFW Official Medical Documentation"

# configs
PDF_FOLDER_PATH = "./Documents"

# Load the document
print('Loading PDF files...')
# loader = PyPDFLoader(file_path=PDF_FOLDER_PATH)
loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# print("The first document is: ", docs[0])

# Semantic Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(documents=docs)
print(f"Doc splitted into {len(split_docs)} chunks")

# Add a stable chunk ID to every chunk
for chunk_index, doc in enumerate(split_docs):
    source = doc.metadata.get("source", "unknown")
    doc.metadata["chunk_id"] = f"{source}::chunk_{chunk_index}"

print("Chunk IDs added")

# generate vector Embeddings
# EMBEDDING_MODEL = "text-embedding-3-large"
# embedding_model = OpenAIEmbeddings(
#     model = EMBEDDING_MODEL
# )

# Load embedding model for Dense embeddings
# using 768 embedding model
print("Loading Dense Embedding model...")
dense_embedding_model = HuggingFaceEmbeddings(
    model_name = DENSE_EMBEDDING_MODEL_NAME,
    model_kwargs = {
        "device": "mps"
    },
    encode_kwargs = {
        "normalize_embeddings": True
    }
)
print("Dense embedding model loaded")

# Sparse embedding model
print("Loading sparse BM25 embedding model...")
sparse_embedding_model = FastEmbedSparse(
    model_name = SPARSE_EMBEDDING_MODEL_NAME
)
print("Sparse BM25 model loaded")

# Embeddings storage inside QdrantVectorDB
try:
    vector_store = QdrantVectorStore.from_documents(
        documents = split_docs,
        url = QDRANT_URL,
        collection_name = COLLECTION_NAME,
        embedding = dense_embedding_model,
        sparse_embedding = sparse_embedding_model,
        retrieval_mode = RetrievalMode.HYBRID,
    )
    print(f"Created collection {COLLECTION_NAME}")
except Exception as e:
    print(f"Error creating the vector store {e}")


print("Indexing of the provided document is done...")

