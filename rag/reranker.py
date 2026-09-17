from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config.config import CROSS_ENCODER_MODEL_NAME

# Load Cross Encoder model
cross_encoder_reranker_model = CrossEncoder(
    CROSS_ENCODER_MODEL_NAME,
    device = "mps"
)

def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 4
) -> list[Document]:

    if not documents:
        return []

    # Create query-document pairs
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    # Get relevance scores
    scores = cross_encoder_reranker_model.predict(pairs)

    # Combine documents with their scores
    scored_documents = list(zip(documents, scores))

    # Sort by relevance score - highest first
    scored_documents.sort(
        key = lambda x: x[1],
        reverse = True
    )

    # Return top K documents
    reranked_documents = []
    for doc, score in scored_documents[:top_k]:
        doc.metadata["reranker_score"] = float(score)
        reranked_documents.append(doc)

    return reranked_documents

