from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from clients.qdrant_vectordb_client import hybrid_retriever
from rag.reranker import rerank_documents

# this does the hybrid search in the DB and uses the cross encoder to re-rank the fecthed docs
def retrieve_and_rerank(query: str) -> list[Document]:
    # Step 1: Retrieve candidate documents from Qdrant
    retrieved_docs = hybrid_retriever.invoke(query)

    # Step 2: Rerank retrieved documents using Cross Encoder
    reranked_docs = rerank_documents(
        query = query,
        documents = retrieved_docs,
        top_k = 4
    )
    print('\nDocuments retrieved from the DB and reranked successfully')
    return reranked_docs

# Give the complete retrieval + reranking step a meaningful name
retrieval_and_reranking = RunnableLambda(
    retrieve_and_rerank
).with_config(
    run_name = "Hybrid Retrieval + Cross Encoder Reranking"
)

