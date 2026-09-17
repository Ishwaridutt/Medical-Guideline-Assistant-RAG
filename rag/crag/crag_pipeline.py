# crag/pipeline.py

from langchain_core.documents import Document

from rag.crag.crag_evaluator import (
    evaluate_fetched_doc_using_crag,
)
from rag.rag_retrieval import retrieval_and_reranking

def crag_retrieval(
    original_query: str,
    retrieval_query: str,
) -> list[Document]:
    """
    Retrieve, rerank and evaluate documents using CRAG.
    Returns only documents that passed CRAG.
    """
    # Initial retrieval
    # documents = retrieval_and_reranking(retrieval_query)
    documents = retrieval_and_reranking.invoke(retrieval_query)

    # CRAG evaluation + retry loop
    crag_result = evaluate_fetched_doc_using_crag(
        original_query = original_query,
        retrieval_query = retrieval_query,
        documents = documents,
        retriever_fn = retrieval_and_reranking,
    )

    # re-attach the original user query
    crag_result["original_query"] = original_query
    print(f"\nCRAG retrieval result: {crag_result["status"]} ")
    return crag_result

