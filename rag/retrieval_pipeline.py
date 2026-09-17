from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from clients.qdrant_vectordb_client import hybrid_retriever
from clients.groq_client import llm, output_parser
from prompts.medical_assistant_prompt import prompt_template
from rag.reranker import rerank_documents
from utils.utils import format_docs
from query_router.query_router import query_router
from operator import itemgetter

def retrieve_and_rerank(query: str) -> list[Document]:
    # Step 1: Retrieve candidate documents from Qdrant
    retrieved_docs = hybrid_retriever.invoke(query)

    # Step 2: Rerank retrieved documents using Cross Encoder
    reranked_docs = rerank_documents(
        query = query,
        documents = retrieved_docs,
        top_k = 5
    )
    print('\nDocuments retrieved from the DB and reranked successfully')
    return reranked_docs

# Give the complete retrieval + reranking step a meaningful name
retrieval_and_reranking = RunnableLambda(
    retrieve_and_rerank
).with_config(
    run_name = "Hybrid Retrieval + Cross Encoder Reranking"
)

# check the relevance of the fetched document against the query using CRAG
# TODO

rag_chain = (
    query_router
    | {
        "context": itemgetter("retrieval_query") | retrieval_and_reranking | format_docs,
        "user_question": itemgetter("user_question"),
        # "user_question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | output_parser
)






