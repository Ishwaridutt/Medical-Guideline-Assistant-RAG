import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from qdrant_vectordb_client import hybrid_retriever
from groq_client import llm, output_parser
from medical_assistant_prompt import prompt_template
from reranker import rerank_documents
from safety.input_guard import input_guard

def format_docs(docs: list[Document]) -> str:
  return "\n\n\n".join([
      f"Page Content: {doc.page_content}\n"
      f"Page Number: {doc.metadata.get('page_label', 'N/A')}\n"
      f"Chunk Id: {doc.metadata.get('chunk_id', 'N/A')}\n"
      f"File Location: {doc.metadata.get('source', 'N/A')}"
      for doc in docs
  ])

def retrieve_and_rerank(query: str) -> list[Document]:
    # Step 1: Retrieve candidate documents from Qdrant
    retrieved_docs = hybrid_retriever.invoke(query)

    # Step 2: Rerank retrieved documents using Cross Encoder
    reranked_docs = rerank_documents(
        query = query,
        documents = retrieved_docs,
        top_k = 5
    )
    return reranked_docs


# Give the complete retrieval + reranking step a meaningful name
retrieval_and_reranking = RunnableLambda(
    retrieve_and_rerank
).with_config(
    run_name = "Hybrid Retrieval + Cross Encoder Reranking"
)

rag_chain = (
    {
        "context": retrieval_and_reranking | format_docs,
        "user_question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | output_parser
)

while True:

    # User Query
    user_query = input("\nEnter your query (type 'exit' to quit): ").strip()
     # Exit condition
    if user_query.lower() == "exit":
        print("\nExiting RAG application. Goodbye!")
        break
    # input guardrails
    try:
        input_guard.validate(user_query)
        True
    except Exception as e:
        error_message = str(e)
        if "PERSONALIZED_MEDICAL_ADVICE" in error_message:
            print(
                "\nI can provide general medical information "
                "from official guidelines, but I cannot provide "
                "personalized medical advice."
            )
        elif "EMERGENCY_OR_CRISIS" in error_message:
            print(
                "\nThis assistant cannot assess or manage "
                "medical emergencies. Please seek immediate "
                "professional medical care or contact your "
                "local emergency services."
            )
        elif "NON_MEDICAL_INFORMATION" in error_message:
            print(
                "\nThis assistant cannot answer non medical queries. "
                "Please enter any other medical query you may have"
            )
        else:
            print(
                "\nThe query could not be safely processed."
            )
        # Do NOT run retrieval/RAG for a rejected query.
        continue
    # run pipeline
    try:
        llm_response = rag_chain.invoke(
        user_query,
        config={
                "run_name": "RAG Pipeline"
            }
        )
        print(f'Retrieved Result: {llm_response}')
    except Exception as e:
        print(
            f"\nAn error occurred while processing your query: {e}"
        )




