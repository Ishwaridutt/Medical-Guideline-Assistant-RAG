# crag/evaluator.py

import json
from langchain_core.documents import Document
from config.config import (
    CRAG_DOCUMENT_RELEVANCE_THRESHOLD, 
    CRAG_MIN_RELEVANCE_RATIO, 
    CRAG_MAX_RETRIES, 
    SAFE_CRAG_FAILURE_RESPONSE
)
from prompts.crag_prompts import QUERY_REWRITE_PROMPT, DOCUMENT_RELVENCE_EVALUATOR_PROMPT
from clients.groq_client import llm_client

# Evaluate how relevant a document is to the user query.
def get_document_relevance_score(query, doc):
    prompt = DOCUMENT_RELVENCE_EVALUATOR_PROMPT.format(
        user_query = query,
        fetched_document = doc.page_content
    )
    response = llm_client.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    response = response.strip()

    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        print('Failed to parse the JSON for LLM response: ', response)
        return 0
    # get score and protect against malformed model output.
    score = max(0.0, min(1.0, response.get("score", 0.0)))
    return score

    
# rewrites the input query of the user for to be used again in the CRAG pipeline
def rewrite_query(query) -> str:
    prompt = QUERY_REWRITE_PROMPT.format(input_query = query)
    response = llm_client.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    updated_query = response.strip()
    if not updated_query:
        print(
            "CRAG rewrite query rewrite returned an empty query. "
            "Keeping the original query."
        )
        return query
    return updated_query


# evaluates the list of documents and discard the irrelevent docs
def evaluate_documents(
    query: str,
    documents: list[Document],
) -> list[Document]:
    relevant_documents = []
    # iterate all docs to score each of them and discard which does suit our need
    for index, document in enumerate(documents):
        doc_score = get_document_relevance_score(
            query = query,
            doc = document,
        )
        # this chunk is relevant hence append it for the final context
        print(f"\nCRAG evaluate doc no: {index} score is {doc_score} and threshold is: {CRAG_DOCUMENT_RELEVANCE_THRESHOLD}")
        if doc_score >= CRAG_DOCUMENT_RELEVANCE_THRESHOLD:
            relevant_documents.append(document)
    return relevant_documents


def classify_relevent_document_quality(
    relevant_count: int,
    retrieved_count: int
) -> str:
    if relevant_count == 0 or retrieved_count == 0:
        return "incorrect"
    # compute the ratio
    relevance_ratio = relevant_count / retrieved_count
    print(f"\nCRAG relevant ratio is {relevance_ratio} and min required ratio is: {CRAG_MIN_RELEVANCE_RATIO}")
    if relevance_ratio > CRAG_MIN_RELEVANCE_RATIO:
        return "correct"
    return "ambiguous"



def evaluate_fetched_doc_using_crag(
    original_query,
    retrieval_query,
    documents,
    retriever_fn,
):
    retry_count = -1 # -1 so we do not count the first attempt in our retires
    while retry_count < CRAG_MAX_RETRIES:
        print(f"\nCRAG Attempt {retry_count + 1}")
        # evalaute the docs and check if they pass the relevant ratio
        # CRAG will evaluate doc against original user query not HyDE generated query
        relevant_documents = evaluate_documents(query = original_query, documents = documents)
        relevant_quality = classify_relevent_document_quality(
            relevant_count = len(relevant_documents),
            retrieved_count = len(documents)
            )
        if(relevant_quality == "correct"):
            return {
                "status": "CRAG_SUCCESS",
                "documents": relevant_documents,
                "query": retrieval_query,
                "retry_count": retry_count,
                "message": None
            }
        
        # since they did not pass the ratio we need to refetch doc from DB and try again
        # Rewrite query, re fetch from db and retry
        print(f'\nCRAG relevant_quality is: {relevant_quality}. ' 
              'Since required relevance ratio is not met CRAG will regenrate query and doc...')
        # CRAG will rewrite the retrieval(HyDE) query not the original query since it was first used to fetch result
        updated_query = rewrite_query(query = retrieval_query)
        updated_doc = retriever_fn.invoke(updated_query)
        print(f"\nCRAG retrying with the updated query: {updated_query} and old query was: {retrieval_query} ", 
              f"and updated docs of total len: {len(documents)}"
            )
        # reassign the values for looping
        retrieval_query = updated_query
        documents = updated_doc
        retry_count += 1
        

    # CRAG failed to improve the query and fetch relevant documents
    print("\nCRAG failed to improve the qualtiy of reltrieval")
    return {
        "status": "CRAG_FAILED",
        "documents": [],
        "query": retrieval_query,
        "retry_count": retry_count,
        "message": SAFE_CRAG_FAILURE_RESPONSE,
    }

        
