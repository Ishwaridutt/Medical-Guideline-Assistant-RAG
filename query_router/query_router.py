import json
from langchain_core.runnables import RunnableLambda
from rag.hyde import generate_hyde
from config.config import HYDE_TRIGGER_THRESHOLD
from clients.groq_client import llm_client
from prompts.query_router_prompt import get_query_router_hyde_check_prompt


def get_query_quality_score(query: str) -> float:
    """
    Returns a retrieval-suitability score between 0 and 1.
    Higher score: Query is suitable for direct retrieval.
    Lower score: Query may benefit from HyDE.
    This will be replaced by the small transformer model.
    """
    hyde_check_prompt = get_query_router_hyde_check_prompt(query)
    response = llm_client.invoke(hyde_check_prompt)
    # verify and parse the response
    try:
        if hasattr(response, "content"):
            response = response.content

        print('\nQuery quality analyzer model response: ', response)
        result = json.loads(response)

    except json.JSONDecodeError:
        print('Failed to read the llm response for the HyDE score')
        return 0.0

    score = result.get("query_score")
    if not isinstance(score, (int, float)):
        print(
            "LLM returned an invalid HyDE score. "
            f"Result: {result}"
        )
        return 0.0
    
    # Protect the router from malformed LLM output.
    score = max(0.0, min(1.0, float(score)))
    return score


def route_query(query: str) -> dict:

    score = get_query_quality_score(query)

    if score >= HYDE_TRIGGER_THRESHOLD:
        # query may have semantic gap, generate HyDE doc
        print(
            f"\nHyDE is required for user query: {query} "
            f"which has score of: {score} above the specified threshold: {HYDE_TRIGGER_THRESHOLD}")
        retrieval_query = generate_hyde(query)
        print("\nGenerated HyDE query: ", retrieval_query)
        hyde_used = True
    else:
        print("HyDE is not required as user query score is:", score)
        retrieval_query = query
        hyde_used = False
    # in case hyde is not used, we pass back the original query in the retrieval query
    return {
        "original_query": query,
        "retrieval_query": retrieval_query,
        "hyde_used": hyde_used,
        "query_quality_score": score,
    }


query_router = RunnableLambda(
    route_query
).with_config(
    run_name = "Query Router"
)

