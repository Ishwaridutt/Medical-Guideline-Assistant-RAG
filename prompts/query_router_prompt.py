from config.config import HYDE_TRIGGER_THRESHOLD


def get_query_router_hyde_check_prompt(query) -> str:
    return f"""
    You are a query router in a RAG system.
    Evaluate whether the user query requires Hypothetical Document Embeddings (HyDE) expansion.

    Score the query from 0.0 to 1.0 based on how strongly it needs HyDE:

    HIGH SCORE (score >= {HYDE_TRIGGER_THRESHOLD}) -> HyDE REQUIRED:
    - Query is abstract, conceptual, or explanatory (e.g., "how", "why", "best practices").
    - High semantic vocabulary mismatch between query and target documents.
    - Query is vague or short without specific entities.

    LOW SCORE (score < {HYDE_TRIGGER_THRESHOLD}) -> HyDE NOT REQUIRED:
    - Query contains specific entities, error codes, IDs, exact technical terms, or code snippets.
    - Direct keyword/dense search is unambiguous and factual.

    Return ONLY a valid JSON object matching: {{"query_score": float between 0.0 and 1.0, "reason": "short explanation"}}

    user query:
    {query}
    """


