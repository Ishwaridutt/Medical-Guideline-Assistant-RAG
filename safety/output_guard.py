# from citation_guardrail import verify_citations
from safety.output_intent_classification_guard import apply_safety_guardrail
from groq_client import llm

def format_output(response: str) -> str:
    response = response.strip()
    # return f"{response}\n\n{DISCLAIMER}"
    return f"{response}\n"

def run_output_guardrails(
    response: str,
    # retrieved_documents,
) -> str:
    # citation verification
    # citations_valid, invalid_chunk_ids = verify_citations(
    #     response,
    #     retrieved_documents
    # )

    # if not citations_valid:
    #     return format_output(
    #         "I am unable to provide a reliable answer because "
    #         "the generated response contains citations that could "
    #         "not be verified against the retrieved guidelines."
    #     )

    # 2. output intent safety classification
    safe_response = apply_safety_guardrail(
        response,
        llm
    )
    return safe_response["response"]
 