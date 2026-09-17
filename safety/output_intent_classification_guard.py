import json
from prompts.safety_classification_prompt import SAFE_REFUSAL, SAFETY_CLASSIFICATION_PROMPT


def classify_output(response: str, llm) -> str:
    try:
        prompt = SAFETY_CLASSIFICATION_PROMPT.format(response = response)
        # send response to the classifier model
        result = llm.invoke(prompt)        
        classification = json.loads(result.content)
        return classification["category"]
    except Exception as e:
        # Fail closed.
        print('Error in output intent classifier', e)
        return "PERSONALIZED_MEDICAL_ADVICE"

def apply_safety_guardrail(response: str, llm) -> str:
    category = classify_output(response, llm)
    if category != "SAFE":
          return {
            "is_safe": False,
            "response": SAFE_REFUSAL,
            "category": category
        }
    # this passes the guardrail, return the same response
    return {
        "is_safe": True,
        "response": response,
        "category": "SAFE"
    }

