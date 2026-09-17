from guardrails import Guard

from safety.validators.medical_intent import MedicalIntentValidator
from guardrails_ai.prompt_injection_detector import PromptInjectionDetector

# also add prompt injection guard rails
input_guard = Guard().use(
    MedicalIntentValidator(on_fail = "exception"),
    # PromptInjectionDetector(on_fail="exception") requires openai key to work
)


def handle_input_guardrail(user_query: str) -> bool:
    # input guardrails
    try:
        input_guard.validate(user_query)
        return True
    except Exception as err:
        error_message = str(err)
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
                "\nThe input query could not be safely processed.", error_message
            )
        # Do NOT run retrieval/RAG for a rejected query.
        return False


