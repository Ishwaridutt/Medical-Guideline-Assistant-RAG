import json
import re
from guardrails.validators import register_validator, Validator, PassResult, FailResult
from clients.groq_client import llm
from prompts.medical_intent_prompt import get_medical_intent_prompt, get_medical_intent_types

@register_validator(
    name="medical_intent_validator",
    data_type="string"
)
class MedicalIntentValidator(Validator):

    def __init__(self, on_fail="exception"):
        super().__init__(on_fail=on_fail)

    def validate(self, user_query, metadata=None):

        prompt = get_medical_intent_prompt(user_query)
        # call the intent classifier llm model
        print('in medical intent classifier')
        response = llm.invoke(prompt)
        print('in medical intent classifier 2', response)

        # LangChain AIMessage -> string
        if hasattr(response, "content"):
            response = response.content
        
        response = response.strip()
        # Remove markdown code fences if the model returns them
        response = re.sub(
            r"```(?:json)?\s*|\s*```",
            "",
            response,
            flags = re.MULTILINE
        ).strip()

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            return FailResult(
                error_message = (
                    "Medical intent classifier returned invalid JSON. "
                    "Failing closed for safety."
                )
            )
        # user query intent
        intent = result.get("intent")
        print('medical intent:', intent)
        # check if the intent returned is defined inside the allowed intent
        allowed_intents = get_medical_intent_types()
        if intent not in allowed_intents:
            return FailResult(
                error_message=(
                    f"Medical intent classifier returned an "
                    f"unknown intent: {intent}"
                )
            )

        if intent == "GENERAL_MEDICAL_INFORMATION":
            return PassResult()
        # invalid user query cases
        if intent == "PERSONALIZED_MEDICAL_ADVICE":
            return FailResult(
                error_message="PERSONALIZED_MEDICAL_ADVICE"
            )
        if intent == "EMERGENCY_OR_CRISIS":
            return FailResult(
                error_message="EMERGENCY_OR_CRISIS"
            )
        if intent == "NON_MEDICAL_INFORMATION":
            return FailResult(
                error_message="NON_MEDICAL_INFORMATION"
            )