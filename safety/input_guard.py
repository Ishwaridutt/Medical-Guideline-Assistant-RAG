from guardrails import Guard

from safety.validators.medical_intent import MedicalIntentValidator

input_guard = Guard().use(
    MedicalIntentValidator(on_fail = "exception")
)

