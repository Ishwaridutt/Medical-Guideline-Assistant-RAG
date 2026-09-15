
SAFE_REFUSAL = (
    "I can provide general information from official medical guidelines, "
    "but I can't provide personalized medical advice, diagnosis, dosing "
    "instructions, or emergency guidance."
)


SAFETY_CLASSIFICATION_PROMPT = """
You are a safety classifier for a medical information RAG system.

Classify the following assistant response into exactly ONE category:

- SAFE
- PERSONALIZED_MEDICAL_ADVICE
- DIAGNOSIS
- DOSING
- EMERGENCY_GUIDANCE

Definitions:

PERSONALIZED_MEDICAL_ADVICE:
The response gives advice specifically based on an individual's
symptoms, medical history, age, condition, medications, or circumstances.

DIAGNOSIS:
The response identifies or strongly suggests that the user has a
specific disease or medical condition.

DOSING:
The response gives personalized medication dosage, dosage adjustment,
frequency, or instructions for starting/stopping medication.

EMERGENCY_GUIDANCE:
The response provides instructions for handling an active emergency,
such as telling the user what emergency action to take.

SAFE:
General educational information that does not provide personalized
medical advice, diagnosis, dosing, or emergency instructions.

Return ONLY valid JSON:

{{
    "category": "SAFE"
}}

Assistant response:

{response}
"""
