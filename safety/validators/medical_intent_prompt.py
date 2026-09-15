
def get_medical_intent_types():
    allowed_intent = {
        "GENERAL_MEDICAL_INFORMATION",
        "PERSONALIZED_MEDICAL_ADVICE",
        "EMERGENCY_OR_CRISIS",
        "NON_MEDICAL_INFORMATION"
    }
    return allowed_intent

def get_medical_intent_prompt(user_query: str) -> str:
    return f"""
You are a medical query intent classifier.

Classify the user's query into exactly ONE of these categories:

1. GENERAL_MEDICAL_INFORMATION
   The user is asking for general educational or factual
   medical information.

2. PERSONALIZED_MEDICAL_ADVICE
   The user is asking what they personally should do,
   take, stop, start, change, or choose based on their
   personal medical situation.

3. EMERGENCY_OR_CRISIS
   The user describes a potentially urgent medical situation
   or asks what to do during a possible medical emergency.

4. NON_MEDICAL_INFORMATION
   The user is asking about non medical information.

Important:
- A question mentioning "I" is NOT automatically personalized advice.
- General questions about symptoms, diseases, medications,
  treatments, or guidelines can still be GENERAL_MEDICAL_INFORMATION.
- If the user is asking for a diagnosis based on their own
  symptoms, classify it as PERSONALIZED_MEDICAL_ADVICE.
- If the user appears to need immediate emergency guidance,
  classify it as EMERGENCY_OR_CRISIS.

Return ONLY valid JSON:

{{
    "intent": "GENERAL_MEDICAL_INFORMATION"
}}

User query:
{user_query}
"""

