from langchain_core.prompts import ChatPromptTemplate

hyde_prompt = ChatPromptTemplate.from_template(
    """
You are generating a hypothetical document to improve retrieval.

Given the user's medical question, generate a hypothetical passage
that would contain the information needed to answer the question.

The passage should contain relevant medical terminology, concepts,
and factual information that could help retrieve relevant documents.

Requirements:
- Use terminology and style likely to appear in real source documents for this topic.
- Write a concise, self-contained technical passage (around 100–200 words).
- Do not include citations, page numbers, or meta-commentary.
- Do not claim that the information is verified; just write the passage.
- Output only the hypothetical document text, nothing else.
- Do not answer the user directly.
- Do not mention that this is a hypothetical document.

User question:
{question}
"""
)