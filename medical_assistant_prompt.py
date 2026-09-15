from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a helpful AI assistant who answers user query based on the  
available context retrieved from the document along with the 
page_content and page number.

You should only answer the user based on the following context 
and navigate the user to open the right page number to know more.

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{user_question}")
])

