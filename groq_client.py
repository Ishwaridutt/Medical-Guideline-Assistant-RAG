import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

GROQ_MODEL = os.getenv("GROQ_MODEL")
llm = ChatGroq(model=GROQ_MODEL)

output_parser = StrOutputParser()

