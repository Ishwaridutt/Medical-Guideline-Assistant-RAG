import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.config import GROQ_MODEL_NAME, GROQ_LLAMA_MODEL_NAME

llm = ChatGroq(model = GROQ_MODEL_NAME)

# llama_llm = ChatGroq(model = GROQ_LLAMA_MODEL_NAME)

output_parser = StrOutputParser()
