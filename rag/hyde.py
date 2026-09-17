from clients.groq_client import llm, output_parser
from prompts.hyde_prompt import hyde_prompt

hyde_chain = (
    hyde_prompt
    | llm
    | output_parser
).with_config(
    run_name = "HyDE Generation"
)

def generate_hyde(query: str) -> str:
    return hyde_chain.invoke(
        {
            "question": query
        }
    )

