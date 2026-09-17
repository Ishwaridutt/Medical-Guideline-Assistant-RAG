from clients.groq_client import llm_client, output_parser
from prompts.hyde_prompt import hyde_prompt

hyde_chain = (
    hyde_prompt
    | llm_client
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

