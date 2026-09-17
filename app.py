import os
from dotenv import load_dotenv
load_dotenv()

from safety.input_guard import handle_input_guardrail
from safety.output_guard import run_output_guardrails
from rag.retrieval_pipeline import rag_chain


def main():

    while True:

        # User Query
        user_query = input("\nEnter your query (type 'exit' to quit): ").strip()
        # Exit condition
        if user_query.lower() == "exit":
            print("\nExiting RAG application. Goodbye!")
            break

         # Input guardrails
        input_guardrail_check_result = handle_input_guardrail(user_query)
        if not input_guardrail_check_result:
            print('Failed to pass the input guardrails, so could not continue further...')
            continue

        print('\ninput guardrails passed, invoking rag pipeline for user query:', user_query)
        # run RAG pipeline
        try:
            # check if better query generation is required 
            # HyDE implementation
            # use a small transformer model to check if the user query meets a specific score
            # if yes then pass the user query to LLM model else generate HyDE for the user query
            
            llm_response = rag_chain.invoke(
                user_query,
                config = {
                        "run_name": "RAG Pipeline"
                    }
            )

            # apply output guardrail
            verified_response = run_output_guardrails(response=llm_response)

            print(f'\nRetrieved Result: {verified_response}')
        except Exception as e:
            print(
                f"\nAn error occurred while processing your query: {e}"
            )


if __name__ == "__main__":
    main()


        