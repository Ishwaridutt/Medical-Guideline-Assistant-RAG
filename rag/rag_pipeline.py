from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
# from operator import itemgetter
from clients.groq_client import llm_client, output_parser
from prompts.medical_assistant_prompt import prompt_template
from utils.utils import format_docs
from query_router.query_router import query_router
# from rag.rag_retrieval import retrieval_and_reranking
from rag.crag.crag_pipeline import crag_retrieval


def run_crag_from_router(router_output):
    return crag_retrieval(
        original_query = router_output["original_query"],
        retrieval_query = router_output["retrieval_query"],
    )

crag_pipeline = RunnableLambda(run_crag_from_router)

# rag_chain = (
#     query_router
#     | {
#         "context": RunnableLambda(run_crag_from_router) | format_docs,
#         # "context": itemgetter("retrieval_query") | retrieval_and_reranking | format_docs,
#         "user_question": itemgetter("user_question"),
#         # "user_question": RunnablePassthrough()
#     }
#     | prompt_template
#     | llm_client
#     | output_parser
# )

rag_chain = (
    query_router
    | crag_pipeline
    | RunnableBranch(
        # CRAG fail case
        (
            lambda x: x["status"] == "CRAG_FAILED",
            RunnableLambda(
                lambda x: x["message"]
            ),
        ),
        # CRAG success
        {
            "context": lambda x: format_docs(
                x["documents"]
            ),
            "user_question": lambda x: x["original_query"],
        }
        | prompt_template
        | llm_client
        | output_parser,
    )
)

