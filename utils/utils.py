from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
  return "\n\n\n".join([
      f"Page Content: {doc.page_content}\n"
      f"Page Number: {doc.metadata.get('page_label', 'N/A')}\n"
      f"Chunk Id: {doc.metadata.get('chunk_id', 'N/A')}\n"
      f"File Location: {doc.metadata.get('source', 'N/A')}"
      for doc in docs
  ])

