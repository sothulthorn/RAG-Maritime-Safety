"""Query ChromaDB for relevant document chunks."""

from langchain_core.documents import Document

from config import RETRIEVAL_K
from ingestion.embedder import get_vectorstore, LocalEmbeddings


def retrieve(query: str, k: int = RETRIEVAL_K, embedding_fn=None) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    vectorstore = get_vectorstore(embedding_fn)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(query)
