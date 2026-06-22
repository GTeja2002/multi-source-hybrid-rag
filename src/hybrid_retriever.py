from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from pdf_ingest import load_chroma_index
from config import TOP_K

def build_hybrid_retriever(weight_dense: float = 0.5, weight_sparse: float = 0.5):
    """
    Builds a hybrid retriever that:
    1. Pulls top_k results from Chroma (dense/embedding similarity)
    2. Pulls top_k results from BM25 (sparse/keyword overlap)
    3. Fuses both ranked lists (EnsembleRetriever uses Reciprocal Rank Fusion
       internally) using the given weights.

    weight_dense + weight_sparse should add up to 1.0
    """
    vectorstore = load_chroma_index()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    all_data = vectorstore.get(include=["documents", "metadatas"])
    from langchain.schema import Document
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    if not documents:
        raise ValueError(
            "[hybrid_retriever] No documents found in Chroma. "
            "Run `python src/pdf_ingest.py` first to build the index."
        )

    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = TOP_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[weight_dense, weight_sparse],
    )

    return hybrid_retriever



if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    query = "example question about your PDF content"
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"{i}. [source={source}, page={page}]")
        print(f"   {doc.page_content[:150]}...\n")
