
from router import route_query
from hybrid_retriever import build_hybrid_retriever
from sql_source import query_sql_source
from web_source import query_web_source
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL


def answer_from_pdf(question: str) -> str:
    """Hybrid search over PDFs + LLM generates the final answer using retrieved chunks."""
    retriever = build_hybrid_retriever()
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the PDFs."

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.content


def answer_question(question: str) -> str:
    source = route_query(question)
    print(f"[main] Routed to source: {source}")

    if source == "sql":
        answer = query_sql_source(question)
        
        if "No results found" in answer or "SQL source error" in answer:
            print("[main] SQL had no results → falling back to PDF")
            answer = answer_from_pdf(question)
            
            if "No relevant information" in answer:
                print("[main] PDF had no results → falling back to Web")
                answer = query_web_source(question)

    elif source == "pdf":
        answer = answer_from_pdf(question)
    
        if "No relevant information" in answer:
            print("[main] PDF had no results → falling back to Web")
            answer = query_web_source(question)

    elif source == "web":
        answer = query_web_source(question)

    else:
        answer = "Could not determine a data source."

    return answer

def main():
    print("=" * 60)
    print("Multi-Source Hybrid RAG  (PDF + SQL Server + Web Search)")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        try:
            answer = answer_question(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"[main] Error: {e}")


if __name__ == "__main__":
    main()
