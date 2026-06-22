"""
router.py
---------
Decides which data source(s) should answer a given question:
- "pdf"  -> hybrid search over your PDF documents
- "sql"  -> query the SQL Server database
- "web"  -> live web search

Uses a simple LLM classification call. This is the "agentic" part of the
project — explain it in interviews as a basic query router (real systems
often use a small/cheap model just for routing to save cost).
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

ROUTER_PROMPT = """You are a query router. Decide which data source is best
suited to answer the user's question. Choose exactly ONE of:

- pdf  -> if the question is about policies, manuals, reports, or any document content
- sql  -> if the question asks for counts, records, numbers, or anything stored in a database/table
- web  -> if the question needs current/live/recent information (news, prices, "latest", etc.)

Respond with ONLY one word: pdf, sql, or web.

Question: {question}
Answer:"""


def route_query(question: str) -> str:
    """Returns 'pdf', 'sql', or 'web' based on the question."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)
    prompt = ROUTER_PROMPT.format(question=question)
    response = llm.invoke(prompt)

    choice = response.content.strip().lower()

    if choice not in ("pdf", "sql", "web"):
        print(f"[router] Unexpected router output '{choice}', defaulting to 'pdf'")
        choice = "pdf"

    return choice


if __name__ == "__main__":
    test_questions = [
        "What does our refund policy say?",
        "How many customers do we have in the database?",
        "What's the latest news in AI today?",
    ]
    for q in test_questions:
        print(f"Q: {q}  ->  route: {route_query(q)}")
