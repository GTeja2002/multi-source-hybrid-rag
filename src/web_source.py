from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from config import TAVILY_API_KEY, GROQ_API_KEY, LLM_MODEL


def get_web_search_tool():
    """Returns a LangChain Tavily search tool that fetches top web results."""
    return TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)


def query_web_source(question: str) -> str:
    """
    1. Search the web for the question
    2. Feed the search snippets to the LLM
    3. LLM writes a final answer citing what it found
    """
    search_tool = get_web_search_tool()
    results = search_tool.invoke({"query": question})

    
    context = "\n\n".join(
        f"Source: {r.get('url', 'unknown')}\nContent: {r.get('content', '')}"
        for r in results
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)

    prompt = (
        f"Question: {question}\n\n"
        f"Web search results:\n{context}\n\n"
        f"Using only the information above, answer the question and mention "
        f"which source(s) you used."
    )
    answer = llm.invoke(prompt)
    return answer.content



if __name__ == "__main__":
    test_question = "What is the latest version of Python?"
    print(query_web_source(test_question))
