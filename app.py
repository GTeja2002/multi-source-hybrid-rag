import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from router import route_query
from hybrid_retriever import build_hybrid_retriever
from sql_source import query_sql_source
from web_source import query_web_source
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

st.set_page_config(page_title="APPSC RAG", page_icon="🎯")
st.title("🎯 APPSC Multi-Source RAG")
st.caption("Answers from PDF | Database | Web Search")

@st.cache_resource
def get_retriever():
    return build_hybrid_retriever()

def answer_from_pdf(question):
    docs = get_retriever().invoke(question)
    if not docs:
        return "NO_RESULTS"
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)
    return llm.invoke(f"Answer using only this context:\n{context}\n\nQuestion: {question}").content

def answer_question(question):
    source = route_query(question)
    if source == "sql":
        answer = query_sql_source(question)
        if not answer or "NO_RESULTS" in answer:
            source, answer = "pdf", answer_from_pdf(question)
    elif source == "pdf":
        answer = answer_from_pdf(question)
        if answer == "NO_RESULTS":
            source, answer = "web", query_web_source(question)
    else:
        answer = query_web_source(question)
    return source, answer

if "history" not in st.session_state:
    st.session_state.history = []

for role, text, src in st.session_state.history:
    with st.chat_message(role):
        if role == "assistant":
            st.caption(f"Source: {src}")
        st.write(text)

question = st.chat_input("Ask about APPSC syllabus...")
if question:
    st.session_state.history.append(("user", question, ""))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            source, answer = answer_question(question)
        st.caption(f"Source: {source.upper()}")
        st.write(answer)

    st.session_state.history.append(("assistant", answer, source.upper()))