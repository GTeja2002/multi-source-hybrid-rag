Multi-Source Hybrid RAG

A Retrieval-Augmented Generation (RAG) application that answers user questions by automatically selecting the best data source:

PDF Documents → Hybrid search (Vector Search + BM25)
SQL Server Database → Text-to-SQL queries
Web Search → Real-time information from the internet

The system uses an LLM-based router to determine where a question should be answered from and generates grounded responses using retrieved information.

Architecture
User Question
      │
      ▼
   LLM Router
      │
 ┌────┼────┐
 │    │    │
 ▼    ▼    ▼
PDF  SQL  Web
 │    │    │
 └────┼────┘
      ▼
   Groq LLM
      ▼
 Final Answer
Features
Automatic source selection (PDF / SQL / Web)
Hybrid document retrieval using Chroma + BM25
Natural language to SQL conversion
Real-time web search integration
Fast response generation with Groq Llama 3.3 70B
Tech Stack
LangChain
Groq (Llama 3.3 70B)
ChromaDB
BM25
SQL Server
Tavily Search
Python
Project Structure
src/
├── pdf_ingest.py
├── hybrid_retriever.py
├── sql_source.py
├── web_source.py
├── router.py
└── main.py
Workflow
User asks a question.
Router identifies the best source.
Relevant information is retrieved.
Groq LLM generates the final answer.
User receives a grounded response.
Resume Bullet

Built a Multi-Source RAG system using LangChain, ChromaDB, SQL Server, and Tavily Search, enabling intelligent routing between document retrieval, database querying, and web search for accurate question answering.

Future Improvements
Cross-encoder re-ranking
RAGAS evaluation
Streamlit UI
Source citations
Multi-document support
One-Line GitHub Description

Intelligent Multi-Source RAG system with PDF Hybrid Search, SQL Querying, and Live Web Search powered by LangChain, Groq, and ChromaDB.

