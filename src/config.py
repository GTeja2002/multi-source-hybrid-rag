import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


TOP_K = 5

PDF_DIR = "data/pdfs"
CHROMA_DB_PATH = "data/vector_store/chroma_db"
CHROMA_COLLECTION_NAME = "pdf_chunks"
