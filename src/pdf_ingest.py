"""
pdf_ingest.py
-------------
Uses LangChain to:
1. Load all PDFs from data/pdfs/        (PyPDFLoader)
2. Split them into chunks               (RecursiveCharacterTextSplitter)
3. Embed + store them in a Chroma DB    (HuggingFaceEmbeddings + Chroma)

Run this file once (or whenever you add new PDFs) to build the vector store:
    python src/pdf_ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
)


def load_pdf_documents(pdf_dir: str = PDF_DIR):
    """Loads every PDF in pdf_dir into LangChain Document objects (one per page)."""
    all_docs = []

    if not os.path.exists(pdf_dir):
        print(f"[pdf_ingest] Folder not found: {pdf_dir}")
        return all_docs

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[pdf_ingest] No PDFs found in {pdf_dir}. Add some and re-run.")
        return all_docs

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        print(f"[pdf_ingest] Loading {filename} ...")
        loader = PyPDFLoader(filepath)
        docs = loader.load() 
        all_docs.extend(docs)

    print(f"[pdf_ingest] Loaded {len(all_docs)} pages total")
    return all_docs


def split_documents(documents):
    """Splits page-level Documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[pdf_ingest] Split into {len(chunks)} chunks")
    return chunks


def build_chroma_index(chunks, persist_dir: str = CHROMA_DB_PATH):
    """Embeds chunks and saves them into a persistent Chroma DB on disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"[pdf_ingest] Chroma DB saved to {persist_dir}")
    return vectorstore


def load_chroma_index(persist_dir: str = CHROMA_DB_PATH):
    """Loads a previously saved Chroma DB from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    docs = load_pdf_documents()
    if docs:
        chunks = split_documents(docs)
        build_chroma_index(chunks)
    else:
        print("[pdf_ingest] Nothing to index. Put PDFs in data/pdfs/ first.")
