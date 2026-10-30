# Phase 1: ingest. Fill in the 5 TODOs, then run: python rag.py data/pdfs
import hashlib
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
embeddings = HuggingFaceEmbeddings(model_name=MODEL, encode_kwargs={"normalize_embeddings": True})
store = Chroma("docchat", embeddings, persist_directory="chroma",
               collection_metadata={"hnsw:space": "cosine"})  # cosine, same as the Colab experiment
# 250, not 256: the model adds 2 special tokens ([CLS], [SEP]) to every input
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=250, chunk_overlap=32)


def ingest(path: str) -> int:
    doc_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    pages = PyPDFLoader(path).load()
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"{path}: no extractable text")
    for c in chunks:
        c.metadata.update(doc_id=doc_id, filename=Path(path).name, page=c.metadata["page"] + 1)
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    store.add_documents(chunks, ids=ids)  # upsert: same ids overwrite, count stays flat
    return len(chunks)


if __name__ == "__main__":
    for pdf in Path(sys.argv[1]).glob("*.pdf"):
        try:
            print(pdf.name, ingest(str(pdf)), "chunks")
        except ValueError as e:
            print("skipped:", e)
