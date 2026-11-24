import io
import os
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME

DB_DIR = "data/vector_db"
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

_cached_embeddings = None
_cached_vectorstore = None
_ephemeral_client = None
_session_lock = threading.Lock()
_session_stores: Dict[str, Dict[str, Any]] = {}


def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _cached_embeddings


def get_vectorstore():
    global _cached_vectorstore
    if _cached_vectorstore is None:
        embeddings = get_embeddings()
        _cached_vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    return _cached_vectorstore


def _get_ephemeral_client():
    global _ephemeral_client
    if _ephemeral_client is None:
        _ephemeral_client = chromadb.EphemeralClient()
    return _ephemeral_client


def _cleanup_expired_sessions() -> None:
    """Removes expired ephemeral PDF collections from memory (caller holds _session_lock)."""
    now = time.time()
    expired_ids = [
        sid for sid, meta in _session_stores.items()
        if now - meta.get("created_at", now) > SESSION_TTL_SECONDS
    ]
    for sid in expired_ids:
        entry = _session_stores.pop(sid, None)
        if entry and _ephemeral_client is not None:
            try:
                _ephemeral_client.delete_collection(entry["collection_name"])
            except Exception:
                pass


def index_uploaded_pdfs(
    files: List[Tuple[str, bytes]],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parses one or more uploaded PDF byte payloads in memory, chunks them,
    and indexes them into a temporary session-only Chroma collection.
    Nothing is written to data/raw_pdfs or data/vector_db.
    """
    if not files:
        raise ValueError("No PDF files were provided.")

    documents: List[Document] = []
    file_summaries: List[Dict[str, Any]] = []

    for raw_name, pdf_bytes in files:
        safe_name = os.path.basename(raw_name or "uploaded_document.pdf").strip() or "uploaded_document.pdf"
        if not pdf_bytes:
            raise ValueError(f"Uploaded file '{safe_name}' is empty.")

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as exc:
            raise ValueError(f"Could not parse '{safe_name}' as a valid PDF: {exc}") from exc

        extracted_pages = 0
        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                extracted_pages += 1
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": safe_name,
                            "page": page_idx,
                            "session_uploaded": True,
                        },
                    )
                )

        if extracted_pages == 0:
            raise ValueError(
                f"No extractable text found in '{safe_name}'. "
                "If this is a scanned image PDF, please run OCR first."
            )

        file_summaries.append({
            "filename": safe_name,
            "total_pages": len(reader.pages),
            "text_pages": extracted_pages,
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Uploaded PDF(s) produced zero text chunks.")

    sid = session_id or str(uuid.uuid4())
    collection_name = f"session_{sid.replace('-', '_')}"
    embeddings = get_embeddings()
    client = _get_ephemeral_client()

    with _session_lock:
        _cleanup_expired_sessions()
        # Replace any previous collection with the same session_id
        if sid in _session_stores:
            old_collection = _session_stores[sid]["collection_name"]
            try:
                client.delete_collection(old_collection)
            except Exception:
                pass

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
        )

        total_pages = sum(item["total_pages"] for item in file_summaries)
        filenames = [item["filename"] for item in file_summaries]
        preview_chunks = [
            {
                "source": c.metadata.get("source", "uploaded.pdf"),
                "page": int(c.metadata.get("page", 0)) + 1,
                "excerpt": c.page_content[:260].replace("\n", " ").strip(),
            }
            for c in chunks[:5]
        ]

        _session_stores[sid] = {
            "vectorstore": vectorstore,
            "collection_name": collection_name,
            "filenames": filenames,
            "file_summaries": file_summaries,
            "total_pages": total_pages,
            "chunk_count": len(chunks),
            "preview_chunks": preview_chunks,
            "created_at": time.time(),
        }

    return {
        "session_id": sid,
        "filenames": filenames,
        "file_summaries": file_summaries,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "storage_mode": "session_ephemeral",
        "preview_chunks": preview_chunks,
    }


def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """Returns metadata for an active ephemeral PDF session, or None if not found."""
    with _session_lock:
        _cleanup_expired_sessions()
        entry = _session_stores.get(session_id)
        if not entry:
            return None
        return {
            "session_id": session_id,
            "filenames": entry["filenames"],
            "file_summaries": entry["file_summaries"],
            "total_pages": entry["total_pages"],
            "total_chunks": entry["chunk_count"],
            "storage_mode": "session_ephemeral",
            "preview_chunks": entry["preview_chunks"],
        }


def delete_session_store(session_id: str) -> bool:
    """Deletes an ephemeral PDF session from memory."""
    with _session_lock:
        entry = _session_stores.pop(session_id, None)
        if not entry:
            return False
        if _ephemeral_client is not None:
            try:
                _ephemeral_client.delete_collection(entry["collection_name"])
            except Exception:
                pass
        return True


def get_retriever(k: int = 4, session_id: Optional[str] = None):
    """
    Returns a LangChain retriever backed by either:
    - The temporary session-only PDF vector store (when session_id is provided), or
    - The persistent local ChromaDB store (default).
    """
    if session_id:
        with _session_lock:
            _cleanup_expired_sessions()
            entry = _session_stores.get(session_id)
            if not entry:
                raise ValueError(f"Uploaded PDF session '{session_id}' was not found or has expired.")
            effective_k = max(1, min(k, entry["chunk_count"]))
            return entry["vectorstore"].as_retriever(
                search_type="similarity",
                search_kwargs={"k": effective_k},
            )

    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def test_retrieval(query):
    """Utility function to test searches directly from the terminal."""
    print(f"\n🔍 Searching for: '{query}'")
    
    retriever = get_retriever(k=3)
    results = retriever.invoke(query)
    
    if not results:
        print("No matching documents found.")
    else:
        for i, doc in enumerate(results):
            # Extract metadata (e.g., the PDF file name and page number)
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            print(f"\n--- Match {i+1} [Source: {source} | Page: {page}] ---")
            print(doc.page_content)

if __name__ == "__main__":
    # Run a quick test search if executed directly
    test_query = "What is the eligibility criteria for small farmers?"
    test_retrieval(test_query)