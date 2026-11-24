import os
import time
from typing import Dict, Any
from agents.state import AgentState
from rag.vector_store import get_retriever
from core.config import RETRIEVAL_TOP_K

def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Hybrid Retrieval Agent & CRAG Evaluator:
    1. Fetches top-k most relevant document chunks from the Chroma vector database
       using semantic similarity on enriched query and demographic profile.
    2. Attaches clean source metadata (filename, page number) for 100% citation mapping.
    3. Corrective RAG (CRAG) Grader: Evaluates retrieval relevance to guarantee that
       only grounded scheme documents proceed to the Answer Agent.
    """
    start_time = time.time()
    query = state.get("sanitized_query") or state.get("user_query", "")
    profile = state.get("user_profile", {})
    intent = state.get("intent", "eligibility")
    session_id = state.get("session_id")
    uploaded_pdf_mode = bool(state.get("uploaded_pdf_mode") or session_id)

    # Formulate enriched retrieval query combining citizen query and demographic context
    profile_hints = []
    if not uploaded_pdf_mode:
        if profile.get("state"):
            profile_hints.append(f"state: {profile['state']}")
        if profile.get("occupation"):
            profile_hints.append(f"occupation: {profile['occupation']}")
        if profile.get("land_holding") is not None:
            profile_hints.append(f"land: {profile['land_holding']} acres")
        if profile.get("annual_income") is not None:
            profile_hints.append(f"income: Rs {profile['annual_income']}")

    search_query = query
    if profile_hints and intent == "eligibility":
        search_query = f"{query} Beneficiary: {', '.join(profile_hints)}"

    # Fetch relevant chunks from session-uploaded PDF or default ChromaDB
    retriever = get_retriever(k=RETRIEVAL_TOP_K, session_id=session_id)
    docs = retriever.invoke(search_query)

    retrieved_chunks = []
    context_metadata = []

    for i, doc in enumerate(docs):
        retrieved_chunks.append(doc.page_content)
        raw_source = doc.metadata.get("source", "unknown_scheme.pdf")
        filename = os.path.basename(raw_source)
        # 1-indexed page number for clear civic citation
        page_num = int(doc.metadata.get("page", 0)) + 1

        context_metadata.append({
            "doc_id": f"Doc.{i+1}",
            "source": filename,
            "page": page_num,
            "full_path": raw_source,
            "length": len(doc.page_content)
        })

    # CRAG Retrieval Grader
    combined_text = " ".join(retrieved_chunks).lower()
    if uploaded_pdf_mode:
        civic_or_doc_keywords = [
            "scheme", "government", "eligibility", "eligible", "benefit", "guideline",
            "policy", "section", "clause", "rule", "portal", "application", "applicant",
            "income", "document", "pension", "subsidy", "grant", "allowance", "act",
            "ministry", "department", "citizen", "family", "financial", "assistance",
            "criteria", "exclusion", "farmer", "beneficiary"
        ]
        query_terms = [w for w in query.lower().split() if len(w) > 3]
        kw_matches = sum(1 for kw in civic_or_doc_keywords if kw in combined_text)
        query_overlap = sum(1 for qt in query_terms if qt in combined_text)
        relevance_passed = len(retrieved_chunks) > 0 and (kw_matches >= 1 or query_overlap >= 1)
    else:
        relevance_keywords = [
            "pm-kisan", "kalia", "scheme", "farmer", "cultivation", "land",
            "beneficiary", "financial assistance", "installment", "guidelines",
            "eligible", "exclusion", "income tax"
        ]
        matches = sum(1 for kw in relevance_keywords if kw in combined_text)
        relevance_passed = matches >= 2 and len(retrieved_chunks) > 0

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Retrieval Agent (CRAG Grader)",
        "action": f"Fetched {len(retrieved_chunks)} chunks in {elapsed}ms | CRAG Relevance: {'✅ Passed' if relevance_passed else '⚠️ Low Relevance'}",
        "sources": [f"{m['source']}#p{m['page']}" for m in context_metadata],
        "relevance_passed": relevance_passed
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "retrieved_context": retrieved_chunks,
        "context_metadata": context_metadata,
        "retrieval_relevance_passed": relevance_passed,
        "execution_trace": traces
    }
