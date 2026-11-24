import os
import re
import time
from typing import Dict, Any
from agents.state import AgentState
from rag.vector_store import get_retriever

def re_retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Targeted Re-retrieval Agent:
    Triggered when the Verification Agent identifies unverified claims during the hallucination audit.
    1. Extracts target entities and propositions from the flagged unverified claims.
    2. Performs focused semantic search against ChromaDB to uncover missing statutory context.
    3. Merges newly discovered evidence chunks into the graph state for the Revision Agent.
    """
    start_time = time.time()
    unverified_claims = state.get("unverified_claims", [])
    if not unverified_claims:
        report = state.get("hallucination_report", {})
        unverified_claims = report.get("unverified_claims", [])

    existing_chunks = list(state.get("retrieved_context", []))
    existing_meta = list(state.get("context_metadata", []))

    new_chunks = []
    new_meta = []

    session_id = state.get("session_id")
    retriever = get_retriever(k=3, session_id=session_id)

    for claim in unverified_claims[:3]:
        # Clean claim to form targeted search query
        clean_claim = re.sub(r"\[Doc\.ID:[^\]]+\]", "", claim).strip()
        docs = retriever.invoke(clean_claim)

        for doc in docs:
            # Check if chunk is already present to avoid duplication
            if not any(doc.page_content.strip()[:60] == c.strip()[:60] for c in existing_chunks):
                raw_source = doc.metadata.get("source", "scheme.pdf")
                filename = os.path.basename(raw_source)
                page_num = int(doc.metadata.get("page", 0)) + 1

                meta_entry = {
                    "doc_id": f"Doc.R{len(existing_meta) + len(new_meta) + 1}",
                    "source": filename,
                    "page": page_num,
                    "full_path": raw_source,
                    "length": len(doc.page_content)
                }

                new_chunks.append(doc.page_content)
                new_meta.append(meta_entry)

    all_chunks = existing_chunks + new_chunks
    all_meta = existing_meta + new_meta

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Targeted Re-retrieval Agent",
        "action": f"Executed targeted query expansion on {len(unverified_claims)} unverified claims in {elapsed}ms -> Found {len(new_chunks)} additional evidence chunks",
        "new_sources": [f"{m['source']}#p{m['page']}" for m in new_meta]
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "retrieved_context": all_chunks,
        "context_metadata": all_meta,
        "re_retrieved_context": new_chunks,
        "execution_trace": traces
    }
