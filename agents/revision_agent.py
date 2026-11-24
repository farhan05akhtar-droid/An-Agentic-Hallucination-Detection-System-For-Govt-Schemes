import time
from typing import Dict, Any
from agents.state import AgentState
from core.prompts import (
    ANSWER_AGENT_SYSTEM_PROMPT,
    ANSWER_REVISION_PROMPT,
    UPLOADED_PDF_ANSWER_SYSTEM_PROMPT,
)
from core.utils import execute_llm_completion
from core.config import LLM_TEMPERATURE

def revision_agent(state: AgentState) -> Dict[str, Any]:
    """
    Revision Agent:
    Triggered after targeted re-retrieval to rewrite the advisory response.
    Specifically removes, replaces, or corrects the claims flagged as unverified,
    basing the new generation strictly on the augmented ground truth evidence.
    Increments revision_count to ensure strict loop budget control.
    """
    start_time = time.time()
    query = state.get("sanitized_query") or state.get("user_query", "")
    profile = state.get("user_profile", {})
    context_chunks = state.get("retrieved_context", [])
    context_metadata = state.get("context_metadata", [])
    eligibility_results = state.get("eligibility_results", {})
    revision_count = state.get("revision_count", 0)
    current_draft = state.get("draft_answer", "")
    unverified_claims = state.get("unverified_claims", [])

    # Format updated evidence context
    formatted_context_list = []
    for i, chunk in enumerate(context_chunks):
        meta = context_metadata[i] if i < len(context_metadata) else {}
        doc_id = meta.get("doc_id", f"Doc.{i+1}")
        source = meta.get("source", "scheme.pdf")
        page = meta.get("page", 1)
        tag = f"[Doc.ID: {source}#p{page}]"
        formatted_context_list.append(f"--- DOCUMENT EXCERPT {doc_id} {tag} ---\n{chunk.strip()}\n")

    combined_context = "\n".join(formatted_context_list)

    # Prepare revision prompt
    revision_header = ANSWER_REVISION_PROMPT.format(
        flagged_claims="\n".join([f"- {c}" for c in unverified_claims])
    )

    user_prompt = f"""{revision_header}

CURRENT REJECTED DRAFT:
{current_draft}

CITIZEN INQUIRY:
"{query}"

AUGMENTED GROUND TRUTH RETRIEVED CONTEXT:
{combined_context}

Please revise the advisory response. COMPLETELY ELIMINATE the flagged ungrounded claims.
Ensure every statement is 100% entailed by the context excerpts with precise `[Doc.ID: Source#pPage]` tags.
"""

    system_prompt = (
        UPLOADED_PDF_ANSWER_SYSTEM_PROMPT
        if (state.get("uploaded_pdf_mode") or state.get("session_id"))
        else ANSWER_AGENT_SYSTEM_PROMPT
    )
    prefix = "UPLOADED_PDF_AUDIT_MODE: TRUE\n\n" if (state.get("uploaded_pdf_mode") or state.get("session_id")) else ""

    new_draft = execute_llm_completion(
        system_prompt=system_prompt,
        user_prompt=prefix + user_prompt,
        temperature=LLM_TEMPERATURE
    )

    new_revision_count = revision_count + 1
    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Revision Agent",
        "action": f"Revised advisory draft (Attempt {new_revision_count}) in {elapsed}ms -> Removed {len(unverified_claims)} flagged assertions",
        "draft_length": len(new_draft)
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "draft_answer": new_draft,
        "revision_count": new_revision_count,
        "execution_trace": traces
    }
