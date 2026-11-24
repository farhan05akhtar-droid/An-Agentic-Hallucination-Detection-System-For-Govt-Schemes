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

def answer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Answer Agent:
    Drafts a comprehensive, grounded civic advisory response.
    Explicitly tags every factual claim with document citation IDs `[Doc.ID: <Source>#p<Page>]`.
    Handles intent-specific routing (Clarification requests, Out-of-scope queries, CRAG fallbacks).
    """
    start_time = time.time()
    query = state.get("sanitized_query") or state.get("user_query", "")
    profile = state.get("user_profile", {})
    intent = state.get("intent", "eligibility")
    retrieval_passed = state.get("retrieval_relevance_passed", True)
    clarification_msg = state.get("clarification_message", "")
    context_chunks = state.get("retrieved_context", [])
    context_metadata = state.get("context_metadata", [])
    eligibility_results = state.get("eligibility_results", {})
    revision_count = state.get("revision_count", 0)
    uploaded_pdf_mode = bool(state.get("uploaded_pdf_mode") or state.get("session_id"))

    # 1. Handle Clarification Intent
    if intent == "clarification_needed":
        draft_answer = (
            "### 🏛️ Civic Profile Clarification Required\n\n"
            f"{clarification_msg}\n\n"
            "Once you provide these details, the system will compute your statutory eligibility, "
            "exact installment amounts, and application steps."
        )
        elapsed = round((time.time() - start_time) * 1000, 2)
        trace_entry = {
            "agent": "Answer Agent",
            "action": f"Generated demographic clarification prompt in {elapsed}ms",
            "intent": intent
        }
        traces = list(state.get("execution_trace", []))
        traces.append(trace_entry)
        return {
            "draft_answer": draft_answer,
            "final_answer": draft_answer,
            "execution_trace": traces
        }

    # 2. Handle Out-of-Scope Intent
    if intent == "out_of_scope":
        draft_answer = (
            "### 🏛️ Out of Scope Query\n\n"
            "I am an authorized Civic Advisory Assistant specialized in **Government Welfare Schemes** "
            "(such as PM-KISAN, KALIA, and allied agricultural/livelihood programs).\n\n"
            "Your inquiry appears outside the scope of civic welfare policies. Please ask a question related to "
            "scheme eligibility, financial benefits, or application procedures."
        )
        elapsed = round((time.time() - start_time) * 1000, 2)
        trace_entry = {
            "agent": "Answer Agent",
            "action": f"Generated out-of-scope guidance in {elapsed}ms",
            "intent": intent
        }
        traces = list(state.get("execution_trace", []))
        traces.append(trace_entry)
        return {
            "draft_answer": draft_answer,
            "final_answer": draft_answer,
            "execution_trace": traces
        }

    # 3. Handle CRAG Relevance Failure (Epistemic Fallback)
    if not retrieval_passed and len(context_chunks) == 0:
        draft_answer = (
            "### 🏛️ Information Not Available in Official Guidelines\n\n"
            "The official government scheme guidelines currently indexed in our verified database do not contain "
            "sufficient documentation to answer this specific query with 100% verified certainty.\n\n"
            "To avoid generating ungrounded advice, please refer to the National Government Services Portal at "
            "[https://services.india.gov.in](https://services.india.gov.in) or visit your nearest Common Service Center (CSC)."
        )
        elapsed = round((time.time() - start_time) * 1000, 2)
        trace_entry = {
            "agent": "Answer Agent",
            "action": f"Generated CRAG epistemic fallback in {elapsed}ms",
            "intent": intent
        }
        traces = list(state.get("execution_trace", []))
        traces.append(trace_entry)
        return {
            "draft_answer": draft_answer,
            "final_answer": draft_answer,
            "execution_trace": traces
        }

    # 4. Standard Advisory Generation with Ground Truth Citations
    formatted_context_list = []
    for i, chunk in enumerate(context_chunks):
        meta = context_metadata[i] if i < len(context_metadata) else {}
        doc_id = meta.get("doc_id", f"Doc.{i+1}")
        source = meta.get("source", "scheme.pdf")
        page = meta.get("page", 1)
        tag = f"[Doc.ID: {source}#p{page}]"
        formatted_context_list.append(f"--- DOCUMENT EXCERPT {doc_id} {tag} ---\n{chunk.strip()}\n")

    combined_context = "\n".join(formatted_context_list)

    if uploaded_pdf_mode:
        system_prompt = UPLOADED_PDF_ANSWER_SYSTEM_PROMPT
        user_prompt = f"""UPLOADED_PDF_AUDIT_MODE: TRUE

USER INQUIRY:
"{query}"

GROUND TRUTH RETRIEVED CONTEXT:
{combined_context}

Please draft an official, zero-hallucination response adhering strictly to the uploaded PDF excerpts above. Every factual statement MUST include citation tags in format `[Doc.ID: Source#pPage]`.
"""
    else:
        system_prompt = ANSWER_AGENT_SYSTEM_PROMPT
        # Format demographic and rule findings
        profile_summary = f"""
Beneficiary Profile:
- State of Residence: {profile.get('state', 'Central / All India')}
- Occupation: {profile.get('occupation', 'Farmer')}
- Annual Income: ₹{profile.get('annual_income', 150000)}
- Landholding Size: {profile.get('land_holding', 1.5)} acres
- Is Land Cultivable?: {profile.get('is_cultivable', True)}
- Is Land Urban?: {profile.get('is_urban', False)}
- Income Tax Payee?: {profile.get('pays_income_tax', False)}
- Institutional Landholder?: {profile.get('is_institutional', False)}

Rule Engine Eligibility Verdicts:
"""
        for scheme, info in eligibility_results.items():
            profile_summary += f"• {scheme}: {info.get('status')} - Benefit: {info.get('benefit')}. Notes: {'; '.join(info.get('rules_applied', []))}\n"

        user_prompt = f"""CITIZEN QUERY:
"{query}"

{profile_summary}

GROUND TRUTH RETRIEVED CONTEXT:
{combined_context}

Please draft an official, zero-hallucination civic advisory response adhering strictly to the above facts. Every statement MUST include citation tags in format `[Doc.ID: Source#pPage]`.
"""

    draft_answer = execute_llm_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=LLM_TEMPERATURE
    )

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Answer Agent",
        "action": f"Drafted advisory response in {elapsed}ms",
        "draft_length": len(draft_answer)
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "draft_answer": draft_answer,
        "execution_trace": traces
    }
