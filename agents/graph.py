import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.guardrail_agent import guardrail_agent
from agents.query_analyzer_agent import query_analyzer_agent
from agents.eligibility_agent import eligibility_agent
from agents.retrieval_agent import retrieval_agent
from agents.answer_agent import answer_agent
from agents.hallucination_agent import hallucination_agent
from agents.re_retrieval_agent import re_retrieval_agent
from agents.revision_agent import revision_agent
from agents.claim_sanitizer import claim_sanitizer_agent
from agents.citation_checker_agent import citation_checker_agent
from core.config import MAX_LATENCY_BUDGET_SECONDS, MAX_REVISION_COUNT

# -------------------------------------------------------------
# Conditional Edge Evaluators
# -------------------------------------------------------------

def route_after_query_analysis(state: AgentState) -> str:
    """
    Routes based on citizen intent:
    - If clarification needed or out of scope: route directly to answer generator.
    - If querying a session-uploaded PDF: route straight to retrieval (skipping hardcoded scheme rules).
    - If user profile is provided or intent is eligibility: route to Deterministic Rule Engine first.
    - Otherwise: route straight to retrieval.
    """
    intent = state.get("intent", "eligibility")
    if intent in ["clarification_needed", "out_of_scope"]:
        return "answer"

    if state.get("uploaded_pdf_mode") or state.get("session_id"):
        return "retrieval"

    profile = state.get("user_profile", {})
    if profile or intent == "eligibility":
        return "eligibility"
    return "retrieval"


def route_after_answer(state: AgentState) -> str:
    """
    If the response is a clarification prompt or out-of-scope guidance,
    skip the NLI hallucination auditor and go directly to citation checker.
    Otherwise, proceed to the full Claim Extractor & Verification Agent.
    """
    intent = state.get("intent", "eligibility")
    if intent in ["clarification_needed", "out_of_scope"]:
        return "citation_checker"
    return "hallucination"

def route_after_hallucination_check(state: AgentState) -> str:
    """
    Evaluates NLI verification report:
    1. If unverified claims detected AND revision count < cutoff -> Loop to Re-retrieval & Revision.
    2. If cutoff reached with unverified claims -> Route to Sanitizer to strip ungrounded statements.
    3. If all claims verified (faithfulness >= 98%) -> Route to Citation Checker.
    """
    report = state.get("hallucination_report", {})
    decision = report.get("decision", "APPROVED")
    revision_count = state.get("revision_count", 0)

    if decision == "LOOP_REVISION" and revision_count < MAX_REVISION_COUNT:
        return "re_retrieval"
    elif report.get("hallucination_count", 0) > 0:
        return "sanitizer"
    return "citation_checker"

# -------------------------------------------------------------
# Build Multi-Agent LangGraph Workflow
# -------------------------------------------------------------
workflow = StateGraph(AgentState)

# 1. Register Nodes
workflow.add_node("guardrail", guardrail_agent)
workflow.add_node("query_analyzer", query_analyzer_agent)
workflow.add_node("eligibility", eligibility_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("answer", answer_agent)
workflow.add_node("hallucination", hallucination_agent)
workflow.add_node("re_retrieval", re_retrieval_agent)
workflow.add_node("revision", revision_agent)
workflow.add_node("sanitizer", claim_sanitizer_agent)
workflow.add_node("citation_checker", citation_checker_agent)

# 2. Sequential & Entry Edges
workflow.add_edge(START, "guardrail")
workflow.add_edge("guardrail", "query_analyzer")

# 3. Intent Routing Conditional Edge
workflow.add_conditional_edges(
    "query_analyzer",
    route_after_query_analysis,
    {
        "answer": "answer",
        "eligibility": "eligibility",
        "retrieval": "retrieval"
    }
)

# 4. Processing Branches
workflow.add_edge("eligibility", "retrieval")
workflow.add_edge("retrieval", "answer")

# 5. Answer to Verification Conditional Edge
workflow.add_conditional_edges(
    "answer",
    route_after_answer,
    {
        "citation_checker": "citation_checker",
        "hallucination": "hallucination"
    }
)

# 6. Hallucination Verification Conditional Edge (Loop / Sanitize / Pass)
workflow.add_conditional_edges(
    "hallucination",
    route_after_hallucination_check,
    {
        "re_retrieval": "re_retrieval",
        "sanitizer": "sanitizer",
        "citation_checker": "citation_checker"
    }
)

# 7. Revision Feedback Loop (Re-retrieval -> Revision -> Re-verification)
workflow.add_edge("re_retrieval", "revision")
workflow.add_edge("revision", "hallucination")

# 8. Output Pipeline
workflow.add_edge("sanitizer", "citation_checker")
workflow.add_edge("citation_checker", END)

# Compile Graph
gov_scheme_graph = workflow.compile()

# -------------------------------------------------------------
# Main Pipeline Invocation Function
# -------------------------------------------------------------
def run_pipeline(
    user_query: str,
    user_profile: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main invocation entrypoint for the GovScheme Multi-Agent System.
    Initializes graph state, runs end-to-end execution, and tracks overall metrics.
    When session_id is supplied, retrieval and auditing run against the temporary uploaded PDF.
    """
    start_total = time.time()

    initial_state: AgentState = {
        "user_query": user_query,
        "sanitized_query": user_query,
        "pii_redacted": False,
        "user_profile": user_profile or {},
        "extracted_slots": {},
        "detected_language": "English",
        "intent": "document_qa" if session_id else "eligibility",
        "clarification_message": "",
        "retrieval_relevance_passed": True,
        "retrieved_context": [],
        "context_metadata": [],
        "eligibility_results": {},
        "draft_answer": "",
        "atomic_claims": [],
        "hallucination_report": {},
        "unverified_claims": [],
        "re_retrieved_context": [],
        "citation_report": {},
        "final_answer": "",
        "revision_count": 0,
        "execution_trace": [],
        "session_id": session_id,
        "uploaded_pdf_mode": bool(session_id),
    }

    # Execute LangGraph
    final_state = gov_scheme_graph.invoke(initial_state)

    total_duration = round(time.time() - start_total, 2)

    # Attach summary stats to execution trace
    final_state["total_latency_seconds"] = total_duration
    final_state["latency_status"] = "✅ Within Budget (< 8s)" if total_duration < MAX_LATENCY_BUDGET_SECONDS else "⚠️ Exceeded 8s Budget"

    return final_state
