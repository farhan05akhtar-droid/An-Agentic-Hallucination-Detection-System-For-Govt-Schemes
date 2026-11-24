from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """
    Central Graph State for the GovScheme Multi-Agent LangGraph workflow.
    Tracks user input, profile, sanitized queries, intent routing, extracted slots,
    retrieved chunks, CRAG relevance, eligibility logic, draft answer generation,
    hallucination audit metrics, re-retrieval, revision loops, and citation validation.
    """
    user_query: str                  # Raw user question/input
    sanitized_query: str            # Query after PII scrubbing & guardrail sanitization
    pii_redacted: bool               # Flag if sensitive PII was detected and masked
    user_profile: dict               # State, Income, Occupation, Land, Cultivable, Urban, etc.
    extracted_slots: dict            # Demographic slots automatically extracted from query text
    detected_language: str           # Language detected (e.g. English, Hindi, Hinglish)
    intent: str                      # "eligibility", "general_info", "application_docs", "grievance_status", "clarification_needed", "out_of_scope"
    clarification_message: str       # Question asked back to citizen if crucial info is missing
    retrieval_relevance_passed: bool # CRAG check: whether retrieved docs are relevant to the query
    retrieved_context: list[str]     # Raw ChromaDB text chunks
    context_metadata: list[dict]     # Source document paths, page numbers, and doc IDs
    eligibility_results: dict        # Pass/Fail booleans and reasoning per scheme
    draft_answer: str                # Current LLM generation with [Doc.ID] citation tags
    atomic_claims: list[str]         # Claims extracted for verification
    hallucination_report: dict       # Atomic claims, NLI entailment scores, verdict
    unverified_claims: list[str]     # Claims flagged as neutral or contradictory
    re_retrieved_context: list[str]  # Additional chunks fetched during targeted re-retrieval
    citation_report: dict            # Verification status of citations [Doc.ID: ...]
    final_answer: str                # Approved markdown output (or corrected response)
    revision_count: int              # Safety cutoff counter for revision looping
    execution_trace: list[dict]      # Step-by-step trace logs for UI/auditing
    total_latency_seconds: float     # Overall pipeline latency
    latency_status: str              # Latency budget status
    session_id: Optional[str]        # Ephemeral uploaded PDF session ID (if active)
    uploaded_pdf_mode: bool          # True when querying user-uploaded PDF ground truth
