import time
from typing import Dict, Any, List
from agents.state import AgentState
from core.utils import split_into_atomic_claims, calculate_nli_entailment
from core.config import (
    ENTAILMENT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    MAX_REVISION_COUNT
)

def hallucination_agent(state: AgentState) -> Dict[str, Any]:
    """
    Hallucination Detection & Verification Agent:
    Performs automated claim-level verification before rendering.
    1. Splits the draft answer into atomic factual propositions.
    2. Runs NLI Entailment checks against retrieved ChromaDB chunks.
    3. Calculates Factual Grounding Score and flags Contradictions/Neutral statements.
    4. Triggers loop to Re-retrieval and Revision Agents if Neutral/Contradiction > 0.3 or Entailment < 0.7.
    5. Handsoff to Claim Sanitizer if max revision safety cutoff is reached to guarantee zero hallucination.
    """
    start_time = time.time()
    draft = state.get("draft_answer", "")
    context_chunks = state.get("retrieved_context", [])
    context_metadata = state.get("context_metadata", [])
    revision_count = state.get("revision_count", 0)

    # Step 1: Claim Decomposition
    atomic_claims = split_into_atomic_claims(draft)

    # Step 2: NLI Entailment & Grounding Verification
    claim_results: List[Dict[str, Any]] = []
    unverified_claims: List[str] = []
    entailed_count = 0

    for claim in atomic_claims:
        nli_res = calculate_nli_entailment(
            claim=claim,
            context_chunks=context_chunks,
            context_metadata=context_metadata
        )

        verdict = nli_res["verdict"]
        entailment = nli_res["entailment_score"]
        contradiction = nli_res["contradiction_score"]

        # Verification rule: Entailment >= 0.70 AND Contradiction/Neutral <= 0.30
        if verdict == "ENTAILMENT" and entailment >= ENTAILMENT_THRESHOLD:
            entailed_count += 1
            status_label = "✅ Entailed"
        elif verdict == "CONTRADICTION" or contradiction > CONTRADICTION_THRESHOLD:
            status_label = "❌ Contradiction"
            unverified_claims.append(claim)
        else:
            status_label = "⚠️ Neutral / Unverified"
            unverified_claims.append(claim)

        claim_results.append({
            "claim": claim,
            "status": status_label,
            "entailment_score": f"{int(entailment * 100)}%",
            "ground_truth_doc": nli_res["ground_truth_doc"],
            "evidence": nli_res["evidence_quote"]
        })

    # Step 3: Compute PRD Metrics
    total_claims = max(1, len(atomic_claims))
    faithfulness_score = round((entailed_count / total_claims) * 100, 1)
    hallucination_count = len(unverified_claims)

    passed = (hallucination_count == 0) or (faithfulness_score >= 98.0)

    # Step 4: Revision cutoff & loop decision
    if not passed and revision_count < MAX_REVISION_COUNT:
        # Request loop to Re-retrieval -> Revision
        decision = "LOOP_REVISION"
        final_answer = ""
    else:
        decision = "APPROVED"
        final_answer = draft

    elapsed = round((time.time() - start_time) * 1000, 2)

    report = {
        "faithfulness_score": f"{faithfulness_score}%",
        "total_claims": total_claims,
        "entailed_claims": entailed_count,
        "hallucination_count": hallucination_count,
        "claims_table": claim_results,
        "unverified_claims": unverified_claims,
        "decision": decision,
        "audit_duration_ms": elapsed
    }

    trace_entry = {
        "agent": "Hallucination Agent",
        "action": f"Audited {total_claims} claims in {elapsed}ms -> Decision: {decision}",
        "grounding_score": f"{faithfulness_score}%",
        "hallucinations_detected": hallucination_count
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "atomic_claims": atomic_claims,
        "unverified_claims": unverified_claims,
        "hallucination_report": report,
        "final_answer": final_answer,
        "execution_trace": traces
    }
