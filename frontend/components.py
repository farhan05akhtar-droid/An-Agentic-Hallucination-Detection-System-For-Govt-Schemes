import streamlit as st
from typing import List, Dict, Any

def render_header():
    """Renders the top banner and title styling."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 24px; border-radius: 12px; margin-bottom: 24px; color: white; border-left: 6px solid #3b82f6;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff;">🏛️ GovScheme AI Advisory Agent</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.05rem; color: #cbd5e1;">
            Agentic Hallucination Detection System in Government Schemes • Zero-Hallucination Civic Intelligence
        </p>
        <div style="margin-top: 12px; display: flex; gap: 12px; flex-wrap: wrap;">
            <span style="background: rgba(59, 130, 246, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #3b82f6;">LangGraph Multi-Agent RAG</span>
            <span style="background: rgba(16, 185, 129, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #10b981;">NLI Claim Entailment Audit</span>
            <span style="background: rgba(245, 158, 11, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #f59e0b;">Temperature T=0.0 Lock</span>
            <span style="background: rgba(168, 85, 247, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #a855f7;">100% PDF Citations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(
    faithfulness: str,
    latency_sec: float,
    total_claims: int,
    hallucination_count: int
):
    """Renders the top executive KPI metric cards matching PRD goals."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Factual Grounding",
            value=faithfulness,
            delta="Target: ≥ 98%",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="⚡ Execution Latency",
            value=f"{latency_sec}s",
            delta="Target: < 8s",
            delta_color="normal" if latency_sec < 8.0 else "inverse"
        )
    with col3:
        st.metric(
            label="🔬 Atomic Claims Audited",
            value=total_claims,
            delta="NLI Evaluated",
            delta_color="off"
        )
    with col4:
        st.metric(
            label="🛡️ Hallucinations Detected",
            value=hallucination_count,
            delta="0 Permitted" if hallucination_count == 0 else f"{hallucination_count} Stripped",
            delta_color="normal" if hallucination_count == 0 else "inverse"
        )


def render_scheme_card(scheme_name: str, info: Dict[str, Any]):
    """Renders an individual civic scheme eligibility card."""
    status = info.get("status", "Unknown")
    benefit = info.get("benefit", "N/A")
    rules = info.get("rules_applied", [])
    citation = info.get("citation", "Official Records")

    border_color = "#10b981" if status == "Eligible" else "#ef4444"
    bg_color = "rgba(16, 185, 129, 0.06)" if status == "Eligible" else "rgba(239, 68, 68, 0.06)"
    badge_bg = "#10b981" if status == "Eligible" else "#ef4444"

    rules_html = "".join([f"<li>{r}</li>" for r in rules])

    st.markdown(f"""
    <div style="border: 1px solid {border_color}; background-color: {bg_color}; border-radius: 10px; padding: 16px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.3rem;">📜 {scheme_name}</h3>
            <span style="background-color: {badge_bg}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">{status}</span>
        </div>
        <p style="margin: 6px 0; font-size: 1.05rem;"><strong>Financial Benefit:</strong> <span style="color: #2563eb; font-weight: 600;">{benefit}</span></p>
        <div style="margin: 8px 0;">
            <strong>Rule Verification:</strong>
            <ul style="margin: 4px 0 0 16px; padding: 0; font-size: 0.95rem;">
                {rules_html}
            </ul>
        </div>
        <div style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
            📎 <em>Ground Truth Source:</em> <code style="color: #0f172a; background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{citation}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_claims_audit_table(claims: List[Dict[str, Any]]):
    """Renders the detailed claim-by-claim verification table."""
    if not claims:
        st.info("No atomic claims extracted for audit.")
        return

    table_data = []
    for item in claims:
        table_data.append({
            "Atomic Claim": item.get("claim", ""),
            "NLI Status": item.get("status", ""),
            "Entailment Score": item.get("entailment_score", ""),
            "Ground Truth Document": item.get("ground_truth_doc", ""),
            "Supporting Evidence Excerpt": item.get("evidence", "")[:120] + "..." if len(item.get("evidence", "")) > 120 else item.get("evidence", "")
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        column_config={
            "Atomic Claim": st.column_config.TextColumn("Atomic Claim", width="medium"),
            "NLI Status": st.column_config.TextColumn("Audit Verdict", width="small"),
            "Entailment Score": st.column_config.TextColumn("Confidence", width="small"),
            "Ground Truth Document": st.column_config.TextColumn("Source PDF", width="small"),
            "Supporting Evidence Excerpt": st.column_config.TextColumn("Exact Evidence Quote", width="large")
        }
    )


def render_trace_logs(trace_entries: List[Dict[str, Any]]):
    """Renders formatted multi-agent execution traces for policy auditors without nested expanders."""
    if not trace_entries:
        st.write("No trace records available.")
        return

    for i, t in enumerate(trace_entries):
        agent_name = t.get("agent", "Agent")
        action = t.get("action", "")

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.04); border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 6px; margin: 8px 0;">
            <strong style="color: #1e3a8a;">Step {i+1}: 🤖 [{agent_name}]</strong> — <em>{action}</em>
        </div>
        """, unsafe_allow_html=True)
        st.json(t)


def render_uploaded_pdf_banner(session_info: Dict[str, Any]):
    """Renders a status card for the currently active session-only uploaded PDF(s)."""
    filenames = ", ".join(session_info.get("filenames", []))
    pages = session_info.get("total_pages", 0)
    chunks = session_info.get("total_chunks", 0)
    sid = session_info.get("session_id", "")[:8]

    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10b981; border-left: 5px solid #10b981; border-radius: 10px; padding: 14px 18px; margin: 10px 0 14px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong style="font-size: 1.05rem;">📄 Active Ground Truth PDF:</strong>
                <code style="background: rgba(15,23,42,0.08); padding: 2px 8px; border-radius: 4px;">{filenames}</code>
            </div>
            <span style="background: #0f766e; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                Session-Only RAM • ID {sid}
            </span>
        </div>
        <div style="margin-top: 6px; font-size: 0.9rem; color: #334155;">
            Indexed <strong>{pages}</strong> page(s) into <strong>{chunks}</strong> searchable vector chunks (automatically discarded when cleared or session ends).
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_results(pipeline_result: Dict[str, Any], uploaded_pdf_mode: bool = False):
    """Renders the full multi-agent pipeline output, NLI audit table, chunks, and trace logs."""
    final_answer = pipeline_result.get("final_answer", "")
    eligibility = pipeline_result.get("eligibility_results", {})
    hallucination_report = pipeline_result.get("hallucination_report", {})
    metadata = pipeline_result.get("context_metadata", [])
    traces = pipeline_result.get("execution_trace", [])
    latency = pipeline_result.get("total_latency_seconds", 0.0)
    faithfulness = hallucination_report.get("faithfulness_score", "100%")
    total_claims = hallucination_report.get("total_claims", 0)
    hallucinations_detected = hallucination_report.get("hallucination_count", 0)
    pii_masked = pipeline_result.get("pii_redacted", False)
    intent = pipeline_result.get("intent", "eligibility")
    detected_lang = pipeline_result.get("detected_language", "English")
    crag_status = "PASSED" if pipeline_result.get("retrieval_relevance_passed", True) else "LOW RELEVANCE"
    source_mode = "Uploaded PDF (Session-Only)" if uploaded_pdf_mode else "Official Scheme DB"

    if pii_masked:
        st.warning("🔒 **Security Guardrail Active**: Sensitive citizen PII (Aadhaar/PAN/Bank/Phone) was automatically masked before processing.")

    st.caption(
        f"🧭 **Agentic Routing:** Source: `{source_mode}` | Intent: `{intent.upper()}` | "
        f"Language: `{detected_lang}` | CRAG Status: `{crag_status}`"
    )

    st.markdown("### 📊 Operational Quality & PRD Metrics")
    render_metric_cards(
        faithfulness=faithfulness,
        latency_sec=latency,
        total_claims=total_claims,
        hallucination_count=hallucinations_detected
    )

    st.divider()

    if not uploaded_pdf_mode:
        st.subheader("📋 Scheme Entitlement Summary")
        if eligibility:
            cols = st.columns(len(eligibility))
            for i, (scheme_name, info) in enumerate(eligibility.items()):
                with cols[i]:
                    render_scheme_card(scheme_name, info)
        else:
            st.info("No matching schemes found for this profile.")

    st.subheader("🛡️ Verified Advisory Response")
    st.markdown(final_answer)

    with st.expander("🔬 Factual Grounding & NLI Entailment Audit Report", expanded=True):
        st.caption("Every proposition was cross-referenced with retrieved PDF chunks using Natural Language Inference:")
        claims_table = hallucination_report.get("claims_table", [])
        render_claims_audit_table(claims_table)

        if hallucination_report.get("unverified_claims"):
            st.warning(f"⚠️ {len(hallucination_report['unverified_claims'])} claim(s) lacked sufficient entailment evidence and were isolated/sanitized:")
            for uc in hallucination_report["unverified_claims"]:
                st.write(f"- `{uc}`")

    with st.expander("📎 Retrieved PDF Context Chunks (100% Mapped Metadata)", expanded=False):
        st.caption("Ground truth PDF chunks used for verification:")
        retrieved_chunks = pipeline_result.get("retrieved_context", [])
        for i, chunk in enumerate(retrieved_chunks):
            meta = metadata[i] if i < len(metadata) else {}
            doc_name = meta.get("source", "PDF Document")
            page_num = meta.get("page", "1")
            doc_id = meta.get("doc_id", f"Doc.{i+1}")
            st.markdown(f"**[{doc_id}] {doc_name} — Page {page_num}**")
            st.text(chunk)
            st.divider()

    with st.expander("⚙️ LangGraph Multi-Agent Execution Log", expanded=False):
        st.caption("Inspect node-by-node state transitions, latencies, and routing decisions:")
        render_trace_logs(traces)

