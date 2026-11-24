import sys
import os
import hashlib
from pathlib import Path

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from agents.graph import run_pipeline
from core.utils import audit_text_against_source
from rag.vector_store import index_uploaded_pdfs, delete_session_store
from frontend.components import (
    render_header,
    render_metric_cards,
    render_claims_audit_table,
    render_uploaded_pdf_banner,
    render_pipeline_results,
)

st.set_page_config(
    page_title="GovScheme AI - Agentic Hallucination Detection",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Styled Header Banner
render_header()

# -------------------------------------------------------------
# 1. SIDEBAR: BENEFICIARY PROFILE & EDGE CASE CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.header("👤 Beneficiary Profile")
    st.caption("Configure citizen demographics and statutory edge case conditions:")

    # Quick Presets for CSC Operators & Evaluators
    st.subheader("⚡ Quick Test Scenarios")
    preset = st.selectbox(
        "Load Preset Case:",
        [
            "Custom Configuration",
            "1. Odisha Marginal Farmer (1.5 Acres - Both Schemes)",
            "2. PM-KISAN Micro-Holding Edge Case (0.03 Acres)",
            "3. Urban Farming Edge Case (2.0 Acres in City)",
            "4. Income Tax Payee (Statutory Disqualification)",
            "5. Landless Laborer in Odisha (Livelihood Support)"
        ]
    )

    # Set default values based on preset
    def_state = "Odisha"
    def_occ = "Farmer"
    def_income = 120000
    def_land = 1.5
    def_cultivable = True
    def_urban = False
    def_tax = False
    def_inst = False

    if preset == "1. Odisha Marginal Farmer (1.5 Acres - Both Schemes)":
        def_state = "Odisha"
        def_occ = "Farmer"
        def_income = 100000
        def_land = 1.5
    elif preset == "2. PM-KISAN Micro-Holding Edge Case (0.03 Acres)":
        def_state = "Central / All India"
        def_occ = "Farmer"
        def_income = 80000
        def_land = 0.03
        def_cultivable = True
    elif preset == "3. Urban Farming Edge Case (2.0 Acres in City)":
        def_state = "Odisha"
        def_occ = "Farmer"
        def_income = 150000
        def_land = 2.0
        def_urban = True
        def_cultivable = True
    elif preset == "4. Income Tax Payee (Statutory Disqualification)":
        def_state = "Odisha"
        def_occ = "Farmer"
        def_income = 600000
        def_land = 3.0
        def_tax = True
    elif preset == "5. Landless Laborer in Odisha (Livelihood Support)":
        def_state = "Odisha"
        def_occ = "Daily Wage Laborer"
        def_income = 60000
        def_land = 0.0
        def_cultivable = False

    state = st.selectbox(
        "State of Residence",
        ["Odisha", "Central / All India", "Uttar Pradesh", "Maharashtra", "Punjab", "Other"],
        index=["Odisha", "Central / All India", "Uttar Pradesh", "Maharashtra", "Punjab", "Other"].index(def_state)
    )

    occupation = st.selectbox(
        "Primary Occupation",
        ["Farmer", "Daily Wage Laborer", "Small Business Owner", "Student", "Other"],
        index=["Farmer", "Daily Wage Laborer", "Small Business Owner", "Student", "Other"].index(def_occ)
    )

    annual_income = st.number_input(
        "Annual Household Income (₹)",
        min_value=0,
        value=def_income,
        step=10000
    )

    land_holding = st.number_input(
        "Landholding Size (Acres)",
        min_value=0.0,
        max_value=500.0,
        value=float(def_land),
        step=0.05,
        help="Test PM-KISAN Micro-holding Rule: < 0.05 acres is eligible if cultivable!"
    )

    st.subheader("⚖️ Core Legal Rules & Toggles")
    is_cultivable = st.toggle("Land is Cultivable", value=def_cultivable, help="Uncultivable micro-holdings are excluded.")
    is_urban = st.toggle("Land is in Urban Area", value=def_urban, help="PM-KISAN Urban Rule: Urban land qualifies if cultivated.")
    pays_income_tax = st.toggle("Pays Income Tax", value=def_tax, help="Statutory disqualification across all schemes.")
    is_institutional = st.toggle("Institutional Landholder", value=def_inst, help="Statutory disqualification for institutions.")

    st.divider()
    st.subheader("🔑 LLM Configuration")
    user_api_key = st.text_input(
        "Groq / Gemini / OpenAI API Key (Optional):",
        type="password",
        help="Leave blank to use the built-in deterministic grounded engine, or paste your key for live LLM inference."
    )
    if user_api_key:
        if user_api_key.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = user_api_key
            st.success("Configured Groq API Key!")
        elif user_api_key.startswith("AIza"):
            os.environ["GEMINI_API_KEY"] = user_api_key
            st.success("Configured Gemini API Key!")
        elif user_api_key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = user_api_key
            st.success("Configured OpenAI API Key!")

# Build current profile dictionary
current_profile = {
    "state": state,
    "occupation": occupation,
    "annual_income": annual_income,
    "land_holding": land_holding,
    "is_cultivable": is_cultivable,
    "is_urban": is_urban,
    "pays_income_tax": pays_income_tax,
    "is_institutional": is_institutional
}

# -------------------------------------------------------------
# 2. DRAG & DROP GOVERNMENT PDF STUDIO (SESSION-ONLY)
# -------------------------------------------------------------
if "pdf_session_id" not in st.session_state:
    st.session_state.pdf_session_id = None
if "pdf_session_info" not in st.session_state:
    st.session_state.pdf_session_info = None
if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pdf_quick_query" not in st.session_state:
    st.session_state.pdf_quick_query = ""

st.markdown("### 📄 Drag & Drop Government PDF Studio (Session-Only)")
st.caption("Drop any government scheme circular, guidelines, or notification PDF to index it temporarily in memory and verify answers or claims against hallucinations.")

uploaded_files = st.file_uploader(
    "Drag and drop government PDF(s) here",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}",
    help="Session-only: Uploaded PDFs live only in memory and are discarded when cleared or when the session ends."
)

if uploaded_files:
    file_payloads = []
    sig_parts = []
    for f in uploaded_files:
        raw_bytes = f.getvalue()
        digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
        file_payloads.append((f.name, raw_bytes))
        sig_parts.append((f.name, len(raw_bytes), digest))
    current_sig = tuple(sig_parts)

    if current_sig != st.session_state.pdf_signature:
        with st.spinner("📥 Parsing and indexing uploaded PDF(s) into temporary session store..."):
            try:
                info = index_uploaded_pdfs(
                    files=file_payloads,
                    session_id=st.session_state.pdf_session_id,
                )
                st.session_state.pdf_session_id = info["session_id"]
                st.session_state.pdf_session_info = info
                st.session_state.pdf_signature = current_sig
            except Exception as exc:
                st.error(f"Could not index uploaded PDF: {exc}")
elif st.session_state.pdf_session_id is not None:
    delete_session_store(st.session_state.pdf_session_id)
    st.session_state.pdf_session_id = None
    st.session_state.pdf_session_info = None
    st.session_state.pdf_signature = None

use_uploaded_pdf = False
if st.session_state.pdf_session_info:
    render_uploaded_pdf_banner(st.session_state.pdf_session_info)
    col_toggle, col_clear = st.columns([3, 1])
    with col_toggle:
        use_uploaded_pdf = st.toggle(
            "Use Uploaded PDF as Ground Truth Source",
            value=True,
            help="When enabled, Q&A and hallucination audits run exclusively against your uploaded PDF(s)."
        )
    with col_clear:
        if st.button("🗑️ Clear Uploaded PDF", use_container_width=True):
            if st.session_state.pdf_session_id:
                delete_session_store(st.session_state.pdf_session_id)
            st.session_state.pdf_session_id = None
            st.session_state.pdf_session_info = None
            st.session_state.pdf_signature = None
            st.session_state.pdf_quick_query = ""
            st.session_state.uploader_key += 1
            st.rerun()

    with st.expander("👀 Preview Indexed PDF Excerpts", expanded=False):
        for item in st.session_state.pdf_session_info.get("preview_chunks", []):
            st.markdown(f"**{item['source']} — Page {item['page']}**")
            st.caption(item["excerpt"])

active_session_id = st.session_state.pdf_session_id if use_uploaded_pdf else None

st.divider()

# -------------------------------------------------------------
# 3. MAIN INTERACTIVE AUDIT WORKSPACE
# -------------------------------------------------------------
tab_qa, tab_audit = st.tabs([
    "💬 Ask & Audit Grounded Q&A",
    "🧪 Check Text / Claims for Hallucinations"
])

with tab_qa:
    if active_session_id:
        if "pdf_inquiry_input" not in st.session_state:
            st.session_state.pdf_inquiry_input = (
                "Summarize the eligibility criteria, benefits, and exclusion rules from the uploaded government PDF."
            )

        st.caption("⚡ Quick audit prompts for your uploaded PDF:")
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            if st.button("🔍 Summarize Key Rules", use_container_width=True):
                st.session_state.pdf_inquiry_input = "Summarize the key rules, eligibility conditions, and benefits stated in this government document."
                st.rerun()
        with q_col2:
            if st.button("✅ Eligibility & Exclusions", use_container_width=True):
                st.session_state.pdf_inquiry_input = "Who is eligible and who is excluded according to this document?"
                st.rerun()
        with q_col3:
            if st.button("💰 Benefits & Application Steps", use_container_width=True):
                st.session_state.pdf_inquiry_input = "What financial benefits, required documents, and application procedures are described in this PDF?"
                st.rerun()

        user_query = st.text_area(
            "💬 Inquiry Against Uploaded PDF:",
            key="pdf_inquiry_input",
            height=80,
            placeholder="Ask any question about the uploaded government PDF..."
        )
        col_run, col_persona = st.columns([2, 5])
        with col_run:
            execute_button = st.button("🚀 Ask & Audit Uploaded PDF", type="primary", use_container_width=True)
        with col_persona:
            st.info("📄 **Uploaded PDF Mode Active:** Responses are grounded exclusively in your dropped PDF and verified by the NLI Hallucination Auditor.")
    else:
        default_queries = {
            "1. Odisha Marginal Farmer (1.5 Acres - Both Schemes)": "I am a small farmer in Odisha with 1.5 acres of land. Am I eligible for PM-KISAN and KALIA schemes?",
            "2. PM-KISAN Micro-Holding Edge Case (0.03 Acres)": "I only own 0.03 acres of cultivable land. Does the micro-holding rule disqualify me from PM-KISAN?",
            "3. Urban Farming Edge Case (2.0 Acres in City)": "My 2-acre farm is inside municipal urban limits where I grow vegetables. Can I get PM-KISAN benefits?",
            "4. Income Tax Payee (Statutory Disqualification)": "I am a farmer with 3 acres in Odisha but I paid income tax last year. Can I receive KALIA or PM-KISAN?",
            "5. Landless Laborer in Odisha (Livelihood Support)": "I am a landless agricultural worker in Odisha with no farmland. Can I receive financial assistance under KALIA?"
        }

        suggested_query = default_queries.get(preset, "What government welfare schemes apply to my profile and how much financial assistance will I receive?")

        user_query = st.text_area(
            "💬 Citizen Inquiry:",
            value=suggested_query,
            height=80,
            placeholder="Ask questions regarding scheme eligibility, financial assistance amounts, exclusion rules, etc."
        )

        col_run, col_persona = st.columns([2, 5])
        with col_run:
            execute_button = st.button("🚀 Verify Eligibility & Audit Claims", type="primary", use_container_width=True)

        with col_persona:
            st.info("💡 **Persona Active:** Select profile settings in the sidebar to test edge cases, or drag & drop any government PDF above to audit it.")

    if execute_button:
        if not user_query.strip():
            st.warning("Please enter a question or inquiry.")
        else:
            with st.spinner("🤖 Multi-Agent Workflow: Retrieval → Grounded Drafting → Hallucination NLI Audit..."):
                pipeline_result = run_pipeline(
                    user_query=user_query,
                    user_profile={} if active_session_id else current_profile,
                    session_id=active_session_id,
                )
            render_pipeline_results(pipeline_result, uploaded_pdf_mode=bool(active_session_id))

with tab_audit:
    target_label = (
        f"Uploaded PDF ({', '.join(st.session_state.pdf_session_info['filenames'])})"
        if active_session_id and st.session_state.pdf_session_info
        else "Official Built-in Scheme Database (PM-KISAN & KALIA)"
    )
    st.caption(f"Paste any statement, summary, or AI-generated response below to check sentence-by-sentence for hallucinations against **{target_label}**.")

    text_to_audit = st.text_area(
        "📝 Statement or Draft Text to Verify:",
        height=130,
        placeholder="Example: Eligible beneficiaries receive ₹6,000 per year in three equal installments. Income tax payees are excluded."
    )
    audit_btn = st.button("🔬 Scan Text for Hallucinations", type="primary")

    if audit_btn:
        if not text_to_audit.strip():
            st.warning("Please paste at least one claim or sentence to verify.")
        else:
            with st.spinner("🔬 Splitting atomic claims & running NLI entailment verification..."):
                audit_res = audit_text_against_source(
                    text=text_to_audit,
                    session_id=active_session_id,
                )

            render_metric_cards(
                faithfulness=audit_res["faithfulness_score"],
                latency_sec=round(audit_res["audit_duration_ms"] / 1000.0, 2),
                total_claims=audit_res["total_claims"],
                hallucination_count=audit_res["hallucination_count"],
            )
            st.divider()
            render_claims_audit_table(audit_res["claims_table"])
            if audit_res["unverified_claims"]:
                st.error(f"🚨 Detected {len(audit_res['unverified_claims'])} potentially hallucinated / unsupported claim(s):")
                for uc in audit_res["unverified_claims"]:
                    st.write(f"- `{uc}`")
            else:
                st.success("✅ All claims are entailed by the ground truth PDF with zero hallucinations detected.")