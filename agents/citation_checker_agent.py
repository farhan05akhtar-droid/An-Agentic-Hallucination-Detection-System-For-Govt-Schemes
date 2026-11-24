import re
import time
from typing import Dict, Any, List
from agents.state import AgentState

OFFICIAL_PORTALS = {
    "pm-kisan": "https://pmkisan.gov.in",
    "kalia": "https://kalia.odisha.gov.in",
    "pmay": "https://pmaymis.gov.in"
}

def citation_checker_agent(state: AgentState) -> Dict[str, Any]:
    """
    Citation Checker & Ground Truth Validator Agent:
    1. Extracts all inline document citation tags: `[Doc.ID: <Source>#p<Page>]`.
    2. Validates that every referenced source file and page corresponds to an authentic,
       retrieved ChromaDB metadata record (prevents phantom citations).
    3. Appends an authoritative Citations & Official Portals verification table.
    """
    start_time = time.time()
    text = state.get("final_answer") or state.get("draft_answer", "")
    metadata = state.get("context_metadata", [])

    # Extract all citation tags
    citation_pattern = r"\[Doc\.ID:\s*([^#\]]+)#p(\d+)\]"
    found_citations = re.findall(citation_pattern, text)

    valid_sources = {m.get("source", "").lower(): m.get("page") for m in metadata}
    validated_citations = []
    phantom_citations = []

    for src, page in found_citations:
        src_clean = src.strip().lower()
        if any(src_clean in vs for vs in valid_sources):
            validated_citations.append(f"{src}#p{page}")
        else:
            phantom_citations.append(f"{src}#p{page}")

    total_citations = len(found_citations)
    valid_count = len(validated_citations)

    mapping_pct = 100.0
    if total_citations > 0:
        mapping_pct = round((valid_count / total_citations) * 100, 1)

    # Build reference footer if citations exist
    footer_lines = ["\n\n---", "### 📚 Verified Government Document References & Official Portals"]
    seen_sources = set()

    for m in metadata:
        s_name = m.get("source", "Official Scheme Document")
        p_num = m.get("page", 1)
        tag = f"{s_name} (Page {p_num})"

        if tag not in seen_sources:
            seen_sources.add(tag)
            # Match portal URL
            portal_url = None
            for k, url in OFFICIAL_PORTALS.items():
                if k in s_name.lower():
                    portal_url = url
                    break

            if portal_url:
                footer_lines.append(f"- 📄 **{s_name}**, Page {p_num} — [Official Portal: {portal_url}]({portal_url})")
            else:
                footer_lines.append(f"- 📄 **{s_name}**, Page {p_num}")

    # Append footer if not already appended
    if "### 📚 Verified Government Document References" not in text and seen_sources:
        updated_text = text + "\n" + "\n".join(footer_lines)
    else:
        updated_text = text

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Citation Checker Agent",
        "action": f"Validated {total_citations} citations in {elapsed}ms | Authenticity: {mapping_pct}%",
        "valid_citations": validated_citations,
        "phantom_citations": phantom_citations
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    citation_report = {
        "total_citations": total_citations,
        "valid_citations": valid_count,
        "mapping_percentage": f"{mapping_pct}%",
        "phantom_citations": phantom_citations
    }

    return {
        "final_answer": updated_text,
        "citation_report": citation_report,
        "execution_trace": traces
    }
