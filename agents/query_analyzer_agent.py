import re
import time
from typing import Dict, Any
from agents.state import AgentState

INDIAN_STATES = [
    "Odisha", "Orissa", "Punjab", "Haryana", "Uttar Pradesh", "Bihar",
    "Madhya Pradesh", "Rajasthan", "Maharashtra", "West Bengal",
    "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana", "Kerala",
    "Gujarat", "Assam", "Jharkhand", "Chhattisgarh", "Himachal Pradesh",
    "Uttarakhand", "Goa", "Tripura", "Meghalaya", "Manipur", "Nagaland",
    "Mizoram", "Sikkim", "Delhi"
]

def query_analyzer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Query Analyzer & Intent Router Agent:
    1. Language Detection: Identifies English, Hindi, or Hinglish query phrasing.
    2. Demographic Slot Extraction: Extracts State, Landholding, Income, Occupation,
       Urban, and Tax status from natural language queries to enrich the user profile.
    3. Intent Routing: Classifies into 'eligibility', 'application_docs',
       'grievance_status', 'general_info', 'clarification_needed', or 'out_of_scope'.
    """
    start_time = time.time()
    query = state.get("sanitized_query") or state.get("user_query", "")
    profile = dict(state.get("user_profile") or {})
    extracted_slots = {}

    query_lower = query.lower()

    # 1. Language Detection
    hindi_keywords = ["kisan", "yojana", "mera", "paisa", "zamin", "kitna", "milega", "aavedan", "karein", "patra"]
    if any(hk in query_lower for hk in hindi_keywords):
        detected_language = "Hindi / Hinglish"
    else:
        detected_language = "English"

    # 2. Demographic Slot Extraction
    # Extract State
    for st in INDIAN_STATES:
        if re.search(r"\b" + re.escape(st.lower()) + r"\b", query_lower):
            extracted_slots["state"] = "Odisha" if st.lower() in ["odisha", "orissa"] else st
            break

    # Extract Landholding (e.g. 1.5 acres, 0.03 acre, 2 bigha, 5 hectares)
    land_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:acres?|acre|ac|bigha|hectares?|ha)\b", query_lower)
    if land_match:
        extracted_slots["land_holding"] = float(land_match.group(1))

    # Extract Income (e.g. Rs 80,000, 1.5 lakh, income 120000)
    income_match = re.search(r"(?:rs\.?|₹|income(?:\s+of)?)\s*(\d+[\d,]*)\b", query_lower)
    if income_match:
        val_str = income_match.group(1).replace(",", "")
        try:
            extracted_slots["annual_income"] = float(val_str)
        except ValueError:
            pass

    # Extract Occupation
    if any(occ in query_lower for occ in ["farmer", "kisan", "cultivator"]):
        extracted_slots["occupation"] = "Farmer"
    elif any(occ in query_lower for occ in ["laborer", "labourer", "daily wage", "mazdoor"]):
        extracted_slots["occupation"] = "Daily Wage Laborer"

    # Extract Urban / Cultivable / Tax status
    if "urban" in query_lower or "municipal" in query_lower or "city limits" in query_lower:
        extracted_slots["is_urban"] = True
    if "cultivable" in query_lower or "farming" in query_lower:
        extracted_slots["is_cultivable"] = True
    if "uncultivable" in query_lower or "barren" in query_lower:
        extracted_slots["is_cultivable"] = False
    if "tax" in query_lower and ("pay" in query_lower or "paid" in query_lower or "filed" in query_lower):
        extracted_slots["pays_income_tax"] = True

    # Merge extracted slots into profile if profile lacked explicit values
    updated_profile = dict(profile)
    for k, v in extracted_slots.items():
        if k not in updated_profile or updated_profile[k] is None:
            updated_profile[k] = v

    uploaded_pdf_mode = bool(state.get("uploaded_pdf_mode") or state.get("session_id"))

    out_of_scope_terms = [
        "recipe", "cook", "biryani", "dish", "cricket", "football", "game",
        "sport", "python code", "movie", "song", "weather forecast", "bitcoin", "crypto"
    ]
    if not uploaded_pdf_mode:
        out_of_scope_terms.append("food")
    application_terms = ["how to apply", "apply online", "documents required", "registration process", "portal link", "application form", "where to apply"]
    grievance_terms = ["installment stopped", "e-kyc", "ekyc", "rejected", "status check", "beneficiary status", "complaint", "grievance", "not received money"]
    eligibility_terms = [
        "eligible", "eligibility", "qualify", "apply to me", "can i get", "can we get",
        "can i receive", "can we receive", "can get", "can receive", "benefit", "disqualif",
        "micro-holding", "urban", "scheme for me", "schemes apply", "get kalia", "get pm-kisan",
        "exclusion", "entitled"
    ]

    has_elig_phrase = bool(re.search(r"\bcan\s+(?:i|we|a\s+farmer|farmers?)\s+(?:get|receive|avail|claim)\b", query_lower))

    is_out_of_scope = any(
        re.search(r"\b" + re.escape(term) + r"\b", query_lower)
        for term in out_of_scope_terms
    )

    if is_out_of_scope:
        intent = "out_of_scope"
    elif uploaded_pdf_mode:
        intent = "document_qa"
    elif any(term in query_lower for term in grievance_terms):
        intent = "grievance_status"
    elif any(term in query_lower for term in application_terms):
        intent = "application_docs"
    elif any(term in query_lower for term in eligibility_terms) or has_elig_phrase:
        # Check if profile lacks basic info when asking specifically about personal eligibility
        if not updated_profile.get("state") and updated_profile.get("land_holding") is None and len(query.split()) < 6:
            intent = "clarification_needed"
        else:
            intent = "eligibility"
    else:
        intent = "general_info"


    clarification_msg = ""
    if intent == "clarification_needed":
        clarification_msg = (
            "To evaluate your exact scheme eligibility and financial benefit amounts, "
            "please share your **State of Residence**, your **Landholding Size (in acres)**, "
            "and whether you pay income tax."
        )

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Query Analyzer Agent",
        "action": f"Analyzed query in {elapsed}ms -> Intent: '{intent}', Language: '{detected_language}'",
        "extracted_slots": extracted_slots,
        "intent": intent
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "intent": intent,
        "detected_language": detected_language,
        "extracted_slots": extracted_slots,
        "user_profile": updated_profile,
        "clarification_message": clarification_msg,
        "execution_trace": traces
    }
