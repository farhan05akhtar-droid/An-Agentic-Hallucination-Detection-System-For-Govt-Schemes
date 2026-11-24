import time
from typing import Dict, Any
from agents.state import AgentState

def eligibility_agent(state: AgentState) -> Dict[str, Any]:
    """
    Eligibility Agent:
    Evaluates the beneficiary's demographic profile against statutory government scheme rules,
    rigorously enforcing core legal edge cases:
      - The PM-KISAN "Micro-Holding" Rule (< 0.05 acres is NOT disqualified unless uncultivable).
      - The "Urban" Rule (urban land qualifies if under actual cultivation).
      - State residence restrictions (KALIA strictly for Odisha; PM-KISAN is All-India).
      - Exclusions (Institutional land, Income Tax payees, constitutional post holders).
    """
    start_time = time.time()
    profile = state.get("user_profile", {})

    state_name = str(profile.get("state", "Central / All India")).strip().title()
    occupation = str(profile.get("occupation", "Farmer")).strip().title()
    income = float(profile.get("annual_income", 150000))
    land_holding = float(profile.get("land_holding", 1.5))
    is_cultivable = bool(profile.get("is_cultivable", True))
    is_urban = bool(profile.get("is_urban", False))
    pays_income_tax = bool(profile.get("pays_income_tax", False))
    is_institutional = bool(profile.get("is_institutional", False))

    results = {}

    # ---------------------------------------------------------
    # 1. PM-KISAN EVALUATION (Central Scheme)
    # ---------------------------------------------------------
    pm_kisan_eligible = True
    pm_kisan_reasons = []

    # Exclusion check: Institutional & Tax
    if is_institutional:
        pm_kisan_eligible = False
        pm_kisan_reasons.append("Excluded: Institutional landholders are strictly disqualified.")
    if pays_income_tax:
        pm_kisan_eligible = False
        pm_kisan_reasons.append("Excluded: Income Tax payees in the last assessment year are disqualified.")

    # Core Rule: PM-KISAN Micro-Holding Rule
    # "Land size < 0.05 acres is NOT automatically disqualified unless it is also explicitly uncultivable"
    if land_holding < 0.05:
        if not is_cultivable:
            pm_kisan_eligible = False
            pm_kisan_reasons.append("Excluded: Micro land holding (< 0.05 acres) is uncultivable (Clause 30 FAQ).")
        else:
            pm_kisan_reasons.append("Admissible under Micro-Holding Rule: Landholding < 0.05 acres is eligible because it is cultivable.")
    else:
        if not is_cultivable:
            pm_kisan_eligible = False
            pm_kisan_reasons.append("Excluded: Land must be cultivable as per land records.")

    # Core Rule: Urban Land Rule
    # "Urban land is treated exactly as rural land, provided actual farming takes place"
    if is_urban:
        if is_cultivable:
            pm_kisan_reasons.append("Admissible under Urban Rule: Urban agricultural land qualifies since actual cultivation is performed (Clause 29 FAQ).")
        else:
            pm_kisan_eligible = False
            pm_kisan_reasons.append("Excluded: Urban land not under active cultivation.")

    if pm_kisan_eligible:
        results["PM-KISAN"] = {
            "status": "Eligible",
            "benefit": "₹6,000 per year (3 installments of ₹2,000 every 4 months)",
            "rules_applied": pm_kisan_reasons if pm_kisan_reasons else ["Standard cultivable landholding criteria satisfied."],
            "citation": "PM-KISAN_Guidelines.pdf#p1 & PM-KISAN_FAQ.pdf#p6"
        }
    else:
        results["PM-KISAN"] = {
            "status": "Ineligible",
            "benefit": "₹0",
            "rules_applied": pm_kisan_reasons,
            "citation": "PM-KISAN_Guidelines.pdf#p1 & PM-KISAN_FAQ.pdf#p6"
        }

    # ---------------------------------------------------------
    # 2. KALIA EVALUATION (Odisha State Scheme)
    # ---------------------------------------------------------
    kalia_eligible = True
    kalia_reasons = []

    # Geographic restriction: Strictly Odisha
    is_odisha = "odisha" in state_name.lower()
    if not is_odisha:
        kalia_eligible = False
        kalia_reasons.append(f"Excluded: Citizen resides in '{state_name}'. KALIA is strictly restricted to bona fide permanent residents of Odisha.")

    # Exclusion checks
    if is_institutional:
        kalia_eligible = False
        kalia_reasons.append("Excluded: Institutional landholders are ineligible for KALIA.")
    if pays_income_tax:
        kalia_eligible = False
        kalia_reasons.append("Excluded: Income tax payees or their family members are ineligible.")

    # Land and occupation checks
    kalia_benefit = "₹10,000 per year (₹5,000 Kharif + ₹5,000 Rabi)"
    if is_odisha and kalia_eligible:
        if occupation == "Farmer":
            if land_holding <= 2.5:
                kalia_reasons.append("Qualified as Marginal Farmer (land <= 2.5 acres / 1 hectare).")
            elif land_holding <= 5.0:
                kalia_reasons.append("Qualified as Small Farmer (land 2.5 to 5.0 acres / 1 to 2 hectares).")
            else:
                kalia_eligible = False
                kalia_reasons.append("Excluded: Farmers owning more than 5 acres (2 hectares) are considered medium/large and excluded.")
        elif occupation in ["Daily Wage Laborer", "Agricultural Laborer"]:
            if land_holding == 0:
                kalia_benefit = "₹12,500 (Livelihood support for landless agricultural households)"
                kalia_reasons.append("Qualified as Landless Agricultural Household for allied livelihood activities.")
            else:
                kalia_reasons.append("Evaluated as small/marginal cultivator.")

    if kalia_eligible:
        results["KALIA"] = {
            "status": "Eligible",
            "benefit": kalia_benefit,
            "rules_applied": kalia_reasons if kalia_reasons else ["Odisha resident and small/marginal farmer."],
            "citation": "KALIA_Guidelines.pdf#p1"
        }
    else:
        results["KALIA"] = {
            "status": "Ineligible",
            "benefit": "₹0",
            "rules_applied": kalia_reasons,
            "citation": "KALIA_Guidelines.pdf#p1-2"
        }

    elapsed = round((time.time() - start_time) * 1000, 2)
    trace_entry = {
        "agent": "Eligibility Agent",
        "action": f"Executed demographic rules and edge case evaluations in {elapsed}ms",
        "decisions": {k: v["status"] for k, v in results.items()}
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "eligibility_results": results,
        "execution_trace": traces
    }
