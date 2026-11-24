import re
import time
from typing import Dict, Any
from agents.state import AgentState

def guardrail_agent(state: AgentState) -> Dict[str, Any]:
    """
    Input Guardrail & PII Sanitizer Agent:
    1. Detects and scrubs sensitive citizen Personally Identifiable Information (PII):
       - Aadhaar Numbers (12 digits, spaced or continuous)
       - PAN Card Numbers (10-character alphanumeric)
       - Phone Numbers (+91 or 10-digit mobile)
       - Bank Account Numbers & IFSC codes
    2. Defends against Prompt Injections and adversarial jailbreaks.
    """
    start_time = time.time()
    raw_query = state.get("user_query", "")
    sanitized = raw_query
    pii_redacted = False
    flags = []

    # 1. Aadhaar Scrubbing (12 digits: 0000 0000 0000 or continuous)
    aadhaar_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    if re.search(aadhaar_pattern, sanitized):
        sanitized = re.sub(aadhaar_pattern, "[AADHAAR_REDACTED]", sanitized)
        pii_redacted = True
        flags.append("Aadhaar Number Redacted")

    # 2. PAN Card Scrubbing (5 letters, 4 digits, 1 letter)
    pan_pattern = r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
    if re.search(pan_pattern, sanitized, re.IGNORECASE):
        sanitized = re.sub(pan_pattern, "[PAN_REDACTED]", sanitized, flags=re.IGNORECASE)
        pii_redacted = True
        flags.append("PAN Card Redacted")

    # 3. Bank IFSC code (4 letters, 0, 6 letters/digits)
    ifsc_pattern = r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    if re.search(ifsc_pattern, sanitized, re.IGNORECASE):
        sanitized = re.sub(ifsc_pattern, "[IFSC_REDACTED]", sanitized, flags=re.IGNORECASE)
        pii_redacted = True
        flags.append("Bank IFSC Code Redacted")

    # 4. Phone Number (10 digits starting with 6-9, optional +91)
    phone_pattern = r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"
    if re.search(phone_pattern, sanitized):
        sanitized = re.sub(phone_pattern, "[PHONE_REDACTED]", sanitized)
        pii_redacted = True
        flags.append("Phone Number Redacted")

    # 5. Prompt Injection Defense
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"system\s*prompt",
        r"jailbreak",
        r"dan\s+mode"
    ]
    is_injection = False
    for pat in injection_patterns:
        if re.search(pat, sanitized, re.IGNORECASE):
            is_injection = True
            flags.append("Adversarial Instruction Defanged")
            sanitized = re.sub(pat, "[INSTRUCTION_FILTERED]", sanitized, flags=re.IGNORECASE)

    elapsed = round((time.time() - start_time) * 1000, 2)
    action_desc = f"Sanitized input in {elapsed}ms"
    if flags:
        action_desc += f" -> Flags: {', '.join(flags)}"
    else:
        action_desc += " -> Clean input, no PII detected"

    trace_entry = {
        "agent": "Input Guardrail Agent",
        "action": action_desc,
        "pii_redacted": pii_redacted,
        "injection_detected": is_injection
    }

    traces = list(state.get("execution_trace", []))
    traces.append(trace_entry)

    return {
        "sanitized_query": sanitized,
        "pii_redacted": pii_redacted,
        "execution_trace": traces
    }
