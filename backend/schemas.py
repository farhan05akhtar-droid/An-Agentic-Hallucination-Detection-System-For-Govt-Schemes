from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UserProfileSchema(BaseModel):
    state: str = Field(default="Odisha", description="Beneficiary state of residence")
    occupation: str = Field(default="Farmer", description="Primary occupation")
    annual_income: float = Field(default=150000.0, description="Gross annual household income in INR")
    land_holding: float = Field(default=1.5, description="Total landholding size in acres")
    is_cultivable: bool = Field(default=True, description="Whether the landholding is cultivable")
    is_urban: bool = Field(default=False, description="Whether the agricultural land is in an urban area")
    pays_income_tax: bool = Field(default=False, description="Whether applicant or family pays income tax")
    is_institutional: bool = Field(default=False, description="Whether land is owned by an institution")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language civic query")
    profile: Optional[UserProfileSchema] = Field(default_factory=UserProfileSchema)
    session_id: Optional[str] = Field(default=None, description="Optional temporary uploaded PDF session ID")


class PdfUploadRequest(BaseModel):
    filename: str = Field(..., description="Original PDF filename")
    content_base64: str = Field(..., description="Base64-encoded PDF file bytes")
    session_id: Optional[str] = Field(default=None, description="Optional existing session ID to replace")


class PdfPreviewChunk(BaseModel):
    source: str
    page: int
    excerpt: str


class PdfUploadResponse(BaseModel):
    session_id: str
    filenames: List[str]
    total_pages: int
    total_chunks: int
    storage_mode: str = "session_ephemeral"
    preview_chunks: List[PdfPreviewChunk] = []


class AuditTextRequest(BaseModel):
    text: str = Field(..., description="Draft text or claim(s) to audit for hallucinations")
    session_id: Optional[str] = Field(default=None, description="Optional temporary uploaded PDF session ID")

class ClaimAuditItem(BaseModel):
    claim: str
    status: str
    entailment_score: str
    ground_truth_doc: str
    evidence: str

class HallucinationReportSchema(BaseModel):
    faithfulness_score: str
    total_claims: int
    entailed_claims: int
    hallucination_count: int
    claims_table: List[ClaimAuditItem]
    unverified_claims: List[str]
    decision: str
    audit_duration_ms: float


class AuditTextResponse(BaseModel):
    faithfulness_score: str
    total_claims: int
    entailed_claims: int
    hallucination_count: int
    claims_table: List[ClaimAuditItem]
    unverified_claims: List[str]
    audit_duration_ms: float

class QueryResponse(BaseModel):
    user_query: str
    sanitized_query: Optional[str] = None
    pii_redacted: Optional[bool] = False
    intent: Optional[str] = "eligibility"
    final_answer: str
    eligibility_results: Dict[str, Any]
    hallucination_report: HallucinationReportSchema
    citation_report: Optional[Dict[str, Any]] = None
    context_metadata: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
    total_latency_seconds: float
    latency_status: str

