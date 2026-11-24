import base64
import importlib.util
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from backend.schemas import (
    QueryRequest,
    QueryResponse,
    ClaimAuditItem,
    PdfUploadRequest,
    PdfUploadResponse,
    AuditTextRequest,
    AuditTextResponse,
)
from agents.graph import run_pipeline
from core.utils import calculate_nli_entailment, audit_text_against_source
from rag.vector_store import (
    get_retriever,
    index_uploaded_pdfs,
    get_session_info,
    delete_session_store,
)

router = APIRouter()

MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB per uploaded PDF

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "GovScheme AI Multi-Agent System",
        "hallucination_engine": "Active (NLI + Deterministic Judge)",
        "vector_store": "ChromaDB (Local SQLite)",
        "target_faithfulness": ">= 98%"
    }

@router.get("/schemes")
def get_supported_schemes():
    return {
        "schemes": [
            {
                "id": "PM-KISAN",
                "name": "Pradhan Mantri Kisan Samman Nidhi",
                "level": "Central / All India",
                "benefit": "₹6,000 / year in 3 installments",
                "rules": "Cultivable land ownership; micro-holdings admissible; urban agricultural land treated equally."
            },
            {
                "id": "KALIA",
                "name": "Krushak Assistance for Livelihood and Income Augmentation",
                "level": "State (Odisha)",
                "benefit": "₹10,000 / year for cultivation support; ₹12,500 for landless livelihoods",
                "rules": "Odisha residents only; small/marginal farmers (<= 5 acres); landless agricultural households."
            }
        ]
    }

@router.post("/query", response_model=QueryResponse)
def query_advisory(request: QueryRequest):
    try:
        profile_dict = request.profile.model_dump() if request.profile else {}
        result = run_pipeline(
            user_query=request.query,
            user_profile=profile_dict,
            session_id=request.session_id,
        )
        return QueryResponse(
            user_query=request.query,
            sanitized_query=result.get("sanitized_query", request.query),
            pii_redacted=result.get("pii_redacted", False),
            intent=result.get("intent", "eligibility"),
            final_answer=result.get("final_answer", ""),
            eligibility_results=result.get("eligibility_results", {}),
            hallucination_report=result.get("hallucination_report", {}),
            citation_report=result.get("citation_report", {}),
            context_metadata=result.get("context_metadata", []),
            execution_trace=result.get("execution_trace", []),
            total_latency_seconds=result.get("total_latency_seconds", 0.0),
            latency_status=result.get("latency_status", "N/A")
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


@router.post("/pdf/upload", response_model=PdfUploadResponse)
def upload_pdf_base64(request: PdfUploadRequest):
    """
    Accepts a base64-encoded PDF and indexes it into a temporary, session-only
    in-memory vector store for grounded Q&A and hallucination verification.
    """
    if not request.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    try:
        pdf_bytes = base64.b64decode(request.content_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF payload.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 20 MB limit.")

    try:
        info = index_uploaded_pdfs(
            files=[(request.filename, pdf_bytes)],
            session_id=request.session_id,
        )
        return PdfUploadResponse(**info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF indexing failed: {str(e)}")


if importlib.util.find_spec("multipart") is not None:
    from fastapi import File, Form, UploadFile

    @router.post("/pdf/upload-file", response_model=PdfUploadResponse)
    async def upload_pdf_multipart(
        files: List[UploadFile] = File(...),
        session_id: Optional[str] = Form(default=None),
    ):
        """Multipart PDF upload endpoint (available when python-multipart is installed)."""
        payloads = []
        for f in files:
            if not (f.filename or "").lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Unsupported file '{f.filename}'. Only .pdf allowed.")
            data = await f.read()
            if len(data) > MAX_PDF_BYTES:
                raise HTTPException(status_code=413, detail=f"'{f.filename}' exceeds 20 MB limit.")
            payloads.append((f.filename or "uploaded.pdf", data))
        try:
            info = index_uploaded_pdfs(files=payloads, session_id=session_id)
            return PdfUploadResponse(**info)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF indexing failed: {str(e)}")


@router.get("/pdf/{session_id}")
def get_uploaded_pdf_session(session_id: str):
    info = get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return info


@router.delete("/pdf/{session_id}")
def clear_uploaded_pdf_session(session_id: str):
    deleted = delete_session_store(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or already cleared.")
    return {"status": "cleared", "session_id": session_id}


@router.post("/pdf/audit-text", response_model=AuditTextResponse)
def audit_text_endpoint(request: AuditTextRequest):
    """
    Splits a pasted text/draft into atomic claims and checks each claim for hallucinations
    against the uploaded PDF session (or default scheme DB if session_id is omitted).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to audit cannot be empty.")
    try:
        report = audit_text_against_source(text=request.text, session_id=request.session_id)
        return AuditTextResponse(**report)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit error: {str(e)}")


@router.post("/audit-claim", response_model=ClaimAuditItem)
def audit_single_claim(claim: str, session_id: Optional[str] = None):
    """
    Standalone endpoint to audit any individual factual proposition against indexed scheme PDFs
    or a temporary uploaded PDF session.
    """
    try:
        retriever = get_retriever(k=3, session_id=session_id)
        docs = retriever.invoke(claim)
        chunks = [d.page_content for d in docs]
        metas = []
        for d in docs:
            m = dict(d.metadata)
            m["page"] = int(m.get("page", 0)) + 1
            metas.append(m)

        nli_res = calculate_nli_entailment(claim, chunks, metas)
        return ClaimAuditItem(
            claim=claim,
            status=f"✅ {nli_res['verdict']}" if nli_res['verdict'] == "ENTAILMENT" else f"⚠️ {nli_res['verdict']}",
            entailment_score=f"{int(nli_res['entailment_score'] * 100)}%",
            ground_truth_doc=nli_res["ground_truth_doc"],
            evidence=nli_res["evidence_quote"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
