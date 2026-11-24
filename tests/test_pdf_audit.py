import base64
import unittest
from fastapi.testclient import TestClient

from backend.main import app
from rag.vector_store import index_uploaded_pdfs, get_session_info, delete_session_store
from agents.graph import run_pipeline
from core.utils import audit_text_against_source


def make_test_pdf_bytes(text_lines):
    """Creates a minimal, valid single-page PDF containing the given lines of text."""
    escaped = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        for line in text_lines
    ]
    ops = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
    for i, line in enumerate(escaped):
        if i == 0:
            ops.append(f"({line}) Tj")
        else:
            ops.append(f"T* ({line}) Tj")
    ops.append("ET")
    stream_data = "\n".join(ops).encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        (
            f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return bytes(pdf)


class TestUploadedPdfAudit(unittest.TestCase):
    def setUp(self):
        self.sample_pdf = make_test_pdf_bytes([
            "National Solar Rooftop Subsidy Scheme 2026 Guidelines.",
            "Eligible residential households receive Rs 30000 subsidy per kW up to 2 kW.",
            "Commercial buildings and government offices are excluded from residential subsidy.",
            "Applicants must submit electricity bill and Aadhaar verification on the national portal.",
        ])

    def test_ephemeral_session_pipeline_and_hallucination_scan(self):
        info = index_uploaded_pdfs([("Solar_Subsidy_2026.pdf", self.sample_pdf)])
        sid = info["session_id"]
        try:
            self.assertEqual(info["storage_mode"], "session_ephemeral")
            self.assertEqual(info["total_pages"], 1)
            self.assertGreaterEqual(info["total_chunks"], 1)
            self.assertIsNotNone(get_session_info(sid))

            pipe_res = run_pipeline(
                user_query="What subsidy amount and exclusions apply in this solar scheme?",
                user_profile={},
                session_id=sid,
            )
            self.assertTrue(pipe_res.get("uploaded_pdf_mode"))
            self.assertEqual(pipe_res.get("eligibility_results"), {})
            self.assertIn("Solar_Subsidy_2026.pdf", pipe_res.get("final_answer", ""))
            self.assertEqual(pipe_res["hallucination_report"]["hallucination_count"], 0)

            audit_res = audit_text_against_source(
                text=(
                    "Eligible residential households receive Rs 30000 subsidy per kW up to 2 kW. "
                    "The government gives a free electric car to every solar applicant."
                ),
                session_id=sid,
            )
            self.assertEqual(audit_res["total_claims"], 2)
            self.assertEqual(audit_res["entailed_claims"], 1)
            self.assertEqual(audit_res["hallucination_count"], 1)
        finally:
            self.assertTrue(delete_session_store(sid))
            self.assertIsNone(get_session_info(sid))

    def test_fastapi_pdf_session_endpoints(self):
        client = TestClient(app)
        b64_pdf = base64.b64encode(self.sample_pdf).decode("ascii")

        up_resp = client.post(
            "/api/pdf/upload",
            json={"filename": "Solar_Subsidy_2026.pdf", "content_base64": b64_pdf},
        )
        self.assertEqual(up_resp.status_code, 200, up_resp.text)
        api_sid = up_resp.json()["session_id"]

        q_resp = client.post(
            "/api/query",
            json={"query": "Who is excluded from residential subsidy?", "session_id": api_sid},
        )
        self.assertEqual(q_resp.status_code, 200, q_resp.text)
        self.assertIn("Solar_Subsidy_2026.pdf", q_resp.json()["final_answer"])

        audit_resp = client.post(
            "/api/pdf/audit-text",
            json={
                "text": "Commercial buildings are excluded from residential subsidy. Monthly Rs 90000 cash is paid.",
                "session_id": api_sid,
            },
        )
        self.assertEqual(audit_resp.status_code, 200, audit_resp.text)
        self.assertGreaterEqual(audit_resp.json()["hallucination_count"], 1)

        del_resp = client.delete(f"/api/pdf/{api_sid}")
        self.assertEqual(del_resp.status_code, 200)
        get_after_del = client.get(f"/api/pdf/{api_sid}")
        self.assertEqual(get_after_del.status_code, 404)


if __name__ == "__main__":
    unittest.main()
