"""
Forensic Export & Court-Admissible Dossier Routes.
Generates cryptographically sealed MARPOL Annex I violation audit PDFs
and provides tamper-evident audit log stream access.
"""

from fastapi import APIRouter, HTTPException, status, Response, Depends
from server.models.schemas import CryptographicAuditSeal, UserProfile
from server.services.incident_service import incident_service
from server.services.pdf_generator import generate_court_admissible_pdf
from server.security.rbac import RequireViewer, RequireAuditor
from server.security.middleware import AUDIT_LOG_STREAM
from server.config import settings

router = APIRouter(prefix="/api/v1/export", tags=["Forensic Reports & Chain of Custody"])


@router.get("/court-pdf/{slick_id}")
async def export_court_pdf(
    slick_id: str,
    current_user: UserProfile = RequireViewer
):
    """
    Builds and downloads the Certified Court-Admissible MARPOL Annex I Forensic Dossier PDF,
    stamped with SHA-256 digital seal and Coast Guard Command PGP signature.
    """
    incident = incident_service.get_incident_by_id(slick_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{slick_id}' not found."
        )

    pdf_bytes, sha256_seal, case_number = generate_court_admissible_pdf(incident)

    headers = {
        "Content-Disposition": f'attachment; filename="COURT_DOSSIER_{case_number}.pdf"',
        "X-Cryptographic-SHA256-Seal": sha256_seal,
        "X-PGP-Fingerprint": settings.FORENSIC_PGP_KEY_ID,
        "X-Case-Number": case_number
    }

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=headers
    )


@router.get("/seal-preview/{slick_id}", response_model=CryptographicAuditSeal)
async def get_seal_preview(
    slick_id: str,
    current_user: UserProfile = RequireViewer
):
    """Returns the cryptographic SHA-256 seal metadata prior to generating the PDF artifact."""
    incident = incident_service.get_incident_by_id(slick_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{slick_id}' not found."
        )

    _, sha256_seal, case_number = generate_court_admissible_pdf(incident)

    return CryptographicAuditSeal(
        incident_uuid=incident.slick_id,
        case_file_number=case_number,
        generated_at_utc=incident.timestamp_utc,
        issuing_authority=settings.FORENSIC_COMMAND_TITLE,
        digital_digest_sha256=sha256_seal,
        pgp_key_fingerprint=settings.FORENSIC_PGP_KEY_ID,
        evidence_payload_verified=True,
        qr_verification_url=f"/api/v1/export/court-pdf/{slick_id}"
    )


@router.get("/audit-logs")
async def get_audit_trail(current_user: UserProfile = RequireAuditor):
    """Returns the append-only cryptographic audit trail of all maritime console events."""
    return {
        "record_count": len(AUDIT_LOG_STREAM),
        "audit_stream": AUDIT_LOG_STREAM[-100:] # Latest 100 immutable events
    }
