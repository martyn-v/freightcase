from email import policy
import email
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pydantic import BaseModel

from datetime import datetime
from bs4 import BeautifulSoup


class IntakeError(Exception):
    """Raised when there is an error parsing the .eml file."""


class EmailAddress(BaseModel):
    display_name: str | None
    address: str  # the actual email address, parsed


class EmailAttachment(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    content_ref: str  # path or key, NOT the bytes — keep state light
    extracted_text: str | None = None  # filled on Day 3 for PDFs


class IntakeResult(BaseModel):
    message_id: str | None = None
    subject: str | None = None
    sender: EmailAddress | None = None  # display name + address, parsed
    date: datetime | None = None
    body_text: str  # the cleaned, extraction-ready text
    attachments: list[EmailAttachment]
    intake_warnings: list[str]  # everything weird, surfaced not swallowed


def _parse_sender(msg: EmailMessage, warnings: list[str]) -> EmailAddress | None:
    sender = msg.get("from")
    if sender is None or sender.addresses is None or len(sender.addresses) == 0:
        warnings.append("Missing or invalid From header")
        return None
    return EmailAddress(
        display_name=sender.addresses[0].display_name,
        address=sender.addresses[0].addr_spec,
    )


def _parse_subject(msg: EmailMessage, warnings: list[str]) -> str | None:
    subject = msg.get("subject")
    if subject is None:
        warnings.append("Missing Subject header")
    return subject


def _parse_date(msg: EmailMessage, warnings: list[str]) -> datetime | None:
    date = msg.get("date")
    if date is None:
        warnings.append("Missing Date header")
        return None
    try:
        return parsedate_to_datetime(date)
    except Exception:
        warnings.append("Invalid Date header")
        return None


def _extract_body_text(msg: EmailMessage, warnings: list[str]) -> str:
    body = msg.get_body(preferencelist=("plain", "html"))
    if not body:
        raise IntakeError("No suitable body part found in the email")

    if body.get_content_type() == "text/html":
        body_text = BeautifulSoup(body.get_content(), "html.parser").get_text().strip()
    else:
        body_text = body.get_content().strip()

    if not body_text:
        warnings.append("Empty body")
    return body_text


def _extract_attachments(msg: EmailMessage) -> list[EmailAttachment]:
    attachments = []
    for part in msg.iter_attachments():
        filename = part.get_filename()
        attachments.append(
            EmailAttachment(
                filename=filename or "unknown",
                content_type=part.get_content_type(),
                size_bytes=len(part.get_content()),
                content_ref=f"attachments/{filename}"
                if filename
                else "attachments/unknown",
                extracted_text=None,  # Placeholder for future PDF text extraction
            )
        )
    return attachments


def parse_eml(raw: bytes) -> IntakeResult:
    """Parse a raw .eml file into an IntakeResult.

    Args:
        raw: The raw bytes of the .eml file.

    Returns:
        An IntakeResult object containing the parsed email data.
    """
    msg = email.message_from_bytes(raw, policy=policy.default)

    warnings: list[str] = []
    sender = _parse_sender(msg, warnings)
    subject = _parse_subject(msg, warnings)
    date = _parse_date(msg, warnings)
    body_text = _extract_body_text(msg, warnings)

    return IntakeResult(
        message_id=msg.get("message-id"),
        subject=subject,
        sender=sender,
        date=date,
        body_text=body_text,
        attachments=_extract_attachments(msg),
        intake_warnings=warnings,
    )
