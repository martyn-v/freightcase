from email import policy
import email
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


def parse_eml(raw: bytes) -> IntakeResult:
    """Parse a raw .eml file into an IntakeResult.

    Args:
        raw: The raw bytes of the .eml file.

    Returns:
        An IntakeResult object containing the parsed email data.
    """
    msg = email.message_from_bytes(raw, policy=policy.default)

    warnings = []

    message_id = msg.get("message-id", "")

    sender = msg.get("from")
    if sender is None or sender.addresses is None or len(sender.addresses) == 0:
        warnings.append("Missing or invalid From header")
    else:
        sender = EmailAddress(
            display_name=sender.addresses[0].display_name,
            address=sender.addresses[0].addr_spec,
        )

    subject = msg.get("subject")
    if subject is None:
        warnings.append("Missing Subject header")

    date = msg.get("date")
    if date is not None:
        try:
            date = parsedate_to_datetime(date)
        except Exception:
            warnings.append(f"Invalid Date header: {date}")
            date = None
    else:
        warnings.append("Missing Date header")

    body = msg.get_body(preferencelist=("plain", "html"))
    if not body:
        raise IntakeError("No suitable body part found in the email")

    body_text = None
    if body.get_content_type() == "text/plain":
        body_text = body.get_content()
    elif body.get_content_type() == "text/html":
        body_text = BeautifulSoup(body.get_content(), "html.parser").get_text()
    else:
        raise IntakeError("No suitable body part found in the email")

    attachments = []

    return IntakeResult(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body_text=body_text,
        attachments=attachments,
        intake_warnings=warnings,
    )
