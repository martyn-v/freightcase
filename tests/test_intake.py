import email
from email import policy
import pytest
from pathlib import Path
from freightcase.intake import IntakeError
from freightcase.intake import parse_eml

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def load(name: str) -> bytes:
    return (FIXTURES / f"{name}.eml").read_bytes()


CASES = {
    "quote_road_plain_es": {
        "message_id": "<20260729142700.AB123@agroexpressco.example.com>",
        "subject": "Solicitud de oferta - transporte terrestre Bogotá a Medellín",
        "sender_display_name": "Camila Torres",
        "sender_address": "ctorres@agroexpressco.example.com",
        "date": "2025-07-29T13:27:00",
        "body_str": "Buenas tardes,\n\nPor favor tu mejor oferta para el siguiente envío terrestre:",
    },
    "quote_lcl_htmlonly_en": {
        "subject": "Rate request - LCL Hamburg to Veracruz",
        "sender_display_name": "Jonas Weber",
        "sender_address": "j.weber@nordkontor.example.de",
        "date": "2025-07-30T13:27:00",
        "body_str": "Dear team,\nWe would appreciate your best rate",
    },
    "quote_fcl_with_attachment_en": {
        "subject": "FCL quote needed - see attached packing list",
        "attachments_0_filename": "packing_list_SAE-2026-0812.pdf",
        "attachments_0_content_type": "application/pdf",
        "attachments_0_size_bytes": 352,
        "attachments_0_content_ref": "attachments/packing_list_SAE-2026-0812.pdf",
        "attachments_0_extracted_text": None,
    },
}


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_parse_eml(name):
    raw = load(name)
    result = parse_eml(raw)

    if "message_id" in CASES[name]:
        assert result.message_id == CASES[name]["message_id"]
    if "subject" in CASES[name]:
        assert result.subject == CASES[name]["subject"]
    if "sender_display_name" in CASES[name]:
        assert result.sender is not None
        assert result.sender.display_name == CASES[name]["sender_display_name"]
    if "sender_address" in CASES[name]:
        assert result.sender is not None
        assert result.sender.address == CASES[name]["sender_address"]
    if "date" in CASES[name]:
        assert result.date is not None
        assert result.date.isoformat() == CASES[name]["date"]
    if "body_str" in CASES[name]:
        assert result.body_text.strip().startswith(CASES[name]["body_str"].strip())

    if "attachments_0_filename" in CASES[name]:
        assert len(result.attachments) > 0
        assert result.attachments[0].filename == CASES[name]["attachments_0_filename"]
        assert (
            result.attachments[0].content_type
            == CASES[name]["attachments_0_content_type"]
        )
        assert (
            result.attachments[0].size_bytes == CASES[name]["attachments_0_size_bytes"]
        )
        assert (
            result.attachments[0].content_ref
            == CASES[name]["attachments_0_content_ref"]
        )
        assert (
            result.attachments[0].extracted_text
            == CASES[name]["attachments_0_extracted_text"]
        )


def test_parse_eml_missing_subject():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the Subject header to simulate a missing subject
    del msg["Subject"]

    result = parse_eml(msg.as_bytes())
    assert "Missing Subject header" in result.intake_warnings


def test_parse_eml_missing_date():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the Date header to simulate a missing date
    del msg["Date"]

    result = parse_eml(msg.as_bytes())
    assert "Missing Date header" in result.intake_warnings


def test_parse_eml_invalid_date():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Set an invalid Date header to simulate an invalid date
    msg.replace_header("Date", "invalid-date")

    result = parse_eml(msg.as_bytes())
    assert "Invalid Date header" in result.intake_warnings


def test_parse_email_missing_from():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the From header to simulate a missing sender
    del msg["From"]

    result = parse_eml(msg.as_bytes())
    assert "Missing or invalid From header" in result.intake_warnings


def test_parse_eml_empty_body():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Blank the body while keeping the text/plain part in place
    msg.set_content("")

    result = parse_eml(msg.as_bytes())
    assert "Empty body" in result.intake_warnings
    assert result.body_text.strip() == ""


def test_parse_eml_no_body():
    raw = load("quote_road_plain_es")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Re-type the only part so the email has no plain/html body part
    msg.replace_header("Content-Type", "application/octet-stream")

    with pytest.raises(IntakeError) as excinfo:
        parse_eml(msg.as_bytes())
    assert "No suitable body part found in the email" in str(excinfo.value)
