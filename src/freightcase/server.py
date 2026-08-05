"""Reference HTTP deployment of the freightcase graph.

A small FastAPI app over the compiled graph with a persistent SQLite
checkpointer: an uploaded .eml opens a case (POST /ingest/), paused cases
are listed (GET /pending/) and decided (POST /confirm/{thread_id}). The
checkpointer makes the HITL pause durable — the process can restart
between ingestion and confirmation.

Run it:

    uv run fastapi dev src/freightcase/server.py

Interactive API docs are served at /docs. Set FREIGHTCASE_CHECKPOINT_DB
to move the checkpoint database (default: logs/checkpoints.sqlite).
See README.md ("Serving the pipeline") for the tour and the limitations.
"""

import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from mcp import StdioServerParameters
from pydantic import BaseModel

from freightcase.contracts import ConfirmationPayload, ConfirmationResume
from freightcase.execution import MCPToolExecutor
from freightcase.graph import build_graph, checkpoint_serde, pending_interrupts
from freightcase.intake import IntakeError

DB_PATH = os.environ.get("FREIGHTCASE_CHECKPOINT_DB", "logs/checkpoints.sqlite")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Megabytes
# Browsers and curl send .eml uploads as application/octet-stream by default.
ALLOWED_MIME_TYPES = {"message/rfc822", "application/octet-stream"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once per process, over a durable checkpointer.

    check_same_thread=False is required: sync endpoints run in FastAPI's
    threadpool, so multiple threads share this connection. SqliteSaver
    serializes access with its own lock. The serde allowlist keeps state
    models rehydrating as models, not dicts (see graph.checkpoint_serde).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=checkpoint_serde())
    checkpointer.setup()
    # Approvals run the full MCP path: each execute call spawns the stub
    # TMS for one tool call. Point server_command at the adopter's TMS
    # MCP server to integrate. MCPToolExecutor uses asyncio.run, which
    # needs these sync (threadpool) endpoints; an async endpoint would
    # crash it with a nested-loop RuntimeError.
    app.state.graph = build_graph(
        executor=MCPToolExecutor(
            server_command=StdioServerParameters(
                command="uv", args=["run", "python", "-m", "freightcase.tms_stub"]
            )
        ),
        checkpointer=checkpointer,
    )
    yield
    conn.close()


app = FastAPI(lifespan=lifespan)


class PausedCase(BaseModel):
    """A case stopped at the HITL gate; decide it via /confirm/{thread_id}."""

    thread_id: str
    status: Literal["awaiting_confirmation"] = "awaiting_confirmation"
    payload: ConfirmationPayload


class IngestCompleted(BaseModel):
    """The run finished without pausing. HITL gates every write (rule 4),
    so an unpaused run can only be a dead letter or a failed extraction —
    'failed' is the one reachable status, and the Literal keeps that
    invariant loud if the wiring ever changes."""

    thread_id: str
    status: Literal["failed"]
    error: str | None


class ConfirmCompleted(BaseModel):
    """The run finished after a decision: executed against the TMS,
    rejected by the human, or failed in the TMS write itself."""

    thread_id: str
    status: Literal["executed", "rejected", "failed"]
    execution_ref: str | None
    error: str | None


def _paused(thread_id: str, interrupt_value: dict) -> PausedCase:
    """Map a raw interrupt value to the typed pause response.

    The one home for the invariant that a pause value is a dumped
    ConfirmationPayload; /ingest, /confirm, and /pending all route
    through it, so the contract cannot drift per endpoint.
    """
    return PausedCase(
        thread_id=thread_id,
        payload=ConfirmationPayload.model_validate(interrupt_value),
    )


def _validate_eml_upload(file: UploadFile) -> None:
    """Reject uploads that are not a plausibly-sized .eml. Transport-level
    checks only; whether the bytes parse as email is intake's verdict."""
    if not (file.filename or "").lower().endswith(".eml"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported extension: only .eml files are accepted.",
        )
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type: '{file.content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )
    if (file.size or 0) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE} bytes.",
        )


@app.post("/ingest/")
def ingest_email(request: Request, file: UploadFile) -> PausedCase | IngestCompleted:
    """Open a case from an uploaded .eml and run it to its first stop:
    paused at the HITL gate, or completed as a dead letter / failure.
    An unparseable email is a 422; the case never opens."""
    _validate_eml_upload(file)

    thread_id = str(uuid.uuid4())
    # Graph state carries a file path, never bytes; bridge the upload
    # through a temp file that lives only for this first invoke.
    temp_filename = os.path.join(tempfile.gettempdir(), thread_id)
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        result = request.app.state.graph.invoke(
            {"eml_file_path": temp_filename}, config=config
        )
    except IntakeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Ingestion failed: {e}",
        ) from e
    finally:
        # Resume replays intake from the checkpoint, not the file, so the
        # temp copy can go immediately.
        os.remove(temp_filename)

    if "__interrupt__" in result:
        (pause,) = result["__interrupt__"]
        return _paused(thread_id, pause.value)

    # No pause: dead-lettered unknown or failed extraction; finalize
    # graduated the outcome into results.
    final = result["results"][-1]
    return IngestCompleted(thread_id=thread_id, status=final.status, error=final.error)


@app.post("/confirm/{thread_id}")
def confirm_case(
    request: Request, thread_id: str, resume: ConfirmationResume
) -> PausedCase | ConfirmCompleted:
    """Answer a paused case: approve (with optional edits) or reject.

    Validation is two-layered by design. Bodies that fail the
    ConfirmationResume shape stop here as a 422 and never reach the
    graph; semantically bad answers (invalid edits, remaining gaps)
    come back as a re-prompt with `problems` explaining why.
    """
    graph = request.app.state.graph
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Covers unknown threads and already-decided cases in one check;
    # resuming a thread with nothing to resume would raise deep in
    # LangGraph instead of 404ing.
    if not graph.get_state(config).interrupts:
        raise HTTPException(status_code=404, detail="No pending confirmation")

    # Resume with a plain dict: LangGraph caches the resume value inside
    # the checkpoint, and ConfirmationResume is not serde-allowlisted.
    result = graph.invoke(Command(resume=resume.model_dump()), config=config)

    if "__interrupt__" in result:  # re-prompt: bad edit or gaps remain
        (pause,) = result["__interrupt__"]
        return _paused(thread_id, pause.value)

    final = result["results"][-1]
    return ConfirmCompleted(
        thread_id=thread_id,
        status=final.status,
        execution_ref=final.execution_ref,
        error=final.error,
    )


@app.get("/pending/")
def get_pending_cases(request: Request) -> list[PausedCase]:
    """List every case paused at the HITL gate, newest first.

    Scans the full checkpoint history on each call (see
    graph.pending_interrupts) — a deliberate limitation at this scope.
    """
    return [
        _paused(thread_id, value)
        for thread_id, value in pending_interrupts(request.app.state.graph)
    ]
