# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Freightcase: an extensible LangGraph agent that turns logistics emails into structured, human-confirmed actions against the adopter's own TMS, via MCP tools. Portfolio project, not a product: one flow done excellently over many flows done partially.

v1 scope: quote request flow end to end, booking instruction second. No third specialist. No .msg files, no charset heroics, one attachment type (PDF).

## Commands

```bash
uv run pytest                                          # full suite (~35s: extraction tests call a live LLM)
uv run pytest tests/test_intake.py tests/test_schemas.py   # fast, no LLM needed
uv run pytest tests/test_intake.py::test_parse_eml_no_body # single test
uv run python scripts/timing.py                        # latency probe for extraction
```

Python 3.12, managed by uv (`uv sync` to set up). No linter is configured.

`tests/test_extraction.py` is a live integration test against Ollama: it needs a running Ollama server with the model pulled. Configure via `.env`: `AGENT_MODEL` (default `gemma4:31b`) and `OLLAMA_BASE_URL` (default `http://localhost:11434`).

## Architecture

Target pipeline: `.eml` in → intake (deterministic MIME parsing) → classifier routes to a specialist subgraph → LLM extraction against a Pydantic schema → `interrupt()` for human confirmation → on resume, execute MCP tool call.

Specialists are self-contained subgraphs registered behind a small registry; adding a function = new schema + subgraph + eval set, zero edits to existing code.

(As of now only intake and extraction exist; graph, classifier, and HITL wiring are upcoming work.)

### Layout

- `src/freightcase/schemas.py` — extraction contracts only (serialized into LLM prompts; validators interpret output)
- `src/freightcase/intake.py` — `parse_eml`, `IntakeResult` (pipeline state, never shown to the model)
- `src/freightcase/extraction.py` — `extract_quote_request`, `ExtractionError`
- `src/freightcase/llm.py` — model construction, timing helpers
- `src/freightcase/graph.py` — state, nodes, wiring only; nodes are thin wrappers over plain functions
- `tests/` at root, src-layout imports through the installed package

## Core design rules (enforce these in all code)

1. **The model transcribes evidence; code interprets it.** The LLM emits values exactly as stated in the email. All normalization (unit aliases, incoterm variants, LOCODE resolution) lives in deterministic Pydantic validators that fail loudly on unrecognized input. Never move interpretation into prompts.
2. **Absence is representable, never synthesized.** Fields the email may not state are `X | None = None` (annotation AND default — `| None` without `= None` is still required and pressures the model to fabricate). Completeness is checked separately and deterministically (`missing_for_quoting()`), feeding HITL per-field flags.
3. **Fatal vs degraded.** Errors (typed: `ExtractionError`, `IntakeError`) are for "processing is impossible" (unparseable message, no body, invalid JSON). Everything else proceeds with the gap recorded in a warnings list. Original values are preserved; canonical forms are computed fields (e.g. `weight.kg`), never overwrites.
4. **HITL gates every write.** Mutations pause via `interrupt()` with a standardized payload (proposed action, fields, per-field confidence, what confirm executes); resume via checkpointer + `Command(resume=...)`.
5. **No new behavior without an eval or test case pinning it.** Fixtures live in `tests/fixtures/emails/`, each earns its place with one distinct failure mode. Partial assertions, not golden-object equality.

### Implementation notes (repo-specific)

- Extraction deliberately does **not** use LangChain's `with_structured_output`: the schema goes in the system prompt and the response is parsed/validated in our code so `ValidationError` details survive inside `ExtractionError` (`.raw`, `.validation_error`) for the repair loop and per-field HITL confidence.
- Email content is untrusted: it is passed only as the `HumanMessage`, never interpolated into instructions.
- Attachment metadata carries a `content_ref` path, never the bytes — keep state light.

## Conventions

- pytest with parametrize; LLM calls injectable (`model: BaseChatModel | None = None`) so tests/evals swap models
- Local dev model: Ollama gemma4:31b with `reasoning=False, format="json", temperature=0` (reasoning-on silently consumed the whole token budget; 44.7s → 5.4s). Eval benchmark target: API Haiku.
- The cargo list field on `QuoteRequest` is named `cargo`
- Fixtures use invented parties and .example domains only; no real customer data ever, including in git history
- Fixture-driven tests with partial expectations: each test module has a `CASES` dict asserting only the fields that matter per scenario. `.eml` fixtures exercise intake, `.txt` fixtures exercise extraction. Header/edge-case tests mutate a loaded `.eml` in memory rather than adding fixture variants.
