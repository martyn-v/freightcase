# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Freightcase: an extensible LangGraph agent that turns logistics emails into structured, human-confirmed actions against the adopter's own TMS, via MCP tools. Portfolio project, not a product: one flow done excellently over many flows done partially.

v1 scope: quote request flow end to end, booking instruction second. No third specialist. No .msg files, no charset heroics, one attachment type (PDF).

## Commands

```bash
uv run pytest                                          # full suite (~50s: extraction/graph tests call a live LLM)
uv run pytest tests/test_intake.py tests/specialists tests/test_contracts.py   # fast, no LLM
uv run pytest tests/test_intake.py::test_parse_eml_no_body # single test
uv run ruff check src tests scripts && uv run ruff format src tests scripts    # lint + format
uvx pyright --pythonpath .venv/bin/python src tests    # type check (venv-aware)
uv run python scripts/run_evals.py                     # local evals (stages: classification/extraction/pipeline)
uv run langgraph dev                                   # LangGraph Studio (module graph has no checkpointer; server injects its own)
```

Python 3.12, managed by uv (`uv sync` to set up). Ruff is configured (defaults + import sorting + pyupgrade).

`tests/test_extraction.py` is a live integration test against Ollama: it needs a running Ollama server with the model pulled. Configure via `.env`: `AGENT_MODEL` (default `gemma4:31b`) and `OLLAMA_BASE_URL` (default `http://localhost:11434`).

## Architecture

Pipeline (fully built): `.eml` in → intake (deterministic MIME parsing) → classifier (registry-driven prompt) routes to a specialist subgraph or dead-letters unknowns → LLM extraction against the specialist's schema (with one repair round) → `interrupt()` HITL loop (edits, re-prompts, completeness gate) → execute via MCP tool → finalize graduates the result into `results`.

Specialists are self-contained subgraphs registered behind a small registry; adding a function = new schema + subgraph + eval set, zero edits to existing code.

### Layout

- `src/freightcase/specialists/` — `base.py` (SpecialistSchema ABC: abstract `summarize`/`missing_for_execution`, concrete `confidence` walker), `common.py` (shared freight vocabulary: Weight, Dimensions, Location, Incoterm, alias tables), `quote.py` (QuoteRequest)
- `src/freightcase/registry.py` — REGISTRY: single source for function → description/subgraph/tool/schema; feeds the classifier prompt, execute dispatch, serde allowlist, and output rehydration
- `src/freightcase/contracts.py` — generic envelope + HITL machinery (SpecialistResult, ConfirmationPayload/Resume, process_resume, apply_edits); imports no specialist module
- `src/freightcase/intake.py` — `parse_eml`, `IntakeResult` (pipeline state, never shown to the model)
- `src/freightcase/extraction.py` — `extract_specialist_schema(email, schema=...)` (generic over SpecialistSchema), repair loop, `ExtractionError`
- `src/freightcase/classification.py` — `classify_email`; prompt composed from registry descriptions
- `src/freightcase/execution.py` — `ToolExecutor` protocol, `StubToolExecutor`, `MCPToolExecutor`
- `src/freightcase/tms_stub.py` — demo MCP server standing in for the adopter's TMS
- `src/freightcase/evals.py` — framework-free eval core (case loader, staged runners, scorers); thin runners in `scripts/`
- `src/freightcase/llm.py` — model construction (`default_model`, `model_from_spec`)
- `src/freightcase/graph.py` — state, nodes, wiring only; nodes are thin wrappers over plain functions
- `tests/` at root, src-layout imports through the installed package

## Core design rules (enforce these in all code)

1. **The model transcribes evidence; code interprets it.** The LLM emits values exactly as stated in the email. All normalization (unit aliases, incoterm variants, LOCODE resolution) lives in deterministic Pydantic validators that fail loudly on unrecognized input. Never move interpretation into prompts.
2. **Absence is representable, never synthesized.** Fields the email may not state are `X | None = None` (annotation AND default — `| None` without `= None` is still required and pressures the model to fabricate). Completeness is checked separately and deterministically (`missing_for_execution()`), feeding HITL per-field flags.
3. **Fatal vs degraded.** Errors (typed: `ExtractionError`, `IntakeError`) are for "processing is impossible" (unparseable message, no body, invalid JSON). Everything else proceeds with the gap recorded in a warnings list. Original values are preserved; canonical forms are computed fields (e.g. `weight.kg`), never overwrites.
4. **HITL gates every write.** Mutations pause via `interrupt()` with a standardized payload (proposed action, fields, per-field confidence, what confirm executes); resume via checkpointer + `Command(resume=...)`.
5. **No new behavior without an eval or test case pinning it.** Fixtures live in `tests/fixtures/emails/`, each earns its place with one distinct failure mode. Partial assertions, not golden-object equality.

### Implementation notes (repo-specific)

- Extraction deliberately does **not** use LangChain's `with_structured_output`: the schema goes in the system prompt and the response is parsed/validated in our code so `ValidationError` details survive inside `ExtractionError` (`.raw`, `.validation_error`) for the repair loop and per-field HITL confidence.
- Email content is untrusted: it is passed only as the `HumanMessage`, never interpolated into instructions.
- Attachment metadata carries a `content_ref` path, never the bytes — keep state light.
- Human input inside the interrupt loop must NEVER raise: LangGraph caches resume values in the checkpoint and replays them, so an exception wedges the thread permanently. Every bad answer becomes a re-prompt with `problems` set.
- Checkpoint serde allowlist (`CHECKPOINT_ALLOWED_TYPES` in graph.py) is exact (module, class) pairs: a state model left out silently deserializes as a plain dict. New/moved state models must be added; verify with `LANGGRAPH_STRICT_MSGPACK=true uv run pytest tests/test_graph.py`.
- `SpecialistResult.output` is abstract-typed: it needs `SerializeAsAny` to dump and the registry-driven `_rehydrate_output` validator to restore from checkpoints — polymorphic state always needs a discriminator (`function`).
- `functools.partial` hides a node's `Command[Literal[...]]` return annotation from LangGraph's edge inference — declare `destinations=` at `add_node` or the rendered graph silently loses edges.
- Pydantic + pyright: pass `Field(default=..., ...)` keyword (positional defaults aren't seen); dict literals assigned to variables need explicit TypedDict/Literal annotations to type-check.
- mcp 2.0 renamed FastMCP → `mcp.server.mcpserver.MCPServer`; typed tool returns (pydantic model) are required for `structured_content`.

## Conventions

- pytest with parametrize; LLM calls injectable (`model: BaseChatModel | None = None`) so tests/evals swap models
- Local dev model: Ollama gemma4:31b with `reasoning=False, format="json", temperature=0` (reasoning-on silently consumed the whole token budget; 44.7s → 5.4s). Eval benchmark target: API Haiku.
- The cargo list field on `QuoteRequest` is named `cargo`
- Fixtures use invented parties and .example domains only; no real customer data ever, including in git history
- Fixture-driven tests with partial expectations: each test module has a `CASES` dict asserting only the fields that matter per scenario. `.eml` fixtures exercise intake, `.txt` fixtures exercise extraction. Header/edge-case tests mutate a loaded `.eml` in memory rather than adding fixture variants.
- Graph tests: journey tests (one flow per test, staged comments) + direct node-call tests; `fake_graph(*responses)` scripts the model deterministically (classification response is prepended — the classify node consumes the first one). GenericFakeChatModel raises StopIteration when its message iterator exhausts — script one response per expected model call, including repair rounds.
- Evals ≠ tests: evals count (fraction of expectations met), tests gate. Eval cases live in `evals/cases/` as `.eml`/`.txt` + `.expected.json` sidecars; dotted-path vocabulary shared with `missing`/`confidence`/edits.
- Dev-only deps go in the `dev` dependency group; `logs/` is gitignored runtime output.
