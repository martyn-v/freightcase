# Freightcase: an extensible agent framework for logistics inboxes

## Design Decisions

- The model transcribes evidence, the code interprets it: the LLM emits values exactly as stated in the source email, and all normalization (unit aliases, incoterm variants, date formats) lives in deterministic validators that fail loudly on unrecognized input, so ambiguity surfaces as a per-field review flag in HITL instead of a silent, confidently wrong conversion.
- MCP is the integration boundary, not an LLM affordance: by the time execute fires, the human has confirmed a validated payload and deterministic code makes the tool call — no model chooses tools or fills arguments (intelligence ends where writes begin). MCP earns its place as a driver-style contract instead of a bespoke adapter interface: the adopter points freightcase at their own TMS's MCP server exposing `create_quote`, and the pipeline gets typed, discoverable tool schemas with zero TMS-specific code.
- Absence degrades, form fails: schema validation polices *what was stated* (unknown units, malformed values, invalid JSON stay fatal), while *what wasn't stated* extracts as `None`, is recorded by a deterministic completeness check (`missing_for_quoting()`), and surfaces in the confirmation payload for the human to remedy — an incomplete email becomes a conversation, not a dead-letter ticket.
- No enums in the transcription contract: a `Literal["kg","lb","t","g"]` in the prompt schema pressures the model to convert "toneladas" → "t" — interpretation smuggled in through type constraints. Unit fields are plain strings in the schema the model sees; the alias table normalizes deterministically afterward, keeping the raw transcription available for per-field provenance (stated / normalized / missing / edited) shown to the reviewer.
- Human input inside an interrupt loop must never raise: LangGraph caches the resume value in the checkpoint and replays it on every retry, so an exception thrown on a malformed resume wedges the thread permanently — no later, corrected answer can ever get through. Every invalid answer (unparseable resume, bad edit path, failed validation, remaining gaps) becomes a re-prompt with the reason attached; the only exits are the human's own approve or reject.
- The demo fakes the TMS, not the integration: `tms_stub.py` is a real MCP server (in-memory quotes, incrementing references) standing in for the adopter's system, so dev and demo runs exercise the genuine client-transport-tool path end to end — the only pretend layer is the system of record, which a demo was always going to pretend. The in-process `StubToolExecutor` exists one layer up for unit tests, where spawning a server subprocess would be waste; the seam between them is the `ToolExecutor` protocol, which is also exactly where an adopter's real TMS plugs in.
- The eval core is framework-free; runners on top are disposable: `freightcase/evals.py` owns the case format, the staged pipeline runners, and the deterministic scorers, with zero eval-framework imports — and two thin runners sit on top. `scripts/run_evals.py` is the zero-dependency local runner (the adopter's eval set is their own production emails, which never leave the machine); `scripts/langsmith_evals.py` runs the same cases as LangSmith experiments, the ecosystem-native choice for this LangChain/LangGraph stack, where traces, tokens and model comparisons come free from the tracing already in place. (An earlier iteration used Inspect AI; it was abandoned because Inspect wants to *own* the model layer, and bridging a pipeline that brings its own models took a cross-thread async adapter — framework-fit friction that thick is a design smell. Frameworks that treat the system under test as a function fit; frameworks that want to be its model layer don't.)
- Evals are staged for attribution: a composite score can't tell a routing failure from an extraction failure, so classification (one cheap call per labelled email — the high-N measurement), extraction (dispatched by the sidecar's declared function, classifier bypassed), and pipeline (classify → extract, the compounding measurement) run and report as separate stages over one case folder. The economics differ by stage — classification labels cost seconds to author, field expectations cost minutes — so the dataset can grow where growth is cheap without diluting the expensive cases.

## Running evals

```bash
uv run python scripts/run_evals.py                          # all stages, local model
uv run python scripts/run_evals.py --stage classification   # one stage
uv run python scripts/run_evals.py --cases /path/to/my-evals --json results.json
uv run python scripts/run_evals.py --model anthropic:claude-haiku-4-5 --max-repairs 0
```

`--model` takes any langchain `init_chat_model` string (matching provider
package and API key are yours); omitted = the local Ollama default. Nothing
leaves the machine.

The same cases run as LangSmith experiments — one per stage, with full
traces and token usage attached — when you want shareable, comparable runs
(note: this uploads the case emails as LangSmith datasets):

```bash
uv run python scripts/langsmith_evals.py --stage all
```

A case is a sidecar pair — `name.eml` (full email, usable by every stage) or
`name.txt` (bare body, extraction stage only) plus `name.expected.json`:

```json
{
  "classification": "quote_request",
  "fields": { "mode": "road", "cargo.0.weight.kg": 8400 },
  "missing": ["origin.locode", "cargo.0.dimensions"]
}
```

Every key is optional and doubles as stage membership: a `classification`
label enters the case in the classification stage, `fields`/`missing` enter
it in extraction. Paths use the same dotted vocabulary as `missing`,
`confidence`, and HITL edits, resolved against the extracted model's dump
(canonical computed fields like `weight.kg` are addressable). Numbers compare
with relative tolerance; `missing` asserts membership in
`missing_for_execution()`. Each case scores the fraction of its expectations
met — evals count, they don't gate; a failed extraction is a scored data
point, not a crashed run. The starter set in `evals/cases/` doubles as
format documentation.
