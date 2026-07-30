# Freightcase: an extensible agent framework for logistics inboxes

## Design Decisions

- The model transcribes evidence, the code interprets it: the LLM emits values exactly as stated in the source email, and all normalization (unit aliases, incoterm variants, date formats) lives in deterministic validators that fail loudly on unrecognized input, so ambiguity surfaces as a per-field review flag in HITL instead of a silent, confidently wrong conversion.
