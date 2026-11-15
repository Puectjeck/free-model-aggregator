# Free Model Aggregator

Entry for the make.fun competition **free-model-aggregator-01**.

A single-file Python 3 server (`main.py`, standard library only) that acts as
one local model endpoint speaking both client APIs:

- **OpenAI Chat Completions** — `POST /v1/chat/completions` (JSON + SSE)
- **Anthropic Messages** — `POST /v1/messages` (JSON + SSE)
- **Model list** — `GET /v1/models` (OpenAI or Anthropic shape, by auth header)

On startup it reads the trial brief from `$MAKEFUN_PROMPT_FILE`, probes every
provider's catalog, registers its port and catalog with
`$MAKEFUN_ENV_URL/register`, then serves the trial's requests until the env
reports done via `$MAKEFUN_ENV_URL/work`, and exits 0.

## How it routes

For each request it takes the requirements (tools / vision / json / minimum
context) from the workload manifest, collects every **free** model that meets
them across all providers — OpenAI-, Anthropic- and Gemini-wire upstreams —
and translates the request through a canonical intermediate representation
that preserves the echo-critical fields (system text, last-user text, tool
names, `max_tokens`, stop sequences, image bytes, json-mode); same-wire
requests are forwarded near-verbatim with only the model swapped. Because a
trial's tie-break is the number of provider calls, it then picks the
candidate that minimises the expected total calls, pricing each one with:

1. A posterior refusal hazard for the provider's next call. The trial's
   budgets, 1–2 scheduled single-call refusals per provider and one
   provider's outage are drawn per trial and never published, but their
   shape is fixed — a provider's first call always answers — so every
   observed answer and refusal updates the hazard for the calls that
   remain, including the trap that one provider's schedule always
   includes its second call.
2. A scarcity price for spending a provider that later, more constrained
   requests in the manifest (known in full from the start) will need.
3. A small price for quality given up, bounded by a hard total loss
   budget and a per-request floor, so the quality column cannot fall.

On a refusal it reads the provider's own error message: *budget spent* or
*is out* marks the provider dead for the rest of the trial (and says
which); a scheduled single-call refusal is now spent, which usually makes
the same provider the cheapest way to finish the request.

Upstream calls are always non-streaming; the client's requested wire format
(JSON or SSE) is rendered locally from the provider's answer, carrying the
provider's exact text, usage and stop reason. Everything rendered uses fixed
ids/timestamps and deterministic ordering — no wall clock, no randomness —
so a replay produces byte-identical state.

## Usage

```
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> python3 main.py
python3 main.py --list   # print the catalog document (empty without a brief)
```

Run `python3 main.py --list` with the trial env vars set to print exactly the
catalog object the server registers.

## License

MIT — see `LICENSE`.
