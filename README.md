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
and ranks them by the brief's quality rating. It then:

1. Calls the best candidate, translating the request through a canonical
   intermediate representation that preserves the echo-critical fields
   (system text, last-user text, tool names, `max_tokens`, stop sequences,
   image bytes, json-mode). Same-wire requests are forwarded near-verbatim
   with only the model swapped.
2. Keeps a per-provider soft budget cap (the low end of the provider's
   free-tier budget) so no single provider's budget is burned early.
3. On a refusal, reads the provider's own error message: *budget spent* or
   *outage* marks the provider dead for the rest of the trial; a one-call
   refusal is retried once on the same provider (the next call number
   usually succeeds) and then fails over to the next-best candidate.

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
