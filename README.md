[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Puectjeck/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Puectjeck/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

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
trial's tie-break is the number of provider calls, it does not choose one
request at a time: it plans the whole remaining manifest at once and
re-plans after every request with the evidence so far, leaning on the
original plan (a small commitment bonus for its pick) so near-ties do not
oscillate between providers. A provider that takes k requests in a plan
physically makes its next k calls, so the plan prices whole prefixes:

1. A prefix refusal price: for each provider, the sum over its planned
   calls of -log(1 - hazard), where the hazard of a future call number
   integrates the published budget range, the hypergeometric
   refusal-count posterior given the calls survived and refused so far,
   the outage prior and the call-2 trap prior — plus a small surcharge
   per level of depth on one provider. The assignment is built greedily
   in manifest order and improved by exact local moves and swaps.
2. A top-tier rule: providers whose best free model is within 5 milli of
   the trial's best rating serve at most one call in any plan. The
   quality their repeats add over the next tier is a few milli, while
   their second call is where the drawn schedules concentrate and where
   the guaranteed call-2 trap sits — the deep, cheap providers carry
   the repeats instead.
3. A small price for quality given up, with the price escalated until
   the plan fits a hard total loss budget (and a per-request floor), so
   the quality column cannot fall.

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
