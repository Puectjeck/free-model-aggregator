# Build notes — free-model-aggregator-01

- Competition: make.fun `free-model-aggregator-01` ($500 pool; deadline
  16 Nov 2026 09:30 WIB).
- Built in agent session `29555f88-af30-469c-a18c-628d264bab4a`
  (label cc882a33-5c6c-4c72-9273-03c6e15f2b7d). This session is the
  submission's transcript-bundle session; `main.py` was authored and then
  revised across multiple messages in it.
- Language: Python 3 (stdlib only), entrypoint `main.py` at repo root.
  Chosen over the TypeScript starter for exact control of the wire
  translations; the pod runs `python3 -P -s /bot/main.py`.
- Architecture: canonical IR translation; same-wire near-verbatim forward;
  upstream always non-streaming, client format (JSON/SSE) rendered locally;
  rating-ranked routing with per-provider soft budget caps (calls_min) and
  refusal-message classification (budget/outage = provider dead, scheduled
  refusal = one same-provider retry, then fail over).
- Local testing: kit practice trials run against the kit's own env server
  and scorer (process driver, the kit's grade-local equivalent without
  Docker). Results, 2026-11-14:
  - trial-0: 20/20 valid, 25 upstream calls — score 99
    (quality 99, served 100, wire 100, failover 100, catalog 100), exit 0.
  - trial-1: 20/20 valid, 25 upstream calls — score 99 (same columns).
  - trial-2: 20/20 valid, 29 upstream calls — score 99 (same columns).
  - Replay determinism: trial-0 run twice; the two exported state.json
    files are byte-identical (cmp clean).
  - `--list` with the trial env vars prints exactly the catalog object the
    server registers (compared against the state export's catalog field);
    without a brief it prints `{"providers": []}`.
  - Quality 99 is the budget-feasible ceiling on these draws: the residual
    is weight-1 requests served at 1470 vs the 1475 bound once the top
    provider's minimum budget is spent.

## Lean revision (2026-11-15 night)

- Built in agent session `7257e3ea-b08c-4024-8bf5-e66bdab8a59e` (subagent of
  the main session), working on this copy; the submitted repo was not edited.
- Change: only the Router. Candidate choice now minimises a cost =
  posterior refusal hazard (budget range posterior, hypergeometric
  refusal-count posterior, outage prior, call-2 trap prior) + 0.1 x
  future-scarcity pressure from the manifest + 0.001 x weighted quality
  loss, with a hard quality guard (total weighted loss cap 700, per-request
  floor best-90). Refusal classification now splits dead into budget vs
  outage. Everything else (parsing, IR, rendering, catalog, server) is
  byte-identical to the submitted version.
- Method: the routing policy was tuned in a Monte-Carlo replica of the
  kit's generator + provider semantics (base-policy replica reproduced the
  live entry exactly: 25 calls / quality 99 on practice trial-0). Over 500
  drawn instances the lean policy averages 21.9 calls (P(<=20) 6%,
  P(<=21) 37%, P(<=22) 76%), quality >= 98 on every draw. Pre-trial probing
  was tested and rejected: probes burn the guaranteed-clean early calls
  and raise the mean (27.5 at depth 2).
- Real kit harness results (kit env + scorer), lean vs submitted bot:
  - practice trial-0: 21 calls, score 99 (quality 98, rest 100) [base: 25/99]
  - practice trial-1: 22 calls, score 99 [base: 25/99]
  - practice trial-2: 22 calls, score 99 [base: 29/99]
  - 7 fresh drawn instances: 25, 21, 21, 21, 22, 24, 23 calls, all score 99,
    served/wire/failover/catalog 100 [base on 3 of them: 30, 21, 26]
  - 10-instance lean mean 22.2 calls vs base mean 26.0 on its 6.
  - Replay determinism: trial-0 run twice, exported state.json byte-identical.
- Why not 20 on every instance: 20 requests over 10 providers force ~10
  repeat calls (call #2+); each repeat carries a 10-40% refusal hazard
  from schedules that are drawn per trial and unknowable in advance, so
  the expected refusal count of any policy is ~2 and P(zero refusals) is
  only ~6-15%. The dev board's 20-call entries are the lucky tail of the
  single fixed dev instance. Expected gain of this revision in the finals:
  ~25 -> ~22 calls at the same score 99.
