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
