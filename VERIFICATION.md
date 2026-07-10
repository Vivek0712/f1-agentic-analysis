# Verification report

End-to-end verification of this repository: environment rebuild, full pipeline
rerun and diff against published outputs, file-by-file audit, dashboard rebuild
with headless visual verification, and cross-document consistency checks.
Environment: Python 3.12.11 venv, pandas 3.0.3, numpy 2.5.1, scipy 1.18.0,
strands-agents 1.45.0, bedrock-agentcore 1.17.0, boto3 1.43.40, pytest 9.1.1.
No version conflicts; the only install constraint is Python >= 3.10 for
strands-agents (the macOS system python3 is 3.9 and cannot install it).

## 1. Pass/fail table

| File | Verdict | Notes |
|---|---|---|
| requirements.txt | FIXED | matplotlib was missing (used by scripts/build_figures.py); added |
| scripts/fetch_data.sh | PASS | all URLs reachable, idempotent (curl -O overwrite); fetches the 9 loader tables plus sprint_results |
| src/f1agents/data/loader.py | PASS | clean_lap_mask applied at every fit site (stints, strategy, backtest, profiles); no mutable defaults, no bare except |
| src/f1agents/data/openf1.py | PASS | normalize_laps drops null lap_duration and emits the loader schema (plus is_pit_out_lap, consumed by stops_from_laps); stop lap = out-lap - 1, lap 1 excluded, fixture yields stops at 8 and 31; 0.4s inter-call sleep, exponential backoff on 429, bounded at 5 attempts then RuntimeError |
| src/f1agents/analysis/stints.py | PASS | fuel_slope clamps to [-0.12, 0]; first and last stint laps dropped before the Theil-Sen fit; StintFit carries CI bounds and residual IQR; deg_anomalies uses MAD z with threshold 1.5. See open issue 4 (dead loop in build_stints) |
| src/f1agents/analysis/strategy.py | PASS | pit loss = in-lap + out-lap excess over clean median, stop counts recorded; undercut_events dedups mirrored pairs and requires response within 5 laps; counterfactual_pit holds pit loss fixed, states assumptions, guards shifts within 3 laps of stint edges. See open issue 5 |
| src/f1agents/analysis/backtest.py | PASS with critical caveat | line-by-line lookahead review: pre_laps/pre_pits filter at the freeze before every fit; fuel slope, clean masks, defender fit, attacker fresh pace, and the observed gap component are all pre-freeze; the stint-2 prior pools pre-freeze stint-2 laps with the first three laps dropped; the undercut forecast includes the defender's pit loss and the same settle lap (stop + 2) as undercut_events; prediction_at_horizon_boundary is set and surfaced. The one lookahead is the pit_loss_s argument itself: see open issue 1 |
| src/f1agents/analysis/profiles.py | PASS | driver metrics read clean laps and fitted slopes only; track_profiles reads fits and pit loss (the neutralisation rate reads all laps by definition: it is the share of slow laps) |
| src/f1agents/compliance/rules.py | PASS | every rule returns a rule id and detail; advisory-only fails closed on auto_execute; audit records carry a hash over the full record (recommendation payload included), model ID, and RULESET_VERSION = "2026.1" |
| src/f1agents/agents/prompts/ (7 files) | FIXED | each states its payload contract and restricts claims to payload numbers; all seven match strands_roster.TIERS. Typo fixed in deg_explainer.md ("slints" -> "stints") |
| src/f1agents/agents/strands_roster.py | PASS | all seven agents built with tools=[]; Haiku for slow_loop, Sonnet for post_session and cross_season; build_message adds previous_brief and instruction only when a previous brief exists |
| src/f1agents/agents/agentcore_app.py | FIXED | entrypoint rejects non-slow-loop agents; get_last_k_turns uses actor_id = agent name, session_id = race session; create_event stores payload as USER and brief as ASSISTANT; _previous_brief and _record were wrapped, session_summary was not: retrieve_memories is now wrapped so a Memory failure returns [] (static check only, no AWS calls made) |
| src/f1agents/agents/base.py | PASS | batch_input.jsonl parses line by line, 9 records, each a valid Bedrock batch record. Note: the record body is the Anthropic-native Messages format (anthropic_version bedrock-2023-05-31), which is what CreateModelInvocationJob requires; it is not the Converse API shape |
| src/f1agents/pipeline/run_race_analysis.py | PASS | CLI matches the README exactly; output path is anchored to the package location so it resolves from src/ or the repo root |
| src/f1agents/pipeline/run_backtest.py | PASS as code, source of open issue 1 | CLI matches the README; output path handles both working directories |
| scripts/build_dashboard.py | FIXED | injects RESULTS, BRIEFS, and BACKTESTS placeholders; results.json, sample_briefs.json, and template.html already failed loudly, but an empty backtest glob passed silently: it now raises FileNotFoundError |
| scripts/build_figures.py | PASS | regenerates all six PNGs from outputs/2024_spanish with no warnings |
| scripts/create_memory.py | PASS | create_memory_and_wait with summaryMemoryStrategy on /summaries/{actorId}/{sessionId} |
| infra/architecture.md | FIXED | tier table matches strands_roster.TIERS exactly; the ASCII diagram labeled the slow-loop runtime "Claude Sonnet" while the roster routes slow_loop to Haiku; label corrected. See open issue 6 |
| infra/cdk/app.py | PASS | cdk synth succeeds (CDK CLI 2.x + aws-cdk-lib), producing F1AgenticAnalysis.template.json |
| dashboard/template.html | FIXED (3 defects) | see defects 1-3 below |
| dashboard/index.html | REBUILT | 61 KB, rebuilt from the fixed template |
| tests/test_core.py | PASS | 16 tests, all module coverage confirmed except profiles.py (open issue 3) |
| tests/fixtures/openf1_laps_9839_44.csv | PASS | pytest -q -k openf1: 1 passed; stops at laps 8 and 31 |
| outputs/2024_spanish/*.json, batch_input.jsonl | PASS | untouched; reproduction diff below |
| README.md | PASS | every decimal cited near pit loss/stint/swing/MAE/bias/z/table keywords exists in outputs JSON; all image refs resolve; quickstart commands match the CLIs; repo layout section matches the tree |
| blog/aws-ml-blog-draft.md | PASS | same numeric and image checks pass; HAM L16 vs PIA L21 15.87s, HAM vs LEC 10.54s, NOR 0.063 s/lap, BOT 0.275/z 4.79/field 0.069, backtest table all verified against outputs |
| blog/figures/*.png (6) | PASS | regenerated without warnings from unchanged outputs |
| docs/architecture.drawio | PASS | parses as XML; contains the AgentCore Memory node ("AgentCore Memory / previous brief + session summary / /summaries/{actorId}/{sessionId}") |
| docs/img/dashboard.png | FIXED | the published image was a broken render (empty stat strip, blank charts: captured before JavaScript ran); replaced with a correct full-page render of the rebuilt dashboard |
| LICENSE | PASS | MIT, matches the README license section |
| .gitignore | PASS | data/raw/ and caches ignored |

## 2. Defects found and fixed

1. dashboard/template.html: the prediction audit section never rendered.
   Top-level `const BACKTESTS` does not attach to `window`, so the guard
   always returned early.
   `if (!window.BACKTESTS || !BACKTESTS.length) return;` -> `if (typeof BACKTESTS === 'undefined' || !BACKTESTS.length) return;`
2. dashboard/template.html: backtest stat labels used class "k", which has
   no CSS rule (the strip style is `.stat .l`), so they rendered unstyled.
   `<div class="k">` -> `<div class="l">` (3 occurrences).
3. dashboard/template.html: the 2.22s finishing gap was hardcoded, violating
   the footer claim that every number on the page is in results.json. It is
   now computed from `R.lap_traces.NOR.gap_s` (final lap), which equals 2.22.
   `textContent = '2.22s'` -> `textContent = norTrace.gap_s[norTrace.gap_s.length-1].toFixed(2)+'s'`
4. dashboard/template.html: compliance verdict pills for failed rules
   rendered FAIL; relabeled BLOCKED to match the fails-closed semantics.
   `${r.passed?'PASS':'FAIL'}` -> `${r.passed?'PASS':'BLOCKED'}`
5. requirements.txt: matplotlib was imported by scripts/build_figures.py but
   not listed.
   Added `matplotlib>=3.7    # blog/README figures (scripts/build_figures.py)`
6. src/f1agents/agents/agentcore_app.py: session_summary called
   retrieve_memories unwrapped, so a Memory outage would raise instead of
   degrading to stateless.
   Wrapped in `try: ... except Exception: return []`
7. scripts/build_dashboard.py: an empty backtest glob built a dashboard with
   a silently empty prediction audit.
   Added `if not backtest_files: raise FileNotFoundError(...)`
8. src/f1agents/agents/prompts/deg_explainer.md: typo.
   `late-race slints` -> `late-race stints`
9. infra/architecture.md: the ASCII diagram labeled the slow-loop runtime
   Claude Sonnet; the roster routes slow_loop to Haiku.
   `| Sonnet)      |` -> `| Haiku)       |`
10. docs/img/dashboard.png: replaced the broken published render with a
    full-page screenshot of the rebuilt dashboard (same content, correct
    render; no numbers changed).

## 3. Open issues not fixed, with reasoning

1. CRITICAL: pit-loss lookahead in the backtest CLI.
   src/f1agents/pipeline/run_backtest.py line 57 computes
   `pit_loss(laps, pits)` on the full race and passes it into
   `undercut_calls` and `pit_window` as a prediction input, while the
   backtest docstring promises "the pre-freeze pit loss estimate" and the
   module header claims no lookahead anywhere. Measured pre-freeze values:
   21.46s at freeze 20, 22.12s at 30, 22.22s at 40, vs 22.61s full-race.
   Rerunning with the pre-freeze estimate moves the published undercut
   forecasts: freeze 20 HAM vs PIA predicted swing 14.76s -> 13.61s
   (abs error 1.11s -> 2.26s); freeze 40 STR vs RIC predicted swing
   9.14s -> 8.75s (abs error 1.10s -> 0.71s). The README claim "both
   undercut forecasts ... landed within 1.11s" does not survive the fix.
   Not fixed because the correction changes published numbers in
   outputs/, README, blog, and dashboard; per the verification brief the
   diff is reported instead. The one-line fix when the numbers may be
   republished: `ploss = pit_loss(*_prefreeze(laps, pits, a.freeze))["pit_loss_s"]`
   (with _prefreeze imported or inlined). The degradation-continuation
   results (the MAE/bias table) do not use pit loss and are unaffected.
2. The compliance checklist wording vs the record hash: audit records carry
   `record_hash`, a SHA-256 over the full record including the
   recommendation payload, timestamp, model ID, and verdicts. There is no
   separate payload-only hash, so the hash changes on every run even for an
   identical payload (this is exactly the only diff the rerun produced in
   results.json). A stable payload hash would make audits reproducible;
   changing the audit schema alters results.json, so left as is.
3. Test coverage gap: src/f1agents/analysis/profiles.py has no test. Every
   other module in analysis/ and data/ is exercised. Not fixed because the
   README and blog both publish "16 tests" and adding one changes that
   count; noting it here instead.
4. Dead code in src/f1agents/analysis/stints.py build_stints (lines 70-72):
   a nested iterrows loop that executes `+= 0` per stop before the real
   stint assignment below it. Provably a no-op, but removing code from the
   function that produces published stint fits was not worth the risk in a
   verification pass.
5. src/f1agents/analysis/strategy.py counterfactual_pit: if the actual pit
   lap falls within 3 laps of a stint edge, the shift-0 row is excluded by
   the guard and `next(r for r in results if r["shift"] == 0)` raises
   StopIteration. Cannot happen for the published VER/NOR runs; a guard
   returning viable=False would be the fix.
6. infra/architecture.md cost paragraph prices the slow loop at
   "Sonnet-class pricing" while slow_loop routes to Haiku. As an upper
   bound the estimate still holds, so the prose was left alone beyond the
   diagram label fix.
7. dashboard/template.html loads Chart.js from a CDN, so the "standalone"
   dashboard requires network access to render charts; this is also the
   likely cause of the broken published dashboard.png. Vendoring Chart.js
   (~200 KB) into the template would fix it at the cost of the 61 KB page.
8. data/openf1/9839/ is an empty committed directory (the OpenF1 fixture
   actually used by tests lives at tests/fixtures/openf1_laps_9839_44.csv).
   Harmless; left in place.
9. src/f1agents/data/openf1.py references urllib.error without importing
   it. Works at runtime because urllib.request imports urllib.error
   internally, but it is an implicit dependency a linter would flag.
10. The pipeline rerun writes trivially different float tails (1e-15) in
    two track_profiles medians inside batch_input.jsonl under
    pandas 3.0.3/numpy 2.5.1 vs the published file. No rounded or published
    value changes; noted for anyone diffing byte-for-byte on new stacks.

## Final state

- Tests: 16 passed (pytest tests/ -q, strands and bedrock-agentcore
  installed so nothing was skipped), 1 pre-existing RuntimeWarning from an
  empty nanmean slice in test_backtest_report_shape.
- Dashboard: rebuilt at 61 KB; rendered headlessly at 1440px with zero
  console errors and zero page errors; all 23 content checks pass (header,
  stat strip, six-line pit-wall trace with markers, NOR 4.38s vs 2.22s,
  undercut ledger rows, BOT z 4.79, seven populated brief cards, PASS and
  BLOCKED compliance verdicts, three-freeze prediction audit with two
  undercut calls, footer stack line).
- Screenshots: docs/img/verify_dashboard_top.png,
  docs/img/verify_dashboard_bottom.png (plus the regenerated full-page
  docs/img/dashboard.png).
- Reproduced results diff vs published outputs/2024_spanish/: empty for all
  published values. backtest_20/30/40.json byte-identical; results.json
  differs only in the two compliance audit timestamps and their record
  hashes; batch_input.jsonl differs only in two 1e-15 float tails. All 11
  spot-check values match exactly (66 laps, 62 stint fits, 22.61s pit loss,
  18 undercut events, VER 0.06, NOR 4.38, BOT z 4.79, backtest_20 MAE
  1.198 and HAM vs PIA error 1.11, backtest_40 STR vs RIC error 1.1 and
  bias 0.102).
