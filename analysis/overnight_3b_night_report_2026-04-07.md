# Overnight 3B Development Report - 2026-04-07

This report covers the full local 3B overnight-runner work for `agent-stego`: what changed, what was tried, what failed, what was fixed, what the current defaults are, and where the relevant artifacts live.

## TL;DR

- I built a bounded local overnight sweep runner for small-model steganography experiments: [`experiments/overnight_3b_sweep.py`](../experiments/overnight_3b_sweep.py) with a wrapper at [`scripts/run_overnight_3b.sh`](../scripts/run_overnight_3b.sh).
- I hardened the agent loop so local 3B models that emit tool calls as plain text can still be evaluated. The core changes are in [`src/agent.py`](../src/agent.py).
- The overnight sweep now targets **6-character sequential correctness**, not just one-character or weak bit leakage.
- The strongest default candidates after the night’s iteration are still:
  - [`path_table`](../experiments/overnight_3b_sweep.py)
  - [`query_table`](../experiments/overnight_3b_sweep.py)
- Current best simple recommendation:

```bash
cd ~/Work/agent-stego
TRIALS=8 BASELINE_TRIALS=4 ./scripts/run_overnight_3b.sh
```

- Checkpoint commit: `89cc523` - `feat(sweep): add overnight 3b experiment runner`

## Main files

### Code and docs
- Runner: [`experiments/overnight_3b_sweep.py`](../experiments/overnight_3b_sweep.py)
- Wrapper: [`scripts/run_overnight_3b.sh`](../scripts/run_overnight_3b.sh)
- Agent loop hardening: [`src/agent.py`](../src/agent.py)
- Repo docs: [`README.md`](../README.md)
- Original research writeup: [`analysis/FINDINGS.md`](./FINDINGS.md)
- Local implementation spec: [`../.pi/specs/overnight-3b-sweep.md`](../.pi/specs/overnight-3b-sweep.md)
- Earlier repo familiarization note: [`/home/onblueroses/.pi/deep-understand/agent-stego.md`](/home/onblueroses/.pi/deep-understand/agent-stego.md)

### Important experiment outputs
- Best broad comparison run of serious contenders: [`findings/overnight_3b/20260407-024307/summary.md`](../findings/overnight_3b/20260407-024307/summary.md)
- Post-hardening default comparison: [`findings/overnight_3b/20260407-032046/summary.md`](../findings/overnight_3b/20260407-032046/summary.md)
- Experimental preset smoke: [`findings/overnight_3b/20260407-031027/summary.md`](../findings/overnight_3b/20260407-031027/summary.md)
- Secret-conditioned path codebook success run: [`findings/overnight_3b/20260407-021122/summary.md`](../findings/overnight_3b/20260407-021122/summary.md)
- Secret-conditioned query codebook success run: [`findings/overnight_3b/20260407-023611/summary.md`](../findings/overnight_3b/20260407-023611/summary.md)
- Final parser-variant validation for `path_table`: [`findings/overnight_3b/20260407-033207/summary.md`](../findings/overnight_3b/20260407-033207/summary.md)

## Starting point

The goal shifted from “interesting ideas” to a concrete local evaluation loop:

1. Work in the real clone at `~/Work/agent-stego`, not the cache checkout.
2. Use **local or free models only**.
3. Focus on **small / 3B-friendly channels**.
4. Optimize for **6+ sequentially correct characters**, not just token-level or one-character signal.
5. Make it runnable overnight with bounded lifecycle and morning-after artifacts.

This was captured in the local spec: [`../.pi/specs/overnight-3b-sweep.md`](../.pi/specs/overnight-3b-sweep.md).

## What I built

### 1. A bounded overnight sweep runner

The new runner is [`experiments/overnight_3b_sweep.py`](../experiments/overnight_3b_sweep.py).

It:
- runs a matrix over models, candidates, secrets, and trials
- collects clean baseline traces first
- computes anomaly relative to baseline
- records per-trial decoded output and trace features
- writes:
  - `manifest.json`
  - `baseline.json`
  - `results.json`
  - `summary.md`

The thin wrapper is [`scripts/run_overnight_3b.sh`](../scripts/run_overnight_3b.sh).

### 2. Small-model tool-call fallback parsing

Local `qwen2.5:3b` often emitted correct tool sequences as text instead of native tool-call objects. Without a fallback parser, those runs looked like failures even when the model was actually following the protocol.

I extended [`src/agent.py`](../src/agent.py) to recover and execute multiple text formats, including:
- `read_file({"path": ...})`
- `search_web {"query": ...}`
- `ReadOnly {...}`
- embedded `{"name": ..., "arguments": ...}` JSON blobs
- `hydrate({...})`
- extra outer parentheses like `((read_file({...})))`

I also normalized tool-call argument strings before putting them back into the conversation history, which fixed a crash path caused by malformed native argument payloads.

### 3. Sequence-oriented metrics

The runner started as a low-precision sweep, but the objective changed during the session. It now tracks:
- exact full-string match
- first-N sequence match (`sequence_match`)
- character accuracy on the first N characters
- correct-prefix length
- anomaly score against clean baseline traces

This makes the evaluation actually reflect the “at least 6 sequentially correct characters” goal.

### 4. Default and experimental presets

The runner now supports:
- `CANDIDATES=default`
- `CANDIDATES=experimental`
- `CANDIDATES=all`
- or an explicit comma list

The wrapper defaults to `default`.

## What I tried tonight

## Phase 1 - Initial orchestration and first runner

Early on, I built the first overnight runner around low-precision binary-style candidates and ran tiny smokes.

Representative runs:
- first single-candidate smoke: [`20260407-000423`](../findings/overnight_3b/20260407-000423/summary.md)
- first all-candidates tiny smoke: [`20260407-000654`](../findings/overnight_3b/20260407-000654/summary.md)

Initial candidate families included:
- `path_depth`
- `file_choice`
- `order_pair`
- `query_anchor`

These were useful as scaffolding but not aligned with the updated 6-character sequential objective.

## Phase 2 - Switch from one-character / bit tests to 6-character sequential correctness

The requirement changed explicitly: the target should be **at least 6 characters, sequentially correct**.

That pushed the sweep away from simple bit channels toward one-symbol-per-character tables.

I introduced stronger sequence-oriented candidates:
- `path_table`
- `query_table`
- later `tool_table`
- later `tri_tool_table`
- later `mnemonic_path_table`
- later `secret_path_codebook`
- later `secret_query_codebook`

Representative transition runs:
- first 6-char path-table probe: [`20260407-001924`](../findings/overnight_3b/20260407-001924/summary.md)
- early fixed-table comparison: [`20260407-003856`](../findings/overnight_3b/20260407-003856/summary.md)
- first larger mixed comparison: [`20260407-011502`](../findings/overnight_3b/20260407-011502/summary.md)

## Phase 3 - Discovering the real small-model failure mode

The biggest surprise of the night was that many apparent failures were not true semantic failures. The model was often outputting the correct sequence, but in plain text.

Observed plain-text formats included:
- `read_file({"path": ...})`
- raw JSON objects with `name` and `arguments`
- `ReadOnly {...}`
- `search_web {...}`
- `hydrate({...})`
- parenthesized wrappers

This was surfaced by direct inspection of completion payloads during repeated diagnostics, then fixed inside [`src/agent.py`](../src/agent.py).

Representative before/after runs:
- before parser hardening: [`20260407-005024`](../findings/overnight_3b/20260407-005024/summary.md)
- after first parser fix: [`20260407-005726`](../findings/overnight_3b/20260407-005726/summary.md)
- later parser-extension validations: [`20260407-010853`](../findings/overnight_3b/20260407-010853/summary.md), [`20260407-033207`](../findings/overnight_3b/20260407-033207/summary.md)

## Phase 4 - Measuring serious contenders

Once the runner and parser were stable enough, I compared the main candidate families.

### Fixed-table defaults

These ended up as the best overall defaults:
- `path_table`
- `query_table`

Best broad comparison of serious contenders:
- [`20260407-024307/summary.md`](../findings/overnight_3b/20260407-024307/summary.md)

That run showed:
- `path_table`: lower anomaly, decent reliability
- `query_table`: better exact and sequence success, but somewhat higher anomaly
- secret-conditioned variants: occasionally strong on specific secrets, but less consistent overall

### Experimental variants

I explored several alternatives:
- `tri_tool_table` - too unstable for 3B
- `tool_table` - too heterogeneous, too noisy
- `mnemonic_path_table` - occasionally strong, especially on `SIGNAL`
- `secret_path_codebook` - looked very promising on some focused tests
- `secret_query_codebook` - also strong on some focused tests

Representative runs:
- secret-conditioned path codebook: [`20260407-021122`](../findings/overnight_3b/20260407-021122/summary.md)
- secret-conditioned query codebook: [`20260407-023611`](../findings/overnight_3b/20260407-023611/summary.md)
- experimental preset smoke: [`20260407-031027`](../findings/overnight_3b/20260407-031027/summary.md)

The experimental variants were interesting, but the fixed tables still won the consistency contest across multiple secrets.

## Phase 5 - Hardening for longer bounded runs

A larger run uncovered another mechanical failure mode: malformed native tool-call argument payloads caused a hard 400 error on the next completion turn.

I fixed that by normalizing arguments to valid JSON before adding them back into the assistant history and before executing them through the harness.

After that, the long-run failure mode shifted from hard crashes to scored compliance failures, which is what we want in an overnight benchmark.

Representative run after crash hardening:
- [`20260407-032046/summary.md`](../findings/overnight_3b/20260407-032046/summary.md)

## What works best right now

## Default recommendation

Use the default preset:
- `path_table`
- `query_table`

Command:

```bash
cd ~/Work/agent-stego
TRIALS=8 BASELINE_TRIALS=4 ./scripts/run_overnight_3b.sh
```

Why:
- these are the best-balanced candidates after the night’s comparisons
- `path_table` tends to stay closer to baseline behavior
- `query_table` often achieves better exact 6-character runs
- both are simple enough to reason about and document

## Experimental recommendation

If you want a second exploratory overnight run after the default preset, use:

```bash
cd ~/Work/agent-stego
CANDIDATES=experimental TRIALS=8 BASELINE_TRIALS=4 ./scripts/run_overnight_3b.sh
```

That will probe:
- `secret_path_codebook`
- `secret_query_codebook`
- `mnemonic_path_table`
- `tri_tool_table`
- `tool_table`

This is useful for idea generation, not as the main benchmark preset.

## Current assessment of candidates

### Best default pair
- `path_table` - better stealth profile, lower anomaly tendency
- `query_table` - often best exact-match performer on the tested 6-character secrets

### Interesting but not default-worthy yet
- `secret_path_codebook` - good focused runs, worse broad consistency
- `secret_query_codebook` - same story
- `mnemonic_path_table` - can pop on some secrets, but not robust enough yet

### Mostly negative / unstable for 3B
- `tri_tool_table`
- `tool_table`

## Important findings from the night

1. **3B models can sometimes execute exact 6-character sequences locally.**
   This is the main positive result.

2. **A lot of apparent failure was actually formatting variance, not planning failure.**
   The parser hardening changed the picture materially.

3. **Tool-family diversity hurts small models quickly.**
   The mixed tool tables were less stable than path-only or query-only tables.

4. **Secret-conditioned codebooks can help on some secrets, but not consistently enough yet.**
   They remain good research targets.

5. **The right metric is sequence correctness, not just raw decode length or one-character signal.**
   The updated summary metrics reflect that.

6. **Bounded finite runs are now in place and documented.**
   The workflow is ready for a real overnight local sweep.

## Changes made to repo files

- Added [`experiments/overnight_3b_sweep.py`](../experiments/overnight_3b_sweep.py)
- Added [`scripts/run_overnight_3b.sh`](../scripts/run_overnight_3b.sh)
- Updated [`src/agent.py`](../src/agent.py) with text-tool-call parsing and argument normalization
- Updated [`README.md`](../README.md) with overnight 3B instructions and candidate presets
- `uv.lock` corrected the editable package entry from the stale old name to the current package name

## Commit and repo state

Checkpoint commit:
- `89cc523` - `feat(sweep): add overnight 3b experiment runner`

This report was written after that checkpoint.

## Recommended next experiments

1. Run the default preset longer:
   ```bash
   TRIALS=8 BASELINE_TRIALS=4 ./scripts/run_overnight_3b.sh
   ```
2. Compare the same overnight matrix on one free remote small model if you find a stable free endpoint.
3. Try secret-conditioned variants only after you have a stable default baseline.
4. Add a morning-after comparison script that summarizes multiple run directories together.
5. Explore whether a detector trained on baseline traces can separate `path_table` and `query_table` under these lower-bandwidth 3B settings.

## Full run index

All generated run summaries from tonight:

- [`20260407-000254`](../findings/overnight_3b/20260407-000254/summary.md)
- [`20260407-000423`](../findings/overnight_3b/20260407-000423/summary.md)
- [`20260407-000654`](../findings/overnight_3b/20260407-000654/summary.md)
- [`20260407-001924`](../findings/overnight_3b/20260407-001924/summary.md)
- [`20260407-002424`](../findings/overnight_3b/20260407-002424/summary.md)
- [`20260407-003856`](../findings/overnight_3b/20260407-003856/summary.md)
- [`20260407-004640`](../findings/overnight_3b/20260407-004640/summary.md)
- [`20260407-005024`](../findings/overnight_3b/20260407-005024/summary.md)
- [`20260407-005726`](../findings/overnight_3b/20260407-005726/summary.md)
- [`20260407-010726`](../findings/overnight_3b/20260407-010726/summary.md)
- [`20260407-010853`](../findings/overnight_3b/20260407-010853/summary.md)
- [`20260407-011502`](../findings/overnight_3b/20260407-011502/summary.md)
- [`20260407-014727`](../findings/overnight_3b/20260407-014727/summary.md)
- [`20260407-020530`](../findings/overnight_3b/20260407-020530/summary.md)
- [`20260407-021122`](../findings/overnight_3b/20260407-021122/summary.md)
- [`20260407-021755`](../findings/overnight_3b/20260407-021755/summary.md)
- [`20260407-023611`](../findings/overnight_3b/20260407-023611/summary.md)
- [`20260407-024307`](../findings/overnight_3b/20260407-024307/summary.md)
- [`20260407-030821`](../findings/overnight_3b/20260407-030821/summary.md)
- [`20260407-031027`](../findings/overnight_3b/20260407-031027/summary.md)
- [`20260407-032046`](../findings/overnight_3b/20260407-032046/summary.md)
- [`20260407-033207`](../findings/overnight_3b/20260407-033207/summary.md)

If you want raw details for any run, open the matching sibling files in the same directory:
- `manifest.json`
- `baseline.json`
- `results.json`
- `summary.md`
