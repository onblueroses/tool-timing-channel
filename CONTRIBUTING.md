# Contributing

## Setup

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego
cp .env.example .env  # add your OPENROUTER_API_KEY
uv sync --extra dev   # includes pytest and dev tools
```

## Running tests

```bash
uv run pytest tests/ -v
```

Tests use mocks - no GPU or API keys needed. Some tests require the `openai` package
and will be skipped if it's not installed.

## Running experiments

Each experiment is a standalone script in `experiments/`. Set the `MODEL` environment variable to target a specific model via OpenRouter:

```bash
MODEL="qwen/qwen3.5-27b" uv run python experiments/capability_ladder.py
```

Raw traces are saved to `findings/` (gitignored). Aggregated results go in `analysis/`.

## Adding a new channel

1. Create `src/channels/yourname.py` implementing `BaseChannel` (see `src/channels/base.py`)
2. Add an encoder prompt in `prompts/yourname_encoder_v1.txt`
3. Create `experiments/run_yourname.py` following the pattern in existing runners
4. Run at least N=10 trials and document results in `analysis/`

## Adding a new experiment

Experiment scripts live in `experiments/`. The structure is straightforward: configure via environment variables, run the agent, decode traces, print results, save raw data to `findings/`.

## Code style

- Python 3.12+, type hints on function signatures
- No linter enforced - match surrounding code style
- Docstrings on modules and public classes, not on every function

## Pull requests

- One experiment or feature per PR
- Include results (CER numbers, model used, number of trials) in the PR description
- Raw trace files stay in `findings/` and are gitignored - commit only aggregated analysis
