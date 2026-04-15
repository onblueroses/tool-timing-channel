"""Multi-model token-likelihood steganography comparison.

Runs token-likelihood stego across Qwen 2.5 1.5B/3B/7B with the same
secrets used in tool-call experiments for direct comparison.

Usage:
    TRIALS=3 python experiments/token_model_comparison.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/token_likelihood")

MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

# Same secrets as tool-call experiments for direct comparison
SECRETS = ["HI", "HELLO", "SEARCH", "HELLOWORLD", "CAT", "KEY", "SIGNAL"]

PROMPT = "The history of computing began with"


def run_comparison() -> None:
    models_str = os.environ.get("MODELS", ",".join(MODELS))
    model_names = [m.strip() for m in models_str.split(",")]
    n_trials = int(os.environ.get("TRIALS", "3"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "200"))
    secrets_str = os.environ.get("SECRETS", ",".join(SECRETS))
    secrets = [s.strip() for s in secrets_str.split(",")]

    all_results: list[dict] = []

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        dtype = torch.float32
        if "7B" in model_name or "7b" in model_name:
            dtype = torch.float16

        print("Loading model...")
        t0 = time.time()
        model = StegoModel(model_name, dtype=dtype)
        model.load()
        print(f"Loaded in {time.time() - t0:.1f}s\n")

        channel = ArithmeticStegoChannel(model)

        for secret in secrets:
            num_bits = len(secret) * 8
            # Skip secrets that need more tokens than max_tokens
            # (rough estimate: ~1 bit/token for arithmetic coding)
            if num_bits > max_tokens:
                print(f"  {secret}: SKIPPED (needs ~{num_bits} tokens)")
                continue

            exact_count = 0
            trial_results: list[dict] = []

            for trial in range(1, n_trials + 1):
                t0 = time.time()
                output = channel.encode(secret, PROMPT, max_tokens=max_tokens)
                metrics = channel.measure(secret, output.text, PROMPT)
                elapsed = time.time() - t0

                if metrics.exact_recovery:
                    exact_count += 1

                trial_results.append(
                    {
                        "trial": trial,
                        "bit_error_rate": metrics.bit_error_rate,
                        "bits_per_token": metrics.bits_per_token,
                        "perplexity": metrics.perplexity,
                        "exact_recovery": metrics.exact_recovery,
                        "tokens_generated": metrics.tokens_generated,
                        "elapsed": elapsed,
                    }
                )

            mean_ber = sum(r["bit_error_rate"] for r in trial_results) / len(
                trial_results
            )
            mean_bpt = sum(r["bits_per_token"] for r in trial_results) / len(
                trial_results
            )
            mean_ppl = sum(r["perplexity"] for r in trial_results) / len(trial_results)

            status = (
                "OK"
                if exact_count == n_trials
                else "PARTIAL"
                if exact_count > 0
                else "FAIL"
            )
            print(
                f"  [{status}] {secret:12s} | "
                f"BER: {mean_ber:.3f} | "
                f"bits/tok: {mean_bpt:.2f} | "
                f"ppl: {mean_ppl:.1f} | "
                f"exact: {exact_count}/{n_trials}"
            )

            all_results.append(
                {
                    "model": model_name,
                    "secret": secret,
                    "n_trials": n_trials,
                    "mean_ber": mean_ber,
                    "mean_bits_per_token": mean_bpt,
                    "mean_perplexity": mean_ppl,
                    "exact_count": exact_count,
                    "trials": trial_results,
                }
            )

        # Free model memory before loading next
        del model
        del channel
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print(f"\n{'=' * 80}")
    print(
        f"{'Model':25s} | {'Secret':12s} | {'BER':>6s} | {'b/tok':>5s} | {'ppl':>8s} | {'Exact':>8s}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['model']:25s} | {r['secret']:12s} | "
            f"{r['mean_ber']:6.3f} | {r['mean_bits_per_token']:5.2f} | "
            f"{r['mean_perplexity']:8.1f} | "
            f"{r['exact_count']}/{r['n_trials']}"
        )
    print("=" * 80)

    # Save
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"comparison_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "models": model_names,
                "secrets": secrets,
                "n_trials": n_trials,
                "max_tokens": max_tokens,
                "prompt": PROMPT,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_comparison()
