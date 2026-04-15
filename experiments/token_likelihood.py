"""Token-likelihood steganography experiment.

Encodes secret messages into generated text using arithmetic coding
over token probability distributions. Measures capacity, naturalness,
and recovery accuracy.

Usage:
    MODEL=Qwen/Qwen2.5-3B SECRET=HELLO TRIALS=5 python experiments/token_likelihood.py
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

# Default cover prompts for text generation
DEFAULT_PROMPTS = [
    "The history of computing began with",
    "In recent years, advances in machine learning have",
    "The weather forecast for tomorrow indicates",
    "A comprehensive review of the literature shows",
]


def run_experiment() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    secret = os.environ.get("SECRET", "HELLO")
    n_trials = int(os.environ.get("TRIALS", "5"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "200"))
    prompt = os.environ.get("PROMPT", DEFAULT_PROMPTS[0])

    print(f"Model: {model_name}")
    print(f"Secret: {secret} ({len(secret)} chars, {len(secret) * 8} bits)")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Trials: {n_trials}, Max tokens: {max_tokens}")
    print()

    # Determine dtype based on model size
    dtype = torch.float32
    if "7B" in model_name or "7b" in model_name:
        dtype = torch.float16
        print("Using float16 for 7B model")

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = StegoModel(model_name, dtype=dtype)
    model.load()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    channel = ArithmeticStegoChannel(model)

    results: list[dict] = []
    exact_count = 0

    for trial in range(1, n_trials + 1):
        t0 = time.time()

        # Encode
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        encode_time = time.time() - t0

        # Measure
        t1 = time.time()
        metrics = channel.measure(secret, output.text, prompt)
        measure_time = time.time() - t1

        total_time = encode_time + measure_time

        if metrics.exact_recovery:
            exact_count += 1

        status = "OK" if metrics.exact_recovery else "FAIL"
        print(
            f"  [{status}] Trial {trial}/{n_trials} | "
            f"BER: {metrics.bit_error_rate:.3f} | "
            f"bits/tok: {metrics.bits_per_token:.2f} | "
            f"ppl: {metrics.perplexity:.1f} | "
            f"tokens: {metrics.tokens_generated} | "
            f"time: {total_time:.1f}s"
        )

        # Show a snippet of generated text
        snippet = output.text[:80].replace("\n", " ")
        print(f"         Text: {snippet}...")
        print()

        results.append(
            {
                "trial": trial,
                "model": model_name,
                "secret": secret,
                "prompt": prompt,
                "text": output.text,
                "tokens_generated": output.tokens_generated,
                "bits_encoded": output.bits_encoded,
                "bit_error_rate": metrics.bit_error_rate,
                "bits_per_token": metrics.bits_per_token,
                "perplexity": metrics.perplexity,
                "exact_recovery": metrics.exact_recovery,
                "encode_time": encode_time,
                "measure_time": measure_time,
            }
        )

    # Summary
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Secret: {secret}")
    mean_ber = sum(r["bit_error_rate"] for r in results) / len(results)
    mean_bpt = sum(r["bits_per_token"] for r in results) / len(results)
    mean_ppl = sum(r["perplexity"] for r in results) / len(results)
    print(f"Mean BER: {mean_ber:.3f}")
    print(f"Mean bits/tok: {mean_bpt:.2f}")
    print(f"Mean perplexity: {mean_ppl:.1f}")
    print(f"Exact recovery: {exact_count}/{n_trials}")
    print("=" * 60)

    # Save results
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "secret": secret,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "n_trials": n_trials,
                "summary": {
                    "mean_ber": mean_ber,
                    "mean_bits_per_token": mean_bpt,
                    "mean_perplexity": mean_ppl,
                    "exact_count": exact_count,
                },
                "trials": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
