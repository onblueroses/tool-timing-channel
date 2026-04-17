"""Detectability scaling experiment: does probe AUC converge to 0.5?

Tests the theoretical guarantee of arithmetic coding steganography:
if the implementation is correct and encryption is enabled, the output
token sequence is drawn from the model's own distribution. Hidden states
from processing stego text should be statistically indistinguishable
from normal text hidden states.

The experiment generates stego and normal samples at increasing scale,
trains probes at each scale, and plots AUC vs sample size. If AUC
converges to 0.5, the system is provably undetectable at the activation
level. If it stays high, there's an implementation leak.

Two probe variants tested at each scale:
- Last-layer probe (mean-pooled last hidden layer)
- Multi-layer probe (concatenated mean-pooled features from all captured layers)
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.token_stego.activation_probe import (
    evaluate_probe,
    train_probe,
)
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
ENCRYPTION_KEY = b"scaling-test-key-2026"

# Scaling schedule: (n_train, n_test) per class
SCALES = [
    (10, 5),
    (25, 10),
    (50, 20),
    (100, 40),
    (200, 80),
]

# Diverse prompts to avoid overfitting to prompt structure
PROMPTS = [
    "Explain how computers process information.",
    "Describe the water cycle in simple terms.",
    "What are the benefits of regular exercise?",
    "How does photosynthesis work?",
    "Explain the concept of supply and demand.",
    "What causes the seasons to change?",
    "How do vaccines help the immune system?",
    "Describe the process of making bread.",
    "What is the role of DNA in living organisms?",
    "How does electricity reach homes from power plants?",
    "Explain how airplanes stay in the air.",
    "What are the main components of a healthy diet?",
    "How do earthquakes occur?",
    "Describe how a search engine works.",
    "What is the greenhouse effect?",
    "How does the human heart pump blood?",
    "Explain how batteries store energy.",
    "What causes thunder and lightning?",
    "How do plants absorb water from soil?",
    "Describe the process of recycling plastic.",
]

# Secret messages of varying lengths
SECRET_POOL = [
    "HI",
    "OK",
    "NO",
    "GO",
    "UP",
    "YES",
    "THE",
    "AND",
    "FOR",
    "BUT",
    "TEST",
    "SEND",
    "HELP",
    "STOP",
    "WAIT",
    "HELLO",
    "WORLD",
    "AGENT",
    "STEGO",
    "COVERT",
    "SECRET",
    "HIDDEN",
    "SIGNAL",
    "DECODE",
    "ENCODE",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def generate_samples(
    model: StegoModel,
    n: int,
    mode: str,  # "normal" or "stego"
) -> tuple[list[LatentSnapshot], list[tuple[list[int], list[int]]]]:
    """Generate n samples with latent capture.

    Returns (snapshots, pairs) where pairs are (context_ids, token_ids).
    """
    snapshots: list[LatentSnapshot] = []
    pairs: list[tuple[list[int], list[int]]] = []

    channel = ArithmeticStegoChannel(
        model=model, key=ENCRYPTION_KEY, temperature=TEMPERATURE, top_p=TOP_P
    )

    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        context_ids = model.tokenize(prompt)

        if mode == "stego":
            secret = SECRET_POOL[i % len(SECRET_POOL)]
            output = channel.encode(secret, prompt, max_tokens=MAX_TOKENS)
            token_ids = output.tokens
        else:
            token_ids = _sample_normal(model, context_ids, MAX_TOKENS)

        # Capture activations via get_distribution (handles device internally)
        full_ids = list(context_ids) + list(token_ids)
        with LatentCapture(model) as cap:
            model.get_distribution(full_ids[:-1], TEMPERATURE, TOP_P)
            snap = cap.snapshot()
        snapshots.append(snap)
        pairs.append((list(context_ids), list(token_ids)))

    return snapshots, pairs


def _sample_normal(
    model: StegoModel, context_ids: list[int], max_tokens: int
) -> list[int]:
    """Sample tokens from the model's distribution (not stego)."""
    current = list(context_ids)
    tokens: list[int] = []
    for _ in range(max_tokens):
        dist = model.get_distribution(current, TEMPERATURE, TOP_P)
        # True stochastic sampling from the distribution
        token_id = random.choices(range(len(dist)), weights=dist)[0]
        tokens.append(token_id)
        current.append(token_id)

        eos = getattr(model.tokenizer, "eos_token_id", None)
        if eos is not None and token_id == eos:
            break
    return tokens


def run_scale(
    model: StegoModel,
    n_train: int,
    n_test: int,
) -> dict:
    """Run probe training and evaluation at a given scale."""
    total = n_train + n_test
    log(f"\n--- Scale: {n_train} train / {n_test} test per class ---")

    # Generate samples
    log(f"  Generating {total} normal samples...")
    normal_snaps, normal_pairs = generate_samples(model, total, "normal")

    log(f"  Generating {total} stego samples...")
    stego_snaps, stego_pairs = generate_samples(model, total, "stego")

    # Split
    train_normal = normal_snaps[:n_train]
    test_normal = normal_snaps[n_train:]
    train_stego = stego_snaps[:n_train]
    test_stego = stego_snaps[n_train:]

    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)

    # Determine available layers
    last_layer = max(normal_snaps[0].hidden_states.keys())
    all_layers = sorted(normal_snaps[0].hidden_states.keys())

    results = {}

    # --- Last-layer probe ---
    log(f"  Training last-layer probe (layer={last_layer})...")
    probe_last = train_probe(
        train_normal, train_stego, layer=last_layer, epochs=300, lr=1e-3
    )
    metrics_last = evaluate_probe(probe_last, test_snaps, test_labels, layer=last_layer)
    results["last_layer"] = {
        "auc": metrics_last["auc"],
        "accuracy": metrics_last["accuracy"],
        "layer": last_layer,
    }
    log(f"  Last-layer AUC: {metrics_last['auc']:.3f}")

    # --- Multi-layer probe ---
    log(f"  Training multi-layer probe ({len(all_layers)} layers)...")
    probe_multi = train_probe(
        train_normal, train_stego, layer=None, epochs=300, lr=1e-3
    )
    metrics_multi = evaluate_probe(probe_multi, test_snaps, test_labels, layer=None)
    results["multi_layer"] = {
        "auc": metrics_multi["auc"],
        "accuracy": metrics_multi["accuracy"],
        "n_layers": len(all_layers),
    }
    log(f"  Multi-layer AUC: {metrics_multi['auc']:.3f}")

    results["n_train"] = n_train
    results["n_test"] = n_test
    return results


def main():
    log(f"Model: {MODEL_NAME}")
    log(f"Temperature: {TEMPERATURE}, top_p: {TOP_P}")
    log(f"Max tokens: {MAX_TOKENS}")
    log(f"Encryption: enabled (key={ENCRYPTION_KEY.decode()})")
    log(f"Scales: {SCALES}")

    model = StegoModel(MODEL_NAME, dtype=torch.bfloat16)
    log("\nLoading model...")
    model.load()
    log("Model loaded.")

    all_results: list[dict] = []

    for n_train, n_test in SCALES:
        t0 = time.time()
        result = run_scale(model, n_train, n_test)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        all_results.append(result)
        log(f"  Completed in {elapsed:.1f}s")

    # Summary table
    log("\n" + "=" * 70)
    log("SCALING RESULTS")
    log("=" * 70)
    log(f"{'n_train':>8} {'n_test':>7} {'last_AUC':>9} {'multi_AUC':>10} {'time':>7}")
    log("-" * 70)
    for r in all_results:
        log(
            f"{r['n_train']:>8} {r['n_test']:>7} "
            f"{r['last_layer']['auc']:>9.3f} {r['multi_layer']['auc']:>10.3f} "
            f"{r['elapsed_seconds']:>6.1f}s"
        )

    # Verdict
    final_last = all_results[-1]["last_layer"]["auc"]
    final_multi = all_results[-1]["multi_layer"]["auc"]
    log("\n" + "=" * 70)
    if final_last < 0.6 and final_multi < 0.6:
        log("VERDICT: Probe AUC converges toward 0.5 at scale.")
        log("The theoretical guarantee holds: encrypted arithmetic stego")
        log("is undetectable at the activation level.")
    elif final_last > 0.7 or final_multi > 0.7:
        log("VERDICT: Probe AUC remains HIGH at scale.")
        log("There is a real implementation leak - the theoretical")
        log("guarantee is broken somewhere in the pipeline.")
    else:
        log("VERDICT: Inconclusive. AUC is between 0.6-0.7.")
        log("May need even larger scale or the leak is subtle.")
    log("=" * 70)

    # Save results
    output = {
        "model": MODEL_NAME,
        "config": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "encryption": True,
            "key": ENCRYPTION_KEY.decode(),
        },
        "scales": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    findings_dir = Path(__file__).parent.parent / "findings"
    findings_dir.mkdir(exist_ok=True)
    out_path = (
        findings_dir / f"detectability_scaling_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(output, indent=2))
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
