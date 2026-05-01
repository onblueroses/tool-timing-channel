"""Activation probe under distribution coupling.

The existential question: does AUC=1.0 activation detection survive
when the entropy gap is closed via distribution coupling?

Previous probe experiment used greedy normal baselines, creating an
artificial gap. This experiment matches sampling conditions:
- Normal: stochastic sample from coupled distribution (temp + top_p)
- Stego: arithmetic coding over the same coupled distribution

If the probe STILL gets AUC~1.0, activation probes detect something
deeper than entropy bias. If AUC drops toward 0.5, coupling defeats
both token-level AND activation-level detection.

Usage:
    MODEL=Qwen/Qwen2.5-3B python experiments/coupled_probe.py
    MODEL=Qwen/Qwen2.5-7B python experiments/coupled_probe.py
"""

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import torch

from src.token_stego.activation_probe import evaluate_probe, train_probe
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/coupled_probe")

PROMPTS = [
    "The history of computing began with",
    "In recent years, advances in machine learning have",
    "The weather forecast for tomorrow indicates",
    "A comprehensive review of the literature shows",
    "Modern software engineering practices emphasize",
    "The development of quantum computing has",
    "Recent studies in neuroscience suggest that",
    "The economic impact of automation on",
    "Climate change research has demonstrated that",
    "The evolution of programming languages reflects",
    "Artificial intelligence applications in healthcare include",
    "Space exploration missions have revealed",
]

# Grid of coupling parameters to sweep
COUPLING_GRID = [
    {"temperature": 1.0, "top_p": 1.0},  # Raw (no coupling) - baseline
    {"temperature": 0.9, "top_p": 1.0},  # Temperature only
    {"temperature": 1.0, "top_p": 0.9},  # Top-p only
    {"temperature": 0.9, "top_p": 0.9},  # Both (the combo that worked in Phase 8b)
    {"temperature": 0.8, "top_p": 0.9},  # Tighter temperature
    {"temperature": 0.7, "top_p": 0.85},  # Aggressive coupling
]


def stochastic_sample(dist: list[float], rng: random.Random) -> int:
    """Sample a token from a probability distribution."""
    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(dist):
        cumsum += p
        if cumsum >= r:
            return i
    return len(dist) - 1


def generate_normal_stochastic(
    model: StegoModel,
    prompt: str,
    n_tokens: int,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> list[int]:
    """Generate tokens via stochastic sampling with coupling params."""
    context = model.tokenize(prompt)
    tokens: list[int] = []
    for _ in range(n_tokens):
        dist = model.get_distribution(context + tokens, temperature, top_p)
        tokens.append(stochastic_sample(dist, rng))
    return tokens


def compute_entropy_auc(
    normal_entropies: list[float], stego_entropies: list[float]
) -> float:
    """Compute oriented AUC from entropy lists (higher = more detectable)."""
    labels = [0] * len(normal_entropies) + [1] * len(stego_entropies)
    scores = normal_entropies + stego_entropies

    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
        i = j

    # Return oriented AUC (always >= 0.5)
    return max(auc, 1 - auc)


def mean_token_entropy(
    model: StegoModel,
    tokens: list[int],
    context_ids: list[int],
    temperature: float,
    top_p: float,
) -> float:
    """Compute mean per-token entropy for a generated sequence."""
    current = list(context_ids)
    entropies = []
    for tok in tokens:
        dist = model.get_distribution(current, temperature, top_p)
        p = dist[tok]
        entropies.append(-math.log(p) if p > 0 else 10.0)
        current.append(tok)
    return sum(entropies) / len(entropies) if entropies else 0.0


def run_single_condition(
    model: StegoModel,
    temperature: float,
    top_p: float,
    n_samples: int,
    max_tokens: int,
    secret: str,
    key: bytes | None,
) -> dict:
    """Run probe experiment for one coupling configuration."""
    key_label = "none" if key is None else key.decode("utf-8", errors="replace")
    print(f"\n--- T={temperature}, top_p={top_p}, key={key_label} ---")

    rng = random.Random(42)
    channel = ArithmeticStegoChannel(
        model, key=key, temperature=temperature, top_p=top_p
    )

    normal_snaps: list[LatentSnapshot] = []
    stego_snaps: list[LatentSnapshot] = []
    normal_entropies: list[float] = []
    stego_entropies: list[float] = []
    stego_lengths: list[int] = []

    # Stego first - its actual length determines the matched-normal length.
    print("  Stego (coupled arithmetic coding)...")
    stego_runs: list[tuple[str, list[int], list[int]]] = []
    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        context_ids = model.tokenize(prompt)
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        stego_runs.append((prompt, context_ids, output.tokens))
        stego_lengths.append(len(output.tokens))

        with LatentCapture(model) as cap:
            full_ids = context_ids + output.tokens
            cap.clear()
            model.get_distribution(full_ids[:-1], temperature, top_p)
            stego_snaps.append(cap.snapshot())

        ent = mean_token_entropy(model, output.tokens, context_ids, temperature, top_p)
        stego_entropies.append(ent)

    # Normal: per-pair length-matched to stego sample i.
    print("  Normal (stochastic, length-matched)...")
    for i, (prompt, context_ids, stego_toks) in enumerate(stego_runs):
        n_tok = len(stego_toks)
        tokens = generate_normal_stochastic(
            model, prompt, n_tok, temperature, top_p, rng
        )

        with LatentCapture(model) as cap:
            full_ids = context_ids + tokens
            model.get_distribution(full_ids[:-1], temperature, top_p)
            normal_snaps.append(cap.snapshot())

        ent = mean_token_entropy(model, tokens, context_ids, temperature, top_p)
        normal_entropies.append(ent)

    # Train and evaluate activation probe
    print("  Training probe...")
    indices = list(range(n_samples))
    random.Random(42).shuffle(indices)
    split = n_samples * 3 // 4
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_normal = [normal_snaps[i] for i in train_idx]
    train_stego = [stego_snaps[i] for i in train_idx]
    test_normal = [normal_snaps[i] for i in test_idx]
    test_stego = [stego_snaps[i] for i in test_idx]

    probe = train_probe(train_normal, train_stego, epochs=200)
    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)
    metrics = evaluate_probe(probe, test_snaps, test_labels)

    # Token-level entropy AUC
    entropy_auc = compute_entropy_auc(normal_entropies, stego_entropies)

    result = {
        "temperature": temperature,
        "top_p": top_p,
        "key": key_label,
        "probe_auc": metrics["auc"],
        "probe_accuracy": metrics["accuracy"],
        "entropy_auc": entropy_auc,
        "normal_entropy_mean": sum(normal_entropies) / len(normal_entropies),
        "stego_entropy_mean": sum(stego_entropies) / len(stego_entropies),
        "entropy_gap": abs(
            sum(normal_entropies) / len(normal_entropies)
            - sum(stego_entropies) / len(stego_entropies)
        ),
        "stego_len_mean": sum(stego_lengths) / len(stego_lengths),
        "stego_len_min": min(stego_lengths),
        "stego_len_max": max(stego_lengths),
        "n_samples": n_samples,
    }

    print(f"  Probe AUC: {metrics['auc']:.3f}")
    print(f"  Probe accuracy: {metrics['accuracy']:.3f}")
    print(f"  Entropy AUC: {entropy_auc:.3f}")
    print(
        f"  Entropy gap: {result['entropy_gap']:.3f} "
        f"(normal={result['normal_entropy_mean']:.3f}, "
        f"stego={result['stego_entropy_mean']:.3f})"
    )

    return result


def run_experiment() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    n_samples = int(os.environ.get("N", "20"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "50"))
    # Default secret long enough to fill max_tokens at any reasonable bpt,
    # so length-match is naturally tight rather than dominated by the
    # encoder's early-stop boundary at 16 bits.
    secret = os.environ.get(
        "SECRET",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG IN THE FIELD AT NIGHT",
    )
    key_env = os.environ.get("KEY", "probe-test")
    key: bytes | None = (
        None if key_env.lower() in ("", "none", "off") else key_env.encode("utf-8")
    )

    # Use bfloat16 for all models (avoids NaN on Qwen at float16)
    dtype = torch.bfloat16
    print(f"Model: {model_name}")
    print(f"Samples per condition: {n_samples} normal + {n_samples} stego")
    print(f"Secret ({len(secret) * 8} bits): {secret!r}")
    print(f"Max tokens: {max_tokens}")
    print(f"Key: {'NONE (unencrypted)' if key is None else key_env}")
    print(f"Coupling grid: {len(COUPLING_GRID)} conditions")
    print(f"dtype: {dtype}")
    print()

    model = StegoModel(model_name, dtype=dtype)
    print("Loading model...")
    model.load()
    print("Model loaded.\n")

    results = []
    for params in COUPLING_GRID:
        result = run_single_condition(
            model=model,
            temperature=params["temperature"],
            top_p=params["top_p"],
            n_samples=n_samples,
            max_tokens=max_tokens,
            secret=secret,
            key=key,
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Temp':>5} {'TopP':>5} {'Probe AUC':>10} {'Ent AUC':>10} {'Ent Gap':>10}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['temperature']:>5.1f} {r['top_p']:>5.2f} "
            f"{r['probe_auc']:>10.3f} {r['entropy_auc']:>10.3f} "
            f"{r['entropy_gap']:>10.3f}"
        )

    # Verdict
    best = min(results, key=lambda r: r["probe_auc"])
    print(
        f"\nBest probe AUC: {best['probe_auc']:.3f} at T={best['temperature']}, top_p={best['top_p']}"
    )
    if best["probe_auc"] > 0.9:
        print("VERDICT: Activation probes SURVIVE coupling. Detection is robust.")
    elif best["probe_auc"] > 0.7:
        print("VERDICT: Activation probes WEAKENED but still effective.")
    elif best["probe_auc"] > 0.55:
        print("VERDICT: Activation probes DEGRADED. Coupling partially defeats them.")
    else:
        print(
            "VERDICT: Activation probes DEFEATED. Coupling breaks both detection layers."
        )

    # Save
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "secret": secret,
                "key": "none" if key is None else key_env,
                "length_match": "per-pair (normal matched to stego length)",
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "dtype": str(dtype),
                "results": results,
                "best_probe_auc": best["probe_auc"],
                "best_params": {
                    "temperature": best["temperature"],
                    "top_p": best["top_p"],
                },
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
