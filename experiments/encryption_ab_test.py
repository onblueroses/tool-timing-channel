"""A/B test: detection with vs without encrypt-then-encode.

Runs the same detection experiment twice on the same model:
  A) No encryption (baseline - matches GPU run 3 results)
  B) With SHAKE-256 encryption (should reduce AUC toward 0.5)

Usage:
    MODEL=Qwen/Qwen2.5-3B N=20 python experiments/encryption_ab_test.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.detector import StegoDetector
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/encryption_ab")

PROMPTS = [
    "The history of computing began with",
    "In recent years, advances in machine learning have",
    "The weather forecast for tomorrow indicates",
    "A comprehensive review of the literature shows",
    "Modern software engineering practices emphasize",
    "The development of quantum computing has",
    "Recent studies in neuroscience suggest that",
    "The economic impact of automation on",
]


def generate_normal_text(
    model: StegoModel,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Generate text via stochastic sampling from the model distribution."""
    import random

    context_ids = model.tokenize(prompt)
    generated: list[int] = []

    for _ in range(max_tokens):
        dist = model.get_distribution(context_ids + generated, temperature, top_p)
        # Sample from distribution (not greedy) to match what arithmetic
        # coding competes against. Greedy has artificially low entropy.
        r = random.random()
        cumulative = 0.0
        token_id = len(dist) - 1  # fallback to last token
        for idx, prob in enumerate(dist):
            cumulative += prob
            if r < cumulative:
                token_id = idx
                break
        generated.append(token_id)
        if (
            hasattr(model.tokenizer, "eos_token_id")
            and token_id == model.tokenizer.eos_token_id
        ):
            break

    return model.detokenize(generated)


def compute_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC with proper tie handling."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
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
        prev_fpr = fpr
        prev_tpr = tpr
        i = j
    return auc


def run_condition(
    label: str,
    model: StegoModel,
    detector: StegoDetector,
    key: bytes | None,
    secret: str,
    n_samples: int,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict:
    """Run one arm of the A/B test."""
    channel = ArithmeticStegoChannel(
        model, key=key, temperature=temperature, top_p=top_p
    )

    labels: list[int] = []
    entropy_scores: list[float] = []
    cdf_scores: list[float] = []

    print(f"\n--- {label} ---")
    print("Generating stego texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        ent = detector.score_entropy(output.text, prompt)
        cdf = detector.score_cdf_uniformity(output.text, prompt)
        labels.append(1)
        entropy_scores.append(ent.score)
        cdf_scores.append(cdf.score)
        print(
            f"  Stego {i + 1}/{n_samples}: entropy={ent.score:.3f} cdf={cdf.score:.3f}"
        )

    print("Generating normal texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        text = generate_normal_text(model, prompt, max_tokens, temperature, top_p)
        ent = detector.score_entropy(text, prompt)
        cdf = detector.score_cdf_uniformity(text, prompt)
        labels.append(0)
        entropy_scores.append(ent.score)
        cdf_scores.append(cdf.score)
        print(
            f"  Normal {i + 1}/{n_samples}: entropy={ent.score:.3f} cdf={cdf.score:.3f}"
        )

    auc_entropy = compute_auc(labels, entropy_scores)
    auc_cdf = compute_auc(labels, cdf_scores)
    # Oriented AUC: if raw < 0.5, the signal is inverted
    oriented_entropy = max(auc_entropy, 1 - auc_entropy)
    oriented_cdf = max(auc_cdf, 1 - auc_cdf)

    stego_ent = entropy_scores[:n_samples]
    normal_ent = entropy_scores[n_samples:]
    mean_stego = sum(stego_ent) / len(stego_ent)
    mean_normal = sum(normal_ent) / len(normal_ent)

    print(f"\n  Raw AUC entropy:      {auc_entropy:.3f}")
    print(f"  Raw AUC CDF:          {auc_cdf:.3f}")
    print(f"  Oriented AUC entropy: {oriented_entropy:.3f}")
    print(f"  Oriented AUC CDF:     {oriented_cdf:.3f}")
    print(f"  Mean entropy (stego): {mean_stego:.3f}")
    print(f"  Mean entropy (norm):  {mean_normal:.3f}")

    return {
        "condition": label,
        "key_used": key is not None,
        "auc_entropy_raw": auc_entropy,
        "auc_cdf_raw": auc_cdf,
        "auc_entropy_oriented": oriented_entropy,
        "auc_cdf_oriented": oriented_cdf,
        "mean_entropy_stego": mean_stego,
        "mean_entropy_normal": mean_normal,
        "entropy_scores": entropy_scores,
        "cdf_scores": cdf_scores,
        "labels": labels,
    }


def run_ab_test() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    n_samples = int(os.environ.get("N", "20"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    secret = os.environ.get("SECRET", "HI")

    dtype = torch.float32
    if "7B" in model_name or "7b" in model_name:
        dtype = torch.float16

    print("Encryption A/B Test")
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples} stego + {n_samples} normal per condition")
    print(f"Secret: {secret}")

    print("\nLoading model...")
    model = StegoModel(model_name, dtype=dtype)
    model.load()
    detector = StegoDetector(model)

    # A: No encryption (baseline)
    result_a = run_condition(
        "A: No encryption",
        model,
        detector,
        key=None,
        secret=secret,
        n_samples=n_samples,
        max_tokens=max_tokens,
    )

    # B: With encryption only
    result_b = run_condition(
        "B: SHAKE-256 encryption",
        model,
        detector,
        key=b"experiment-shared-key-2026",
        secret=secret,
        n_samples=n_samples,
        max_tokens=max_tokens,
    )

    # C: Encryption + distribution coupling (top-p=0.9, temp=0.9)
    result_c = run_condition(
        "C: Encrypt + top-p=0.9 temp=0.9",
        model,
        detector,
        key=b"experiment-shared-key-2026",
        secret=secret,
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=0.9,
    )

    # D: Distribution coupling only (no encryption)
    result_d = run_condition(
        "D: top-p=0.9 temp=0.9 (no encrypt)",
        model,
        detector,
        key=None,
        secret=secret,
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=0.9,
    )

    # Summary
    conditions = [
        ("A: Raw", result_a),
        ("B: Encrypt", result_b),
        ("C: Enc+Coupled", result_c),
        ("D: Coupled", result_d),
    ]

    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Metric':<25} ", end="")
    for name, _ in conditions:
        print(f"{name:>13}", end="")
    print()
    print(f"{'-' * 80}")

    for metric, key_name in [
        ("AUC entropy (oriented)", "auc_entropy_oriented"),
        ("AUC CDF (oriented)", "auc_cdf_oriented"),
        ("Mean entropy (stego)", "mean_entropy_stego"),
        ("Mean entropy (normal)", "mean_entropy_normal"),
    ]:
        print(f"{metric:<25} ", end="")
        for _, r in conditions:
            print(f"{r[key_name]:>13.3f}", end="")
        print()

    print(f"{'=' * 80}")

    # Best result
    best_name, best_result = min(conditions, key=lambda x: x[1]["auc_entropy_oriented"])
    print(
        f"\nLowest entropy AUC: {best_name} ({best_result['auc_entropy_oriented']:.3f})"
    )

    # Save
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"ab_test_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "secret": secret,
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "conditions": {name: r for name, r in conditions},
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_ab_test()
