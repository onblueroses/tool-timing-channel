"""Detection experiment for token-likelihood steganography.

Generates stego and normal texts, runs detector on both,
computes ROC curve and AUC.

Usage:
    MODEL=Qwen/Qwen2.5-3B N=20 python experiments/token_detection.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.detector import StegoDetector
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/token_likelihood")

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


def generate_normal_text(model: StegoModel, prompt: str, max_tokens: int) -> str:
    """Generate text via greedy sampling (no steganography)."""
    context_ids = model.tokenize(prompt)
    generated: list[int] = []

    for _ in range(max_tokens):
        dist = model.get_distribution(context_ids + generated)
        # Greedy: pick highest probability token
        token_id = max(range(len(dist)), key=lambda i: dist[i])
        generated.append(token_id)

        # Stop on EOS
        if (
            hasattr(model.tokenizer, "eos_token_id")
            and token_id == model.tokenizer.eos_token_id
        ):
            break

    return model.detokenize(generated)


def compute_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC from labels (0=normal, 1=stego) and scores."""
    # Sort by score only (not label) to avoid tie-breaking bias
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    i = 0
    while i < len(pairs):
        # Process all tied scores together
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


def find_rate_at_threshold(
    labels: list[int], scores: list[float], target_fpr: float
) -> float:
    """Find TPR at a given FPR threshold."""
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / total_neg
        if fpr > target_fpr:
            return (tp - (1 if label == 1 else 0)) / total_pos
    return tp / total_pos


def run_detection() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    n_samples = int(os.environ.get("N", "20"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    secret = os.environ.get("SECRET", "HI")

    dtype = torch.float32
    if "7B" in model_name or "7b" in model_name:
        dtype = torch.float16

    print(f"Model: {model_name}")
    print(f"Samples: {n_samples} stego + {n_samples} normal")
    print(f"Secret: {secret}")
    print()

    print("Loading model...")
    model = StegoModel(model_name, dtype=dtype)
    model.load()

    channel = ArithmeticStegoChannel(model)
    detector = StegoDetector(model)

    labels: list[int] = []
    entropy_scores: list[float] = []
    cdf_scores: list[float] = []
    combined_scores: list[float] = []

    # Generate stego texts
    print("Generating stego texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        output = channel.encode(secret, prompt, max_tokens=max_tokens)

        result = detector.score(output.text, prompt)
        labels.append(1)
        entropy_scores.append(detector.score_entropy(output.text, prompt).score)
        cdf_scores.append(detector.score_cdf_uniformity(output.text, prompt).score)
        combined_scores.append(result.score)

        print(
            f"  Stego {i + 1}/{n_samples}: score={result.score:.3f} p={result.p_value:.3f}"
        )

    # Generate normal texts
    print("\nGenerating normal texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        text = generate_normal_text(model, prompt, max_tokens)

        result = detector.score(text, prompt)
        labels.append(0)
        entropy_scores.append(detector.score_entropy(text, prompt).score)
        cdf_scores.append(detector.score_cdf_uniformity(text, prompt).score)
        combined_scores.append(result.score)

        print(
            f"  Normal {i + 1}/{n_samples}: score={result.score:.3f} p={result.p_value:.3f}"
        )

    # Compute ROC metrics
    auc_entropy = compute_auc(labels, entropy_scores)
    auc_cdf = compute_auc(labels, cdf_scores)
    auc_combined = compute_auc(labels, combined_scores)

    tpr_at_005 = find_rate_at_threshold(labels, combined_scores, 0.05)

    print(f"\n{'=' * 60}")
    print("Detection Results")
    print(f"{'=' * 60}")
    print(f"AUC (entropy):   {auc_entropy:.3f}")
    print(f"AUC (CDF):       {auc_cdf:.3f}")
    print(f"AUC (combined):  {auc_combined:.3f}")
    print(f"TPR@FPR=0.05:    {tpr_at_005:.3f}")
    print()

    if auc_combined < 0.55:
        print("RESULT: Near-undetectable (AUC ~0.5)")
    elif auc_combined < 0.7:
        print("RESULT: Weakly detectable")
    elif auc_combined < 0.9:
        print("RESULT: Moderately detectable")
    else:
        print("RESULT: Highly detectable - check implementation for distribution leaks")
    print(f"{'=' * 60}")

    # Save results
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"detection_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "secret": secret,
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "auc_entropy": auc_entropy,
                "auc_cdf": auc_cdf,
                "auc_combined": auc_combined,
                "tpr_at_fpr_005": tpr_at_005,
                "labels": labels,
                "entropy_scores": entropy_scores,
                "cdf_scores": cdf_scores,
                "combined_scores": combined_scores,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_detection()
