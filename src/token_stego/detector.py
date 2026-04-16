"""Steganography detector for token-likelihood encoded text.

Two detection methods:
1. Entropy-based: analyzes per-token entropy and log-probability patterns
2. CDF uniformity test: under normal sampling, CDF values should be uniform on [0,1]

If arithmetic coding is implemented correctly (zero KL-divergence), both
methods should produce AUC ~0.5 (no better than random).
"""

import math
import random
from dataclasses import dataclass, field

from .model import StegoModel


@dataclass
class DetectionResult:
    """Result of a steganography detection attempt."""

    score: float  # 0.0 = normal, 1.0 = stego
    per_token_scores: list[float] = field(default_factory=list)
    method: str = ""
    p_value: float = 1.0  # for statistical tests


class StegoDetector:
    """Detects token-likelihood steganography in generated text."""

    def __init__(self, model: StegoModel) -> None:
        self._model = model

    def score_entropy(self, text: str, prompt: str) -> DetectionResult:
        """Entropy-based detection.

        Analyzes whether per-token entropy and log-probability patterns
        differ from what normal sampling would produce.
        """
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        per_token_scores: list[float] = []
        current_context = list(context_ids)

        for token_id in stego_ids:
            dist = self._model.get_distribution(current_context)

            # Entropy of the distribution at this position
            entropy = -sum(p * math.log(p + 1e-30) for p in dist if p > 0)

            # Log-prob of the chosen token
            prob = dist[token_id]
            log_prob = math.log(prob + 1e-30)

            # Surprise: how unexpected is this token?
            # Under normal sampling, surprise ~ entropy
            # Under stego, surprise may deviate systematically
            surprise = -log_prob
            normalized_surprise = surprise / (entropy + 1e-10)

            per_token_scores.append(normalized_surprise)
            current_context.append(token_id)

        # Aggregate: mean normalized surprise
        # Normal text: ~1.0, Stego text: may differ
        if per_token_scores:
            mean_score = sum(per_token_scores) / len(per_token_scores)
            # Map to [0, 1] range where 1 = likely stego
            # Values far from 1.0 are suspicious
            score = min(1.0, abs(mean_score - 1.0))
        else:
            score = 0.0

        return DetectionResult(
            score=score,
            per_token_scores=per_token_scores,
            method="entropy",
        )

    def score_cdf_uniformity(
        self, text: str, prompt: str, seed: int | None = None
    ) -> DetectionResult:
        """CDF uniformity test for steganography detection.

        Under normal sampling from distribution P, the CDF value
        F(x) = sum(P(t) for t <= x) is uniform on [0,1].

        Under arithmetic coding, this should also hold (provably
        zero KL-divergence). If it doesn't, the implementation has a bug.

        Args:
            seed: If provided, use a deterministic RNG for the randomized
                PIT, making results reproducible.
        """
        rng = random.Random(seed) if seed is not None else random
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        cdf_values: list[float] = []
        current_context = list(context_ids)

        for token_id in stego_ids:
            dist = self._model.get_distribution(current_context)

            # Randomized PIT for discrete distributions:
            # U ~ Uniform(F(x-1), F(x)] to avoid discrete CDF artifacts
            cdf_upper = sum(dist[: token_id + 1])
            cdf_lower = sum(dist[:token_id]) if token_id > 0 else 0.0
            cdf = cdf_lower + rng.random() * (cdf_upper - cdf_lower)
            cdf_values.append(cdf)
            current_context.append(token_id)

        if not cdf_values:
            return DetectionResult(score=0.0, method="cdf_uniformity", p_value=1.0)

        # Kolmogorov-Smirnov test against uniform [0,1]
        # Sort CDF values and compare against expected uniform quantiles
        n = len(cdf_values)
        sorted_vals = sorted(cdf_values)

        ks_stat = 0.0
        for i, val in enumerate(sorted_vals):
            expected = (i + 1) / n
            ks_stat = max(ks_stat, abs(val - expected))
            expected_prev = i / n
            ks_stat = max(ks_stat, abs(val - expected_prev))

        # Approximate p-value for KS test
        # Using the asymptotic formula: P(D > x) ~ 2*exp(-2*n*x^2)
        lambda_val = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks_stat
        if lambda_val > 0:
            p_value = 2.0 * math.exp(-2.0 * lambda_val * lambda_val)
            p_value = min(1.0, max(0.0, p_value))
        else:
            p_value = 1.0

        # Score: low p-value = likely stego (or broken implementation)
        score = 1.0 - p_value

        return DetectionResult(
            score=score,
            per_token_scores=cdf_values,
            method="cdf_uniformity",
            p_value=p_value,
        )

    def score(self, text: str, prompt: str, seed: int | None = None) -> DetectionResult:
        """Combined detection score using both methods."""
        entropy_result = self.score_entropy(text, prompt)
        cdf_result = self.score_cdf_uniformity(text, prompt, seed=seed)

        # Use the stronger signal
        combined_score = max(entropy_result.score, cdf_result.score)

        return DetectionResult(
            score=combined_score,
            per_token_scores=entropy_result.per_token_scores,
            method="combined",
            p_value=cdf_result.p_value,
        )
