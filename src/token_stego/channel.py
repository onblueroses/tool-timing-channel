"""Arithmetic coding steganographic channel using HuggingFace models.

Encodes secret messages into generated text by choosing among tokens
based on their probability distributions. The output reads naturally
because all chosen tokens are high-probability.
"""

import math

from .base import StegoOutput, TokenStegoChannel, TokenStegoMetrics
from .coding import decode_message
from .model import StegoModel


class ArithmeticStegoChannel(TokenStegoChannel):
    """Steganographic channel using arithmetic coding over token distributions."""

    def __init__(self, model: StegoModel) -> None:
        self._model = model

    def encode(self, secret: str, prompt: str, max_tokens: int = 200) -> StegoOutput:
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))

        # Build context incrementally - each step sees all prior tokens
        context_ids = list(self._model.tokenize(prompt))

        # Generate tokens one at a time, building distributions incrementally
        # so encode and decode see identical distributions
        from .coding import ArithmeticEncoder

        encoder = ArithmeticEncoder(secret_bits)
        distributions: list[list[float]] = []
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._model.get_distribution(context_ids)
            distributions.append(dist)
            idx = encoder.encode_step(dist)
            tokens.append(idx)
            context_ids.append(idx)
            # Stop when we've encoded enough bits
            if encoder.bits_consumed >= len(secret_bits):
                break

        # Build the generated text - keep special tokens to preserve round-trip
        text = self._model.detokenize(tokens)

        return StegoOutput(
            text=text,
            tokens=tokens,
            bits_encoded=len(secret_bits),
            tokens_generated=len(tokens),
        )

    def decode(self, text: str, prompt: str, num_bits: int) -> str:
        # Tokenize the stego text
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        # Replay: for each token, recompute the distribution from context
        distributions: list[list[float]] = []
        current_context = list(context_ids)
        for token_id in stego_ids:
            dist = self._model.get_distribution(current_context)
            distributions.append(dist)
            current_context.append(token_id)

        # Map token IDs to indices in the vocabulary
        # The distribution is over the full vocab, so token_id IS the index
        token_indices = stego_ids

        return decode_message(token_indices, distributions, num_bits)

    def measure(self, secret: str, text: str, prompt: str) -> TokenStegoMetrics:
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        num_bits = len(secret_bits)

        # Decode
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        distributions: list[list[float]] = []
        log_probs: list[float] = []
        current_context = list(context_ids)

        for token_id in stego_ids:
            dist = self._model.get_distribution(current_context)
            distributions.append(dist)
            # Track log probability of each chosen token
            prob = dist[token_id]
            log_probs.append(math.log(prob) if prob > 0 else -100.0)
            current_context.append(token_id)

        # Recover message
        recovered_bits = decode_message(stego_ids, distributions, num_bits)

        # Bit error rate
        errors = sum(a != b for a, b in zip(secret_bits, recovered_bits))
        errors += abs(len(secret_bits) - len(recovered_bits))
        total = max(len(secret_bits), len(recovered_bits))
        ber = errors / total if total > 0 else 1.0

        # Perplexity: exp(mean(-log_prob))
        if log_probs:
            avg_neg_log = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(min(avg_neg_log, 100.0))  # cap to avoid overflow
        else:
            perplexity = float("inf")

        # Bits per token
        bpt = num_bits / len(stego_ids) if stego_ids else 0.0

        return TokenStegoMetrics(
            bit_error_rate=ber,
            bits_per_token=bpt,
            perplexity=perplexity,
            kl_divergence=0.0,  # TODO: measure against baseline sampling
            exact_recovery=(recovered_bits == secret_bits),
            tokens_generated=len(stego_ids),
        )
