"""Arithmetic coding steganographic channel using HuggingFace models.

Encodes secret messages into generated text by choosing among tokens
based on their probability distributions. The output reads naturally
because all chosen tokens are high-probability.
"""

import math

from .base import StegoOutput, TokenStegoChannel, TokenStegoMetrics
from .coding import decode_message
from .crypto import decrypt_bits, encrypt_bits
from .model import StegoModel


class ArithmeticStegoChannel(TokenStegoChannel):
    """Steganographic channel using arithmetic coding over token distributions.

    When a key is provided, message bits are encrypted before encoding
    and decrypted after decoding. This makes the arithmetic coder's
    target point pseudorandom, eliminating the entropy bias that lets
    a calibrated detector distinguish stego from normal sampling.
    """

    def __init__(
        self,
        model: StegoModel,
        key: bytes | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self._model = model
        self._key = key
        self._temperature = temperature
        self._top_p = top_p

    def encode(self, secret: str, prompt: str, max_tokens: int = 200) -> StegoOutput:
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        nonce = None
        if self._key is not None:
            secret_bits, nonce = encrypt_bits(secret_bits, self._key)

        # Build context incrementally - each step sees all prior tokens
        context_ids = list(self._model.tokenize(prompt))

        # Generate tokens one at a time, building distributions incrementally
        # so encode and decode see identical distributions
        from .coding import ArithmeticEncoder

        encoder = ArithmeticEncoder(secret_bits)
        distributions: list[list[float]] = []
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._model.get_distribution(
                context_ids, self._temperature, self._top_p
            )
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
            bits_encoded=min(encoder.bits_consumed, len(secret_bits)),
            tokens_generated=len(tokens),
            nonce=nonce,
        )

    def decode(
        self, text: str, prompt: str, num_bits: int, nonce: bytes | None = None
    ) -> str:
        # Tokenize the stego text
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        # Replay: for each token, recompute the distribution from context
        distributions: list[list[float]] = []
        current_context = list(context_ids)
        for token_id in stego_ids:
            dist = self._model.get_distribution(
                current_context, self._temperature, self._top_p
            )
            distributions.append(dist)
            current_context.append(token_id)

        # Map token IDs to indices in the vocabulary
        # The distribution is over the full vocab, so token_id IS the index
        token_indices = stego_ids

        recovered_bits = decode_message(token_indices, distributions, num_bits)
        if self._key is not None:
            if nonce is None:
                raise ValueError("Key is set but nonce is missing; cannot decrypt.")
            recovered_bits = decrypt_bits(recovered_bits, self._key, nonce)
        return recovered_bits

    def measure(
        self, secret: str, text: str, prompt: str, nonce: bytes | None = None
    ) -> TokenStegoMetrics:
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        num_bits = len(secret_bits)

        # Decode
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        distributions: list[list[float]] = []
        log_probs: list[float] = []
        current_context = list(context_ids)

        for token_id in stego_ids:
            dist = self._model.get_distribution(
                current_context, self._temperature, self._top_p
            )
            distributions.append(dist)
            # Track log probability of each chosen token
            prob = dist[token_id]
            log_probs.append(math.log(prob) if prob > 0 else -100.0)
            current_context.append(token_id)

        # Recover message
        recovered_bits = decode_message(stego_ids, distributions, num_bits)
        if self._key is not None and nonce is not None:
            recovered_bits = decrypt_bits(recovered_bits, self._key, nonce)

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
