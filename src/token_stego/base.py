"""Token-level steganography abstractions.

Parallel to src/channels/base.py but operates on text + logits
rather than tool call traces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StegoOutput:
    """Result of encoding a secret into generated text."""

    text: str
    tokens: list[int]
    bits_encoded: int
    tokens_generated: int
    nonce: bytes | None = None  # encryption nonce, needed for decryption


@dataclass
class TokenStegoMetrics:
    """Metrics for token-level steganographic encoding."""

    bit_error_rate: float  # 0.0 = perfect recovery
    bits_per_token: float  # encoding capacity
    perplexity: float  # naturalness of generated text
    kl_divergence: float  # deviation from normal sampling
    exact_recovery: bool  # did we get the exact message back?
    tokens_generated: int


class TokenStegoChannel(ABC):
    """Abstract base for token-level steganographic channels."""

    @abstractmethod
    def encode(self, secret: str, prompt: str, max_tokens: int = 200) -> StegoOutput:
        """Encode a secret message into generated text.

        Args:
            secret: The hidden message (ASCII string).
            prompt: The cover prompt - what the model should write about.
            max_tokens: Maximum tokens to generate.

        Returns:
            StegoOutput with generated text and encoding metadata.
        """

    @abstractmethod
    def decode(self, text: str, prompt: str, num_bits: int) -> str:
        """Decode a hidden message from generated text.

        Args:
            text: The generated text (from encode).
            prompt: The same cover prompt used during encoding.
            num_bits: Number of message bits to extract.

        Returns:
            The recovered bitstring.
        """

    def measure(self, secret: str, text: str, prompt: str) -> TokenStegoMetrics:
        """Measure encoding quality by decoding and comparing.

        Args:
            secret: The original secret message (ASCII string).
            text: The generated text to decode from.
            prompt: The cover prompt used during encoding.

        Returns:
            TokenStegoMetrics with quality measurements.
        """
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        num_bits = len(secret_bits)

        recovered_bits = self.decode(text, prompt, num_bits)

        # Bit error rate
        errors = sum(a != b for a, b in zip(secret_bits, recovered_bits))
        errors += abs(len(secret_bits) - len(recovered_bits))
        total = max(len(secret_bits), len(recovered_bits))
        ber = errors / total if total > 0 else 1.0

        return TokenStegoMetrics(
            bit_error_rate=ber,
            bits_per_token=0.0,  # filled by subclass
            perplexity=0.0,  # filled by subclass
            kl_divergence=0.0,  # filled by subclass
            exact_recovery=(recovered_bits == secret_bits),
            tokens_generated=0,  # filled by subclass
        )
