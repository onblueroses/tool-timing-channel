"""Tests for arithmetic coding steganography encoder/decoder."""

import random

from src.token_stego.coding import (
    decode_message,
    encode_message,
)


class TestUniformDistribution:
    """Uniform distribution over 4 tokens: each step encodes exactly 2 bits."""

    def test_round_trip(self) -> None:
        secret = "10110100"
        # 4 tokens uniform -> 2 bits per step -> need 4 steps for 8 bits
        dist = [0.25, 0.25, 0.25, 0.25]
        distributions = [dist] * 4

        tokens = encode_message(secret, distributions)
        recovered = decode_message(tokens, distributions, num_bits=len(secret))

        assert recovered == secret

    def test_encoder_selects_correct_tokens(self) -> None:
        """With uniform 4-token dist, selections [0,1,2,3] map to bit pairs."""
        dist = [0.25, 0.25, 0.25, 0.25]

        # Encode "00011011" -> should produce tokens [0, 1, 2, 3]
        # 00 -> token 0, 01 -> token 1, 10 -> token 2, 11 -> token 3
        secret = "00011011"
        distributions = [dist] * 4
        tokens = encode_message(secret, distributions)

        assert tokens == [0, 1, 2, 3]


class TestSkewedDistribution:
    """Skewed distribution: one token gets most probability mass."""

    def test_round_trip(self) -> None:
        secret = "1101001011"
        # Skewed: first token dominates
        dist = [0.7, 0.1, 0.1, 0.1]
        # Need more steps since high-prob tokens carry fewer bits
        distributions = [dist] * 20

        tokens = encode_message(secret, distributions)
        recovered = decode_message(tokens, distributions, num_bits=len(secret))

        assert recovered == secret


class TestSingleDominantToken:
    """One token has ~97% probability - very few bits per step."""

    def test_round_trip(self) -> None:
        secret = "101010101010"
        dist = [0.97, 0.01, 0.01, 0.01]
        # Dominant token barely narrows per step (~0.04 bits when selected).
        # Need many steps since most selections will be token 0.
        distributions = [dist] * 500

        tokens = encode_message(secret, distributions)
        recovered = decode_message(tokens, distributions, num_bits=len(secret))

        assert recovered == secret


class TestStress:
    """100+ token vocabulary with random distributions."""

    def test_round_trip_large_vocab(self) -> None:
        rng = random.Random(42)
        secret = "".join(rng.choice("01") for _ in range(200))

        distributions: list[list[float]] = []
        for _ in range(120):
            raw = [rng.random() for _ in range(128)]
            total = sum(raw)
            distributions.append([p / total for p in raw])

        tokens = encode_message(secret, distributions)
        recovered = decode_message(tokens, distributions, num_bits=len(secret))

        assert recovered == secret, (
            f"Bit errors in stress test: "
            f"{sum(a != b for a, b in zip(secret, recovered))} "
            f"out of {len(secret)}"
        )
