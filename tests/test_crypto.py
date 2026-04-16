"""Tests for encrypt-then-encode steganography layer."""

import random

from src.token_stego.coding import decode_message, encode_message
from src.token_stego.crypto import decrypt_bits, encrypt_bits


class TestCryptoRoundTrip:
    """Verify encrypt/decrypt is a no-op (XOR self-inverse)."""

    def test_short_message(self) -> None:
        key = b"test-key-123"
        bits = "10110100"
        encrypted, nonce = encrypt_bits(bits, key)
        assert encrypted != bits  # should be different
        decrypted = decrypt_bits(encrypted, key, nonce)
        assert decrypted == bits

    def test_long_message(self) -> None:
        key = b"another-key"
        bits = "".join(random.Random(42).choice("01") for _ in range(1000))
        encrypted, nonce = encrypt_bits(bits, key)
        decrypted = decrypt_bits(encrypted, key, nonce)
        assert decrypted == bits

    def test_different_keys_produce_different_ciphertext(self) -> None:
        bits = "10110100"
        nonce = b"\x00" * 16
        enc1, _ = encrypt_bits(bits, b"key-a", nonce)
        enc2, _ = encrypt_bits(bits, b"key-b", nonce)
        assert enc1 != enc2

    def test_different_nonces_produce_different_ciphertext(self) -> None:
        bits = "10110100"
        key = b"same-key"
        enc1, n1 = encrypt_bits(bits, key)
        enc2, n2 = encrypt_bits(bits, key)
        # Random nonces should differ, producing different ciphertext
        assert n1 != n2
        assert enc1 != enc2

    def test_explicit_nonce_is_deterministic(self) -> None:
        bits = "10110100"
        key = b"key"
        nonce = b"fixed-nonce-1234"
        enc1, _ = encrypt_bits(bits, key, nonce)
        enc2, _ = encrypt_bits(bits, key, nonce)
        assert enc1 == enc2


class TestEncryptedArithmeticCoding:
    """Verify arithmetic coding works identically with encrypted bits."""

    def test_round_trip_with_encryption(self) -> None:
        key = b"shared-secret"
        secret = "10110100"
        encrypted, nonce = encrypt_bits(secret, key)

        dist = [0.25, 0.25, 0.25, 0.25]
        distributions = [dist] * 4

        tokens = encode_message(encrypted, distributions)
        recovered_encrypted = decode_message(
            tokens, distributions, num_bits=len(secret)
        )
        recovered = decrypt_bits(recovered_encrypted, key, nonce)

        assert recovered == secret

    def test_encrypted_tokens_differ_from_unencrypted(self) -> None:
        """With encryption, the same message produces different token selections."""
        secret = "10110100"
        dist = [0.25, 0.25, 0.25, 0.25]
        distributions = [dist] * 4

        tokens_plain = encode_message(secret, distributions)

        encrypted, _ = encrypt_bits(secret, b"some-key")
        tokens_encrypted = encode_message(encrypted, distributions)

        assert tokens_plain != tokens_encrypted

    def test_stress_with_encryption(self) -> None:
        rng = random.Random(99)
        key = b"stress-test-key"
        secret = "".join(rng.choice("01") for _ in range(200))
        encrypted, nonce = encrypt_bits(secret, key)

        distributions: list[list[float]] = []
        for _ in range(120):
            raw = [rng.random() for _ in range(128)]
            total = sum(raw)
            distributions.append([p / total for p in raw])

        tokens = encode_message(encrypted, distributions)
        recovered_encrypted = decode_message(
            tokens, distributions, num_bits=len(secret)
        )
        recovered = decrypt_bits(recovered_encrypted, key, nonce)

        assert recovered == secret
