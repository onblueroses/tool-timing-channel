"""Encrypt-then-encode layer for arithmetic steganography.

XORs message bits with a SHAKE-256 keystream derived from key + nonce.
This makes the target point in [0, 1) pseudorandom, eliminating the
entropy bias that makes naive arithmetic coding detectable.

Uses only stdlib (hashlib.shake_256, os.urandom). No external dependencies.
"""

import hashlib
import os


def _keystream(key: bytes, nonce: bytes, num_bits: int) -> str:
    """Generate a pseudorandom bitstring from key || nonce using SHAKE-256."""
    num_bytes = (num_bits + 7) // 8
    stream = hashlib.shake_256(key + nonce).digest(num_bytes)
    bits = "".join(format(b, "08b") for b in stream)
    return bits[:num_bits]


def encrypt_bits(
    plaintext_bits: str, key: bytes, nonce: bytes | None = None
) -> tuple[str, bytes]:
    """XOR plaintext bits with SHAKE-256 keystream.

    Returns (ciphertext_bits, nonce). If nonce is None, generates a fresh one.
    The nonce must be transmitted alongside the stegotext for decryption.
    """
    if nonce is None:
        nonce = os.urandom(16)
    ks = _keystream(key, nonce, len(plaintext_bits))
    ciphertext = "".join("1" if a != b else "0" for a, b in zip(plaintext_bits, ks))
    return ciphertext, nonce


def decrypt_bits(ciphertext_bits: str, key: bytes, nonce: bytes) -> str:
    """XOR ciphertext bits with SHAKE-256 keystream."""
    ks = _keystream(key, nonce, len(ciphertext_bits))
    return "".join("1" if a != b else "0" for a, b in zip(ciphertext_bits, ks))
