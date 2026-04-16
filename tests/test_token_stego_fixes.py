"""Tests for token stego fixes. Uses mock StegoModel - no GPU or real model needed."""

from unittest.mock import MagicMock

import pytest
import torch

from src.token_stego.base import StegoOutput
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.detector import StegoDetector
from src.token_stego.latent import LatentSnapshot
from src.token_stego.activation_probe import snapshot_to_features


def _make_mock_model(vocab_size=10):
    """Create a mock StegoModel with a small vocabulary."""
    model = MagicMock()
    model.vocab_size = vocab_size

    def mock_tokenize(text):
        # Return one token per character
        return list(range(min(len(text), vocab_size)))

    def mock_detokenize(ids):
        return "".join(chr(65 + (i % 26)) for i in ids)

    def mock_get_distribution(input_ids, temperature=1.0, top_p=1.0):
        # Return a roughly uniform distribution over vocab
        dist = [1.0 / vocab_size] * vocab_size
        return dist

    model.tokenize = mock_tokenize
    model.detokenize = mock_detokenize
    model.get_distribution = mock_get_distribution
    return model


class TestArithmeticStegoChannelBitsConsumed:
    """encode should return actual bits_consumed, not len(secret_bits)."""

    def test_bits_encoded_le_consumed(self):
        mock_model = _make_mock_model(vocab_size=4)
        channel = ArithmeticStegoChannel(model=mock_model)

        # Encode with max_tokens=1 - will likely not consume all bits
        result = channel.encode(secret="AB", prompt="test", max_tokens=1)

        # bits_encoded should be <= total secret bits (16 for "AB")
        total_secret_bits = 16  # 2 chars * 8 bits
        assert result.bits_encoded <= total_secret_bits
        assert isinstance(result, StegoOutput)
        assert result.tokens_generated == 1


class TestArithmeticStegoChannelDecryptNonce:
    """decode should raise ValueError when key set but nonce missing."""

    def test_raises_without_nonce(self):
        mock_model = _make_mock_model(vocab_size=4)
        channel = ArithmeticStegoChannel(model=mock_model, key=b"testkey")

        with pytest.raises(ValueError, match="nonce"):
            channel.decode(text="AB", prompt="test", num_bits=16, nonce=None)


class TestStegoDetectorDeterminism:
    """score_cdf_uniformity should be deterministic with seed."""

    def test_seeded_determinism(self):
        mock_model = _make_mock_model(vocab_size=4)
        detector = StegoDetector(model=mock_model)

        r1 = detector.score_cdf_uniformity("AB", "test", seed=42)
        r2 = detector.score_cdf_uniformity("AB", "test", seed=42)

        assert r1.score == r2.score
        assert r1.p_value == r2.p_value
        assert r1.per_token_scores == r2.per_token_scores


class TestSnapshotToFeaturesMultiLayer:
    """snapshot_to_features should concatenate mean-pooled features from all layers."""

    def test_multi_layer_concatenation(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 3),
                6: torch.randn(1, 5, 3),
                12: torch.randn(1, 5, 3),
            },
        )
        features = snapshot_to_features(snap)
        # 3 layers * 3 (hidden_dim after mean over seq_len) = 9
        assert features.shape == (9,)

    def test_single_layer_extraction(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 3),
                6: torch.randn(1, 5, 3),
                12: torch.randn(1, 5, 3),
            },
        )
        features = snapshot_to_features(snap, layer=6)
        # Single layer: hidden_dim = 3
        assert features.shape == (3,)

    def test_single_layer_still_works(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 7),
            },
        )
        features = snapshot_to_features(snap, layer=0)
        assert features.shape == (7,)
