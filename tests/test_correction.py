"""Tests for error correction strategies."""

from src.correction import InterleavedRepetitionCode


class TestInterleavedRepetitionCodeStateless:
    """InterleavedRepetitionCode should not have a _last_k attribute."""

    def test_no_last_k_attribute(self):
        code = InterleavedRepetitionCode()
        assert not hasattr(code, "_last_k")


class TestInterleavedRepetitionCodeRoundTrip:
    """Round-trip with explicit original_length."""

    def test_add_redundancy(self):
        code = InterleavedRepetitionCode(n=3)
        assert code.add_redundancy("HI") == "HIHIHI"

    def test_correct_with_length(self):
        code = InterleavedRepetitionCode(n=3)
        result = code.correct("HIHIHI", original_length=2)
        assert result == "HI"

    def test_correct_without_length(self):
        code = InterleavedRepetitionCode(n=3)
        # Without original_length, estimates k = len(decoded) // n = 6 // 3 = 2
        result = code.correct("HIHIHI")
        assert result == "HI"

    def test_error_correction(self):
        code = InterleavedRepetitionCode(n=3)
        redundant = code.add_redundancy("AB")
        assert redundant == "ABABAB"
        # Introduce error: change second char of first repetition
        corrupted = "AXABAB"
        result = code.correct(corrupted, original_length=2)
        assert result == "AB", f"Expected 'AB', got '{result}'"
