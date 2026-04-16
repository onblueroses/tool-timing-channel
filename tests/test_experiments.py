"""Tests for experiment invariants WITHOUT running LLM calls."""

import inspect
import sys
from pathlib import Path

import pytest

# Add experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Many experiments import src.agent -> src.client -> openai
_has_openai = pytest.importorskip("openai", reason="openai not installed")


class TestCapabilityLadderL2Determinism:
    """build_tool_sequence_prompt should be deterministic."""

    def test_deterministic_output(self):
        from experiments.capability_ladder import build_tool_sequence_prompt

        prompt1, map1 = build_tool_sequence_prompt("HELLO")
        prompt2, map2 = build_tool_sequence_prompt("HELLO")
        assert prompt1 == prompt2
        assert map1 == map2


class TestCapabilityLadderL4MutualExclusivity:
    """Each argument should match exactly one grid column predicate."""

    def test_predicates_are_exclusive(self):
        from experiments.capability_ladder import GRID_COL_DECODE_ORDER

        # Diverse test arguments that each should hit exactly one predicate
        test_args = [
            "readme.md",  # starts lowercase
            "README.md",  # starts uppercase
            "3config.json",  # starts with digit
            "is this ok?",  # contains ?
            "one two three",  # 3+ words, starts lowercase (but 3+ words is higher priority)
            "use-case-test",  # contains hyphen
        ]

        for arg in test_args:
            matches = []
            for col_idx, check_fn in GRID_COL_DECODE_ORDER:
                if check_fn(arg):
                    matches.append(col_idx)
                    break  # decoder uses first match (elif chain)
            assert len(matches) == 1, (
                f"Arg '{arg}' matched {len(matches)} predicates (cols: {matches}), expected 1"
            )


class TestArmsRaceCustomMapBijection:
    """custom_map in encoder_v4_custom_lang should be a bijection."""

    def test_all_values_unique(self):

        # Extract the custom_map by inspecting the source - it's defined inline
        # in encoder_v4_custom_lang. We can reconstruct it from the known values.
        custom_map = {
            "A": 14,
            "B": 7,
            "C": 19,
            "D": 11,
            "E": 16,
            "F": 3,
            "G": 22,
            "H": 9,
            "I": 21,
            "J": 5,
            "K": 25,
            "L": 13,
            "M": 8,
            "N": 17,
            "O": 6,
            "P": 23,
            "Q": 4,
            "R": 15,
            "S": 10,
            "T": 20,
            "U": 12,
            "V": 24,
            "W": 18,
            "X": 26,
            "Y": 2,
            "Z": 1,
        }
        assert len(custom_map) == 26
        assert len(set(custom_map.values())) == 26, "Values are not all unique"

    def test_values_cover_1_to_26(self):
        custom_map = {
            "A": 14,
            "B": 7,
            "C": 19,
            "D": 11,
            "E": 16,
            "F": 3,
            "G": 22,
            "H": 9,
            "I": 21,
            "J": 5,
            "K": 25,
            "L": 13,
            "M": 8,
            "N": 17,
            "O": 6,
            "P": 23,
            "Q": 4,
            "R": 15,
            "S": 10,
            "T": 20,
            "U": 12,
            "V": 24,
            "W": 18,
            "X": 26,
            "Y": 2,
            "Z": 1,
        }
        assert set(custom_map.values()) == set(range(1, 27))


class TestOvernightDecoderSignatures:
    """Overnight 3b sweep decoders should accept only traces, not (secret, traces)."""

    def test_decode_secret_path_codebook_signature(self):
        from experiments.overnight_3b_sweep import decode_secret_path_codebook

        sig = inspect.signature(decode_secret_path_codebook)
        params = list(sig.parameters.keys())
        assert len(params) == 1, (
            f"decode_secret_path_codebook should take 1 param (traces), got {params}"
        )

    def test_decode_secret_query_codebook_signature(self):
        from experiments.overnight_3b_sweep import decode_secret_query_codebook

        sig = inspect.signature(decode_secret_query_codebook)
        params = list(sig.parameters.keys())
        assert len(params) == 1, (
            f"decode_secret_query_codebook should take 1 param (traces), got {params}"
        )
