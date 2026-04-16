"""Tests for tool-call based covert channels."""

import pytest

from src.channels.ordering import char_to_tool_pair, tool_pair_to_char
from src.channels.scaled_table import ScaledTableChannel
from src.channels.table_multiturn import MultiTurnTableChannel
from src.channels.timing import TimingChannel
from src.channels.response_length import ResponseLengthChannel
from src.harness import ToolTrace


def _make_trace(
    tool_name: str = "read_file",
    arguments: dict | None = None,
    latency: float = 0.0,
    timestamp: float = 0.0,
) -> ToolTrace:
    return ToolTrace(
        timestamp=timestamp,
        tool_name=tool_name,
        arguments=arguments or {},
        response="ok",
        success=True,
        latency_from_previous=latency,
    )


class TestOrderingChannelRoundTrip:
    """OrderingChannel should round-trip all 26 letters via tool pair encoding."""

    def test_all_letters_round_trip(self):
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            t1, t2 = char_to_tool_pair(letter)
            recovered = tool_pair_to_char(t1, t2)
            assert recovered == letter, f"Failed on {letter}: got {recovered}"

    def test_z_specifically(self):
        """Z was previously broken - verify the fix."""
        t1, t2 = char_to_tool_pair("Z")
        assert tool_pair_to_char(t1, t2) == "Z"

    def test_lowercase_normalizes(self):
        t1, t2 = char_to_tool_pair("z")
        assert tool_pair_to_char(t1, t2) == "Z"


class TestTimingChannelDecode:
    """TimingChannel.decode() should skip the first trace entry."""

    def test_skips_first_entry(self):
        traces = [
            _make_trace(latency=0.0, timestamp=0.0),
            _make_trace(latency=0.5, timestamp=0.5),
            _make_trace(latency=2.0, timestamp=2.5),
            _make_trace(latency=0.3, timestamp=2.8),
            _make_trace(latency=1.5, timestamp=4.3),
        ]
        channel = TimingChannel(threshold=1.0)
        decoded = channel.decode(traces)
        # Skips first (0.0), decodes: 0.5<1.0=0, 2.0>1.0=1, 0.3<1.0=0, 1.5>1.0=1
        assert len(decoded) == 4, f"Expected 4 bits, got {len(decoded)}: {decoded}"
        assert decoded == "0101"


class TestResponseLengthChannelDecode:
    """ResponseLengthChannel.decode() should skip the first trace entry."""

    def test_skips_first_entry(self):
        traces = [
            _make_trace(latency=0.0, timestamp=0.0),
            _make_trace(latency=0.5, timestamp=0.5),
            _make_trace(latency=2.0, timestamp=2.5),
            _make_trace(latency=0.3, timestamp=2.8),
            _make_trace(latency=1.5, timestamp=4.3),
        ]
        channel = ResponseLengthChannel(threshold=1.0)
        decoded = channel.decode(traces)
        assert len(decoded) == 4, f"Expected 4 bits, got {len(decoded)}: {decoded}"
        assert decoded == "0101"


class TestScaledTableChannelValidation:
    """ScaledTableChannel should reject table_size < 26."""

    def test_rejects_size_25(self):
        with pytest.raises(ValueError):
            ScaledTableChannel(25)

    def test_rejects_size_5(self):
        with pytest.raises(ValueError):
            ScaledTableChannel(5)

    def test_accepts_size_26(self):
        channel = ScaledTableChannel(26)
        assert channel.table_size == 26


class TestScaledTableChannelRoundTrip:
    """ScaledTableChannel round-trip for a short message."""

    def test_encode_decode_hi(self):
        channel = ScaledTableChannel(26)
        message = "HI"

        # Build fake traces that match the expected tool calls
        # The encoder maps each letter to a table entry via char_to_entry
        traces = []
        for ch in message.upper():
            letter_idx = ord(ch) - ord("A")
            entry_idx = channel.char_to_entry[letter_idx]
            tool_name, args = channel.forward_map[entry_idx]
            traces.append(
                _make_trace(
                    tool_name=tool_name,
                    arguments=args,
                    timestamp=len(traces) * 1.0,
                )
            )

        decoded = channel.decode(traces)
        assert decoded == "HI"


class TestMultiTurnTableChannelEncode:
    """MultiTurnTableChannel.encode should return prompts for all characters."""

    def test_encode_contains_all_chars(self):
        channel = MultiTurnTableChannel()
        result = channel.encode("HI")
        # encode() joins per-char prompts with "\n---\n"
        assert "---" in result, "Expected separator between character prompts"
        parts = result.split("\n---\n")
        assert len(parts) == 2, f"Expected 2 parts for 'HI', got {len(parts)}"

    def test_get_turn_prompts_length(self):
        channel = MultiTurnTableChannel()
        prompts = channel.get_turn_prompts("HELLO")
        assert len(prompts) == 5
