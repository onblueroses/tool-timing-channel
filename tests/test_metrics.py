"""Tests for channel metrics / BER computation."""

from src.channels.ordering import OrderingChannel, INDEX_TO_TOOL
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


class TestBaseChannelMeasureBER:
    """BaseChannel.measure computes true BER for bit strings."""

    def test_perfect_decode_ber_zero(self):
        """Traces that decode to matching bits should give BER = 0.0."""
        from src.channels.timing import TimingChannel

        # Encode "A" = 01000001 in bits
        # TimingChannel decodes from latencies: bit 1 if lat > threshold, else 0
        # "A" bits: 0 1 0 0 0 0 0 1
        # Use threshold=1.0. Need latencies: 0.5, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0
        # Plus a first trace (skipped)
        traces = [_make_trace(latency=0.0, timestamp=0.0)]  # skipped
        latencies = [0.5, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0]
        t = 0.0
        for lat in latencies:
            t += lat
            traces.append(_make_trace(latency=lat, timestamp=t))

        channel = TimingChannel(threshold=1.0)
        decoded = channel.decode(traces)
        assert decoded == "01000001"

        metrics = channel.measure("A", traces)
        assert metrics.bit_error_rate == 0.0

    def test_all_wrong_ber_positive(self):
        """Traces that decode to all-wrong bits should give BER > 0."""
        from src.channels.timing import TimingChannel

        # "A" bits: 01000001
        # Decode all 1s instead: latencies all > threshold
        traces = [_make_trace(latency=0.0, timestamp=0.0)]  # skipped
        t = 0.0
        for _ in range(8):
            t += 2.0
            traces.append(_make_trace(latency=2.0, timestamp=t))

        channel = TimingChannel(threshold=1.0)
        decoded = channel.decode(traces)
        assert decoded == "11111111"

        metrics = channel.measure("A", traces)
        assert metrics.bit_error_rate > 0


class TestOrderingChannelBER:
    """OrderingChannel BER computation for character-based channels."""

    def _make_ordering_traces(self, message: str) -> list[ToolTrace]:
        """Build traces that perfectly encode the given message."""
        from src.channels.ordering import char_to_tool_pair

        traces = []
        t = 0.0
        for ch in message.upper():
            t1, t2 = char_to_tool_pair(ch)
            t += 1.0
            traces.append(
                _make_trace(
                    tool_name=INDEX_TO_TOOL[t1],
                    timestamp=t,
                    latency=1.0,
                )
            )
            t += 1.0
            traces.append(
                _make_trace(
                    tool_name=INDEX_TO_TOOL[t2],
                    timestamp=t,
                    latency=1.0,
                )
            )
        return traces

    def test_identical_strings_ber_zero(self):
        channel = OrderingChannel()
        traces = self._make_ordering_traces("HI")
        metrics = channel.measure("HI", traces)
        assert metrics.bit_error_rate == 0.0

    def test_one_char_off_ber_between_zero_and_one(self):
        channel = OrderingChannel()
        # Build traces for "HI" but measure against "HJ" - one char different
        traces = self._make_ordering_traces("HI")
        metrics = channel.measure("HJ", traces)
        # I vs J differ by 1 bit at the bit level (I=01001001, J=01001010)
        # So BER should be small but > 0, and definitely < 1
        assert 0 < metrics.bit_error_rate < 1
