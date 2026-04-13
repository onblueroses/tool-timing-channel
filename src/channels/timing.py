from pathlib import Path

import numpy as np

from ..harness import ToolTrace
from .base import BaseChannel, str_to_bits

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class TimingChannel(BaseChannel):
    def __init__(self, threshold: float | None = None):
        self.threshold = threshold  # None = auto (median split)

    def encode(self, message: str) -> str:
        bits = str_to_bits(message)
        template = (PROMPTS_DIR / "timing_encoder_v1.txt").read_text()
        return template.replace("{MESSAGE}", message).replace(
            "{NUM_BITS}", str(len(bits))
        )

    def decode(self, traces: list[ToolTrace]) -> str:
        if not traces:
            return ""

        latencies = [t.latency_from_previous for t in traces]

        threshold = self.threshold
        if threshold is None:
            nonzero = [lat for lat in latencies if lat > 0.01]
            if not nonzero:
                return ""
            threshold = float(np.median(nonzero))

        bits = []
        for lat in latencies:
            bits.append("1" if lat > threshold else "0")

        return "".join(bits)


class TimingChannelTokenCount(BaseChannel):
    """Variant: decode based on token count between tool calls, not wall-clock time.

    Uses completion token count from the response preceding each tool call
    as the signal, since models control token output more reliably than
    wall-clock timing (which includes network latency).
    """

    def __init__(self, threshold: int = 50):
        self.threshold = threshold  # tokens: above = 1, below = 0

    def encode(self, message: str) -> str:
        bits = str_to_bits(message)
        template = (PROMPTS_DIR / "timing_encoder_v1.txt").read_text()
        return template.replace("{MESSAGE}", message).replace(
            "{NUM_BITS}", str(len(bits))
        )

    def decode(self, traces: list[ToolTrace]) -> str:
        if not traces:
            return ""

        # Use token counts when available, fall back to latency
        token_counts = [t.token_count for t in traces if t.token_count is not None]
        if token_counts and len(token_counts) == len(traces):
            threshold = self.threshold or int(np.median(token_counts))
            return "".join(
                "1" if (t.token_count or 0) > threshold else "0" for t in traces
            )

        latencies = [t.latency_from_previous for t in traces]
        nonzero = [lat for lat in latencies if lat > 0.01]
        if not nonzero:
            return ""
        threshold = float(np.median(nonzero))

        return "".join("1" if lat > threshold else "0" for lat in latencies)
