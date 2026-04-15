"""Arithmetic coding for token-likelihood steganography.

Encodes hidden message bits into token selections by partitioning probability
intervals. Uses decimal.Decimal with precision >= 50 to avoid floating-point
drift across long sequences.

The message bitstring is interpreted as a binary fraction 0.b1b2b3... defining
a target point in [0, 1). At each generation step, the probability distribution
partitions the current interval into token sub-intervals. The encoder selects
the token whose sub-interval contains the target point, narrowing the interval.
The decoder reverses this by narrowing the same interval from observed tokens,
then reads off the binary expansion.

To guarantee exact recovery, the caller must provide enough generation steps
that the interval narrows below 2^(-num_message_bits).
"""

from collections.abc import Sequence
from decimal import Decimal, localcontext

# Default precision: 50 decimal digits (~166 bits).
# Functions scale up automatically for longer messages.
_BASE_PRECISION = 50

Distribution = list[Decimal]


def _precision_for_bits(num_bits: int) -> int:
    """Decimal digits needed to represent num_bits of binary precision."""
    # 1 binary bit ~ 0.301 decimal digits; add margin for intermediate ops
    return max(_BASE_PRECISION, int(num_bits * 0.302) + 30)


def _normalize(dist: Sequence[float | Decimal]) -> Distribution:
    """Convert a distribution to Decimal and normalize so it sums to exactly 1."""
    d = [Decimal(str(p)) for p in dist]
    total = sum(d)
    if total == 0:
        raise ValueError("Distribution sums to zero")
    d = [p / total for p in d]
    residual = Decimal(1) - sum(d)
    d[-1] += residual
    return d


def _bits_to_point(bits: str) -> Decimal:
    """Convert a bitstring to a Decimal in [0, 1) via binary fraction.

    To avoid boundary issues, we place the target at the midpoint of the
    message's binary interval [val, val + 2^-len), i.e. val + 2^-(len+1).
    This ensures the target never sits exactly on a sub-interval boundary.
    """
    value = Decimal(0)
    scale = Decimal("0.5")
    for b in bits:
        if b == "1":
            value += scale
        scale /= 2
    # scale is now 2^(-(len+1)); add it to reach midpoint
    value += scale
    return value


def _point_to_bits(value: Decimal, num_bits: int) -> str:
    """Convert a Decimal in [0, 1) to a bitstring of given length."""
    bits: list[str] = []
    remainder = value
    threshold = Decimal("0.5")
    for _ in range(num_bits):
        if remainder >= threshold:
            bits.append("1")
            remainder -= threshold
        else:
            bits.append("0")
        threshold /= 2
    return "".join(bits)


def _partition_step(
    low: Decimal,
    high: Decimal,
    probs: Distribution,
) -> list[tuple[Decimal, Decimal]]:
    """Partition [low, high) into sub-intervals proportional to probs."""
    width = high - low
    n = len(probs)
    intervals: list[tuple[Decimal, Decimal]] = []
    cum_lo = low
    for i in range(n):
        if i == n - 1:
            cum_hi = high  # avoid rounding gap
        else:
            cum_hi = cum_lo + width * probs[i]
        intervals.append((cum_lo, cum_hi))
        cum_lo = cum_hi
    return intervals


class ArithmeticEncoder:
    """Encodes message bits into token index selections."""

    def __init__(self, message_bits: str) -> None:
        self.message_bits = message_bits
        self.target = _bits_to_point(message_bits)
        self.low = Decimal(0)
        self.high = Decimal(1)

    @property
    def bits_consumed(self) -> int:
        """Approximate number of message bits encoded so far."""
        width = self.high - self.low
        if width <= 0:
            return len(self.message_bits)
        count = 0
        threshold = Decimal(1)
        while threshold > width and count < 10000:
            threshold /= 2
            count += 1
        return count

    def encode_step(self, dist: Sequence[float | Decimal]) -> int:
        """Given a probability distribution, return the token index to select."""
        probs = _normalize(dist)
        intervals = _partition_step(self.low, self.high, probs)

        for i, (lo, hi) in enumerate(intervals):
            if self.target >= lo and self.target < hi:
                self.low = lo
                self.high = hi
                return i

        # Target at upper boundary - pick last token
        lo, hi = intervals[-1]
        self.low = lo
        self.high = hi
        return len(probs) - 1


class ArithmeticDecoder:
    """Decodes token index selections back into the original message bits."""

    def __init__(self) -> None:
        self.low = Decimal(0)
        self.high = Decimal(1)

    def decode_step(self, token_index: int, dist: Sequence[float | Decimal]) -> None:
        """Narrow the interval based on an observed token selection."""
        probs = _normalize(dist)
        intervals = _partition_step(self.low, self.high, probs)
        lo, hi = intervals[token_index]
        self.low = lo
        self.high = hi

    def extract_bits(self, num_bits: int) -> str:
        """Extract message bits from the narrowed interval.

        Uses the midpoint of [low, high) for extraction. Provided the interval
        is narrow enough (width < 2^-(num_bits+1)), the midpoint's first
        num_bits binary digits match the original message.
        """
        mid = (self.low + self.high) / 2
        return _point_to_bits(mid, num_bits)


def encode_message(
    secret_bits: str, distributions: Sequence[Sequence[float | Decimal]]
) -> list[int]:
    """Encode a bitstring into token selections using arithmetic coding.

    Args:
        secret_bits: The hidden message as a string of '0' and '1' characters.
        distributions: One probability distribution per generation step.

    Returns:
        List of token indices - one per distribution/step.

    Raises:
        ValueError: If distributions do not provide enough capacity for secret_bits.
    """
    prec = _precision_for_bits(len(secret_bits))
    with localcontext() as ctx:
        ctx.prec = prec

        encoder = ArithmeticEncoder(secret_bits)
        tokens: list[int] = []
        for dist in distributions:
            idx = encoder.encode_step(dist)
            tokens.append(idx)

        if encoder.bits_consumed < len(secret_bits):
            raise ValueError(
                f"Insufficient capacity: distributions encoded only "
                f"{encoder.bits_consumed} bits but {len(secret_bits)} were required"
            )

    return tokens


def decode_message(
    token_indices: list[int],
    distributions: Sequence[Sequence[float | Decimal]],
    num_bits: int,
) -> str:
    """Decode token selections back into the hidden bitstring.

    Args:
        token_indices: The observed token indices (from encode_message).
        distributions: The same distributions used during encoding.
        num_bits: Number of message bits to extract.

    Returns:
        The recovered bitstring.

    Raises:
        ValueError: If token counts mismatch or interval width insufficient for num_bits.
    """
    if len(token_indices) != len(distributions):
        raise ValueError(
            f"Got {len(token_indices)} tokens but {len(distributions)} distributions"
        )

    prec = _precision_for_bits(num_bits)
    with localcontext() as ctx:
        ctx.prec = prec

        decoder = ArithmeticDecoder()
        for idx, dist in zip(token_indices, distributions):
            decoder.decode_step(idx, dist)

        # Verify the interval has narrowed enough to reliably extract num_bits.
        # Required width: < 2^-num_bits (so midpoint uniquely determines all bits).
        width = decoder.high - decoder.low
        required_width = Decimal(1) / (Decimal(2) ** num_bits)
        if width > required_width:
            raise ValueError(
                f"Insufficient capacity: interval width {width} is not narrow enough "
                f"to extract {num_bits} bits (need width < 2^-{num_bits} = {required_width})"
            )

        return decoder.extract_bits(num_bits)
