"""Multi-agent covert conversation via token-likelihood steganography.

Two instances of the same model (Alice and Bob) exchange hidden messages
embedded in a natural cover conversation. Each turn is one arithmetic
coding encode/decode cycle. An observer sees a normal dialogue.
"""

from dataclasses import dataclass, field

from .coding import ArithmeticEncoder
from .model import StegoModel


@dataclass
class TurnResult:
    """Result of one conversation turn."""

    role: str  # "alice" or "bob"
    text: str  # the visible cover text
    tokens: list[int]  # generated token IDs
    covert_sent: str  # covert message encoded this turn (ASCII)
    covert_sent_bits: str  # bitstring sent
    covert_recovered: str  # what the other side decoded (ASCII, filled after decode)
    covert_recovered_bits: str  # bitstring recovered
    bits_encoded: int
    bits_per_token: float
    exact_recovery: bool  # filled after decode


@dataclass
class ConversationResult:
    """Full result of a covert conversation."""

    turns: list[TurnResult] = field(default_factory=list)
    alice_secret: str = ""
    bob_secret: str = ""
    alice_recovered_by_bob: str = ""
    bob_recovered_by_alice: str = ""
    total_bits_exchanged: int = 0
    overall_exact: bool = False


class ConversationContext:
    """Manages growing chat history as structured messages and token IDs.

    Operates on token IDs throughout to avoid string round-trip drift.
    """

    def __init__(self, model: StegoModel, system_prompt: str = "") -> None:
        self._model = model
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._messages.append({"role": role, "content": content})

    def get_context_ids(self, add_generation_prompt: bool = True) -> list[int]:
        """Get the full context as token IDs via the chat template."""
        return self._model.tokenize_chat(
            self._messages, add_generation_prompt=add_generation_prompt
        )


def _bits_to_ascii(bits: str) -> str:
    """Convert bitstring to ASCII, dropping incomplete bytes."""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = int(bits[i : i + 8], 2)
        if 32 <= byte <= 126:
            chars.append(chr(byte))
    return "".join(chars)


class CovertConversation:
    """Orchestrates a covert conversation between Alice and Bob.

    Both agents use the same model weights. The shared "key" is the
    conversation history - both sides see all prior messages and can
    recompute identical token distributions.
    """

    def __init__(
        self,
        model: StegoModel,
        cover_topic: str = "Discuss recent advances in computing",
    ) -> None:
        self._model = model
        self._cover_topic = cover_topic

    def run(
        self,
        alice_secret: str,
        bob_secret: str = "RECEIVED",
        num_turns: int = 6,
        max_tokens_per_turn: int = 150,
    ) -> ConversationResult:
        """Run a covert conversation.

        Alice encodes alice_secret across her turns.
        Bob decodes Alice's message, then encodes bob_secret back.
        """
        alice_bits = "".join(format(b, "08b") for b in alice_secret.encode("ascii"))
        bob_bits = "".join(format(b, "08b") for b in bob_secret.encode("ascii"))

        ctx = ConversationContext(self._model, system_prompt=self._cover_topic)
        # Seed with a user message to start the conversation
        ctx.add_message("user", "Let's have a conversation about this topic.")

        result = ConversationResult(alice_secret=alice_secret, bob_secret=bob_secret)

        alice_bits_sent = 0
        bob_bits_sent = 0

        for turn_idx in range(num_turns):
            is_alice = turn_idx % 2 == 0

            if is_alice:
                remaining = alice_bits[alice_bits_sent:]
                role_name = "alice"
            else:
                remaining = bob_bits[bob_bits_sent:]
                role_name = "bob"

            if not remaining:
                # No more covert bits to send, generate normally
                remaining = ""

            # Encode: get context, generate tokens with stego
            context_ids = ctx.get_context_ids(add_generation_prompt=True)

            if remaining:
                tokens, bits_used = self._encode_turn(
                    context_ids, remaining, max_tokens_per_turn
                )
            else:
                tokens = self._generate_normal(context_ids, max_tokens_per_turn)
                bits_used = 0

            text = self._model.detokenize(tokens)

            # Record what was sent
            sent_bits = remaining[:bits_used] if remaining else ""
            turn = TurnResult(
                role=role_name,
                text=text,
                tokens=tokens,
                covert_sent=_bits_to_ascii(sent_bits),
                covert_sent_bits=sent_bits,
                covert_recovered="",
                covert_recovered_bits="",
                bits_encoded=bits_used,
                bits_per_token=bits_used / len(tokens) if tokens else 0.0,
                exact_recovery=False,
            )

            if is_alice:
                alice_bits_sent += bits_used
            else:
                bob_bits_sent += bits_used

            # Add to conversation context as assistant turn
            ctx.add_message("assistant", text)
            # Add a brief user prompt to keep conversation going
            if turn_idx < num_turns - 1:
                ctx.add_message("user", "Please continue.")

            result.turns.append(turn)

        # Decode phase: Bob decodes Alice's turns, Alice decodes Bob's turns
        alice_all_bits = ""
        bob_all_bits = ""

        # Rebuild context for decoding
        decode_ctx = ConversationContext(self._model, system_prompt=self._cover_topic)
        decode_ctx.add_message("user", "Let's have a conversation about this topic.")

        for turn in result.turns:
            context_ids = decode_ctx.get_context_ids(add_generation_prompt=True)

            if turn.covert_sent_bits:
                recovered = self._decode_turn(
                    context_ids, turn.tokens, len(turn.covert_sent_bits)
                )
                turn.covert_recovered_bits = recovered
                turn.covert_recovered = _bits_to_ascii(recovered)
                turn.exact_recovery = recovered == turn.covert_sent_bits

                if turn.role == "alice":
                    alice_all_bits += recovered
                else:
                    bob_all_bits += recovered

            decode_ctx.add_message("assistant", turn.text)
            # Mirror the "Please continue" messages
            if turn != result.turns[-1]:
                decode_ctx.add_message("user", "Please continue.")

        result.alice_recovered_by_bob = _bits_to_ascii(alice_all_bits)
        result.bob_recovered_by_alice = _bits_to_ascii(bob_all_bits)
        result.total_bits_exchanged = sum(t.bits_encoded for t in result.turns)
        result.overall_exact = (
            result.alice_recovered_by_bob == alice_secret
            and result.bob_recovered_by_alice == bob_secret
        )

        return result

    def _encode_turn(
        self, context_ids: list[int], secret_bits: str, max_tokens: int
    ) -> tuple[list[int], int]:
        """Encode covert bits into a turn's tokens. Returns (tokens, bits_used)."""
        encoder = ArithmeticEncoder(secret_bits)
        current_ids = list(context_ids)
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._model.get_distribution(current_ids)
            idx = encoder.encode_step(dist)
            tokens.append(idx)
            current_ids.append(idx)

            if encoder.bits_consumed >= len(secret_bits):
                break

        return tokens, min(encoder.bits_consumed, len(secret_bits))

    def _decode_turn(
        self, context_ids: list[int], tokens: list[int], num_bits: int
    ) -> str:
        """Decode covert bits from a turn's tokens."""
        from .coding import decode_message

        # Rebuild distributions by replaying token by token
        distributions: list[list[float]] = []
        current_ids = list(context_ids)

        for token_id in tokens:
            dist = self._model.get_distribution(current_ids)
            distributions.append(dist)
            current_ids.append(token_id)

        return decode_message(tokens, distributions, num_bits)

    def _generate_normal(self, context_ids: list[int], max_tokens: int) -> list[int]:
        """Generate tokens via greedy sampling (no stego)."""
        current_ids = list(context_ids)
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._model.get_distribution(current_ids)
            token_id = max(range(len(dist)), key=lambda i: dist[i])
            tokens.append(token_id)
            current_ids.append(token_id)

            # Stop on EOS
            eos = getattr(self._model.tokenizer, "eos_token_id", None)
            if eos is not None and token_id == eos:
                break

        return tokens
