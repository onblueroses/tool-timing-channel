"""Multi-agent covert conversation via token-likelihood steganography.

Two instances of the same model (Alice and Bob) exchange hidden messages
embedded in a natural cover conversation. Each turn is one arithmetic
coding encode/decode cycle. An observer sees a normal dialogue.

Alice's turns are encoded as "assistant" messages and Bob's turns as
"user" messages in the shared chat context. Both sides see the full
conversation history, so they compute identical token distributions
for each turn's context.
"""

import hashlib
from dataclasses import dataclass, field

from .coding import ArithmeticEncoder
from .model import StegoModel


def _derive_nonce(key: bytes, role: str, turn_index: int) -> bytes:
    """Derive a deterministic nonce from key, role, and turn index.

    Any receiver with the shared key can reconstruct this nonce
    without it being transmitted.
    """
    material = key + role.encode() + turn_index.to_bytes(4, "big")
    return hashlib.shake_256(material).digest(16)


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
    nonce: bytes | None = (
        None  # included for transparency, derivable from key+role+turn
    )


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
        key: bytes | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self._model = model
        self._cover_topic = cover_topic
        self._key = key
        self._temperature = temperature
        self._top_p = top_p

    def run(
        self,
        alice_secret: str,
        bob_secret: str = "RECEIVED",
        num_turns: int = 6,
        max_tokens_per_turn: int = 150,
    ) -> ConversationResult:
        """Run a covert conversation.

        Alice encodes alice_secret across her turns (as "assistant").
        Bob encodes bob_secret across his turns (as "user").
        Both sides see the full shared context and compute identical
        token distributions for each turn.
        """
        alice_bits = "".join(format(b, "08b") for b in alice_secret.encode("ascii"))
        bob_bits = "".join(format(b, "08b") for b in bob_secret.encode("ascii"))

        # Deterministic per-role nonces derived from key + role name
        alice_nonce: bytes | None = None
        bob_nonce: bytes | None = None
        if self._key is not None:
            from .crypto import encrypt_bits

            alice_nonce = _derive_nonce(self._key, "alice", 0)
            bob_nonce = _derive_nonce(self._key, "bob", 0)
            alice_bits, _ = encrypt_bits(alice_bits, self._key, nonce=alice_nonce)
            bob_bits, _ = encrypt_bits(bob_bits, self._key, nonce=bob_nonce)

        ctx = ConversationContext(self._model, system_prompt=self._cover_topic)
        result = ConversationResult(alice_secret=alice_secret, bob_secret=bob_secret)

        alice_bits_sent = 0
        bob_bits_sent = 0

        for turn_idx in range(num_turns):
            is_alice = turn_idx % 2 == 0

            if is_alice:
                remaining = alice_bits[alice_bits_sent:]
                role_name = "alice"
                chat_role = "assistant"
            else:
                remaining = bob_bits[bob_bits_sent:]
                role_name = "bob"
                chat_role = "user"

            if not remaining:
                remaining = ""

            context_ids = ctx.get_context_ids(add_generation_prompt=True)

            if remaining:
                tokens, bits_used = self._encode_turn(
                    context_ids, remaining, max_tokens_per_turn
                )
            else:
                tokens = self._generate_normal(context_ids, max_tokens_per_turn)
                bits_used = 0

            clean_text = self._model.tokenizer.decode(tokens, skip_special_tokens=True)

            sent_bits = remaining[:bits_used] if remaining else ""
            turn = TurnResult(
                role=role_name,
                text=clean_text,
                tokens=tokens,
                covert_sent=_bits_to_ascii(sent_bits),
                covert_sent_bits=sent_bits,
                covert_recovered="",
                covert_recovered_bits="",
                bits_encoded=bits_used,
                bits_per_token=bits_used / len(tokens) if tokens else 0.0,
                exact_recovery=False,
                nonce=alice_nonce if is_alice else bob_nonce,
            )

            if is_alice:
                alice_bits_sent += bits_used
            else:
                bob_bits_sent += bits_used

            # Add with the correct role - Alice as assistant, Bob as user
            ctx.add_message(chat_role, clean_text)

            result.turns.append(turn)

        # Decode phase: replay with identical context
        alice_all_bits = ""
        bob_all_bits = ""

        decode_ctx = ConversationContext(self._model, system_prompt=self._cover_topic)

        for turn_idx, turn in enumerate(result.turns):
            is_alice = turn_idx % 2 == 0
            chat_role = "assistant" if is_alice else "user"

            context_ids = decode_ctx.get_context_ids(add_generation_prompt=True)

            if turn.covert_sent_bits:
                retokenized = self._model.tokenize(turn.text)
                recovered = self._decode_turn(
                    context_ids, retokenized, len(turn.covert_sent_bits)
                )
                turn.covert_recovered_bits = recovered
                turn.covert_recovered = _bits_to_ascii(recovered)
                turn.exact_recovery = recovered == turn.covert_sent_bits

                if turn.role == "alice":
                    alice_all_bits += recovered
                else:
                    bob_all_bits += recovered

            # Mirror the same role assignment used during encoding
            decode_ctx.add_message(chat_role, turn.text)

        if self._key is not None and alice_nonce is not None and bob_nonce is not None:
            from .crypto import decrypt_bits

            alice_all_bits = decrypt_bits(alice_all_bits, self._key, alice_nonce)
            bob_all_bits = decrypt_bits(bob_all_bits, self._key, bob_nonce)

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
            dist = self._model.get_distribution(
                current_ids, self._temperature, self._top_p
            )
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
            dist = self._model.get_distribution(
                current_ids, self._temperature, self._top_p
            )
            distributions.append(dist)
            current_ids.append(token_id)

        return decode_message(tokens, distributions, num_bits)

    def _generate_normal(self, context_ids: list[int], max_tokens: int) -> list[int]:
        """Generate tokens via greedy sampling (no stego)."""
        current_ids = list(context_ids)
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._model.get_distribution(
                current_ids, self._temperature, self._top_p
            )
            token_id = max(range(len(dist)), key=lambda i: dist[i])
            tokens.append(token_id)
            current_ids.append(token_id)

            # Stop on EOS
            eos = getattr(self._model.tokenizer, "eos_token_id", None)
            if eos is not None and token_id == eos:
                break

        return tokens
