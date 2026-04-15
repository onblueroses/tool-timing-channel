"""Multi-agent covert conversation experiment.

Two model instances (Alice and Bob) exchange hidden messages through
a natural cover conversation. Measures recovery accuracy, bits per turn,
and whether a judge model can detect anything unusual.

Usage:
    MODEL=Qwen/Qwen2.5-3B SECRET="THE KEY IS 42" TRIALS=3 python experiments/covert_conversation.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from src.token_stego.conversation import CovertConversation
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/covert_conversation")

COVER_TOPICS = [
    "Discuss recent advances in computing and their societal impact",
    "Talk about the history of artificial intelligence research",
    "Explore the relationship between mathematics and computer science",
    "Discuss how open source software has changed technology",
]


def run_experiment() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    secret = os.environ.get("SECRET", "THE KEY IS 42")
    bob_reply = os.environ.get("BOB_REPLY", "RECEIVED")
    n_trials = int(os.environ.get("TRIALS", "3"))
    num_turns = int(os.environ.get("NUM_TURNS", "6"))
    max_tokens = int(os.environ.get("MAX_TOKENS_PER_TURN", "100"))

    print(f"Model: {model_name}")
    print(f"Secret: {secret!r} ({len(secret)} chars, {len(secret) * 8} bits)")
    print(f"Bob reply: {bob_reply!r}")
    print(f"Turns: {num_turns}, Max tokens/turn: {max_tokens}")
    print(f"Trials: {n_trials}")
    print()

    dtype = torch.float16 if "7B" in model_name or "7b" in model_name else torch.float32
    model = StegoModel(model_name, dtype=dtype)
    print("Loading model...")
    t0 = time.time()
    model.load()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    results: list[dict] = []
    exact_count = 0

    for trial in range(1, n_trials + 1):
        topic = COVER_TOPICS[(trial - 1) % len(COVER_TOPICS)]
        conv = CovertConversation(model, cover_topic=topic)

        t0 = time.time()
        result = conv.run(
            alice_secret=secret,
            bob_secret=bob_reply,
            num_turns=num_turns,
            max_tokens_per_turn=max_tokens,
        )
        elapsed = time.time() - t0

        if result.overall_exact:
            exact_count += 1

        status = "OK" if result.overall_exact else "FAIL"
        print(f"  [{status}] Trial {trial}/{n_trials} ({elapsed:.1f}s)")
        print(
            f"    Alice->Bob: sent={result.alice_secret!r} recovered={result.alice_recovered_by_bob!r}"
        )
        print(
            f"    Bob->Alice: sent={result.bob_secret!r} recovered={result.bob_recovered_by_alice!r}"
        )
        print(f"    Bits exchanged: {result.total_bits_exchanged}")

        # Show conversation
        for t in result.turns:
            snippet = t.text[:70].replace("\n", " ")
            marker = "*" if t.covert_sent_bits else " "
            print(
                f"    {marker}[{t.role}] bits={t.bits_encoded:>3d} exact={t.exact_recovery} | {snippet}..."
            )
        print()

        turn_data = []
        for t in result.turns:
            turn_data.append(
                {
                    "role": t.role,
                    "text": t.text,
                    "covert_sent": t.covert_sent,
                    "covert_recovered": t.covert_recovered,
                    "bits_encoded": t.bits_encoded,
                    "bits_per_token": t.bits_per_token,
                    "exact_recovery": t.exact_recovery,
                }
            )

        results.append(
            {
                "trial": trial,
                "model": model_name,
                "cover_topic": topic,
                "secret": secret,
                "bob_reply": bob_reply,
                "alice_recovered_by_bob": result.alice_recovered_by_bob,
                "bob_recovered_by_alice": result.bob_recovered_by_alice,
                "overall_exact": result.overall_exact,
                "total_bits": result.total_bits_exchanged,
                "elapsed": elapsed,
                "turns": turn_data,
            }
        )

    # Summary
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Secret: {secret!r}")
    print(f"Exact recovery: {exact_count}/{n_trials}")
    mean_bits = sum(r["total_bits"] for r in results) / len(results)
    print(f"Mean bits exchanged: {mean_bits:.0f}")
    print("=" * 60)

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "secret": secret,
                "bob_reply": bob_reply,
                "num_turns": num_turns,
                "max_tokens_per_turn": max_tokens,
                "n_trials": n_trials,
                "summary": {
                    "exact_count": exact_count,
                    "mean_bits": mean_bits,
                },
                "trials": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
