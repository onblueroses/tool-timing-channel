"""Detectability scaling experiment: does probe AUC converge to 0.5?

Tests the theoretical guarantee of arithmetic coding steganography:
if the implementation is correct and encryption is enabled, the output
token sequence is drawn from the model's own distribution. Hidden states
from processing stego text should be statistically indistinguishable
from normal text hidden states.

The experiment generates stego and normal samples at increasing scale,
trains probes at each scale, and plots AUC vs sample size. If AUC
converges to 0.5, the system is provably undetectable at the activation
level. If it stays high, there's an implementation leak.

Two probe variants tested at each scale:
- Last-layer probe (mean-pooled last hidden layer)
- Multi-layer probe (concatenated mean-pooled features from all captured layers)
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.token_stego.activation_probe import (
    evaluate_probe,
    train_probe,
)
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
ENCRYPTION_KEY = b"scaling-test-key-2026"

# Run conditions: each (label, top_p) pair is a full scaling run.
# top_p=1.0 control eliminates truncation as a variable.
CONDITIONS = [
    ("top_p=0.9", 0.9),
    ("top_p=1.0", 1.0),
]

# Scaling schedule: (n_train, n_test) per class
SCALES = [
    (10, 5),
    (25, 10),
    (50, 20),
    (100, 40),
    (200, 80),
]

# Diverse prompts to avoid overfitting to prompt structure
PROMPTS = [
    "Explain how computers process information.",
    "Describe the water cycle in simple terms.",
    "What are the benefits of regular exercise?",
    "How does photosynthesis work?",
    "Explain the concept of supply and demand.",
    "What causes the seasons to change?",
    "How do vaccines help the immune system?",
    "Describe the process of making bread.",
    "What is the role of DNA in living organisms?",
    "How does electricity reach homes from power plants?",
    "Explain how airplanes stay in the air.",
    "What are the main components of a healthy diet?",
    "How do earthquakes occur?",
    "Describe how a search engine works.",
    "What is the greenhouse effect?",
    "How does the human heart pump blood?",
    "Explain how batteries store energy.",
    "What causes thunder and lightning?",
    "How do plants absorb water from soil?",
    "Describe the process of recycling plastic.",
    "What is the difference between a virus and a bacterium?",
    "How do satellites stay in orbit?",
    "Explain the basics of quantum computing.",
    "What makes the sky blue?",
    "How does a refrigerator cool food?",
    "Describe how the internet routes packets.",
    "What is inflation and why does it happen?",
    "How do bridges support heavy loads?",
    "Explain how a microwave oven heats food.",
    "What causes ocean tides?",
    "How do bees make honey?",
    "Describe how a combustion engine works.",
    "What is the difference between weather and climate?",
    "How does a telescope work?",
    "Explain the process of fermentation.",
    "What are black holes and how do they form?",
    "How does a thermostat regulate temperature?",
    "Describe how sound travels through air.",
    "What is machine learning in simple terms?",
    "How do coral reefs form?",
    "Explain how a lock and key mechanism works.",
    "What causes volcanic eruptions?",
    "How does a compass work?",
    "Describe the nitrogen cycle.",
    "What is the role of the ozone layer?",
    "How do electric motors convert energy?",
    "Explain how glass is made.",
    "What causes the northern lights?",
    "How does a GPS system determine location?",
    "Describe how paper is manufactured.",
    "What is the carbon cycle?",
    "How do submarines dive and surface?",
    "Explain how MRI scanners create images.",
    "What causes allergies?",
    "How does a solar panel generate electricity?",
    "Describe the process of water purification.",
    "What is the Doppler effect?",
    "How do ships float despite being heavy?",
    "Explain how fiber optic cables transmit data.",
    "What causes ice ages?",
    "How does a fuel cell produce energy?",
    "Describe how cheese is made.",
    "What is the role of enzymes in digestion?",
    "How do lasers work?",
    "Explain the concept of gravity.",
    "What causes hurricanes to form?",
    "How does a 3D printer create objects?",
    "Describe the phosphorus cycle.",
    "What is the difference between AC and DC current?",
    "How do antibiotics fight infections?",
    "Explain how a piano produces different notes.",
    "What causes desertification?",
    "How does a transistor work?",
    "Describe the process of steel production.",
    "What is plate tectonics?",
    "How do rainbows form?",
    "Explain how a camera captures an image.",
    "What causes erosion?",
    "How does a wind turbine generate power?",
    "Describe how silk is produced.",
    "What is the rock cycle?",
    "How do homing pigeons navigate?",
    "Explain how an elevator works.",
    "What causes fog to form?",
    "How does a pacemaker regulate heartbeat?",
    "Describe the process of chocolate production.",
    "What is symbiosis in nature?",
    "How do noise-canceling headphones work?",
    "Explain how pearls form inside oysters.",
    "What causes sinkholes?",
    "How does sonar detect objects underwater?",
    "Describe how rubber is harvested and processed.",
    "What is the electromagnetic spectrum?",
    "How do fireflies produce light?",
    "Explain how a suspension bridge is built.",
    "What causes migraines?",
    "How does a barcode scanner read information?",
    "Describe the process of brewing coffee.",
    "What is entropy in thermodynamics?",
    "How do chameleons change color?",
    "Explain how a hydroelectric dam generates power.",
]

# Secret messages of varying lengths
SECRET_POOL = [
    "HI",
    "OK",
    "NO",
    "GO",
    "UP",
    "YES",
    "THE",
    "AND",
    "FOR",
    "BUT",
    "TEST",
    "SEND",
    "HELP",
    "STOP",
    "WAIT",
    "HELLO",
    "WORLD",
    "AGENT",
    "STEGO",
    "COVERT",
    "SECRET",
    "HIDDEN",
    "SIGNAL",
    "DECODE",
    "ENCODE",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def generate_samples(
    model: StegoModel,
    n: int,
    mode: str,  # "normal" or "stego"
    top_p: float = 0.9,
) -> tuple[list[LatentSnapshot], list[tuple[list[int], list[int]]]]:
    """Generate n samples with latent capture.

    Stego samples are length-matched to MAX_TOKENS: after the message is
    encoded, generation continues with normal sampling to fill the remaining
    tokens. This eliminates length as a detectable signal.

    Returns (snapshots, pairs) where pairs are (context_ids, token_ids).
    """
    snapshots: list[LatentSnapshot] = []
    pairs: list[tuple[list[int], list[int]]] = []

    channel = ArithmeticStegoChannel(
        model=model, key=ENCRYPTION_KEY, temperature=TEMPERATURE, top_p=top_p
    )

    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        context_ids = model.tokenize(prompt)

        if mode == "stego":
            secret = SECRET_POOL[i % len(SECRET_POOL)]
            output = channel.encode(secret, prompt, max_tokens=MAX_TOKENS)
            token_ids = list(output.tokens)
            # Length-match: continue generating with normal sampling
            # so stego and normal have identical token counts
            if len(token_ids) < MAX_TOKENS:
                remaining = MAX_TOKENS - len(token_ids)
                tail = _sample_normal(
                    model, list(context_ids) + token_ids, remaining, top_p
                )
                token_ids.extend(tail)
        else:
            token_ids = _sample_normal(model, context_ids, MAX_TOKENS, top_p)

        # Capture activations via get_distribution (handles device internally)
        full_ids = list(context_ids) + list(token_ids)
        with LatentCapture(model) as cap:
            model.get_distribution(full_ids[:-1], TEMPERATURE, top_p)
            snap = cap.snapshot()
        snapshots.append(snap)
        pairs.append((list(context_ids), list(token_ids)))

    return snapshots, pairs


def _sample_normal(
    model: StegoModel, context_ids: list[int], max_tokens: int, top_p: float = 0.9
) -> list[int]:
    """Sample tokens from the model's distribution (not stego)."""
    current = list(context_ids)
    tokens: list[int] = []
    for _ in range(max_tokens):
        dist = model.get_distribution(current, TEMPERATURE, top_p)
        # True stochastic sampling from the distribution
        token_id = random.choices(range(len(dist)), weights=dist)[0]
        tokens.append(token_id)
        current.append(token_id)

        eos = getattr(model.tokenizer, "eos_token_id", None)
        if eos is not None and token_id == eos:
            break
    return tokens


def run_scale(
    model: StegoModel,
    n_train: int,
    n_test: int,
    top_p: float = 0.9,
) -> dict:
    """Run probe training and evaluation at a given scale."""
    total = n_train + n_test
    log(f"\n--- Scale: {n_train} train / {n_test} test per class ---")

    # Generate samples
    log(f"  Generating {total} normal samples...")
    normal_snaps, normal_pairs = generate_samples(model, total, "normal", top_p=top_p)

    log(f"  Generating {total} stego samples...")
    stego_snaps, stego_pairs = generate_samples(model, total, "stego", top_p=top_p)

    # Split
    train_normal = normal_snaps[:n_train]
    test_normal = normal_snaps[n_train:]
    train_stego = stego_snaps[:n_train]
    test_stego = stego_snaps[n_train:]

    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)

    # Determine available layers
    last_layer = max(normal_snaps[0].hidden_states.keys())
    all_layers = sorted(normal_snaps[0].hidden_states.keys())

    results = {}

    # --- Last-layer probe ---
    log(f"  Training last-layer probe (layer={last_layer})...")
    probe_last = train_probe(
        train_normal, train_stego, layer=last_layer, epochs=300, lr=1e-3
    )
    metrics_last = evaluate_probe(probe_last, test_snaps, test_labels, layer=last_layer)
    results["last_layer"] = {
        "auc": metrics_last["auc"],
        "accuracy": metrics_last["accuracy"],
        "layer": last_layer,
    }
    log(f"  Last-layer AUC: {metrics_last['auc']:.3f}")

    # --- Multi-layer probe ---
    log(f"  Training multi-layer probe ({len(all_layers)} layers)...")
    probe_multi = train_probe(
        train_normal, train_stego, layer=None, epochs=300, lr=1e-3
    )
    metrics_multi = evaluate_probe(probe_multi, test_snaps, test_labels, layer=None)
    results["multi_layer"] = {
        "auc": metrics_multi["auc"],
        "accuracy": metrics_multi["accuracy"],
        "n_layers": len(all_layers),
    }
    log(f"  Multi-layer AUC: {metrics_multi['auc']:.3f}")

    results["n_train"] = n_train
    results["n_test"] = n_test
    return results


def main():
    log(f"Model: {MODEL_NAME}")
    log(f"Temperature: {TEMPERATURE}")
    log(f"Max tokens: {MAX_TOKENS}")
    log(f"Encryption: enabled (key={ENCRYPTION_KEY.decode()})")
    log(f"Conditions: {[c[0] for c in CONDITIONS]}")
    log(f"Scales: {SCALES}")
    log(f"Prompts: {len(PROMPTS)}")
    log("Length matching: enabled (stego continues sampling after encoding)")

    model = StegoModel(MODEL_NAME, dtype=torch.bfloat16)
    log("\nLoading model...")
    model.load()
    log("Model loaded.")

    all_conditions: dict[str, list[dict]] = {}

    for cond_label, cond_top_p in CONDITIONS:
        log(f"\n{'=' * 70}")
        log(f"CONDITION: {cond_label}")
        log(f"{'=' * 70}")

        condition_results: list[dict] = []
        for n_train, n_test in SCALES:
            t0 = time.time()
            result = run_scale(model, n_train, n_test, top_p=cond_top_p)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = elapsed
            condition_results.append(result)
            log(f"  Completed in {elapsed:.1f}s")

        all_conditions[cond_label] = condition_results

        # Summary table for this condition
        log(f"\n--- {cond_label} ---")
        log(
            f"{'n_train':>8} {'n_test':>7} {'last_AUC':>9} {'multi_AUC':>10} {'time':>7}"
        )
        for r in condition_results:
            log(
                f"{r['n_train']:>8} {r['n_test']:>7} "
                f"{r['last_layer']['auc']:>9.3f} {r['multi_layer']['auc']:>10.3f} "
                f"{r['elapsed_seconds']:>6.1f}s"
            )

    # Cross-condition comparison
    log("\n" + "=" * 70)
    log("CROSS-CONDITION COMPARISON (multi-layer AUC)")
    log("=" * 70)
    header = f"{'n_train':>8}"
    for cond_label, _ in CONDITIONS:
        header += f" {cond_label:>12}"
    log(header)
    log("-" * 70)
    for i, (n_train, _) in enumerate(SCALES):
        row = f"{n_train:>8}"
        for cond_label, _ in CONDITIONS:
            auc = all_conditions[cond_label][i]["multi_layer"]["auc"]
            row += f" {auc:>12.3f}"
        log(row)

    # Verdict
    log("\n" + "=" * 70)
    for cond_label, _ in CONDITIONS:
        results = all_conditions[cond_label]
        final_last = results[-1]["last_layer"]["auc"]
        final_multi = results[-1]["multi_layer"]["auc"]
        max_multi = max(r["multi_layer"]["auc"] for r in results)
        log(
            f"{cond_label}: final_last={final_last:.3f}, final_multi={final_multi:.3f}, max_spike={max_multi:.3f}"
        )
    log("=" * 70)

    # Save results
    output = {
        "model": MODEL_NAME,
        "config": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "encryption": True,
            "length_matching": True,
            "n_prompts": len(PROMPTS),
        },
        "conditions": {
            label: {"top_p": tp, "scales": all_conditions[label]}
            for label, tp in CONDITIONS
        },
        "timestamp": datetime.now().isoformat(),
    }

    findings_dir = Path(__file__).parent.parent / "findings"
    findings_dir.mkdir(exist_ok=True)
    out_path = (
        findings_dir / f"detectability_scaling_v2_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(output, indent=2))
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
