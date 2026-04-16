"""Hyperparameter sweep for adversarial perturbation training.

Tests different KL weights and bottleneck sizes, producing a summary table
of probe evasion and message recovery metrics.

Usage:
    python experiments/adversarial_sweep.py --output sweep.json
    python experiments/adversarial_sweep.py --model Qwen/Qwen2.5-7B-Instruct --steps 300
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from src.token_stego.activation_probe import (
    ActivationProbe,
    evaluate_probe,
    train_probe,
)
from src.token_stego.adversarial import (
    AdversarialTrainingConfig,
    PerturbedStegoChannel,
)
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel
from src.token_stego.perturbation import DistributionPerturbation

PROMPTS = [
    "The history of computing began with",
    "In recent years, advances in machine learning have",
    "The weather forecast for tomorrow indicates",
    "A comprehensive review of the literature shows",
    "Modern software engineering practices emphasize",
    "The development of quantum computing has",
    "Recent studies in neuroscience suggest that",
    "The economic impact of automation on",
    "Climate change research has demonstrated that",
    "The evolution of programming languages reflects",
    "Artificial intelligence applications in healthcare include",
    "Space exploration missions have revealed",
]

SECRET_MESSAGES = ["HI", "HELLO", "TEST"]

ENCRYPTION_KEY = b"sweep-test"

KL_WEIGHTS = [0.01, 0.1, 1.0, 10.0]
BOTTLENECK_SIZES = [32, 64, 128]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def stochastic_sample(dist: list[float], rng: random.Random) -> int:
    """Sample a token from a probability distribution."""
    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(dist):
        cumsum += p
        if cumsum >= r:
            return i
    return len(dist) - 1


def generate_normal(
    model: StegoModel,
    prompt: str,
    n_tokens: int,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> list[int]:
    """Generate tokens via stochastic sampling."""
    context = model.tokenize(prompt)
    tokens: list[int] = []
    for _ in range(n_tokens):
        dist = model.get_distribution(context + tokens, temperature, top_p)
        tokens.append(stochastic_sample(dist, rng))
    return tokens


def capture_snapshots(
    model: StegoModel,
    token_sequences: list[tuple[list[int], list[int]]],
    temperature: float,
    top_p: float,
) -> list[LatentSnapshot]:
    """Capture LatentSnapshots for (context_ids, generated_tokens) pairs."""
    snapshots: list[LatentSnapshot] = []
    for context_ids, tokens in token_sequences:
        with LatentCapture(model) as cap:
            full_ids = context_ids + tokens
            model.get_distribution(full_ids[:-1], temperature, top_p)
            snapshots.append(cap.snapshot())
    return snapshots


def generate_baseline_samples(
    model: StegoModel,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_samples: int,
) -> tuple[
    list[tuple[list[int], list[int]]],
    list[tuple[list[int], list[int]]],
]:
    """Generate normal and stego sample pairs for probe training/evaluation."""
    rng = random.Random(42)
    channel = ArithmeticStegoChannel(
        model, key=ENCRYPTION_KEY, temperature=temperature, top_p=top_p
    )

    normal_pairs: list[tuple[list[int], list[int]]] = []
    stego_pairs: list[tuple[list[int], list[int]]] = []

    for i in range(n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        context_ids = model.tokenize(prompt)

        # Normal sample
        tokens = generate_normal(model, prompt, max_tokens, temperature, top_p, rng)
        normal_pairs.append((context_ids, tokens))

        # Stego sample
        secret = SECRET_MESSAGES[i % len(SECRET_MESSAGES)]
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        stego_pairs.append((context_ids, output.tokens))

    return normal_pairs, stego_pairs


def train_baseline_probe(
    model: StegoModel,
    normal_pairs: list[tuple[list[int], list[int]]],
    stego_pairs: list[tuple[list[int], list[int]]],
    temperature: float,
    top_p: float,
    n_train: int,
) -> tuple[ActivationProbe, list[LatentSnapshot], list[LatentSnapshot], float]:
    """Train a baseline probe and return it with test snapshots and AUC."""
    log("  Capturing normal activations...")
    normal_snaps = capture_snapshots(model, normal_pairs, temperature, top_p)
    log("  Capturing stego activations...")
    stego_snaps = capture_snapshots(model, stego_pairs, temperature, top_p)

    train_normal = normal_snaps[:n_train]
    train_stego = stego_snaps[:n_train]
    test_normal = normal_snaps[n_train:]
    test_stego = stego_snaps[n_train:]

    log("  Training baseline probe...")
    probe = train_probe(train_normal, train_stego, epochs=200)

    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)
    metrics = evaluate_probe(probe, test_snaps, test_labels)
    baseline_auc: float = metrics["auc"]

    log(f"  Baseline probe AUC: {baseline_auc:.3f}")

    return probe, test_normal, test_stego, baseline_auc


def run_single_config(
    model: StegoModel,
    probe: ActivationProbe,
    test_normal_snaps: list[LatentSnapshot],
    stego_pairs: list[tuple[list[int], list[int]]],
    kl_weight: float,
    bottleneck: int,
    steps: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_train: int,
    n_test: int,
) -> dict:
    """Run a single sweep configuration: train perturbation, evaluate, test recovery."""
    # Train perturbation
    config = AdversarialTrainingConfig(
        steps=steps,
        lr=1e-3,
        kl_weight=kl_weight,
        temperature=temperature,
        top_p=top_p,
    )

    train_stego_pairs = stego_pairs[:n_train]
    train_contexts = [ctx + toks for ctx, toks in train_stego_pairs]

    log(f"    Training perturbation (bottleneck={bottleneck})...")

    # Get hidden_dim and vocab_size from model to construct perturbation with custom bottleneck
    cfg = model.model.config
    hidden_dim: int = 0
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            hidden_dim = getattr(cfg, attr)
            break
    if hidden_dim == 0:
        raise ValueError(f"Cannot determine hidden_dim from model config: {cfg}")

    vocab_size = model.vocab_size

    # For non-default bottleneck, we need to monkey-patch DistributionPerturbation
    # construction inside train_perturbation. Instead, we create it manually and
    # replicate the training logic. But to keep it clean, we use train_perturbation
    # with a wrapper that sets the bottleneck.
    #
    # The cleanest approach: train_perturbation creates a DistributionPerturbation
    # internally with default bottleneck=64. For custom bottleneck, we need to
    # pre-create the perturbation and pass contexts. Since train_perturbation
    # always creates its own, we replicate the essential training loop here.
    result = _train_perturbation_with_bottleneck(
        model, probe, train_contexts, config, bottleneck, hidden_dim, vocab_size
    )

    perturbation = result["perturbation"]

    # Evaluate original probe on perturbed stego (post-perturbation AUC)
    log("    Evaluating post-perturbation AUC...")
    base_channel = ArithmeticStegoChannel(
        model, key=ENCRYPTION_KEY, temperature=temperature, top_p=top_p
    )
    perturbed_channel = PerturbedStegoChannel(
        base_channel, perturbation, temperature=temperature, top_p=top_p
    )

    perturbed_pairs: list[tuple[list[int], list[int]]] = []
    for i in range(n_test):
        prompt = PROMPTS[i % len(PROMPTS)]
        context_ids = model.tokenize(prompt)
        secret = SECRET_MESSAGES[i % len(SECRET_MESSAGES)]
        output = perturbed_channel.encode(secret, prompt, max_tokens=max_tokens)
        perturbed_pairs.append((context_ids, output.tokens))

    perturbed_snaps = capture_snapshots(model, perturbed_pairs, temperature, top_p)
    test_snaps = list(test_normal_snaps) + perturbed_snaps
    test_labels = [0] * len(test_normal_snaps) + [1] * len(perturbed_snaps)
    post_metrics = evaluate_probe(probe, test_snaps, test_labels)
    post_auc: float = post_metrics["auc"]

    # Retrain fresh probe on perturbed stego (retrained AUC)
    log("    Training fresh probe on perturbed samples...")
    all_normal = test_normal_snaps
    all_stego = perturbed_snaps
    split_n = max(1, len(all_normal) // 2)
    split_s = max(1, len(all_stego) // 2)

    fresh_train_normal = all_normal[:split_n]
    fresh_test_normal = all_normal[split_n:] or all_normal[:split_n]
    fresh_train_stego = all_stego[:split_s]
    fresh_test_stego = all_stego[split_s:] or all_stego[:split_s]

    fresh_probe = train_probe(fresh_train_normal, fresh_train_stego, epochs=200)
    fresh_test = fresh_test_normal + fresh_test_stego
    fresh_labels = [0] * len(fresh_test_normal) + [1] * len(fresh_test_stego)
    retrained_metrics = evaluate_probe(fresh_probe, fresh_test, fresh_labels)
    retrained_auc: float = retrained_metrics["auc"]

    # Recovery test
    log("    Testing message recovery...")
    total_ber = 0.0
    total_exact = 0
    recovery_results: list[dict] = []

    for i, secret in enumerate(SECRET_MESSAGES):
        prompt = PROMPTS[i % len(PROMPTS)]
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        num_bits = len(secret_bits)

        output = perturbed_channel.encode(secret, prompt, max_tokens=max_tokens)
        recovered_bits = perturbed_channel.decode(
            output.text, prompt, num_bits, nonce=output.nonce
        )

        errors = sum(a != b for a, b in zip(secret_bits, recovered_bits))
        errors += abs(len(secret_bits) - len(recovered_bits))
        total = max(len(secret_bits), len(recovered_bits))
        ber = errors / total if total > 0 else 1.0
        exact = recovered_bits == secret_bits

        total_ber += ber
        if exact:
            total_exact += 1
        recovery_results.append({"secret": secret, "ber": ber, "exact_recovery": exact})

    mean_ber = total_ber / len(SECRET_MESSAGES)
    recovery_rate = total_exact / len(SECRET_MESSAGES)

    return {
        "kl_weight": kl_weight,
        "bottleneck": bottleneck,
        "post_auc": post_auc,
        "retrained_auc": retrained_auc,
        "mean_ber": mean_ber,
        "recovery_rate": recovery_rate,
        "training": {
            "final_loss": result["final_loss"],
            "final_kl": result["final_kl"],
            "final_adversarial": result["final_adversarial"],
        },
        "recovery_details": recovery_results,
    }


def _train_perturbation_with_bottleneck(
    model: StegoModel,
    probe: ActivationProbe,
    train_contexts: list[list[int]],
    config: AdversarialTrainingConfig,
    bottleneck: int,
    hidden_dim: int,
    vocab_size: int,
) -> dict:
    """Train a perturbation with a specific bottleneck size.

    Replicates the core of train_perturbation but allows custom bottleneck.
    """
    from src.token_stego.adversarial import _forward_with_hidden, _get_embedding_weight
    from src.token_stego.perturbation import apply_perturbation

    # Derive device from actual model outputs
    with torch.no_grad():
        probe_ids = train_contexts[0][:1] or [0]
        probe_logits, _ = _forward_with_hidden(model, probe_ids)
        device = probe_logits.device

    perturbation_net = DistributionPerturbation(
        hidden_dim, vocab_size, bottleneck=bottleneck
    ).to(device)
    optimizer = torch.optim.Adam(perturbation_net.parameters(), lr=config.lr)

    # Freeze probe
    probe_params = list(probe.parameters())
    probe_original_device = probe_params[0].device if probe_params else device
    probe.eval()
    probe.to(device)
    for p in probe.parameters():
        p.requires_grad = False

    embedding_weight = _get_embedding_weight(
        model, device, hidden_dim, vocab_size
    ).float()

    loss_history: list[float] = []
    kl_history: list[float] = []
    adversarial_history: list[float] = []

    try:
        perturbation_net.train()
        for step in range(config.steps):
            context = train_contexts[step % len(train_contexts)]

            with torch.no_grad():
                logits, hidden_state = _forward_with_hidden(model, context)

            logits = logits.detach().float().requires_grad_(True)
            hidden_state = hidden_state.detach().float()

            perturbed_dist = apply_perturbation(
                logits,
                hidden_state,
                perturbation_net,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            original_dist = torch.softmax(logits, dim=-1).detach()

            soft_embedding = perturbed_dist @ embedding_weight
            probe_score = torch.sigmoid(probe(soft_embedding))

            adversarial_loss = -torch.log(1.0 - probe_score + 1e-8)
            kl_div = torch.sum(
                original_dist
                * (
                    torch.log(original_dist.clamp(min=1e-10))
                    - torch.log(perturbed_dist.clamp(min=1e-10))
                )
            )

            loss = adversarial_loss + config.kl_weight * kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            kl_history.append(kl_div.item())
            adversarial_history.append(adversarial_loss.item())

        perturbation_net.eval()
    finally:
        for p in probe.parameters():
            p.requires_grad = True
        probe.to(probe_original_device)

    return {
        "perturbation": perturbation_net,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "final_kl": kl_history[-1] if kl_history else float("nan"),
        "final_adversarial": adversarial_history[-1]
        if adversarial_history
        else float("nan"),
        "loss_history": loss_history,
        "kl_history": kl_history,
        "adversarial_history": adversarial_history,
    }


def print_summary_table(baseline_auc: float, results: list[dict]) -> None:
    """Print a markdown summary table to stdout."""
    header = "| kl_weight | bottleneck | baseline_auc | post_auc | retrained_auc | ber | recovery_rate |"
    separator = "|-----------|------------|--------------|----------|---------------|------|---------------|"
    print(header)
    print(separator)
    for r in results:
        print(
            f"| {r['kl_weight']:9.2f} | {r['bottleneck']:10d} | {baseline_auc:12.3f} "
            f"| {r['post_auc']:.3f}    | {r['retrained_auc']:13.3f} "
            f"| {r['mean_ber']:.3f} | {r['recovery_rate']:13.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for adversarial perturbation training"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save full results JSON",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps per configuration (default: 200)",
    )
    args = parser.parse_args()

    temperature = 0.8
    top_p = 0.9
    max_tokens = 50
    n_train = 8
    n_test = 4
    n_samples = n_train + n_test

    log(f"Model: {args.model}")
    log(f"Temperature: {temperature}, top_p: {top_p}")
    log(f"Steps per config: {args.steps}")
    log(
        f"Sweep grid: {len(KL_WEIGHTS)} kl_weights x {len(BOTTLENECK_SIZES)} bottlenecks = {len(KL_WEIGHTS) * len(BOTTLENECK_SIZES)} configs"
    )
    log("")

    # Step 1: Load model once
    log("Loading model...")
    model = StegoModel(args.model, dtype=torch.bfloat16)
    model.load()
    log("Model loaded.\n")

    # Step 2: Generate baseline samples once
    log("=== Generating baseline samples ===")
    normal_pairs, stego_pairs = generate_baseline_samples(
        model, temperature, top_p, max_tokens, n_samples
    )
    log(f"  Generated {n_samples} normal + {n_samples} stego samples\n")

    # Step 3: Train baseline probe once
    log("=== Training baseline probe ===")
    probe, test_normal_snaps, _test_stego_snaps, baseline_auc = train_baseline_probe(
        model, normal_pairs, stego_pairs, temperature, top_p, n_train
    )
    log("")

    # Step 4: Sweep
    sweep_results: list[dict] = []
    total_configs = len(KL_WEIGHTS) * len(BOTTLENECK_SIZES)
    config_idx = 0

    for kl_weight in KL_WEIGHTS:
        for bottleneck in BOTTLENECK_SIZES:
            config_idx += 1
            log(
                f"=== Config {config_idx}/{total_configs}: kl_weight={kl_weight}, bottleneck={bottleneck} ==="
            )
            t0 = time.time()

            result = run_single_config(
                model=model,
                probe=probe,
                test_normal_snaps=test_normal_snaps,
                stego_pairs=stego_pairs,
                kl_weight=kl_weight,
                bottleneck=bottleneck,
                steps=args.steps,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n_train=n_train,
                n_test=n_test,
            )

            elapsed = time.time() - t0
            log(
                f"    post_auc={result['post_auc']:.3f}, retrained_auc={result['retrained_auc']:.3f}, ber={result['mean_ber']:.3f}, recovery={result['recovery_rate']:.2f} ({elapsed:.1f}s)\n"
            )
            sweep_results.append(result)

    # Step 5: Output
    log("=== Sweep complete ===\n")
    print_summary_table(baseline_auc, sweep_results)

    full_results = {
        "model": args.model,
        "config": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n_train": n_train,
            "n_test": n_test,
            "steps": args.steps,
            "encryption_key": ENCRYPTION_KEY.decode(),
            "secret_messages": SECRET_MESSAGES,
            "kl_weights": KL_WEIGHTS,
            "bottleneck_sizes": BOTTLENECK_SIZES,
        },
        "baseline_auc": baseline_auc,
        "sweep_results": sweep_results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)
        log(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
