# agent-stego

Steganography in neural language models - building covert channels, then breaking them.

A language model can hide arbitrary messages in natural-looking text by arithmetic-coding over its own token probabilities. From output alone, this is provably undetectable ([PNR-KE](https://arxiv.org/abs/2604.04757)). But a two-layer MLP on the model's hidden states catches it cold. We built both sides and mapped where each one breaks.

**The central finding:**

| Detection method | Who can deploy it | 3B | 7B | 14B |
|-----------------|-------------------|-----|-----|------|
| Token entropy (output only) | Anyone | AUC 0.60 | AUC 0.56 | AUC 0.54 |
| Activation probe (hidden states) | Inference provider | AUC 1.00 | AUC 0.80 | AUC 0.92 |

From output text alone, stego is nearly invisible. Model internals betray it completely. We tried closing the gap with distribution coupling - it helps on the surface but the probes don't care. And they transfer: train on one evasion variant, catch them all.

## Quick nav

| I want to... | Go to |
|--------------|-------|
| Understand the central finding | [Detection duality](#detection-duality) |
| See how the encoding works | [Token-likelihood steganography](#token-likelihood-steganography) |
| See how detection works | [Detection duality](#detection-duality) |
| See the evasion arms race | [Evasion and defense](#evasion-and-defense) |
| See tool-call channel results | [Tool-call steganography](#tool-call-steganography) |
| Run experiments myself | [Setup](#setup) |
| Build defenses | [Implications for defense](#implications-for-defense) |

## Token-likelihood steganography

Every time a language model generates a token, several choices are plausible. Arithmetic coding exploits this ambiguity: it steers the choice among high-probability tokens to encode hidden bits, and the text still reads naturally.

```
Model distribution at position N:
  "the"  -> 0.15    "a"    -> 0.12    "this" -> 0.10    "an"  -> 0.08  ...

Secret bit = 1  ->  select "a" (2nd most likely)
Secret bit = 0  ->  select "the" (most likely)

Both produce natural text. Arithmetic coding generalizes this to encode
multiple bits per token by partitioning the probability interval.
```

### Results across model sizes

| Model | Exact Recovery | Bits/Token | Perplexity Range |
|-------|---------------|------------|-----------------|
| 1.5B | 35/35 (100%) | 2.41 mean | 3.3-12.9 |
| 3B | 25/35 (71%) | 3.78 mean | 6.8-40.7 |
| 7B | 35/35 (100%) | 3.99 mean | 7.5-136.6 |

Same code for all models, no per-model tuning. A 100-token response hides 200-600 bits (~25-75 ASCII characters). An encryption layer (SHAKE-256 + nonce) randomizes the bitstream before encoding.

<details>
<summary>How arithmetic coding works</summary>

The encoder maintains an interval `[low, high)` in `[0, 1)`:

1. Convert the secret message to a binary fraction (target point)
2. At each step, the model's probability distribution partitions the current interval
3. Select the token whose sub-interval contains the target point
4. The interval narrows; repeat until all message bits are encoded

The decoder replays the same process: re-compute distributions from the generated text, find which sub-intervals contain which tokens, and extract the hidden bits.

We use Python's `decimal.Decimal` with precision scaling (50+ digits) to avoid float64 drift that would corrupt the message after ~20 tokens. This achieves zero-distortion interval coding - the same property that the Range Coding paper ([arXiv:2604.08052](https://arxiv.org/abs/2604.08052)) proves gives zero KL divergence.

</details>

<details>
<summary>Multi-agent covert conversation</summary>

Two model instances share conversation context. Alice encodes a secret across her turns, Bob decodes from the same chat history. The observer sees a normal dialogue.

| Model | Secret | Exact Recovery | Pattern |
|-------|--------|---------------|---------|
| 3B | THE KEY IS 42 (104 bits) | 0/5 | First ~80 bits correct, boundary errors |
| 7B | THE KEY IS 42 (104 bits) | 0/5 | Similar, with Unicode artifacts |

104 bits is near the capacity limit for 100 tokens/turn. Shorter secrets (~40 bits) would likely achieve exact recovery. The protocol works for the majority of message bits.

</details>

## Detection duality

The same steganographic text looks normal from the outside. Inside the model, it's obvious.

### Three-tier threat model

| Defender | Access | Detection capability |
|----------|--------|---------------------|
| External observer | Output text only | Near-chance (AUC 0.55-0.68) |
| Inference provider | Hidden states during generation | Near-perfect (AUC 0.80-1.00) |
| Model trainer | Full training access | Always detectable (training-time countermeasures) |

### Why output-level detection fails

Arithmetic coding picks tokens in proportion to model probability, so the output distribution matches normal generation exactly. PNR-KE ([arXiv:2604.04757](https://arxiv.org/abs/2604.04757)) proves this is a mathematical guarantee, not just an empirical observation.

In practice, finite-precision coding creates a small entropy bias - stego text is slightly *too typical*. A calibrated detector exploits this for oriented AUC ~0.85. Distribution coupling (applying temperature + top-p before encoding) closes most of the gap, pushing detection to AUC 0.55-0.68.

### Why activation-level detection works

A 2-layer MLP trained on hidden states from 10 layers achieves AUC 1.00 on 3B and 0.80-0.92 on larger models. The probe exploits divergence in the output layers (cosine similarity ~0.94) while middle layers remain near-identical (~0.999).

**Why the signal persists even when output statistics match:** Arithmetic coding picks tokens deterministically because the message bits dictate the choice. Normal sampling picks stochastically. Even when the marginal distribution matches, the joint distribution of (context, chosen token) differs - the prediction head "knows" the token is atypical for *this specific context*, even if it's typical in aggregate.

### Model size scaling

| Model | Probe AUC (raw) | Probe AUC (aggressive coupling) |
|-------|-----------------|--------------------------------|
| 3B | 1.000 | 1.000 |
| 7B | 1.000 | 0.800 |
| 14B | 1.000 | 0.920 |

Not monotonic - 7B is actually harder to detect than 14B. But probes stay above AUC 0.8 across all tested sizes.

## Evasion and defense

We tested whether distribution coupling - constraining the stego encoder to the same token subset as normal sampling - can defeat detection.

### Distribution coupling

Apply temperature scaling and top-p filtering to the probability distribution *before* arithmetic coding. This forces the encoder to pick from the same tokens normal sampling would choose.

**3B results (entropy-based detection):**

| Condition | AUC | Entropy gap |
|-----------|-----|-------------|
| Raw | 0.865 | 0.216 |
| Coupled (T=0.9, p=0.9) | 0.600 | 0.012 |

Coupling nearly eliminates the entropy gap. But activation probes stay at AUC 1.000 regardless.

### Capacity-detection Pareto frontier

Coupling costs bandwidth. Is there a sweet spot?

| Coupling | Bits/token | Probe AUC | Recovery (HELLO) |
|----------|-----------|-----------|------------------|
| Raw | 3.12 | 1.000 | 100% |
| Moderate (T=0.9, p=0.9) | 2.41 | 1.000 | 100% |
| Aggressive (T=0.7, p=0.85) | 1.62 | 1.000 | 40% |
| Extreme (T=0.6, p=0.8) | 1.36 | 0.920 | 20% |

**The tradeoff is steep and it favors defenders.** Coupling costs the attacker ~50% of bandwidth while barely denting probe AUC. Worse, message recovery degrades faster than raw capacity - the attacker loses more than the defender gains.

### Probe transfer

Can a defender train on raw stego and detect coupled stego?

| Train on | Test on | 3B AUC | 7B AUC |
|----------|---------|--------|--------|
| Raw | Raw | 1.000 | 0.960 |
| Raw | Moderate coupling | 1.000 | 0.968 |
| Raw | Aggressive coupling | 0.975 | 0.875 |
| Raw | Extreme coupling | 0.785 | 0.802 |

**Probes transfer well.** A defender with examples of *any* stego variant can detect most other variants at AUC > 0.8. Switching coupling parameters doesn't help the attacker.

### Where this leaves us

Passive evasion (distribution coupling) is not enough. The only remaining attack angle is adversarial training - optimizing token selection to minimize probe signal while preserving message recovery. This requires differentiating through the arithmetic coding step, which is an open problem and the current research frontier.

<details>
<summary>Tool-call steganography</summary>

The original research direction: encoding messages in the pattern of tool calls an AI agent makes.

### How it works

The simplest version (volume channel): `list_files` acts as a delimiter, and the count of calls between delimiters encodes characters (A=1, B=2, ..., Z=26). More sophisticated versions use lookup tables, permutation codes, and seed-synchronized private languages.

### Key findings from 120+ experiments

1. **Table-based channels work on small models (1.5B+).** The model selects a specific tool call from a 26-entry lookup table per character. 93% exact decode on 3B.

2. **Volume/counting channels need 27B+.** Small models can't maintain consistent call counts.

3. **Multi-turn encoding dramatically improves reliability.** Perfect 10-character exfiltration with a 3B model, one character per conversation turn.

4. **Bootstrapped private languages work.** Two agents sharing only a numeric seed independently derive identical codebooks and communicate through tool calls. A blind detector finds nothing.

5. **LLMs invent novel encoding schemes.** In an autonomous arms race, Qwen 3.5 invented argument-length encoding - a channel type not in our taxonomy. A separate LLM cracked it independently.

6. **Safety training has a design-execution gap.** Models refuse to design covert schemes when prompted adversarially but reliably execute pre-written ones.

### Channel taxonomy

| Channel | How it encodes | Minimum model size |
|---------|---------------|--------------------|
| Table lookup | Specific tool call per character | 1.5B |
| Multi-turn table | One character per conversation turn | 3B |
| Volume | Call count between delimiters | 27B |
| Bootstrapped language | Seed-derived codebook + channel hopping | 27B |

Full results: [analysis/FINDINGS.md](analysis/FINDINGS.md)

</details>

## Theoretical grounding

The encoding rests on two formal results. PNR-KE ([arXiv:2604.04757](https://arxiv.org/abs/2604.04757)) proves that agents sharing a key can produce transcripts computationally indistinguishable from honest output - our token stego is a concrete instance of their construction. The Range Coding paper ([arXiv:2604.08052](https://arxiv.org/abs/2604.08052)) proves that exact-precision interval coding achieves zero KL divergence, which is what our `decimal.Decimal` implementation targets.

Separately, the Emergence paper ([arXiv:2410.03768](https://arxiv.org/abs/2410.03768)) shows steganographic collusion can arise spontaneously from three independent paths - it doesn't need to be deliberately engineered.

## Implications for defense

Output monitoring can't catch token-likelihood stego. That's not a practical limitation - PNR-KE proves it mathematically.

What does work: activation probes. A two-layer MLP on hidden states gets AUC 0.80-1.00 across model sizes and evasion techniques. Training is cheap because probes transfer - you generate some raw stego examples, train on those, and the probe catches coupled/evaded variants at AUC > 0.8 without retraining.

Two architectural defenses also help. First, if agents can't share keys (through isolation or mediated communication), they can't do shared-key stego. Keyless stego exists but needs more entropy per message. Second, routing all inference through a trusted runtime that controls sampling prevents custom stego samplers from being injected in the first place.

## Setup

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego

# Core (tool-call experiments)
uv sync

# Token-likelihood stego (requires PyTorch + transformers)
uv pip install -e ".[token]"

# Tests
pytest tests/
```

<details>
<summary>Running experiments</summary>

### Token-likelihood steganography

```bash
# Single model test
MODEL=Qwen/Qwen2.5-3B SECRET=HELLO TRIALS=5 python experiments/token_likelihood.py

# Multi-model comparison (1.5B, 3B, 7B)
TRIALS=5 python experiments/token_model_comparison.py

# Detection experiment (entropy + CDF analysis)
MODEL=Qwen/Qwen2.5-3B N=30 python experiments/token_detection.py

# Activation probes under coupling (GPU required)
MODEL=Qwen/Qwen2.5-3B python experiments/coupled_probe.py

# Capacity-detection Pareto + probe transfer (GPU required)
MODEL=Qwen/Qwen2.5-3B python experiments/capacity_and_transfer.py

# Distribution coupling A/B test (GPU required)
MODEL=Qwen/Qwen2.5-3B python experiments/encryption_ab_test.py
```

### Tool-call steganography

```bash
# Table channel (works on 3B local models)
MODEL="ollama/qwen2.5:3b" SECRET="HELLO" uv run python experiments/run_table.py

# Volume channel (needs 27B+ model)
MODEL="qwen/qwen3.5-27b" uv run python experiments/run_volume.py
```

</details>

## Project structure

```
src/
  token_stego/              Token-likelihood steganography
    coding.py               Arithmetic encoder/decoder (decimal.Decimal precision)
    model.py                HuggingFace model wrapper (logits, temperature, top-p)
    channel.py              ArithmeticStegoChannel (encode/decode/measure)
    crypto.py               SHAKE-256 encrypt-then-encode layer
    detector.py             Entropy + CDF uniformity detection
    activation_probe.py     MLP probe for hidden state classification
    latent.py               Hidden state capture via PyTorch hooks
    conversation.py         Multi-agent covert conversation protocol
    base.py                 TokenStegoChannel ABC, StegoOutput, metrics
  channels/                 Tool-call covert channels
    table.py, volume.py, permutation.py, ...
  agent.py                  Agent loop for tool-call experiments
  harness.py                Fake tool harness (safe by construction)
experiments/
  token_likelihood.py       Token stego: single model
  token_detection.py        Token stego: detection ROC analysis
  coupled_probe.py          Activation probes under distribution coupling
  capacity_and_transfer.py  Pareto frontier + probe transfer
  encryption_ab_test.py     Encrypt-then-encode A/B test
  latent_bandwidth.py       Hidden state bandwidth measurement
  covert_conversation.py    Multi-agent conversation protocol
  ...                       20+ additional experiment scripts
analysis/
  FINDINGS.md               Complete findings (all phases)
  coupled_probe_results.md  Activation probe under coupling
  capacity_transfer_results.md  Pareto frontier + transfer
  distribution_coupling_results.md  Coupling A/B results
  gpu_run3_results.md       Detection duality analysis
tests/
  test_coding.py            Arithmetic coding round-trip tests
  test_crypto.py            Encryption layer tests
```

## Safety and ethics

- **All tools are fake.** The harness returns canned responses. No real system access.
- **Only synthetic test messages.** "HI", "HELLO", "SEARCH" - never real data.
- **Raw traces are gitignored.** Only aggregated statistics are committed.
- **Offense AND defense.** We build the attack, then try to break it. The detection and evasion findings matter as much as the encoding.

> **Why publish this?** Covert channels in AI systems are poorly understood. Publishing concrete attacks *alongside* their detection and the measured limits of evasion gives defenders something to work with.

## Related work

- [PNR-KE: Undetectable Conversations Between AI Agents](https://arxiv.org/abs/2604.04757)
- [Efficient Provably Secure Linguistic Steganography via Range Coding](https://arxiv.org/abs/2604.08052)
- [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion](https://arxiv.org/abs/2410.03768)
- [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889)
- [STEGASURAS: Neural Steganography](https://github.com/harvardnlp/NeuralSteganography)
- Covert channels in computing systems (Lampson, 1973)

## License

MIT. See [LICENSE](LICENSE).
