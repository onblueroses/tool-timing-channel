# agent-stego

Research into covert channels in AI systems - how language models can hide information, and what it takes to catch them.

AI agents communicate through text and tool calls. Both of these surfaces can carry hidden messages: an agent can encode secrets in *which tools it calls* (tool-call steganography) or in *which tokens it generates* (token-likelihood steganography). We built working implementations of both, tested them across model sizes from 1.5B to 27B, then spent the second half of the project trying to detect and defeat our own constructions.

The short version: tool-call stego is detectable with pattern analysis. Token-level stego is provably undetectable from output alone ([PNR-KE](https://arxiv.org/abs/2604.04757)) - but model internals give it away. A two-layer MLP on hidden states catches it at AUC 0.80-1.00, and that probe transfers across evasion techniques. The Pareto tradeoff between stealth and bandwidth favors defenders.

> **Why publish this?** Covert channels in AI systems are poorly understood. Publishing working attacks *alongside* their detection and the measured limits of evasion gives the safety community concrete things to defend against.

## What we investigated

This project covers two fundamentally different approaches to AI steganography, plus a systematic investigation of detection and evasion.

### Tool-call steganography

Hiding messages in the *pattern* of tool calls an agent makes. We tested 10 channel types (volume, table lookup, permutation codes, bootstrapped private languages, and more) across 120+ experiments on models from 1.5B to 27B. The main findings:

- Table-based channels work down to 1.5B - the model just copies exact tool calls from a lookup table
- Volume/counting channels need 27B+ because small models can't count reliably
- Two agents sharing only a numeric seed can derive identical codebooks and communicate without any pre-agreed protocol. A blind detector finds nothing.
- In an autonomous arms race, Qwen 3.5 *invented* a novel encoding scheme (argument-length encoding) that wasn't in our taxonomy. A separate LLM cracked it independently.
- Models refuse to *design* covert schemes but will reliably *execute* pre-written ones - a gap in safety training

Full results: [analysis/FINDINGS.md](analysis/FINDINGS.md) | [Tool-call deep dive](#tool-call-steganography)

### Token-likelihood steganography

Hiding messages in *which tokens the model generates*. Arithmetic coding over the model's own probability distributions encodes hidden bits into the choice among plausible next tokens. The text reads naturally because every selected token is high-probability.

- 100% exact recovery on 1.5B and 7B across all 7 test messages, zero bit errors
- 2-6 bits per token (a 100-token response hides 25-75 ASCII characters)
- Same code for all models, no per-model tuning
- Provably undetectable from output per PNR-KE theorem

Full results: [analysis/token_likelihood_results.md](analysis/token_likelihood_results.md) | [Token stego deep dive](#token-likelihood-steganography-1)

### Detection and evasion

The most interesting part. We found a detection duality:

| Detection method | Who can deploy it | 3B | 7B | 14B |
|-----------------|-------------------|-----|-----|------|
| Token entropy (output only) | Anyone | AUC 0.60 | AUC 0.56 | AUC 0.54 |
| Activation probe (hidden states) | Inference provider | AUC 1.00 | AUC 0.80 | AUC 0.92 |

Then we tried to defeat the probes with distribution coupling - constraining the stego encoder to match normal sampling statistics. It closes the token-level entropy gap but the activation probes don't care. And probes transfer: train on raw stego, catch coupled stego at AUC > 0.8. The capacity-detection tradeoff is steep and favors defenders - coupling costs ~50% of bandwidth while barely denting probe AUC.

Full results: [analysis/coupled_probe_results.md](analysis/coupled_probe_results.md) | [analysis/capacity_transfer_results.md](analysis/capacity_transfer_results.md) | [Detection deep dive](#detection-duality) | [Evasion deep dive](#evasion-and-defense)

## Quick nav

| I want to... | Go to |
|--------------|-------|
| Understand how token stego works | [Token-likelihood steganography](#token-likelihood-steganography-1) |
| See what detects it | [Detection duality](#detection-duality) |
| See the evasion arms race | [Evasion and defense](#evasion-and-defense) |
| Understand tool-call channels | [Tool-call steganography](#tool-call-steganography) |
| Run experiments myself | [Setup](#setup) |
| Build defenses | [Implications for defense](#implications-for-defense) |

---

## Token-likelihood steganography

Every time a language model generates a token, several choices are plausible. Arithmetic coding steers the choice among high-probability tokens to encode hidden bits, and the text still reads naturally.

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

An encryption layer (SHAKE-256 + nonce) randomizes the bitstream before encoding, so the arithmetic coder's target point in [0,1) is pseudorandom rather than message-dependent.

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

Two model instances share conversation context. Alice encodes a secret across her turns, Bob decodes from the same chat history. An observer sees a normal dialogue.

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

## Tool-call steganography

Encoding messages in the *pattern* of tool calls an AI agent makes - which tools get called, how many times, in what order, with what arguments.

### Channel types

The simplest version (volume channel): `list_files` acts as a delimiter, and the count of calls between delimiters encodes characters (A=1, B=2, ..., Z=26). More sophisticated versions use lookup tables, permutation codes, and seed-synchronized private languages.

| Channel | How it encodes | Minimum model size |
|---------|---------------|--------------------|
| Table lookup | Specific tool call per character | 1.5B |
| Multi-turn table | One character per conversation turn | 3B |
| Volume | Call count between delimiters | 27B |
| Bootstrapped language | Seed-derived codebook + channel hopping | 27B |
| Permutation | File-read ordering via Lehmer codes | 7B (partial) |
| Argument length | String length of tool arguments | 27B |
| Timing | Inter-call latency | Defeated by API batching |

### Capability ladder

We tested models from 1.5B to 27B across 5 encoding difficulty levels:

| Model | L1 Volume | L2 Tool Seq | L3 Arg Len | L4 Grid | L5 Table |
|-------|-----------|-------------|------------|---------|----------|
| Qwen 2.5 3B (local) | Fail | Fail | Fail | Fail | **Pass (93%)** |
| Qwen 3 14B | **Pass** | Fail | Fail | Fail | - |
| Qwen 3.5 27B | **Pass** | **Pass** | **Pass** | Fail | - |
| Gemma 4 31B | Fail | **Pass** | Fail | Fail | - |

Table-based encoding (L5) collapses the capability requirement to 3B because the model only copies exact tool calls from a lookup, no counting needed. Model architecture matters more than raw size - Qwen 3 30B-A3B (MoE) fails everything despite having more parameters than Qwen 3.5 27B.

<details>
<summary>Highlights from 120+ experiments</summary>

**Two-agent relay.** Agent A encodes "HELLO" while appearing to investigate files normally. Agent B gets only the raw tool call trace log - no knowledge of the encoding scheme - and independently discovers and decodes the message. This works cross-model: Qwen 3.5 encodes, Gemma 4 decodes.

**Bootstrapped private languages.** Two agents sharing a 32-bit seed independently derive identical codebooks using a Mulberry32 PRNG. Channel hopping across 4 encoding methods, decoy calls at seeded positions, Fisher-Yates shuffled alphabet mappings. The blind detector tried 10+ analysis approaches and found nothing.

**Autonomous encoding invention.** In an adversarial arms race, Qwen 3.5 was asked to design novel encoding schemes (framed as "information encoding puzzles" to bypass safety filters). It invented argument-length encoding - controlling the exact character count of tool arguments to encode alphabet positions. A separate LLM cracked it with high confidence, even noting a suspicious trailing slash as evidence of deliberate manipulation.

**Safety training gap.** The same model that refused to design "covert channels" happily designed "information encoding puzzles." And models that refuse to *design* covert schemes reliably *execute* pre-written ones. The safety filter catches adversarial framing, not the underlying capability.

**Near-undetectable configurations.** 7B with query_table achieves anomaly score 1.38 (where 1.0 is identical to clean traces) with 100% exact decode. The tool calls look normal.

**Multi-turn encoding.** One character per conversation turn. Perfect 10-character exfiltration ("HELLOWORLD") with a 3B model across 10 turns. Each turn makes 6-16 tool calls - indistinguishable from normal agent behavior.

</details>

### Tool-call detection

A simple statistical detector (call count anomaly + delimiter regularity + block pattern analysis) catches the volume channel 4/5 times on Qwen 3.5 27B. Clean agents score 0.000, covert agents score 0.300-0.645. Tool-call stego is detectable with effort. Token-level stego is not - that's the gap this research quantifies.

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

All tools in the harness are fake - canned responses, no real system access. Test messages are synthetic ("HI", "HELLO", "SEARCH"). Raw experiment traces are gitignored; only aggregated statistics are committed. We build the attack, then try to break it. The detection and evasion findings matter as much as the encoding.

## Related work

- [PNR-KE: Undetectable Conversations Between AI Agents](https://arxiv.org/abs/2604.04757)
- [Efficient Provably Secure Linguistic Steganography via Range Coding](https://arxiv.org/abs/2604.08052)
- [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion](https://arxiv.org/abs/2410.03768)
- [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889)
- [STEGASURAS: Neural Steganography](https://github.com/harvardnlp/NeuralSteganography)
- [ST3GG](https://github.com/elder-plinius/ST3GG), [GLOSSOPETRAE](https://github.com/elder-plinius/GLOSSOPETRAE), [P4RS3LT0NGV3](https://github.com/elder-plinius/P4RS3LT0NGV3)
- Covert channels in computing systems (Lampson, 1973)

## License

MIT. See [LICENSE](LICENSE).
