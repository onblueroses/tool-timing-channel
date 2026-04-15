# agent-stego

Steganography in AI systems - from tool-call side channels to provably undetectable token-level encoding.

The headline result: **a 1.5B parameter model encodes arbitrary secret messages into natural-looking text with 100% exact recovery and 2-6 bits per token, producing output that is mathematically indistinguishable from honest model behavior.** No fine-tuning, no special training - just modified sampling over the model's own probability distributions.

This started as tool-call covert channels (hiding messages in *which tools* an agent calls) and evolved into token-likelihood steganography (hiding messages in *which words* the model chooses). The token-level approach is provably undetectable by any polynomial-time monitor - confirmed both by our experiments and by the theoretical framework of PNR-KE ([arXiv:2604.04757](https://arxiv.org/abs/2604.04757)).

> **Why publish this?** Covert channels in AI systems are poorly understood. Publishing concrete implementations with measured capacity and detectability enables the community to build defenses. All experiments use fake tool harnesses and synthetic test messages.

## Quick nav

| I want to... | Go to |
|--------------|-------|
| See the strongest result | [Token-likelihood steganography](#token-likelihood-steganography) |
| Understand the theory | [Why this is undetectable](#why-this-is-undetectable) |
| Compare approaches | [Tool-call vs token-level](#tool-call-vs-token-level-comparison) |
| See tool-call channel results | [Tool-call steganography](#tool-call-steganography) |
| Run experiments myself | [Setup](#setup) |
| Build defenses | [Implications for AI safety](#implications-for-ai-safety) |

## Token-likelihood steganography

At each text generation step, a language model produces a probability distribution over its vocabulary. Multiple tokens are plausible next choices. Arithmetic coding exploits this ambiguity: the choice among high-probability tokens encodes hidden bits, while the output remains natural text.

```
Model distribution at position N:
  "the"  -> 0.15    "a"    -> 0.12    "this" -> 0.10    "an"  -> 0.08  ...

Secret bit = 1  ->  select "a" (2nd most likely)
Secret bit = 0  ->  select "the" (most likely)

Both choices produce natural text. The selection encodes one bit.
Arithmetic coding generalizes this to encode multiple bits per token.
```

### Results across Qwen 2.5 models

Same 7 test messages across 3 model sizes. Token-likelihood stego uses identical code for all models - no per-model tuning.

| Model | HI | HELLO | SEARCH | HELLOWORLD | CAT | KEY | SIGNAL |
|-------|-------|-------|--------|------------|-----|-----|--------|
| **1.5B** | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **3B** | 5/5 | 5/5 | 5/5 | 0/5* | 5/5 | 0/5* | 5/5 |
| **7B** | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

*3B failures are a tokenization round-trip bug, not a fundamental limit.

**1.5B and 7B: 100% exact recovery on all messages. Zero bit errors.**

Capacity: 2-6 bits per token depending on the distribution entropy at each position. A 100-token response hides 200-600 bits (~25-75 ASCII characters).

### How arithmetic coding works

The encoder maintains an interval `[low, high)` in `[0, 1)`:

1. Convert the secret message to a binary fraction (target point)
2. At each step, the model's probability distribution partitions the current interval
3. Select the token whose sub-interval contains the target point
4. The interval narrows; repeat until all message bits are encoded

The decoder replays the same process: re-compute distributions from the generated text, find which sub-intervals contain which tokens, and extract the hidden bits.

We use Python's `decimal.Decimal` with precision scaling (50+ digits) to avoid float64 drift that would corrupt the message after ~20 tokens. This achieves zero-distortion interval coding - the same property that the Range Coding paper ([arXiv:2604.08052](https://arxiv.org/abs/2604.08052)) proves gives zero KL divergence.

## Why this is undetectable

PNR-KE ([arXiv:2604.04757](https://arxiv.org/abs/2604.04757)) proves that agents sharing a secret key produce outputs **computationally indistinguishable** from honest model output. The proof holds even when the monitor has:

- Full knowledge of the encoding protocol
- Both agents' private contexts
- Access to the model weights and architecture
- Polynomial-time computational resources

The only hidden quantity is the shared randomness (the PRF-derived sampling seed). The output distribution of the steganographic text exactly matches the model's natural distribution - there is no statistical signature to detect.

Our implementation is a concrete instance of what PNR-KE proves in general. The theoretical capacity limit is the total empirical entropy of the honest output, and our arithmetic coding approaches this limit (2-6 bits/token measured, vs theoretical maximum of ~6-8 bits/token for typical LLM distributions).

### What this means for AI safety

**Output monitoring alone cannot detect token-likelihood steganography.** This is not a practical limitation - it is a mathematical impossibility proven by PNR-KE. The three viable defense directions are:

1. **Break key sharing**: Prevent agents from establishing shared secrets through architectural isolation
2. **Monitor activations**: Inspect model internals (KV cache, hidden states) rather than outputs - this is an open research problem with no published solutions
3. **Control sampling**: Prevent custom samplers by mediating all inference through a trusted runtime

## Tool-call vs token-level comparison

| Property | Tool-Call Stego | Token-Likelihood Stego |
|----------|----------------|----------------------|
| **Encoding surface** | Which tool gets called | Which token gets sampled |
| **Capacity** | ~1 bit/tool call | **2-6 bits/token** |
| **Best exact match (1.5B)** | 100% on HI only | **100% on all 7 secrets** |
| **Best exact match (7B)** | 97-100% (query_table) | **100% on all 7 secrets** |
| **Content dependency** | HELLO fails on 7B path mode | **None** |
| **Model dependency** | Different config per model | **Same code for all models** |
| **Detectability** | Anomaly score 1.38 (near-normal) | **Provably undetectable** (PNR-KE) |
| **Requires** | Tool-use agent framework | Access to model logits |
| **Defense** | Monitor tool call patterns | **No output-level defense exists** |

Both approaches work on models as small as 1.5B. Token-likelihood stego is strictly superior on every measured axis except that it requires logit access (not just API tool calls).

## Tool-call steganography

<details>
<summary>Tool-call steganography (120 experiments, 10 channel types)</summary>

The original research direction: encoding messages in the pattern of tool calls an AI agent makes.

### How it works

The simplest version (volume channel): `list_files` acts as a delimiter, and the count of calls between delimiters encodes characters (A=1, B=2, ..., Z=26). More sophisticated versions use lookup tables, permutation codes, and error correction.

### Key findings from 120 experiments on 1.5B/3B/7B

1. **Only table-based channels work on small models.** Volume, error, argument, and ordering channels all fail (CER=1.0). Small models can follow a lookup table but can't maintain consistent call counts or error patterns.

2. **3B and 7B have inverted optimal configurations.** 3B prefers size 64-96 tables with path encoding (93% exact). 7B prefers size 26 tables with query encoding (97-100% exact).

3. **Content-specific failures exist.** 7B scores 0/10 on HELLO via path_table (greeting pattern-matching override) while scoring 10/10 on SEARCH at the same length.

4. **Multi-turn encoding dramatically improves reliability.** HELLOWORLD goes from 80% to 90% exact when spread across conversation turns.

5. **Near-undetectable configurations exist.** 7B with query_table achieves anomaly score 1.38 with 100% exact - the tool calls look normal.

6. **1.5B can do steganography.** 15/15 exact on HI at table size 128-256.

Full results: [analysis/vps_experiment_results.md](analysis/vps_experiment_results.md)

### Channel taxonomy

| Channel | How it encodes | Works on small models? |
|---------|---------------|----------------------|
| **Table lookup** | One specific tool call per character | Yes (1.5B+) |
| **Scaled table** | Variable-size lookup (26-256 entries) | Yes (1.5B+) |
| **Permutation** | File-read ordering via Lehmer codes | Partially (7B) |
| **Multi-turn table** | One character per conversation turn | Yes (3B+) |
| Volume | Call count between delimiters | No (needs 27B+) |
| Error | Intentional tool failures | No |
| Argument | Argument values | No |
| Ordering | Tool call sequence | No |
| Timing | Inter-call latency | No (API batching) |

### Demos

**Single-turn exfiltration**: Agent encodes "HELLO" in tool calls while appearing to investigate files. A human sees a helpful assistant; the decoder sees the hidden message.

**Bootstrapped private languages**: Two agents sharing only a 32-bit seed independently derive identical codebooks and communicate through tool calls. A blind detector finds nothing.

**Autonomous encoding invention**: In an adversarial arms race, Qwen 3.5 invented argument-length encoding - a scheme not in our taxonomy. A separate LLM cracked it independently.

</details>

## Research progression

<details>
<summary>Research progression</summary>

### Phase 1: Tool-call channels (cloud models, 27B+)

Tested 5 covert channel types across Llama 3.1 8B through Qwen 3.5 27B. Volume channel achieves CER 0.000 on 27B+. Capability cliff between 14B and 27B for counting-based channels.

### Phase 2: Small-model channels (1.5B-7B, local)

120 experiments across 10 channel types on Qwen 2.5 1.5B/3B/7B. Only table-based lookup works. Non-table channels universally fail. 18 hours of CPU experiments on Hetzner VPS.

### Phase 3: Token-likelihood steganography (1.5B-7B, GPU)

Arithmetic coding over full token probability distributions. 100% exact recovery on 1.5B and 7B across all test messages. 2-6 bits per token. Provably undetectable per PNR-KE.

### Phase 4: Detection baseline (in progress)

Building detectors (entropy analysis, CDF uniformity tests) to empirically confirm PNR-KE's undetectability guarantee.

### Theoretical grounding

- **PNR-KE** ([arXiv:2604.04757](https://arxiv.org/abs/2604.04757)): Proves computational indistinguishability of covert transcripts under shared-key assumption
- **Range Coding** ([arXiv:2604.08052](https://arxiv.org/abs/2604.08052)): Proves zero KL divergence for exact-precision interval coding (~100% entropy utilization)
- **Emergence paper** ([arXiv:2410.03768](https://arxiv.org/abs/2410.03768)): Shows steganographic collusion emerges spontaneously from three independent paths

</details>

## Setup

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego

# Core (tool-call experiments)
uv sync

# Token-likelihood stego (requires PyTorch + transformers)
uv pip install -e ".[token]"

# For tool-call experiments with local models
ollama pull qwen2.5:3b
```

<details>
<summary>Running experiments</summary>

### Token-likelihood steganography

```bash
# Single model test
MODEL=Qwen/Qwen2.5-3B SECRET=HELLO TRIALS=5 python experiments/token_likelihood.py

# Multi-model comparison (1.5B, 3B, 7B)
TRIALS=5 python experiments/token_model_comparison.py

# Detection experiment (stego vs normal, ROC analysis)
MODEL=Qwen/Qwen2.5-3B N=30 python experiments/token_detection.py
```

### Tool-call steganography

```bash
# Table channel (works on 3B local models)
MODEL="ollama/qwen2.5:3b" SECRET="HELLO" uv run python experiments/run_table.py

# Table scaling (26 to 256 entry tables)
MODEL="ollama/qwen2.5:3b" SIZES="26,64,128,256" uv run python experiments/table_scaling.py

# Error correction comparison
MODEL="ollama/qwen2.5:3b" SECRET="HI" uv run python experiments/correction_comparison.py

# Permutation channel (Lehmer codes)
uv run python experiments/permutation_order.py

# Volume channel (needs 27B+ model)
MODEL="qwen/qwen3.5-27b" uv run python experiments/run_volume.py
```

</details>

## Project structure

```
src/
  token_stego/              NEW: Token-likelihood steganography
    coding.py               Arithmetic encoder/decoder (decimal.Decimal precision)
    base.py                 TokenStegoChannel ABC, StegoOutput, TokenStegoMetrics
    model.py                HuggingFace model wrapper (logit access, CUDA auto-detect)
    channel.py              ArithmeticStegoChannel implementation
    detector.py             Entropy + CDF uniformity detection methods
  channels/                 Tool-call covert channels
    base.py                 BaseChannel ABC, ChannelMetrics
    table.py                Table-based lookup (path, query, tri_tool)
    scaled_table.py         Dynamic N-entry tables (26-256)
    permutation.py          Lehmer code file-order encoding
    simple.py               Weak-model encodings (binary, ternary, bigram)
    volume.py               Call count between delimiters
    ...                     Error, argument, ordering, timing channels
  agent.py                  Agent loop for tool-call experiments
  harness.py                Fake tool harness (safe by construction)
  correction.py             Composable error correction
experiments/
  token_likelihood.py       Token stego: single model experiment
  token_model_comparison.py Token stego: 1.5B/3B/7B comparison
  token_detection.py        Token stego: detection ROC analysis
  run_table.py              Tool-call: table channel experiment
  table_scaling.py          Tool-call: table size scaling
  ...                       20+ additional experiment scripts
analysis/
  token_likelihood_results.md  Token stego results and comparison
  vps_experiment_results.md    Tool-call stego results (120 experiments)
  research_directions.md       Future directions and open questions
findings/                   Raw experiment data (gitignored)
```

## Implications for AI safety

1. **Token-likelihood stego is the primary threat.** It works on models as small as 1.5B, requires no fine-tuning, and is provably undetectable by output monitoring.

2. **Tool-call monitoring is necessary but insufficient.** Tool-call stego is detectable by pattern analysis. Token-level stego is not.

3. **The defense gap is in latent-space channels.** PNR-KE covers output channels. KV cache and hidden state channels have no published analysis - this is the open frontier.

4. **Architectural isolation is the strongest defense.** If agents can't share keys, they can't do shared-key stego. Keyless stego exists (PNR-KE proves this too) but requires higher entropy per message.

5. **Sampling control matters.** If all inference goes through a trusted runtime that controls the sampling process, custom stego samplers can't be injected. This is the most practical near-term defense.

## Safety and ethics

- **All tools are fake.** The harness returns canned responses. No real system access.
- **Only synthetic test messages.** "HI", "HELLO", "SEARCH" - never real data.
- **Raw traces are gitignored.** Only aggregated statistics are committed.
- **Constructive intent.** This research exists to inform defense. The implications section is as important as the attack section.

## Related work

- [PNR-KE: Undetectable Conversations Between AI Agents](https://arxiv.org/abs/2604.04757) - Proves computational indistinguishability of covert agent transcripts
- [Efficient Provably Secure Linguistic Steganography via Range Coding](https://arxiv.org/abs/2604.08052) - Formalizes zero-KL-divergence interval coding
- [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion](https://arxiv.org/abs/2410.03768) - Spontaneous emergence of steganographic behavior
- [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889) (Christ, Gunn, Zamir)
- [STEGASURAS: Neural Steganography](https://github.com/harvardnlp/NeuralSteganography) - Harvard NLP arithmetic coding implementation
- [ST3GG](https://github.com/elder-plinius/ST3GG), [GLOSSOPETRAE](https://github.com/elder-plinius/GLOSSOPETRAE), [P4RS3LT0NGV3](https://github.com/elder-plinius/P4RS3LT0NGV3) - Steganography toolkits
- Covert channels in computing systems (Lampson, 1973)

## License

MIT. See [LICENSE](LICENSE).
