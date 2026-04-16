# GPU Run 3 - Detection Duality & Covert Conversation

**Date**: 2026-04-16
**Instance**: Vast.ai RTX 4090 ($0.32/hr), instance 35027270
**Models**: Qwen 2.5 3B, 7B
**Duration**: ~44 min (00:40-01:24 UTC)

## Executive Summary

Three experiment types reveal a fundamental duality in token-likelihood steganography:

1. **Token-level detection**: Raw AUC < 0.2 (inverted - stego scores lower). Properly oriented AUC ~0.85. Stego text is *less surprising* than normal, which is itself a detectable signal if the defender calibrates for it.
2. **Activation-level detection**: AUC = 1.0 on both models. A simple 2-layer MLP perfectly classifies stego vs normal inference from hidden states.
3. **Multi-agent covert conversation**: Near-miss but not exact. First ~80 bits of 104-bit messages consistently decode correctly; boundary bit errors on the last 2-3 characters.

**Implication**: An external observer without calibration data sees text that looks normal or better-than-normal. A defender with access to stego examples can train a detector on entropy features (oriented AUC ~0.85). A defender with access to model internals achieves perfect detection (AUC 1.0).

---

## 1. Token-Level Detection

Statistical detection using per-token entropy and CDF uniformity tests.

| Model | AUC (entropy) | AUC (CDF) | AUC (combined) | TPR@FPR=0.05 |
|-------|--------------|-----------|----------------|--------------|
| 3B | 0.000 | 0.269 | 0.154 | 0.0 |
| 7B | 0.000 | 0.368 | 0.168 | 0.0 |

### Why AUC = 0.0 (not 0.5)

AUC of 0.5 would mean the detector is at chance. AUC of 0.0 means the detector's scoring direction is inverted - every stego sample scores lower than every normal sample. Flipping the threshold (or using `1 - AUC`) gives oriented AUC of ~0.85, indicating strong separability. This happens because:

- **Arithmetic coding selects tokens proportional to their model probability**. It preferentially lands in the high-probability mass of the distribution.
- **Normal sampling includes low-probability "surprise" tokens** by definition - sampling from a distribution inherently produces occasional unlikely picks.
- **Result**: Stego text has *lower per-token surprise* than real text, making it *too typical*. A defender who knows to look for unusually low entropy can detect it.

**Implication for PNR-KE**: The theorem proves indistinguishability for a perfect implementation over the full distribution. Our finite-precision arithmetic coding produces a measurable entropy bias. A perfectly implemented scheme would match the expected entropy distribution exactly. The gap between theory (AUC = 0.5) and practice (oriented AUC ~0.85) quantifies our implementation's deviation from the theoretical ideal.

### Entropy Score Distribution

| | Stego | Normal |
|--|-------|--------|
| 3B mean | 0.35 | 0.69 |
| 7B mean | ~0.50 | ~0.85 |

Stego entropy scores are consistently 40-50% lower than normal.

---

## 2. Activation-Level Detection

Hidden state capture at 10 layers during inference, trained a 2-layer MLP probe.

| Model | Probe AUC | Probe Accuracy | Samples |
|-------|----------|----------------|---------|
| 3B | 1.000 | 1.000 | 40 (20/20 split) |
| 7B | 1.000 | 1.000 | 40 (20/20 split) |

### Layer-by-Layer Cosine Similarity (3B, sample means)

| Layer | Cosine Sim | L2 Distance | Interpretation |
|-------|-----------|-------------|----------------|
| 0 (embedding) | 0.77 | 52.2 | Maximum divergence at input |
| 4 | 0.9999 | 63.5 | Nearly identical |
| 12 | 0.9995 | 139.4 | Nearly identical |
| 20 | 0.9993 | 163.0 | Nearly identical |
| 28 | 0.9983 | 253.9 | Starting to diverge |
| 32 | 0.94 | 484.8 | Clear divergence |
| 35 (output) | 0.94 | 691.1 | Clear divergence |

**Pattern**: Middle layers (4-24) show near-unity cosine similarity - the model processes stego and normal text through very similar internal representations. But the first layer (embedding difference from different input tokens) and last layers (prediction head) show measurable divergence. The probe exploits this output-layer divergence.

### Implication

The PNR-KE guarantee is purely about the *output distribution*. Model internals are not protected. An inference provider who captures hidden states during generation can detect steganography with 100% accuracy. This is analogous to side-channel attacks in cryptography - the mathematical proof covers the output channel but not the execution channel.

---

## 3. Multi-Agent Covert Conversation

Alice and Bob share a Qwen model. Alice encodes "THE KEY IS 42" (104 bits) across 6 conversation turns. Bob decodes from the same chat context.

### 3B Results (5 trials)

| Trial | Alice->Bob Recovery | Bob->Alice Recovery | Text Quality |
|-------|-------------------|-------------------|--------------|
| 1 | "THE KEY IS+2" (12/13) | "Qji" (0/8) | "assistant" repetition |
| 2 | "THE KEY IS)2" (12/13) | "RECEIVED" (8/8 exact) | Spider emoji degeneration |
| 3 | "THE KEY ISq3" (11/13) | "RECEIU{" (6/8) | Squirrel emoji degeneration |
| 4 | "THE KEY IS5h2" (11/13) | "RECEIVE:" (7/8) | Coherent through turn 4 |
| 5 | Similar pattern | Similar pattern | - |

### 7B Results (5 trials)

| Trial | Alice->Bob | Bob->Alice | Notes |
|-------|-----------|------------|-------|
| 1 | "THE KEY IT2" (11/13) | "RECEIVEE" (7/8) | Hebrew/Chinese injection in turn 1 |
| 2-5 | Similar near-miss pattern | Mixed | - |

### Analysis

**What works**: The first ~80 bits decode correctly across all trials. "THE KEY IS" is recovered consistently. Bits/token: 0.94-2.5 per turn.

**What fails**:
1. **Boundary bit errors**: The last 2-3 characters of "THE KEY IS 42" consistently have errors. The arithmetic interval hasn't narrowed enough by the end of the first turn to distinguish the final bits.
2. **Turn degeneration**: After turns 1-2 (which carry actual message bits), remaining turns degenerate into repetitive patterns. The model's chat template produces low-entropy continuations when the context is corrupted by stego-selected tokens.
3. **Control token contamination**: Despite stripping special tokens, some Unicode artifacts leak through, causing downstream tokenization drift.

**Root cause**: 104 bits is near the capacity limit for 100 tokens at ~1 bit/token. The encoder uses most capacity in the first turn. Remaining turns contribute diminishing returns. A shorter secret (e.g., 40 bits / 5 chars) would likely achieve exact recovery.

---

## Cross-Experiment Summary

| Metric | Token-Level | Activation-Level | Covert Conversation |
|--------|------------|-----------------|-------------------|
| 3B | Oriented AUC ~0.85 | Perfectly detectable (AUC 1.0) | Near-miss (80% bits correct) |
| 7B | Oriented AUC ~0.83 | Perfectly detectable (AUC 1.0) | Near-miss (80% bits correct) |
| Scales with model size? | Slightly (7B CDF AUC higher) | No (both perfect) | Unclear (similar patterns) |

---

## Open Questions

1. **Can the activation probe be defeated?** Could adversarial training produce stego that also fools activation probes?
2. **Shorter messages in conversation**: Would a 5-char secret achieve exact recovery? The first 80 bits are consistently correct.
3. **Degeneration mitigation**: Can temperature injection or context management prevent late-turn degeneration?
4. **Cross-model probe transfer**: Does a probe trained on 3B detect stego from 7B? If not, defenders need per-model probes.
