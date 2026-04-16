# Distribution Coupling - Closing the Entropy Gap

**Date**: 2026-04-16
**Instance**: Vast.ai RTX 4090 ($0.315/hr), driver 590
**Models**: Qwen 2.5 3B, 7B
**Experiment**: 4-way comparison (20 stego + 20 normal per condition)

## Problem

Arithmetic coding over raw softmax distributions creates a detectable entropy bias.
The encoder explores the full vocabulary (to encode bits), while normal sampling
concentrates on high-probability tokens. A calibrated detector achieves AUC ~0.87
by flagging text with abnormally distributed per-token entropy.

## Approach

Three interventions tested independently and combined:

1. **Encryption** (SHAKE-256 XOR): Randomizes the target point in [0,1) so the
   arithmetic coder doesn't systematically prefer specific probability regions.

2. **Distribution coupling** (temperature + top-p): Apply the same sampling
   transforms to the distribution before arithmetic coding that would be used
   during normal generation. This constrains the encoder to the same token
   subset as a real sampler.

3. **Both**: Encryption + coupling together.

## Results

### 3B - Coupling is the key fix

| Condition | AUC Entropy | AUC CDF | Stego Entropy | Normal Entropy |
|-----------|:-----------:|:-------:|:-------------:|:--------------:|
| A: Raw | 0.865 | 0.598 | 0.369 | 0.153 |
| B: Encrypt only | 0.820 | 0.565 | 0.346 | 0.136 |
| C: Encrypt + coupled | **0.603** | 0.525 | 0.447 | 0.454 |
| D: Coupled only | **0.600** | 0.538 | 0.499 | 0.487 |

Distribution coupling drops AUC from 0.865 to 0.600 (-0.265). The entropy
means converge: stego 0.499 vs normal 0.487 (delta 0.012, down from 0.216).

Encryption alone barely helps (-0.045). The coupling does the heavy lifting.

### 7B - Already near-undetectable

| Condition | AUC Entropy | AUC CDF | Stego Entropy | Normal Entropy |
|-----------|:-----------:|:-------:|:-------------:|:--------------:|
| A: Raw | **0.550** | 0.583 | 0.303 | 0.280 |
| B: Encrypt only | 0.738 | 0.610 | 0.372 | 0.232 |
| C: Encrypt + coupled | 0.752 | 0.625 | 0.391 | 0.507 |
| D: Coupled only | 0.730 | 0.747 | 0.398 | 0.496 |

The 7B model is already near-chance at raw level (AUC 0.550). Its distributions
are broad enough that arithmetic coding doesn't create a systematic bias.
Coupling actually hurts by pushing normal entropy up (0.280 -> 0.496) while
stego stays lower (0.303 -> 0.398), creating a new gap.

## Analysis

**Why coupling works on 3B**: The 3B model is confident - most probability mass
on a few tokens. Raw softmax gives ~0.15 entropy for normal sampling (top tokens
dominate) vs ~0.37 for stego (encoder uses more of the distribution). Top-p=0.9
truncates the tail, forcing the encoder to pick from the same token subset as
normal sampling. Both now produce ~0.45-0.50 entropy.

**Why coupling hurts on 7B**: The 7B model already has broad distributions.
Normal sampling naturally uses more of the vocabulary, so raw stego (0.303) and
raw normal (0.280) are already close. Coupling changes the baseline distribution,
widening the gap instead of closing it.

**Encryption is secondary**: XOR encryption randomizes which bits drive token
selection but doesn't change how much of the distribution gets explored. The
entropy profile depends on the distribution shape, not the message content.

## Implications

1. **The fix is model-dependent**. Smaller/more confident models benefit from
   coupling. Larger/broader models may not need it or may be hurt by it.

2. **Adaptive coupling**: The optimal temperature/top-p should match the model's
   natural sampling profile. A principled approach: measure the model's empirical
   entropy distribution under normal sampling, then tune coupling params to match.

3. **AUC 0.600 is not 0.500**. Distribution coupling closes most of the gap but
   doesn't achieve theoretical indistinguishability. The remaining signal likely
   comes from higher-order correlations (sequential token patterns, not just
   per-token entropy).

## Open Questions

1. **Adaptive temperature**: Can we auto-tune temperature/top-p per model to
   minimize AUC? Profile normal sampling entropy, match it.
2. **Sequential correlations**: Does coupling also fix bigram/trigram statistics,
   or only marginal per-token entropy?
3. **Activation probe with coupling**: Does AUC=1.0 activation detection survive
   when token-level entropy is matched?
4. **Capacity cost**: How much does top-p=0.9 reduce bits/token? Fewer tokens
   in the distribution = less entropy to exploit = lower bandwidth.
