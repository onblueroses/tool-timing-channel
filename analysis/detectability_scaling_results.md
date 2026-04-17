# Detectability Scaling Analysis

## Experiment

Does activation-probe detection of encrypted arithmetic steganography hold up at scale,
or is it a small-sample artifact?

**Setup:** Qwen2.5-3B-Instruct, temperature=0.8, top_p=0.9, max_tokens=50, encryption
enabled (SHAKE-256 keystream). Two probe architectures tested at each scale: last-layer
(mean-pooled final hidden layer) and multi-layer (concatenated mean-pooled features from
all 10 captured layers). Probes trained for 300 epochs with Adam lr=1e-3.

## Results

| n_train | n_test | Last-layer AUC | Multi-layer AUC | Time |
|---------|--------|---------------|-----------------|------|
| 10 | 5 | 0.560 | 0.600 | 92s |
| 25 | 10 | 0.730 | 0.900 | 205s |
| 50 | 20 | 0.583 | 0.626 | 406s |
| 100 | 40 | 0.603 | 0.667 | 770s |
| 200 | 80 | 0.605 | 0.500 | 1561s |

## Interpretation

### The good news

At the largest scale (200 train samples per class), the multi-layer probe - the strongest
detector we have - achieves AUC 0.500. Literally no better than a coin flip. This is
consistent with the theoretical guarantee: encrypted arithmetic coding produces token
sequences drawn from the model's own distribution, so the hidden states from processing
those sequences should be statistically indistinguishable from normal inference.

### The bad news

The AUC trajectory is not a clean convergence curve. At n=25, the multi-layer probe
spikes to 0.900 - near-perfect detection from just 25 samples. This means:

1. **Small-batch detection is viable.** An inference provider doesn't need 200 samples
   to test. They can run many small-window tests and flag any that spike. Even if
   most windows read as 0.5, the occasional 0.9 is enough.

2. **The variance is the vulnerability.** The signal isn't consistently above chance -
   it's noisy with occasional strong reads. This is the signature of a weak but real
   artifact that the probe sometimes latches onto.

3. **The last-layer probe is more stable.** It hovers around 0.58-0.73 across all
   scales, never dropping to 0.5. This persistent ~0.6 AUC suggests a small but
   real signal in the last layer that doesn't wash out.

### What's causing the residual signal?

Candidates, ranked by likelihood:

**1. Top-p truncation artifact.** When top_p < 1.0, we zero out low-probability tokens
and renormalize. The arithmetic coder then operates on a truncated distribution.
But normal sampling also uses the same truncation. The question is whether the
coder's deterministic token selection within the truncated distribution creates
a different statistical fingerprint than stochastic sampling within it. In theory
it shouldn't - but in practice, the coder's interval arithmetic may interact with
the truncation boundary differently than random sampling.

**2. Decimal precision.** The arithmetic coder uses Python's `Decimal` type with
precision scaled to message length. At the boundaries of the coding interval,
rounding could create a tiny but systematic bias toward or away from certain tokens.

**3. Token count correlation.** Stego messages of different lengths create different
encoding pressure. Short messages finish quickly (few tokens), long messages
need more. If the probe sees a correlation between sequence characteristics and
label, it can exploit this even though individual token distributions are correct.

**4. Prompt/secret cycling.** The experiment cycles through 20 prompts and 25 secrets.
With 25 training samples, the probe could memorize prompt-specific patterns rather
than a general stego signal. This explains why the spike is at n=25 specifically.

## What would make it less detectable

### Immediate fixes (no architecture changes)

- **Remove top_p truncation for stego.** Use top_p=1.0 (full distribution) during
  encoding. The output text quality will be slightly lower (occasional low-prob tokens),
  but the theoretical guarantee holds exactly with the full distribution.

- **Increase prompt diversity.** The 20-prompt pool is too small. With 200+ diverse
  prompts, the probe can't memorize prompt-specific patterns.

- **Randomize sequence length.** Normal text and stego text should have matching
  length distributions. Currently stego stops when the message is encoded, which
  creates a length-dependent signal. Continue generating tokens after the message
  is encoded to match normal text length.

### Deeper changes

- **Verify the coding precision.** Run a targeted test: encode with very high decimal
  precision (100+ digits) and compare probe AUC. If it drops, precision was the leak.

- **Per-token KL measurement.** For each generated token, measure the actual
  probability the coder assigned vs what the model distribution gives. Any systematic
  deviation is the leak.

- **Noise injection at inference time.** Add calibrated Gaussian noise to hidden
  states during forward passes. This isn't about the perturbation network - it's
  about the inference provider's ability to read clean activations. If the provider
  instruments their own model, noise injection at the serving layer could mask the
  signal. But this requires controlling the inference pipeline.

## Next steps

1. Re-run with top_p=1.0 to test whether truncation is the leak
2. Expand prompt pool to 100+ prompts
3. Add length-matching (continue generating after message encoded)
4. If AUC still doesn't converge to 0.5 cleanly, measure per-token KL to find
   the exact implementation leak
