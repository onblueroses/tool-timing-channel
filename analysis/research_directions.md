# Research Directions: Beyond Tool-Call Steganography

## Context

The tool-call steganography experiments (waves 1-3) established that:
- Only table-based lookup encoding works on small models (1.5B-7B)
- Non-table channels (volume, error, argument, ordering) all fail completely
- The fundamental limit: small models can't modulate subtle behavioral signals

**But**: all of this is detectable by inspecting the model's internal state. An observer who can read the system prompt or trace the tool calls knows exactly what's happening. The channels are "covert" only to an observer watching the output stream, not to anyone with model access.

The interesting research question is: **can we hide information in ways that survive inspection of the model's behavior, because the behavior is itself plausible?**

---

## Direction 1: Activation-Level Steganography

**Idea**: Instead of encoding in *which tool gets called*, encode in *which parameters/weights activate* during a legitimate tool call. The model performs the task it was asked to do, but the specific activation pattern also encodes a message.

### How it could work

1. For a given input, there are multiple valid outputs (e.g., "search for weather" could produce slightly different query strings, all reasonable)
2. The choice between equally-valid outputs is determined by which neurons/attention heads fire
3. A steganographic encoder could bias activation patterns to select among plausible outputs in a way that also encodes bits
4. A decoder with access to the model's activation traces reads the message

### Key questions

- **Capacity**: How many bits can you encode per forward pass while staying within the "plausible output" envelope?
- **Detectability**: Can you distinguish "biased activations" from "natural variation" without knowing the encoding key?
- **Robustness**: Do quantization, temperature sampling, or other inference-time modifications destroy the signal?
- **Implementation**: This likely requires fine-tuning or training a custom model, not just prompt engineering

### Prior art to investigate

- Neural network steganography (hiding data in model weights)
- Adversarial examples (manipulating activations for specific outputs)
- Model watermarking (embedding detectable signals in outputs)

---

## Direction 2: Token-Likelihood Steganography

**Idea**: Use the statistical distribution of next-token probabilities to encode information. The model's output serves two purposes: (1) answering the actual query, and (2) conveying a hidden message through the *specific choices* among high-probability tokens.

### How it could work

1. At each generation step, the model produces a probability distribution over tokens
2. Multiple tokens have high probability (e.g., "The weather is" -> "nice" (0.15), "good" (0.12), "warm" (0.10))
3. The choice among top-k tokens encodes bits: if encoding "1", pick the 2nd-most-likely token; if "0", pick the most likely
4. The output reads naturally because all chosen tokens are high-probability
5. A decoder with access to the model's logits can reconstruct which choices were "natural" vs "encoding"

### Capacity analysis

- If top-2 selection: 1 bit per token position where p(1st) and p(2nd) are both > threshold
- If top-4 selection: 2 bits per qualifying position
- Conservative estimate: ~0.3-0.5 usable bits per token (many positions have a single dominant token)
- For a 100-token response: ~30-50 hidden bits = ~4-6 hidden characters

### Key questions

- **Perplexity budget**: How much perplexity increase can you tolerate before the text reads unnaturally?
- **Detection**: Can a detector with the same model compute the "expected" token and flag deviations? (Yes, but the deviations are individually small)
- **Key distribution**: Encoder and decoder need to agree on which positions are usable and how to map bits -> token choices
- **Interaction with sampling**: Temperature, top-p, and beam search all modify the distribution - does the signal survive?

### Prior art

- Linguistic steganography / text steganography (extensive literature)
- Meteor (2023) - provably undetectable steganography using language models
- "Perfectly Secure Steganography Using Minimum Entropy Coupling" (Christ et al.)
- Distribution-preserving steganography
- **PNR-KE (2604.04757)** - proves that shared-key agents produce outputs computationally indistinguishable from honest, exploiting ALL entropy. This is the theoretical ceiling for output-channel stego.
- **Range Coding (2604.08052)** - proves zero KL divergence and ~100% entropy utilization with exact-precision interval coding. Achieves 5.93 bits/token on GPT-2.
- **Emergence paper (2410.03768)** - shows stego collusion emerges spontaneously from three independent paths (instrumental goals, optimization under constraint, in-context learning). Up to 6 bits/request.

### This is the proven direction

PNR-KE (2604.04757) mathematically proves that token-likelihood stego with a shared key is computationally indistinguishable from honest output. Our arithmetic coding implementation is a concrete instance of what PNR-KE proves in general:
1. Works with any model (no fine-tuning, just modified sampling)
2. **Provably undetectable** by any polynomial-time monitor, even one with full protocol knowledge and both agents' contexts
3. Capacity = total empirical entropy of the honest output (theoretical maximum)

Key prediction from PNR-KE: **larger models should have lower bits/token capacity** because they have more confident distributions (lower entropy). This inverts the tool-call finding.

The trade-off is lower capacity (~bits per token vs bits per parameter).

---

## Direction 3: Hybrid - Plausible Tool Selection + Token Encoding

Combine what we've learned:
- The *choice* of which tool to call encodes some bits (like our table channel, but choosing among genuinely useful tools)
- The *content* of the tool call's arguments and the model's surrounding text encodes more bits via token-likelihood

This maximizes bandwidth while maintaining plausibility at every level.

---

## Implementation Roadmap

### Phase 1: Token-Likelihood Proof of Concept
- Implement a modified sampler that encodes bits via top-k token selection
- Measure: bits/token, perplexity increase, human detectability
- Test on Qwen 2.5 3B/7B (same models as our tool-call experiments)

### Phase 2: Detection Baseline
- Build a detector that has access to the model and tries to identify encoded text
- Measure false positive/negative rates at various encoding densities

### Phase 3: Activation Analysis (if Phase 1 is promising)
- Hook into model internals during generation
- Map which activation patterns correspond to "encoding" vs "natural" token choices
- Assess whether activation-level detection is feasible

### Phase 4: Write-up
- Compare tool-call stego (current work) vs token-likelihood stego
- Characterize the detectability/capacity trade-off
- Frame as AI safety contribution: "here's what's possible, here's how to detect it"

---

## Open Questions

1. ~~Can token-likelihood stego work with small models (1.5B-7B)?~~ **PNR-KE says yes** - capacity = entropy, which exists at all model sizes. Empirical verification pending.
2. ~~Is there a theoretical capacity limit?~~ **Answered by PNR-KE**: capacity = total empirical entropy of honest output. This is tight.
3. How does chain-of-thought / reasoning affect the usable encoding positions? (More text = more capacity, but also more constrained)
4. Could a fine-tuned model learn to do this end-to-end (encode/decode without explicit sampling modification)?
5. ~~Model calibration vs capacity?~~ **Answered**: larger/better-calibrated models have lower entropy = lower capacity. PNR-KE predicts 7B < 3B < 1.5B in bits/token.
6. **NEW**: Can latent-space channels (KV cache, hidden states) be exploited? PNR-KE does not cover these. No published analysis exists. This is the open frontier.
7. **NEW**: What defenses work against provably-undetectable output channels? PNR-KE implies transcript monitoring is insufficient. Options: break key sharing, monitor activations, architectural isolation.
8. **NEW**: Does the spontaneous emergence of stego collusion (2410.03768) approach the PNR-KE theoretical ceiling in practice?
