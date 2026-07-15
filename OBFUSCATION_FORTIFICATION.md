# Fortifying the obfuscation tier — red-teamed

## ⚠️ Adversarial stress-test update (real weights, adaptive attacks) — `obfuscation_stresstest.py`
Re-run on a **real trained weight** (MiniLM FFN, 1536×384) with a stronger attacker that adds **Sinkhorn
normalization** (alternating row/col L2), which *cancels any positive per-channel diagonal scaling*:

| defense | multiset | corr | **sinkhorn+ms** | **sinkhorn+corr** | verdict |
|---|---|---|---|---|---|
| P (permutation) | 100% | 100% | 100% | 100% | BROKEN |
| **P·D+** (positive scaling) | 3% | 3% | **100%** | **100%** | **BROKEN — Sinkhorn undoes it** |
| **P·D±** (SIGNED scaling) | 0.5% | 0.5% | 6% | 12% | leaks |
| P·D±+Q (+requantize) | 0.8% | 0.5% | 3% | 4% | residual |
| P·D±+Q+δ (+tiny delta) | 0.6% | 0.5% | 1.6% | 2.1% | residual |

**Findings that change the recipe:**
1. **Positive diagonal scaling is NOT safe** — Sinkhorn normalization recovers the permutation 100%. A naive
   red-team (multiset/correlation only) misses this and wrongly reports P·D as holding.
2. **Use SIGNED diagonal scaling** — the per-channel sign flips are an ambiguity Sinkhorn can't normalize away;
   they drop recovery from 100% to ~10%. Signed folding is exact for linear layers; take care through SiLU/RMSNorm.
3. **+ requantize + a small fine-tune delta** push recovery to low single digits, but a **measurable residual
   remains** (~2–4%). Behavior cost: scaling is exact/free; requantize costs ≈ the 4-bit budget you already pay.
4. **Therefore obfuscation is a cost-raising layer, not a guarantee.** The guarantee must come from the MPC
   **share** layer (uniform activations, information-theoretic) — empirically reconfirmed below.


## ⛔ The fundamental limit (why recovery can't reach 0 with obfuscation) — `obfuscation_fortify_v2.py`
The attacker HAS the public checkpoint, so any behavior-exact transform of it is a **solvable inverse**:
- permutation → value-matching (multiset/Sinkhorn),
- **rotation / scaling → least-squares** (a random orthogonal `W·Q` is inverted to **3.9e-15** — proven).

So no amount of clever *relabelling* reaches recovery 0. Only a genuine **weight change** does, and it isn't
cheap: a random delta needs ~**40% of ‖W‖** to kill the match (2–10% barely dents it) — essentially a different
model. A real **self-distillation** fine-tune is more efficient and restores behavior by construction, but its
recovery-vs-steps curve needs a training run to quantify.

**Conclusion:** obfuscation is cheap **defense-in-depth** that raises weight-fingerprinting cost; it is *not*
the path to recovery-0. The **guarantee** — provably recovery 0, information-theoretic, for any attacker
compute — is the **share layer**. Best deployment: **self-distilled weights** (so there's no public checkpoint
to match) **+ the share layer** for the guarantee.

## The break we're fixing
Static **permutation** obfuscation is broken. The deployed weights are the public checkpoint's *exact values,
just rearranged*, so an attacker matches each deployed row's **multiset** of values to a public row and
recovers the permutation in seconds. Measured earlier on real gemma4-31b: ~100% token + key recovery.
Reproduced here (`scripts/obfuscation_redteam.py`, N=256 rows, chance = 0.4%):

| defense | exact-multiset attack | correlation attack | verdict |
|---|---|---|---|
| **P** (permutation only) | **100.0%** | **100.0%** | BROKEN |
| **P·D** (permute + diagonal scaling) | 0.0% | 1.2% | holds (~chance) |
| **P·D + Q** (+ requantize, fresh scales) | 1.6% | 0.4% | holds (~chance) |

## The fix — make the *values* different, cheaply, offline (behavior unchanged)
Ranked by simplicity; the composition is close to free and entirely one-time/offline.

1. **SIGNED diagonal scaling `P·D±` (best lever).** Permute *and* multiply each channel by a random nonzero
   scalar **including sign flips**, with the inverse folded into the adjacent layer — the standard
   network-reparameterization symmetry (`y = (W·D⁻¹)(D·x)`), handled with care at the RMSNorm/SiLU boundaries.
   Every value changes → multiset matches nothing, and (unlike *positive* scaling) the sign flips survive a
   Sinkhorn attacker. **Zero runtime cost, exact.** (Positive-only scaling is broken by Sinkhorn — see the
   stress-test above.)
2. **Requantize after transforming.** Scale, then quantize to int4 with *fresh per-row scales*. Quantization
   noise on top of random scaling destroys exact-value matching and degrades the correlation residual (the
   attacker's library is quantized with *different* scales). Nearly free — you were quantizing anyway.
3. **Tiny random delta (sledgehammer).** Bake in a low-rank delta, or fine-tune a few hundred steps on generic
   data, *before* obfuscating. Now the deployed weights are **not any public checkpoint** — multiset matching
   has nothing to match against. Cost: a few GPU-hours, once. (Model-soup merging is a fancier version; a
   LoRA-style delta is simpler and gives the same property.)
4. **Keep the first and last layers off the mesh.** The client / trusted coordinator holds the embeddings and
   the output head. Even a node that fully inverts its middle shard sees activations with **no direct token
   mapping** at either end. Cheap; stacks with everything above.

**Recommended composition: `D-scale → requantize → per-tensor P`.** Purely offline, ~free, and it turns the
attack from "seconds via multiset lookup" into "recover per-channel scale factors under quantization noise" —
a much harder statistical problem. Add (3) whenever a model matters enough, and it stops being a matching
problem at all.

## The honest framing (put this in the sales/legal docs verbatim)
None of the above is *provable* security — it raises the inversion **cost**. That's fine, because it's the
wrong layer to carry the guarantee:

> **Obfuscation makes the *weights* expensive to fingerprint. Shares make the *activations* meaningless to
> observe. The guarantee lives in the second one.**

- **Obfuscation tier** (fast, cheap): `P·D → requant → P` + off-mesh ends. Bounds *weight* de-anonymization
  cost. Sell it as "computationally private, fast" — never as information-theoretic.
- **Share / MPC tier** (the guarantee): additive shares make each node's view of *activations* uniform →
  information-theoretic, KS≈0. This is what survives an adversary with unlimited compute. Sell the guarantee
  here.
- **Custodian tier** (strongest for regulated data): the record never leaves the device at all; the powerful
  model only asks questions (`docs/SPLIT_DEPTH_CUSTODIAN.md`).

## Serving-engine note (for the full-scale red-team)
Ollama (current Expert host) doesn't expose activations, so the scaled red-team above runs on realistic weight
tensors. To red-team the **real 31B** end-to-end — apply `P·D→Q` to the actual checkpoint, serve with an
activation-accessible engine (vLLM / SGLang / llama.cpp with hooks), and measure both weight-fingerprint
recovery *and* activation-inversion — run it on a box with headroom (the current GPU is VRAM-pressured at
31/32 GB; free the other ~16 GB first). The attack + metric code generalizes directly from
`scripts/obfuscation_redteam.py`.
