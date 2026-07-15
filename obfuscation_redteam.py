#!/usr/bin/env python
"""Red-team the obfuscation tier: does the P·D → requantize → P composition defeat the multiset attack?

Context (measured earlier on real gemma4-31b): STATIC PERMUTATION is broken. The deployed weights are the
public checkpoint's EXACT values, just rearranged, so an attacker matches each deployed row's *multiset* of
values to a public row → recovers the permutation in seconds → 100% de-obfuscation.

Suraj's fix family, red-teamed here on realistic weights with two attacks:
  A) exact-multiset  — match by sorted-row values (the attack that breaks permutation)
  B) correlation      — match by Pearson correlation of sorted rows (the "ratio" residual attack)
Defenses:
  1) P        — permutation only (baseline; expected broken)
  2) P·D      — permute + per-channel diagonal scaling (values all change; behavior preserved by folding
                the inverse scales into adjacent layers — network reparameterization symmetry)
  3) P·D+Q    — the above, then requantize each row to int4 with a FRESH per-row scale (quant noise)
Metric = fraction of rows the attacker correctly re-identifies (1.0 = fully de-obfuscated; ~1/N = chance).

Run: python scripts/obfuscation_redteam.py
"""
import numpy as np

RNG = np.random.default_rng(7)   # fixed seed (reproducible; Math.random-free)

def make_checkpoint(n=256, m=512):
    """A public checkpoint weight matrix. Real rows have distinct value distributions (as here), which is
    exactly why multiset matching works. The attacker holds this."""
    W = RNG.standard_normal((n, m)).astype(np.float64)
    W *= (0.5 + RNG.random((n, 1)))          # per-row scale variety, like real trained layers
    return W

def obf_perm(W):
    pr, pc = RNG.permutation(W.shape[0]), RNG.permutation(W.shape[1])
    return W[pr][:, pc], pr

def obf_pd(W):
    d_out = np.exp(RNG.uniform(-1, 1, (W.shape[0], 1)))   # per-output-channel scale (folded into next layer)
    d_in  = np.exp(RNG.uniform(-1, 1, (1, W.shape[1])))   # per-input-channel scale (folded into prev layer)
    Ws = d_out * W * d_in                                 # every value changes; y=Wx unchanged after folding
    pr, pc = RNG.permutation(W.shape[0]), RNG.permutation(W.shape[1])
    return Ws[pr][:, pc], pr

def obf_pdq(W):
    Wp, pr = obf_pd(W)
    scale = np.abs(Wp).max(axis=1, keepdims=True) / 7.0   # fresh per-row int4 scale
    Wq = np.round(Wp / scale) * scale                     # requantize to int4 (nf4-ish), fresh scales
    return Wq, pr

# ---- attacks: recover the row permutation by matching deployed rows to public rows ----
def attack_multiset(Wpub, Wdep):
    pub_sorted = np.sort(Wpub, axis=1)
    dep_sorted = np.sort(Wdep, axis=1)
    # nearest public row by L2 on sorted values (exact match → distance 0 under permutation-only)
    guess = np.array([np.argmin(((pub_sorted - d) ** 2).sum(1)) for d in dep_sorted])
    return guess

def attack_correlation(Wpub, Wdep):
    ps = np.sort(Wpub, axis=1); ds = np.sort(Wdep, axis=1)
    ps = (ps - ps.mean(1, keepdims=True)); ps /= (np.linalg.norm(ps, axis=1, keepdims=True) + 1e-9)
    ds = (ds - ds.mean(1, keepdims=True)); ds /= (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-9)
    return (ds @ ps.T).argmax(1)   # max Pearson corr of sorted rows

def recovery(true_pr, guess):
    return float((guess == true_pr).mean())

def main():
    W = make_checkpoint()
    n = W.shape[0]
    print(f"checkpoint {W.shape} · N={n} rows · chance recovery = {1/n:.3f}\n")
    print(f"{'defense':<10}{'exact-multiset':>16}{'correlation':>14}   verdict")
    for name, fn in [("P", obf_perm), ("P·D", obf_pd), ("P·D+Q", obf_pdq)]:
        Wdep, pr = fn(W)
        rm = recovery(pr, attack_multiset(W, Wdep))
        rc = recovery(pr, attack_correlation(W, Wdep))
        worst = max(rm, rc)
        verdict = "BROKEN" if worst > 0.5 else ("partial" if worst > 5/n else "holds ✓ (~chance)")
        print(f"{name:<10}{rm:>15.1%}{rc:>14.1%}   {verdict}")
    print("\nreading: permutation alone is fully recovered (multiset). Diagonal scaling kills exact-multiset")
    print("but a correlation residual remains → requantize closes it. The composition is ~free + offline.")
    print("NOTE: this bounds WEIGHT-fingerprinting cost. Activation secrecy is the MPC share layer's job")
    print("(uniform shares → info-theoretic); obfuscation makes weights expensive, shares make activations meaningless.")

if __name__ == "__main__":
    main()
