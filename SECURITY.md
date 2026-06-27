# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do **not** open a public GitHub issue.

- Email **legal@zkagi.ai** (PGP welcome). Include a description, affected files/versions, and a
  proof-of-concept if you have one.
- We aim to acknowledge within **5 business days** and to agree a coordinated-disclosure timeline
  (typically up to 90 days) before any public write-up.
- Please act in good faith: don't access others' data, degrade services, or exfiltrate beyond what's
  needed to demonstrate the issue.

There is no paid bounty at this time; we credit reporters who wish to be named.

## What this system does and does not protect (threat model)

Read this before assessing severity — many "issues" are actually documented properties.

**It protects:** the confidentiality of your prompt, the model's internal activations, and the output
**from the GPU(s) doing the work**. A blind node receives only permuted token IDs (single-node) or
P-framed activations (sharded), holds **no** keys/tokenizer/embedding/head, and cannot reconstruct
your prompt. Keys (`Π`, `P`) are generated on, and never leave, the **trusted** device.

**Its guarantee is computational, not information-theoretic — and CONDITIONAL on the deployed model
not being a public checkpoint the operator can also download.** This is the load-bearing caveat:

> A permutation/rotation **preserves each weight's value-multiset.** So an operator who downloads the
> **same public base model** can sort-and-match the scrambled weights/embeddings against it and
> **reconstruct the key `Π`/`P` with no keys, no labels, and no chosen-plaintext oracle** — a single-shot,
> unsupervised attack. Measured on real weights: **~100% token recovery (73% even through 4-bit quant),
> ~15 seconds.** See `redteam_keyrecovery.py`.

The earlier "~0.033%" figure was measured against an adversary that did **not** cross-reference the
public base model (`phase1_gate.py` Adversary A). It does **not** hold against an operator who does.
**Privacy is therefore real only under one of:**
1. **Secret weights** — deploy a private fine-tune/merge the operator does not have (no public reference
   ⇒ no multiset match). Fast, single-node.
2. **MPC `R_t` tier** — a per-request value mask with interactive non-linearities; information-theoretic
   per request, independent of the key, holds even on the exact public model. Slower (the round-complexity
   cost). This is the real cryptographic guarantee.
3. **TEE-backed node** — confidential compute; permutation is then defense-in-depth.

With a **public** base model on a single node and none of the above, this is **obfuscation** (raises the
attacker's cost), not a confidentiality guarantee. Label it as such.

**Out of scope / known limitations (not vulnerabilities):**
- **Public-base key recovery (the big one).** A static permutation/rotation of a *public* checkpoint is
  recoverable by value-multiset matching against that public model (see above / `redteam_keyrecovery.py`).
  Mitigate with a secret model, the MPC `R_t` tier, or a TEE node — not with more permutation or higher
  precision (higher precision makes the match *easier*).
- **Static key → traffic analysis.** A single fixed `Π` reused across requests is a substitution cipher;
  an operator aggregating traffic can apply frequency analysis. Per-request re-randomization / `R_t` closes this.
- **Structure leakage.** Exact prompt/output length and repeated-token structure survive permutation.
  Mitigate with length-bucketing, fixed batches, and jittered streaming.
- **Trusted device compromise.** If the device holding `perm_keys.pt` is compromised, privacy is lost.
  Protect the keys; prefer ephemeral per-session keys (roadmap: TEE-anchored key ceremony).
- **The base-model license / patents.** See `LEGAL.md`. Not a security matter.
- **Low-bit quantization** (nf4/int8) erodes exactness — serve at bf16/fp8. A correctness, not a
  security, property.
- **Node collusion in deep multi-node splits** may reduce protection; analyze your own topology.
- **Side channels** (timing, memory, traffic analysis) on a malicious node are an active research
  area; we make no guarantee against a fully adversarial host beyond the obfuscation properties above.

**In scope (please report):** key/secret leakage paths in the code; ways a blind node could recover
plaintext tokens *without* the keys beyond the measured bound; receipt forgery or signature bypass;
incorrect `Π`/`P` inversion that leaks data; or any path that lets an untrusted node obtain the keys,
tokenizer, embedding, or head.

## Honesty requirement

Security claims in this project must be accurate. Do not describe it as "unbreakable" or
"information-theoretically private." See `CONTRIBUTING.md`.
