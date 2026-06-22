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

**Its guarantee is computational, not information-theoretic.** Privacy rests on the secrecy of `Π`
(and the hardness of inverting the obfuscation), **not** on the rotation `P` alone — a determined
attacker with labeled examples can partially undo a fixed rotation, so `P` by itself is not the lock.
Measured true-token recovery against a realistic no-keys adversary is ~0.033% (vs a 5% bar); treat
this as an empirical bound, not a proof.

**Out of scope / known limitations (not vulnerabilities):**
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
