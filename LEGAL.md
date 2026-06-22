# Legal notes & due diligence

**This is not legal advice.** It is a plain-language summary of the considerations
we worked through when open-sourcing this code, plus the documentation we put in
place. For anything commercial or high-stakes, have a qualified IP/technology
lawyer review it — especially the patent point in §3.

---

## 1. What is licensed here (and what isn't)

- **The CODE in this repository** — the scramble/key-generation pipeline, the
  shard node, the coordinators, the serving and verification scripts — is original
  work by the ZKAGI Ecosystem Association and is released under the **MIT License**
  (see `LICENSE`). MIT lets anyone use, modify, and redistribute it, including
  commercially, with no warranty.

- **No model weights are included.** The repository ships **code only**. It does
  not contain any base-model weights, any *scrambled* weights, or any Π/P keys.
  This is deliberate (see §2).

## 2. Base-model license — the most common pitfall

The tool **transforms a model you supply**. A scrambled checkpoint is a
**derivative** of the base model, so it is governed by the **base model's license**,
not by ours.

- Download and use the base model (e.g. a DeepSeek-V3-family model such as
  `moonshotai/Moonlight-16B-A3B-Instruct`) under **its own license**, and check
  whether that license permits derivatives, redistribution, and commercial use.
- **Do not redistribute scrambled weights** unless the base model's license clearly
  allows redistribution of derivatives. The safe, default workflow this repo
  encourages is: *each user downloads their own model and runs the scramble
  locally.* That keeps you out of the weight-redistribution question entirely.
- `.gitignore` excludes `*_scrambled_hf/`, `*_bundle.pt`, `*.safetensors`,
  `perm_keys.pt`, etc., so weights/keys cannot be committed by accident.

## 3. The method: covariant obfuscation (the part to actually check)

The core privacy primitive — the covariant rotation `P` applied jointly to the
data and weights — was introduced by **AloePri** (Lin et al., 2026,
arXiv:2603.01499). We credit it prominently (`NOTICE`, `README`).

- **Copyright:** implementing a *published algorithm* in your own code is generally
  permitted — copyright protects a specific expression of code, not the underlying
  mathematical method. This repo is an independent implementation, not a copy of
  any AloePri code.
- **Patents (the real risk):** a published method can still be **patented**, and
  patents cover the **method regardless of independent implementation** — unlike
  copyright, writing your own code does NOT avoid patent infringement.

  **Preliminary search performed (2026-06, not a professional FTO):**
  - The one *issued* US patent that surfaced near this space —
    [US 11,288,379](https://patents.google.com/patent/US11288379B2/en) (UC San Diego)
    — is a **different technique** (Laplace input-noise / differential privacy,
    weights frozen, no weight rotation, no transformers). It does **not** read on
    covariant obfuscation. Not a concern.
  - **No issued or published patent on covariant obfuscation / weight-rotation
    privacy was found** in public databases as of the search date.
  - **But that is not clearance.** AloePri is a **ByteDance + Nanjing University**
    work, described as "integrated into an industrial LMaaS system." ByteDance is a
    prolific patent filer (USPTO, EPO, WIPO/PCT, and especially CNIPA/China). Patents
    publish **~18 months after filing**, so any 2025-2026 filing on covariant
    obfuscation would very likely be **unpublished and invisible to a search today**.
    "Found nothing" here means "nothing public yet," not "nothing exists." The same
    authors' related prior work (ObfusLM, ACL 2025; embedding-obfuscation, EMNLP 2024)
    may also have filings.
  - **Prior art in your favor (for invalidity, not freedom-to-operate):** the
    underlying *rotational invariance* of transformer weights is established public
    prior art (QuaRot, SpinQuant, 2024, used for quantization). This may limit how
    broadly any covariant-obfuscation patent can validly claim, and is a defense if a
    broad claim were ever asserted — but it does not by itself grant you the right to
    operate.

  **What to do:** open-sourcing the *code* (MIT) is low-risk — publishing code is not
  the same as commercially making/using/selling a patented invention. The risk
  concentrates on **commercial deployment** (running it as a paid service). Before
  that, commission a real **freedom-to-operate search** covering **ByteDance and the
  named inventors** (Yu Lin, Qizhi Zhang, Wenqiang Ruan, Jue Hong, Ye Wu) across
  USPTO / EPO / WIPO / **CNIPA**, including the unpublished window an attorney can
  track via priority/family monitoring. As a **Swiss** association, what binds you is
  patents in force where you operate/sell (CH/EU, and the US if you serve US users);
  a China-only patent affects only China operations. Consider contacting the authors
  for a license — they published openly, but publication is not a patent waiver.

## 4. Other prior work

- **leyten/shard** (`https://github.com/leyten/shard`) informed the sharding design.
  No shard code is included; this is an independent implementation. If you ever copy
  from it, comply with its license and keep its notices.

## 5. Dependencies

All runtime dependencies are permissively licensed and MIT-compatible: PyTorch
(BSD-3), Transformers / vLLM / safetensors (Apache-2.0), FastAPI/Uvicorn/Pydantic
(MIT/BSD). They are **installed via pip**, not redistributed here, so their license
obligations are limited. See `NOTICE`.

## 6. Cryptography / export

Verifiable receipts use **ML-DSA (FIPS-204)**, a public, standardized post-quantum
signature scheme. Publicly available open-source cryptographic source code is
generally treated as exportable under most regimes (e.g. the publicly-available
source-code provisions), but if you ship binaries to sanctioned destinations,
check your local export rules.

## 7. Privacy claims — say it accurately

This method provides **computational privacy** (hard to reverse without the secret
keys), **not** information-theoretic privacy (impossible to reverse). The measured
guarantee is an empirical attack bound (token-recovery well under a 5% bar), plus
exactness and a verifiable receipt. Marketing it as "unbreakable" or
"mathematically impossible to reverse" would be inaccurate and could create
misrepresentation exposure. The README and NOTICE state the honest version.

---

## 8. Commercial / API deployment — extended due diligence (2026-06)

This section records a deeper search performed when considering selling the method as a
paid API. **It is still not a professional FTO clearance.**

**Base-model license — CLEARED (this was the biggest concrete risk).**
The reference base model, `moonshotai/Moonlight-16B-A3B-Instruct`, is published under the
**MIT license** (per its Hugging Face model card). MIT permits commercial use, serving via an
API, and creating/distributing derivatives, provided you retain the MIT notice. So serving a
scrambled derivative commercially does **not** breach the base-model license. *(Action: keep a
copy of the model's LICENSE/notice; verify the repo's actual LICENSE file matches the "mit"
tag at the time you ship; if you swap to a different base model, re-check its license.)*

**Patents — no published patent found, across 8+ search vectors.**
Searches covered: "covariant obfuscation" + patent; weight-rotation / invertible-matrix privacy;
ByteDance + obfuscation + inference; each named inventor (Yu Lin, Qizhi Zhang, Wenqiang Ruan,
Jue Hong, Ye Wu); the predecessor work (ObfusLM, ACL 2025; embedding-obfuscation, EMNLP 2024);
and WIPO/Espacenet/CNIPA-style queries. **No issued or published patent reading on covariant
obfuscation / token-permutation weight obfuscation was found.** The only issued patents in the
broad area ([US 11,288,379](https://patents.google.com/patent/US11288379B2/en) family, UCSD) are a
different technique (Laplace input noise) and do not apply.

**Two hard limits on "found nothing":**
1. **18-month publication delay.** AloePri is a 2026 ByteDance industrial system; any patent they
   filed on it in 2025-2026 is very likely **not yet published** and is invisible to *any* search
   today, including this one. "Nothing public" ≠ "nothing filed."
2. **Search reach.** Public web search is US/English-centric; it does not reliably cover **CNIPA
   (China)**, where ByteDance files most heavily. A China patent would bind you only for China
   operations, but it should still be checked by counsel with CNIPA access.

**Commercial / API launch checklist (do before you charge customers):**
- [ ] **FTO opinion** from a patent attorney — ByteDance + the named inventors, across USPTO / EPO /
      CH / WIPO / **CNIPA**, including priority/family monitoring for the unpublished window. A written
      opinion also limits *willful*-infringement (treble) damages.
- [ ] **Base-model license** re-verified for the exact model you ship; MIT notice retained.
- [ ] **Prior-art file** kept (rotational invariance: QuaRot, SpinQuant, 2024) for invalidity defense.
- [ ] **Design-around** plan ready in case a blocking claim surfaces.
- [ ] **Entity + insurance:** operate via the Swiss entity (limited liability); carry tech E&O /
      IP-infringement-defense insurance.
- [ ] **API Terms of Service:** disclaim warranties, cap liability, push base-model-license
      compliance to the customer for their use.
- [ ] **Honest marketing:** "computational privacy," never "unbreakable."
- [ ] **Data protection (GDPR / Swiss FADP):** DPAs + privacy posture for processing customer prompts.

## 9. Patent-conflict / design-around policy

The privacy layer is intentionally **modular**: the secret vocabulary permutation (`Π`) and the
covariant rotation (`P`) are independent components, and the obfuscation transform is **pluggable**.

If a third party identifies a **valid, in-force patent** that the covariant-obfuscation component
(`P`) would infringe, the project's intent is to **modify or replace that component** with a
non-infringing alternative — for example, a different invertible-transform construction, a method
that falls clearly within established public prior art (e.g. the rotational-invariance line of work,
QuaRot / SpinQuant), or greater reliance on the independent `Π` component.

This statement reflects an intent to operate in **good faith** and to respect valid third-party
intellectual property. It is **not** an admission that any patent is infringed, and it does **not**
waive any defense (including invalidity or non-infringement). It is a statement of policy, not a
warranty; the MIT `LICENSE` governs the code "as is." For a commercial deployment, have counsel
review the exact wording before relying on it.

### Bottom line

Open-sourcing **your own code** under MIT is low-risk and well-documented here.
The two things to get right are operational, not licensing: **(a)** don't
redistribute base-model-derived weights, and **(b)** clear the **patent** question
on covariant obfuscation with a lawyer before you commercialize. Everything in this
repo is structured to make (a) automatic and to give you the attribution trail for
(b).
