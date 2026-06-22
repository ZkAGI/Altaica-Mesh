# Contributing to Altaica Mesh

Thanks for your interest. This project is privacy infrastructure — correctness and honesty matter
more than speed of features. Please read these ground rules before opening a PR.

## Ground rules

1. **Never commit secrets, keys, or weights.** No `perm_keys.pt`, no scrambled checkpoints, no
   `*.safetensors`, no API keys, no `.altaica_mesh.json`. `.gitignore` blocks the common cases —
   double-check your diff anyway.
2. **Stay honest about the guarantees.** This is **computational privacy**, not information-theoretic.
   Do not add code, docs, or comments that claim "unbreakable", "impossible to reverse", or
   information-theoretic security. If you change the privacy model, update the measured numbers and
   the caveats together.
3. **Keep exactness.** The scramble must stay **token-exact** vs plaintext at bf16. If you touch the
   transform (`m1_moonlight_mla_scramble.py`) or the sharding path, run `python verify_shard3.py` and
   include the result in your PR.
4. **Respect the [Acceptable Use Policy](ACCEPTABLE_USE.md).** Don't contribute features whose primary
   purpose is to facilitate the prohibited uses listed there.
5. **MIT-compatible only.** Contributions are accepted under the project's MIT License. Don't import
   GPL/AGPL code or paste code from sources whose license you can't honor. Implementing *published
   methods* in your own code is fine; copying others' code is not (see `NOTICE`, `LEGAL.md`).

## How to contribute

1. Fork, branch from `main` (`feature/...` or `fix/...`).
2. Set up: `pip install -r requirements.txt` (NVIDIA GPU required for end-to-end runs).
3. Make focused changes; match the surrounding style.
4. Verify what you touched:
   - transform / sharding → `python verify_shard3.py` (must stay token-exact)
   - privacy claims → `python phase1_gate.py` (recovery vs the 5% bar)
5. Open a PR with: what changed, why, and any measured numbers.

## Developer Certificate of Origin (sign-off)

By contributing, you certify the [DCO](https://developercertificate.org/) — that you wrote the
contribution (or have the right to submit it) and agree it is provided under the MIT License.
Add a sign-off line to each commit:

```
git commit -s -m "your message"      # adds: Signed-off-by: Your Name <you@example.com>
```

## Reporting security issues

Do **not** open a public issue for a vulnerability — see **[SECURITY.md](SECURITY.md)**.

## Conduct

Be respectful and constructive. Harassment or discrimination is not tolerated; maintainers may
remove contributions or contributors that violate this or the Acceptable Use Policy.
