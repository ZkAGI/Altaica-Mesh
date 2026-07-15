# Deploy the blind-GPU mesh — fortified obfuscation + client anonymization, on RunPod + local

Run a model on **someone else's GPU (RunPod) and your own local machines** such that no node can read the
data. Two independent layers, matched to the red-team findings in `OBFUSCATION_FORTIFICATION.md`:

- **Client anonymization** (zkTerminal `web/anonymize.mjs`): the query is pseudonymized *on the user's device*
  before it leaves — names/IDs → realistic fakes, key stays local, answer re-hydrated locally.
- **Weight obfuscation on the shards**: deployed weights are transformed so a node can't fingerprint which
  public checkpoint it holds — **signed `P·D±` + group-wise requantize (+ a tiny fine-tune delta for models
  that matter)**. Positive scaling and plain permutation are BROKEN (Sinkhorn / multiset) — use the signed
  composition. This *raises attacker cost*; it is not the guarantee (see below).

## Pipeline
```
model checkpoint ──(1) fortify-obfuscate──► obfuscated shards ──(2) place──► RunPod serverless + local nodes
                                                                                   ▲
        user device: anonymize query ──(3)──► coordinator (holds embed + output head, off-mesh) ──► shards
                     ◄── rehydrate answer + signed receipt ──────────────────────────────────────┘
```

### 1. Fortify-obfuscate the weights (offline, one-time, behavior-preserving)
The transform is `obfuscate(W, mode="D4"|"D5")` in `obfuscation_stresstest.py` (signed `P·D±` + group-wise
int4; `D5` adds the delta). Fold the inverse scales into adjacent layers — **take care through SiLU/RMSNorm**
(sign flips must be absorbed by the neighboring linear, not the norm). Integrate into the existing scramble
path (`m1_moonlight_mla_scramble.py`, `export_perm_scrambled_hf.py`) — those currently use permutation only,
which the stress-test shows is broken; **switch them to signed `P·D±`**. Verify token-exactness after folding
before shipping (the scramble tools already have exactness gates, e.g. `phase1_gate.py`).

### 2. Place shards — RunPod + local
- **Local nodes** (`node_kv.py` / `mesh_serve.py`): your own GPUs run some middle shards.
- **RunPod**: the middle shards you don't hold. Use the existing `docker/` image + serverless config.
  **Validate with a ~1GB model FIRST** (hard-won lesson: a big model that never runs still bills, and a
  container RAM cgroup < model size OOM-wipes the session). On-demand is ~$15–20/session; do not hold 24/7.
- **Off-mesh ends**: the coordinator (your trusted box) holds the **embedding + output head** so no node sees
  a direct token↔activation mapping at either end (`moonlight_coord.py`).

### 3. Serve, with anonymization + receipts in front
- The zkTerminal gateway takes the **already-anonymized** query, routes it to the coordinator, which drives the
  RunPod+local shards, and returns the answer + a **signed processing receipt** (Ed25519 + ML-DSA); the client
  rehydrates. Batch the receipts on-chain (`scripts/receipt_anchor.py` in zkTerminal).

## The guarantee (be honest in the pitch)
The red-team (`obfuscation_stresstest.py`) shows obfuscation leaves a **measurable residual** (~2–4% row
recovery even fully fortified). So:
- **Obfuscation = cost-raising**, orders of magnitude harder to fingerprint the weights. Sell as "computationally
  private, fast."
- **The guarantee = the MPC share layer** (additive shares → each node's activation view is uniform,
  information-theoretic). Turn this on for the tiers that promise a *guarantee*, not just speed.
- **The strongest posture for regulated data** is the custodian pattern (data never leaves; the mesh only sees
  the orchestrator's questions) — kept with the product.

## Go-live checklist
- [ ] Switch scramble tools permutation → **signed `P·D±` + requantize**; re-run the exactness gate.
- [ ] Build/validate the RunPod image with a **~1GB model first**; confirm no cgroup OOM.
- [ ] Coordinator holds embed + output head (off-mesh); shards split across local + RunPod.
- [ ] Client anonymization on; receipts emitted + optionally anchored on-chain.
- [ ] Region-pin RunPod if data-residency is required; record the region on the receipt.
- [ ] For guarantee-tier customers: enable the MPC share layer (don't rely on obfuscation alone).
