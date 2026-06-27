#!/usr/bin/env python
"""RED-TEAM: the adversary phase1_gate.py's 'realistic mesh' (Adversary A) omitted.

Adversary A in phase1_gate.py recovers the PERMUTED token stream but assumes it cannot map
permuted -> true 'without the secret Π'. That assumption is FALSE when the deployed model is a
PUBLIC download: the node holds the scrambled weights, downloads the SAME public base model, and
recovers Π (and the hidden-axis key P) by VALUE-MULTISET MATCHING — a permutation/rotation preserves
each row/column's multiset of values, so sort-and-match against the public reference reconstructs the
key. No keys, no chosen-plaintext oracle, no labels. Unsupervised, single-shot, seconds.

This tool measures that honest number. Run it as Adversary A' alongside the gate.

  python redteam_keyrecovery.py --scrambled ./perm_scrambled_hf --original moonshotai/Moonlight-16B-A3B-Instruct

Reports:
  * token-recovery (Π) : % of tokens whose true id is recovered from the scrambled embedding rows
  * key-recovery   (P) : % of hidden positions of the column permutation recovered
A static permutation of a public model should report ~100% here. If it does, privacy is obfuscation —
deploy a SECRET model or the MPC R_t tier for a real guarantee (see SECURITY.md).
"""
import argparse, time, numpy as np, torch
from safetensors import safe_open
from transformers import AutoModel, AutoModelForCausalLM

def load_embed(spec):
    """Return the input-embedding matrix [vocab, hidden] as float32 numpy, from a HF dir or model id."""
    import os, json, glob
    # try safetensors index (fast, no full load)
    idx = os.path.join(spec, "model.safetensors.index.json") if os.path.isdir(spec) else None
    if idx and os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        cand = [k for k in wm if k.endswith("embed_tokens.weight")]
        if cand:
            name = cand[0]
            with safe_open(os.path.join(spec, wm[name]), framework="pt") as f:
                return f.get_tensor(name).float().numpy(), name
    # fallback: load the model (CPU) and read input embeddings
    m = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=torch.float32,
                                             trust_remote_code=True, device_map="cpu")
    return m.get_input_embeddings().weight.detach().float().numpy(), "input_embeddings"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrambled", required=True, help="scrambled HF dir (what the node holds)")
    ap.add_argument("--original", required=True, help="public base model id or dir (what the adversary downloads)")
    ap.add_argument("--n", type=int, default=2000, help="tokens to test for Π recovery")
    args = ap.parse_args()

    print("loading scrambled + public embeddings ...", flush=True)
    Es, _ = load_embed(args.scrambled)      # node-held scrambled embed  (rows=Π(tokens), cols=P(hidden))
    Eo, _ = load_embed(args.original)       # public reference embed
    assert Es.shape == Eo.shape, f"shape mismatch {Es.shape} vs {Eo.shape}"
    V, H = Eo.shape
    print(f"vocab={V} hidden={H}", flush=True)

    rng = np.random.default_rng(0)

    # --- Π recovery (token mapping): row-multiset is invariant to the column perm P.
    #     Match each scrambled row's sorted values to a public row -> true token of that node row. ---
    t0 = time.time()
    Eo_sorted_q = np.round(np.sort(Eo, axis=1) * 1024).astype(np.int32)
    index = {Eo_sorted_q[i].tobytes(): i for i in range(V)}   # public offline index
    rows = rng.choice(V, size=args.n, replace=False)
    hit = 0
    for r in rows:
        key = np.round(np.sort(Es[r]) * 1024).astype(np.int32).tobytes()
        if index.get(key, -1) == r:        # node row r really is some true token; exact-match recovers it
            hit += 1
    pi_acc = hit / args.n
    print(f"\nΠ token-mapping recovery (exact multiset): {pi_acc*100:.2f}% of {args.n} rows  ({time.time()-t0:.0f}s)", flush=True)

    # --- P recovery (hidden key): column-multiset match on a row sample ---
    t1 = time.time()
    sub = rng.choice(V, size=min(4096, V), replace=False)
    oc = np.sort(Eo[sub], axis=0); sc = np.sort(Es[sub], axis=0)   # [n,H] column fingerprints
    rec = np.empty(H, dtype=int)
    for c0 in range(0, H, 256):
        blk = sc[:, c0:c0+256]
        d = np.abs(oc[:, :, None] - blk[:, None, :]).sum(0)
        rec[c0:c0+256] = d.argmin(0)
    # P is recovered up to consistency; report how many columns matched a unique public column
    p_acc = len(set(rec.tolist())) / H
    print(f"P hidden-key columns uniquely matched to public columns: {p_acc*100:.2f}% of {H}  ({time.time()-t1:.0f}s)", flush=True)

    print("\nVERDICT:", flush=True)
    if pi_acc > 0.5:
        print(f"  🔴 STATIC PERMUTATION OF A PUBLIC MODEL IS RECOVERABLE ({pi_acc*100:.0f}% token recovery).")
        print("     Privacy here is OBFUSCATION, not a guarantee. Deploy a SECRET model or the MPC R_t tier.")
    else:
        print(f"  🟢 Multiset attack ineffective ({pi_acc*100:.1f}%) — deployed weights are not a public reference.")

if __name__ == "__main__":
    main()
