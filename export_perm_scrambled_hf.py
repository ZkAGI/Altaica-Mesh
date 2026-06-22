#!/usr/bin/env python
"""Export a Π+P obfuscated Moonlight as an HF checkpoint that STANDARD vLLM can serve.

P-scramble (M1) makes layers covariant + embed=E·P + lm_head unscrambles P. Then a secret vocab
permutation Π reorders the embed/lm_head rows, so the node runs ordinary generation on PERMUTED
token IDs (sees gibberish + P-framed noise) while the coordinator maps Π/Π^-1 at the boundary.

Writes:  perm_scrambled_hf/   (HF checkpoint for vLLM, on the untrusted node)
         perm_keys.pt          (Π, Π^-1 — the coordinator's secret, NEVER on the node)
"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import m1_moonlight_mla_scramble as m1

MODEL = os.environ.get("MODEL", "moonshotai/Moonlight-16B-A3B-Instruct")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "perm_scrambled_hf")
SEED_P, SEED_PI = 0, 7


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"== load {MODEL} ==", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map={"": "cpu"}).eval()
    cfg = model.config; V = cfg.vocab_size
    print("== P-scramble (M1) ==", flush=True)
    t = time.time(); m1.scramble(model, cfg, seed=SEED_P); print(f"  {time.time()-t:.0f}s", flush=True)

    print("== apply Π (permute vocab rows of embed + lm_head) ==", flush=True)
    rng = torch.Generator().manual_seed(SEED_PI)
    perm = torch.randperm(V, generator=rng)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(V)
    with torch.no_grad():
        ew = model.model.embed_tokens.weight
        lw = model.lm_head.weight
        ew.copy_(ew[inv].clone())          # row Π(t) := old row t  (E[t]·P)
        lw.copy_(lw[inv].clone())

    print(f"== save HF checkpoint -> {OUT} ==", flush=True)
    t = time.time()
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(OUT)
    print(f"  saved in {time.time()-t:.0f}s", flush=True)
    torch.save({"perm": perm, "inv": inv, "eos_real": tok.eos_token_id,
                "im_end": tok.convert_tokens_to_ids("<|im_end|>")},
               os.path.join(HERE, "perm_keys.pt"))
    print("DONE. node-checkpoint =", OUT, "| coordinator-keys = perm_keys.pt", flush=True)


if __name__ == "__main__":
    main()
