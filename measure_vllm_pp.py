#!/usr/bin/env python
"""Trusted Π client for the vLLM pipeline-parallel scrambled mesh. Tokenize -> Π(token ids) ->
vLLM PP cluster (blind, sees only permuted ids) -> Π^-1 -> detokenize. Measures decode tok/s.

  python measure_vllm_pp.py --base https://<pod>-8000.proxy.runpod.net --prompt "..." --max-new 64
"""
import argparse, json, os, time
import requests
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prompt", default="In three sentences, explain why private AI inference matters.")
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()
    base = a.base.rstrip("/")

    keys = torch.load(os.path.join(HERE, "perm_keys.pt"), weights_only=False)
    PERM, INV = keys["perm"].tolist(), keys["inv"].tolist()
    stop_real = [i for i in {keys.get("im_end"), keys.get("eos_real")} if i is not None and i >= 0]
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "ml_tok"))

    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    print(f"[piclient] model={model}", flush=True)

    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   add_generation_prompt=True, tokenize=False)
    real_ids = tok(text, add_special_tokens=False).input_ids
    perm_ids = [PERM[t] for t in real_ids]                      # Π(prompt) -> blind cluster sees gibberish
    stop_perm = [PERM[i] for i in stop_real]

    def call(maxn):
        body = {"model": model, "prompt": perm_ids, "max_tokens": maxn, "temperature": 0,
                "stop_token_ids": stop_perm, "return_tokens_as_token_ids": True, "logprobs": 1}
        t = time.time()
        r = requests.post(base + "/v1/completions", json=body, timeout=300).json()
        return r, time.time() - t

    call(8)                                  # warmup (cold start + CUDA-graph capture)
    r, wall = call(a.max_new)                # measured (warm)
    ch = r["choices"][0]
    # vLLM returns generated token ids as "token_id:<n>" strings in logprobs (return_tokens_as_token_ids),
    # else parse the text. Be adaptive.
    out_perm = []
    lp = ch.get("logprobs") or {}
    if lp.get("tokens"):
        for t in lp["tokens"]:
            out_perm.append(int(t.split(":")[-1]) if isinstance(t, str) and "token_id" in t else int(t))
    else:
        # fallback: text may be a space-joined id list, or detokenized
        txt = ch.get("text", "")
        try:
            out_perm = [int(x) for x in txt.split()]
        except Exception:
            print("[piclient] RAW response (adapt parser):", json.dumps(r)[:600]); return
    real_out = [INV[t] for t in out_perm]                       # Π^-1 (client only)
    answer = tok.decode(real_out, skip_special_tokens=True)
    n = len(real_out)
    usage = r.get("usage", {})
    print("=== ANSWER (vLLM PP-2 scrambled mesh, obfuscated) ===")
    print(answer)
    print(f"\n{n} tok in {wall:.2f}s -> {n/wall:.1f} tok/s (end-to-end incl network) | usage {usage}")


if __name__ == "__main__":
    main()
