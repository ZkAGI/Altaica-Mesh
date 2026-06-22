#!/usr/bin/env python
"""Moonlight coordinator (TRUSTED, runs on the 5090). Holds the key-frame pieces: scrambled
embed (E·P), lm_head, final norm, and layer block [0:SPLIT]. Generates by running block A
locally, sending the P-framed hidden to the 4090 node (block B), then unscrambling at the head.

Usage:  python moonlight_coord.py --node http://<4090-tailscale-ip>:8009 --prompt "..."
"""
import argparse, base64, os, time
import numpy as np
import torch
import requests
from transformers import AutoTokenizer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding, DeepseekV3RMSNorm)
from transformers.masking_utils import create_causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def enc(t): return {"b64": base64.b64encode(t.detach().cpu().float().numpy().astype(np.float32).tobytes()).decode(),
                    "shape": list(t.shape)}
def dec(o): return torch.from_numpy(np.frombuffer(base64.b64decode(o["b64"]), dtype=np.float32)
                                    .reshape(o["shape"]).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--prompt", default="Explain why secret sharing keeps data private during AI inference.")
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()

    print("[coord] loading coord_bundle.pt ...", flush=True)
    b = torch.load(os.path.join(HERE, "coord_bundle.pt"), map_location="cpu", weights_only=False)
    cfg = b["config"]; SPLIT = b["split"]; L = b["n_layers"]
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "ml_tok"))

    embed_w = b["embed"].to(DEV).to(torch.bfloat16)                 # E·P (scrambled)
    lm_w = b["lm_head"].to(DEV).to(torch.bfloat16)
    norm = DeepseekV3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(DEV).to(torch.bfloat16)
    norm.weight.data.copy_(b["norm"].to(DEV))
    def build_layers(layer_dict, idxs):
        out = []
        for i in idxs:
            with torch.device("meta"):
                lay = DeepseekV3DecoderLayer(cfg, i)
            lay.load_state_dict({k: v.to(DEV, torch.bfloat16) for k, v in layer_dict[i].items()}, assign=True)
            lay.eval()
            out.append(lay)
        return out

    layersA = build_layers(b["layers"], range(SPLIT))            # 5090: block A [0:SPLIT]
    # NODE_HI: the 4090 holds [SPLIT:HI]; the 5090 runs the leftover tail [HI:L] (it has the bundle)
    HI = int(os.environ.get("NODE_HI", str(L)))
    layersC = []
    if HI < L:
        nb = torch.load(os.path.join(HERE, "nodeB_bundle.pt"), map_location="cpu",
                        weights_only=False, mmap=True)
        layersC = build_layers(nb["layers"], range(HI, L))       # 5090: tail block C [HI:L]
    rotary = DeepseekV3RotaryEmbedding(cfg).to(DEV)
    print(f"[coord] ready: 5090 holds [0:{SPLIT}]+[{HI}:{L}], 4090 holds [{SPLIT}:{HI}] @ {a.node}", flush=True)
    print("[coord] node health:", requests.get(a.node.rstrip('/') + "/health", timeout=20).json(), flush=True)

    @torch.no_grad()
    def run_local(layers, h):
        if not layers:
            return h
        S = h.shape[1]; pos = torch.arange(S, device=DEV).unsqueeze(0)
        cm = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None,
                                past_key_values=None, position_ids=pos)
        pe = rotary(h, position_ids=pos)
        for lay in layers:
            r = lay(h, attention_mask=cm, position_embeddings=pe, position_ids=pos,
                    past_key_values=None, use_cache=False)
            h = r[0] if isinstance(r, tuple) else r
        return h

    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    seq = ids[0].tolist()
    print(f"\n[coord] prompt: {a.prompt}\n[coord] generating across 5090(block A) + 4090(block B)...\n", flush=True)
    t0 = time.time(); wire_std = None
    for _ in range(a.max_new):
        cur = torch.tensor([seq], device=DEV)
        h = torch.nn.functional.embedding(cur, embed_w)            # e·P
        h = run_local(layersA, h)                                  # 5090 block A [0:SPLIT]
        if wire_std is None:
            wire_std = float(h.float().std())                      # what crosses to the 4090
        r = requests.post(a.node.rstrip("/") + "/forward", json=enc(h[0]), timeout=120)  # -> 4090 [SPLIT:HI]
        h = dec(r.json()).to(DEV).to(torch.bfloat16).unsqueeze(0)
        h = run_local(layersC, h)                                  # 5090 tail block C [HI:L]
        h = norm(h)
        nxt = int((h[0, -1] @ lm_w.t()).argmax())                 # unscramble at head
        seq.append(nxt)
        if nxt == tok.eos_token_id:
            break
    dt = time.time() - t0
    out = tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)
    n = len(seq) - ids.shape[1]
    print("=== ANSWER (5090 coordinator + 4090 node, obfuscated) ===")
    print(out)
    print(f"\nTPS: {n/dt:.2f} tok/s  | wire to 4090: P-framed (std {wire_std:.2f}, the 4090 sees noise)")
    print(f"node: {a.node}  | block split: 0:{SPLIT} on 5090, {SPLIT}:{L} on 4090")


if __name__ == "__main__":
    main()
