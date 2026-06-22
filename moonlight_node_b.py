#!/usr/bin/env python
"""Moonlight node B (UNTRUSTED, runs on the 4090). Loads ONLY the scrambled layer block
[SPLIT:L] from nodeB_bundle.pt and runs it on the P-framed hidden it receives. No key, no
embedding, no lm_head, no tokenizer -> sees only noise.

  GET  /health
  POST /forward {b64,shape}  ->  {b64,shape}   (hidden after the block)
"""
import base64, os
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding)
from transformers.masking_utils import create_causal_mask

BUNDLE = os.environ.get("NODE_BUNDLE", "/data/nodeB_bundle.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[nodeB] memory-mapping {BUNDLE} (lazy, low-RAM) on {DEV} ...", flush=True)
# mmap=True: file is memory-mapped, tensors load lazily -> peak RAM ~one layer, not 16GB
b = torch.load(BUNDLE, map_location="cpu", weights_only=False, mmap=True)
cfg = b["config"]; SPLIT = b["split"]; L = b["n_layers"]
# NODE_HI lets a memory-limited node hold only layers [SPLIT:HI]; the coordinator runs
# [HI:L] from the same bundle (it has it on disk). Default = all of block B.
HI = int(os.environ.get("NODE_HI", str(L)))
layers = []
for i in range(SPLIT, HI):
    # build on meta (no alloc) then assign weights straight to GPU -> no double allocation
    with torch.device("meta"):
        lay = DeepseekV3DecoderLayer(cfg, i)
    sd = {k: v.to(DEV, torch.bfloat16) for k, v in b["layers"][i].items()}   # one layer at a time
    lay.load_state_dict(sd, assign=True)
    lay.eval()
    for p in lay.parameters():
        p.requires_grad_(False)
    layers.append(lay)
    del sd
    print(f"[nodeB]   loaded layer {i}", flush=True)
rotary = DeepseekV3RotaryEmbedding(cfg).to(DEV)
print(f"[nodeB] ready: scrambled layers {SPLIT}:{HI} ({len(layers)} blocks) on {DEV}", flush=True)

app = FastAPI()


class T(BaseModel):
    b64: str
    shape: list


def enc(t): return {"b64": base64.b64encode(t.detach().cpu().numpy().astype(np.float32).tobytes()).decode(),
                    "shape": list(t.shape)}


@app.get("/health")
def health():
    return {"ok": True, "block": f"{SPLIT}:{HI}", "device": DEV,
            "note": "I hold only scrambled layers; the hidden I receive is P-framed noise."}


@app.post("/forward")
@torch.no_grad()
def forward(t: T):
    h = torch.from_numpy(np.frombuffer(base64.b64decode(t.b64), dtype=np.float32)
                         .reshape(t.shape).copy()).to(DEV).to(torch.bfloat16)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    S = h.shape[1]
    pos = torch.arange(S, device=DEV).unsqueeze(0)
    cm = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None,
                            past_key_values=None, position_ids=pos)
    pe = rotary(h, position_ids=pos)
    print(f"[nodeB] forward S={S} mean={h.float().mean():.3f} std={h.float().std():.3f} (opaque)", flush=True)
    for lay in layers:
        r = lay(h, attention_mask=cm, position_embeddings=pe, position_ids=pos,
                past_key_values=None, use_cache=False)
        h = r[0] if isinstance(r, tuple) else r
    return enc(h[0].float())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8009")))
