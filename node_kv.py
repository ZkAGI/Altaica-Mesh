#!/usr/bin/env python
"""Blind shard node with KV CACHE (UNTRUSTED). Holds ONLY scrambled layers [SPLIT:HI]; sees only
P-framed hidden states. Per-session KV cache so each decode step processes 1 new position instead
of the whole sequence (validated token-exact vs no-cache in verify_shard3_kv.py).

  GET  /health
  POST /forward {b64, shape, sid, reset}  -> {b64, shape}   (hidden after this block)
  POST /reset   {sid}                                        (drop a session's cache)

Env: NODE_BUNDLE, PORT, NODE_REGION (e.g. "Chennai, India" / "Amsterdam, Netherlands"), NODE_HI.
"""
import base64, os
import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding)
from transformers.cache_utils import DynamicCache

BUNDLE = os.environ.get("NODE_BUNDLE", "/models/nodeB3_bundle.pt")
REGION = os.environ.get("NODE_REGION", "unknown")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = torch.finfo(torch.bfloat16).min

print(f"[node] mmap {BUNDLE} on {DEV} (region={REGION}) ...", flush=True)
b = torch.load(BUNDLE, map_location="cpu", weights_only=False, mmap=True)
cfg = b["config"]; SPLIT = b["split"]; L = b["n_layers"]
HI = int(os.environ.get("NODE_HI", str(L)))
layers = []
for i in range(SPLIT, HI):
    with torch.device("meta"):
        lay = DeepseekV3DecoderLayer(cfg, i)
    lay.load_state_dict({k: v.to(DEV, torch.bfloat16) for k, v in b["layers"][i].items()}, assign=True)
    lay.eval()
    for p in lay.parameters():
        p.requires_grad_(False)
    layers.append(lay)
    print(f"[node]   loaded layer {i}", flush=True)
rotary = DeepseekV3RotaryEmbedding(cfg).to(DEV)
SESS = {}                                   # sid -> {"cache": DynamicCache, "seen": int}
print(f"[node] ready: scrambled layers {SPLIT}:{HI} on {DEV} (region={REGION})", flush=True)

app = FastAPI()


class T(BaseModel):
    b64: str
    shape: list
    sid: str = "default"
    reset: bool = False


class R(BaseModel):
    sid: str = "default"


class TR(BaseModel):
    sid: str = "default"
    length: int


def enc(t): return {"b64": base64.b64encode(t.detach().cpu().float().numpy().astype(np.float32).tobytes()).decode(),
                    "shape": list(t.shape)}   # .float() FIRST — numpy has no bfloat16


@app.get("/health")
def health():
    return {"ok": True, "block": f"{SPLIT}:{HI}", "device": DEV, "region": REGION,
            "kv": True, "sessions": len(SESS),
            "note": "I hold only scrambled layers; the hidden I receive is P-framed noise."}


@app.post("/reset")
def reset(r: R):
    SESS.pop(r.sid, None)
    return {"ok": True}


@app.post("/truncate")
def truncate(r: TR):
    """Roll the session KV cache back to `length` positions (drop rejected speculative drafts)."""
    sess = SESS.get(r.sid)
    if not sess:
        return {"ok": True, "seen": 0}
    sess["cache"].crop(r.length)        # transformers 5.x crop: keep first `length` positions (MLA-aware)
    sess["seen"] = r.length
    return {"ok": True, "seen": r.length}


@torch.no_grad()
def _run(h, sid, reset):
    """Core: run the scrambled block on hidden h [1,S,H] with the session KV cache. Returns h [1,S,H]."""
    if reset or sid not in SESS:
        SESS[sid] = {"cache": DynamicCache(), "seen": 0}
    sess = SESS[sid]; cache = sess["cache"]; past = sess["seen"]
    S = h.shape[1]
    cache_position = torch.arange(past, past + S, device=DEV)
    pos = cache_position.unsqueeze(0)
    pe = rotary(h, position_ids=pos)
    total = past + S
    if S > 1:
        qp = torch.arange(past, past + S, device=DEV).unsqueeze(1)
        kp = torch.arange(total, device=DEV).unsqueeze(0)
        m = torch.where(kp <= qp, 0.0, NEG).to(torch.bfloat16).view(1, 1, S, total)
    else:
        m = torch.zeros(1, 1, 1, total, dtype=torch.bfloat16, device=DEV)
    for lay in layers:
        r = lay(h, attention_mask=m, position_embeddings=pe, position_ids=pos,
                past_key_values=cache, use_cache=True, cache_position=cache_position)
        h = r[0] if isinstance(r, tuple) else r
    sess["seen"] = total
    return h


@app.post("/forward")                         # legacy: float32 + base64 + JSON
@torch.no_grad()
def forward(t: T):
    h = torch.from_numpy(np.frombuffer(base64.b64decode(t.b64), dtype=np.float32)
                         .reshape(t.shape).copy()).to(DEV).to(torch.bfloat16)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    return enc(_run(h, t.sid, t.reset)[0])


@app.post("/forward_raw")                      # fast wire: raw bf16 bytes + headers (½ the bytes, no base64/JSON)
@torch.no_grad()
async def forward_raw(request: Request):
    raw = await request.body()
    sid = request.headers.get("x-sid", "default")
    reset = request.headers.get("x-reset", "0") == "1"
    shape = [int(x) for x in request.headers.get("x-shape", "").split(",")]
    h = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).to(DEV)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    out = _run(h, sid, reset)[0].to(torch.bfloat16).cpu().contiguous()
    return Response(content=out.view(torch.uint8).numpy().tobytes(),
                    media_type="application/octet-stream",
                    headers={"x-shape": ",".join(str(x) for x in out.shape)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8009")))
