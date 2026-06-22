#!/usr/bin/env python
"""Blind shard node (UNTRUSTED) that loads ONLY its layer range [SPLIT:HI] straight from the
scrambled HF checkpoint already baked at /models/perm_scrambled_hf — so it deploys on the stock
moonlight-baked image with NO new image and NO bundle transfer. KV cache + truncate, same wire
contract as node_kv.py. Sees only P-framed hidden states; no embedding, no head, no tokenizer, no keys.

Env: CKPT (default /models/perm_scrambled_hf), SPLIT, HI, PORT, NODE_REGION.
"""
import base64, json, os, re
import numpy as np
import requests
import torch
from fastapi import FastAPI, Request, Response

_CHAIN = requests.Session()
from pydantic import BaseModel
from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding)
from transformers.cache_utils import DynamicCache
from safetensors import safe_open

CKPT = os.environ.get("CKPT", "/models/perm_scrambled_hf")
SPLIT = int(os.environ["SPLIT"]); HI = int(os.environ["HI"])
REGION = os.environ.get("NODE_REGION", "unknown")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = torch.finfo(torch.bfloat16).min

print(f"[node-hf] loading scrambled layers {SPLIT}:{HI} from {CKPT} on {DEV} (region={REGION})", flush=True)
cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
_open = {}
def _tensor(k):
    f = idx[k]
    if f not in _open:
        _open[f] = safe_open(os.path.join(CKPT, f), framework="pt", device="cpu")
    return _open[f].get_tensor(k)

N_EXP = getattr(cfg, "n_routed_experts", 64)
_rx = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
layers = []
for i in range(SPLIT, HI):
    pref = f"model.layers.{i}."
    sd = {}; g = {}; u = {}; dn = {}
    for k in idx:
        if not k.startswith(pref): continue
        sub = k[len(pref):]; mm = _rx.match(sub)
        if mm:                                              # per-expert routed weight -> collect to fuse
            e = int(mm.group(1)); t = _tensor(k).to(DEV, torch.bfloat16)
            (g if mm.group(2) == "gate" else u if mm.group(2) == "up" else dn)[e] = t
        else:
            sd[sub] = _tensor(k).to(DEV, torch.bfloat16)
    if g:                                                   # fuse routed experts to the layer's expected layout
        sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([g[e], u[e]], dim=0) for e in range(N_EXP)])
        sd["mlp.experts.down_proj"] = torch.stack([dn[e] for e in range(N_EXP)])
    with torch.device("meta"):
        lay = DeepseekV3DecoderLayer(cfg, i)
    lay.load_state_dict(sd, assign=True); lay.eval()
    for p in lay.parameters():
        p.requires_grad_(False)
    layers.append(lay); del sd
    print(f"[node-hf]   layer {i} ready", flush=True)
rotary = DeepseekV3RotaryEmbedding(cfg).to(DEV)
SESS = {}
print(f"[node-hf] ready: {len(layers)} scrambled layers {SPLIT}:{HI} on {DEV}", flush=True)

app = FastAPI()


class T(BaseModel):
    b64: str; shape: list; sid: str = "default"; reset: bool = False
class R(BaseModel):
    sid: str = "default"
class TR(BaseModel):
    sid: str = "default"; length: int


def enc(t): return {"b64": base64.b64encode(t.detach().cpu().float().numpy().astype(np.float32).tobytes()).decode(),
                    "shape": list(t.shape)}


@torch.no_grad()
def _run(h, sid, reset):
    if reset or sid not in SESS:
        SESS[sid] = {"cache": DynamicCache(), "seen": 0}
    s = SESS[sid]; cache = s["cache"]; past = s["seen"]; S = h.shape[1]
    cp = torch.arange(past, past + S, device=DEV); pos = cp.unsqueeze(0)
    pe = rotary(h, position_ids=pos); total = past + S
    if S > 1:
        qp = torch.arange(past, past + S, device=DEV).unsqueeze(1); kp = torch.arange(total, device=DEV).unsqueeze(0)
        m = torch.where(kp <= qp, 0.0, NEG).to(torch.bfloat16).view(1, 1, S, total)
    else:
        m = torch.zeros(1, 1, 1, total, dtype=torch.bfloat16, device=DEV)
    for lay in layers:
        r = lay(h, attention_mask=m, position_embeddings=pe, position_ids=pos,
                past_key_values=cache, use_cache=True, cache_position=cp)
        h = r[0] if isinstance(r, tuple) else r
    s["seen"] = total
    return h


@app.get("/health")
def health(): return {"ok": True, "block": f"{SPLIT}:{HI}", "device": DEV, "region": REGION,
                      "kv": True, "sessions": len(SESS), "note": "P-framed noise only; no keys."}
@app.post("/reset")
def reset(r: R): SESS.pop(r.sid, None); return {"ok": True}
@app.post("/truncate")
def truncate(r: TR):
    s = SESS.get(r.sid)
    if s: s["cache"].crop(r.length); s["seen"] = r.length
    return {"ok": True, "seen": r.length}
@app.post("/forward")
@torch.no_grad()
def forward(t: T):
    h = torch.from_numpy(np.frombuffer(base64.b64decode(t.b64), dtype=np.float32).reshape(t.shape).copy()).to(DEV).to(torch.bfloat16)
    if h.dim() == 2: h = h.unsqueeze(0)
    return enc(_run(h, t.sid, t.reset)[0])
@app.post("/forward_raw")
@torch.no_grad()
async def forward_raw(request: Request):
    raw = await request.body()
    sid = request.headers.get("x-sid", "default"); reset = request.headers.get("x-reset", "0") == "1"
    shape = [int(x) for x in request.headers.get("x-shape", "").split(",")]
    nxt = request.headers.get("x-next")                         # chain: forward to the NEXT colocated shard
    h = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).to(DEV)
    if h.dim() == 2: h = h.unsqueeze(0)
    out = _run(h, sid, reset)[0].to(torch.bfloat16).cpu().contiguous()
    payload = out.view(torch.uint8).numpy().tobytes()
    if nxt:                                                     # local hop to next node, return ITS output
        r = _CHAIN.post(nxt.rstrip("/") + "/forward_raw", data=payload,
                        headers={"x-sid": sid, "x-reset": "1" if reset else "0",
                                 "x-shape": ",".join(str(x) for x in out.shape),
                                 "content-type": "application/octet-stream"}, timeout=120)
        return Response(content=r.content, media_type="application/octet-stream",
                        headers={"x-shape": r.headers["x-shape"]})
    return Response(content=payload, media_type="application/octet-stream",
                    headers={"x-shape": ",".join(str(x) for x in out.shape)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8010")))
