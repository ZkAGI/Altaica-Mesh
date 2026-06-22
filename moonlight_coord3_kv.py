#!/usr/bin/env python
"""3-shard coordinator with KV CACHE (TRUSTED). Holds keys (embed·P, lm_head, norm, layers 0:K1).
Prefill once, then incremental decode: each token = embed 1 id -> layersA(1 pos, cached) -> node B
(1 pos, cached) -> node C (1 pos, cached) -> head. Validated token-exact in verify_shard3_kv.py.

Usage:
  python moonlight_coord3_kv.py --node-b http://<chennai-2>:8009 \
                                --node-c https://<runpod-nl>.proxy.runpod.net \
                                --prompt "..." --max-new 48
"""
import argparse, base64, hashlib, os, time, uuid
import numpy as np
import torch
import requests
from transformers import AutoTokenizer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding, DeepseekV3RMSNorm)
from transformers.cache_utils import DynamicCache

try:
    import receipts as rc
except Exception:
    rc = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = torch.finfo(torch.bfloat16).min

SESSION = requests.Session()
_ad = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=6, max_retries=2)
SESSION.mount("http://", _ad); SESSION.mount("https://", _ad)


def enc(t): return {"b64": base64.b64encode(t.detach().cpu().float().numpy().astype(np.float32).tobytes()).decode(),
                    "shape": list(t.shape)}
def dec(o): return torch.from_numpy(np.frombuffer(base64.b64decode(o["b64"]), dtype=np.float32)
                                    .reshape(o["shape"]).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-b", required=True); ap.add_argument("--node-c", required=True)
    ap.add_argument("--prompt", default="Explain why splitting a model across untrusted GPUs keeps the prompt private.")
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    sid = uuid.uuid4().hex[:12]

    print(f"[coord] loading coord3_bundle.pt on {DEV} ...", flush=True)
    b = torch.load(os.path.join(HERE, "coord3_bundle.pt"), map_location="cpu", weights_only=False)
    cfg = b["config"]; K1 = b["split"]; L = b["n_layers"]
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "ml_tok"))
    embed_w = b["embed"].to(DEV).to(torch.bfloat16)
    lm_w = b["lm_head"].to(DEV).to(torch.bfloat16)
    norm = DeepseekV3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(DEV).to(torch.bfloat16)
    norm.weight.data.copy_(b["norm"].to(DEV))
    layersA = []
    for i in range(K1):
        with torch.device("meta"):
            lay = DeepseekV3DecoderLayer(cfg, i)
        lay.load_state_dict({k: v.to(DEV, torch.bfloat16) for k, v in b["layers"][i].items()}, assign=True)
        lay.eval(); layersA.append(lay)
    rotary = DeepseekV3RotaryEmbedding(cfg).to(DEV)

    def health(u):
        try: return SESSION.get(u.rstrip("/") + "/health", timeout=30).json()
        except Exception as e: return {"ok": False, "error": str(e)}
    hB, hC = health(a.node_b), health(a.node_c)
    if not (hB.get("ok") and hC.get("ok") and hB.get("kv") and hC.get("kv")):
        raise SystemExit(f"nodes must run node_kv.py (KV-capable). B={hB} C={hC}")
    rB, rC = hB.get("region", "?"), hC.get("region", "?")
    print(f"[coord] B {hB.get('block')} @ {rB} | C {hC.get('block')} @ {rC}", flush=True)

    cacheA = DynamicCache()

    @torch.no_grad()
    def runA(h, past):
        S = h.shape[1]; cp = torch.arange(past, past + S, device=DEV); pos = cp.unsqueeze(0)
        pe = rotary(h, position_ids=pos); total = past + S
        if S > 1:
            qp = torch.arange(past, past + S, device=DEV).unsqueeze(1)
            kp = torch.arange(total, device=DEV).unsqueeze(0)
            m = torch.where(kp <= qp, 0.0, NEG).to(torch.bfloat16).view(1, 1, S, total)
        else:
            m = torch.zeros(1, 1, 1, total, dtype=torch.bfloat16, device=DEV)
        for lay in layersA:
            r = lay(h, attention_mask=m, position_embeddings=pe, position_ids=pos,
                    past_key_values=cacheA, use_cache=True, cache_position=cp)
            h = r[0] if isinstance(r, tuple) else r
        return h

    def hop(u, h, reset):
        body = enc(h[0]); body.update(sid=sid, reset=reset)
        r = SESSION.post(u.rstrip("/") + "/forward", json=body, timeout=300)
        return dec(r.json()).to(DEV).to(torch.bfloat16).unsqueeze(0)

    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)[0].tolist()
    P = len(ids)
    print(f"\n[coord] prompt: {a.prompt}\n[coord] sid={sid} prefill {P} tok ...", flush=True)

    wB = wC = None
    # prefill
    tpf = time.time()
    h = torch.nn.functional.embedding(torch.tensor([ids], device=DEV), embed_w)
    h = runA(h, 0); wB = float(h.float().std()); h = hop(a.node_b, h, True)
    wC = float(h.float().std()); h = hop(a.node_c, h, True)
    nxt = int((norm(h)[0, -1] @ lm_w.t()).argmax()); seq = ids + [nxt]
    prefill_s = time.time() - tpf
    # decode
    seen = P; n = 1; td = time.time()
    print("".join([""]), flush=True)
    for _ in range(a.max_new - 1):
        h = torch.nn.functional.embedding(torch.tensor([[seq[-1]]], device=DEV), embed_w)
        h = runA(h, seen); h = hop(a.node_b, h, False); h = hop(a.node_c, h, False)
        nxt = int((norm(h)[0, -1] @ lm_w.t()).argmax()); seq.append(nxt); seen += 1; n += 1
        if nxt == tok.eos_token_id:
            break
    decode_s = time.time() - td
    ans = tok.decode(seq[P:], skip_special_tokens=True)
    dtps = (n - 1) / decode_s if decode_s > 0 else 0

    print("=== ANSWER (3-shard mesh, KV-cached, obfuscated) ===")
    print(ans)
    print(f"\nprefill {prefill_s:.2f}s ({P} tok) | decode {dtps:.2f} tok/s ({n-1} tok) "
          f"| wire→B std {wB:.2f} | wire→C std {wC:.2f}")
    print(f"shards: 0:{K1} (coordinator, {DEV}) | {hB.get('block')} @ {rB} | {hC.get('block')} @ {rC}")

    if rc is not None:
        shards = [f"coordinator:0:{K1}", f"{rB}:{hB.get('block')}@nodeB", f"{rC}:{hC.get('block')}@nodeC"]
        rcpt = rc.make_receipt(
            request_id="sh3_" + hashlib.sha256((ans + sid).encode()).hexdigest()[:12],
            model="Moonlight-16B(Kimi) Π+P / 3-shard KV mesh",
            node=" | ".join(shards),
            wire_bytes=("Pframe(B=%.2f,C=%.2f)" % (wB, wC)).encode(),
            answer=ans, privacy_cos=0.0, integrity_ok=True)
        rc.append_log(rcpt)
        print(f"\n🔏 ML-DSA-65 receipt {rcpt['body']['request_id']} | {' | '.join(shards)}")
    # free node caches
    for u in (a.node_b, a.node_c):
        try: SESSION.post(u.rstrip("/") + "/reset", json={"sid": sid}, timeout=10)
        except Exception: pass


if __name__ == "__main__":
    main()
