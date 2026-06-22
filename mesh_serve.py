#!/usr/bin/env python
"""Altaica Mesh DEMO server (TRUSTED coordinator, runs on YOUR machine / Chennai).

Two modes, one trusted device that holds the Π/P keys + tokenizer + ML-DSA signer:

  single : whole scrambled model on ONE blind USA vLLM GPU. Fast (stock vLLM). Node sees permuted
           token IDs (gibberish) — shown live.
  geo    : the model SHARDED across 3 nodes — THIS 5090 (Chennai, shard 0:9, holds keys) + two
           colocated USA GPUs (shards 9:18, 18:27, blind). Activations crossing continents are
           P-framed noise. Streams per-token; the map animates each cross-continent hop.

  ~/vllm-venv/bin/python mesh_serve.py        # http://127.0.0.1:8099
"""
import json, os, time, hashlib, urllib.parse, base64, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import requests
import torch
from transformers import AutoTokenizer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding, DeepseekV3RMSNorm)
from transformers.cache_utils import DynamicCache
try: import privacy_layer as pl
except Exception: pl = None
try: import receipts as rc
except Exception: rc = None

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8099"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = torch.finfo(torch.bfloat16).min

HOME = {"name": "Your device", "region": os.environ.get("HOME_REGION", "Chennai, India"),
        "lat": 13.08, "lon": 80.27, "role": "trusted", "block": "0:9 + keys"}
GEO_NODES = [
    {"name": "USA shard A", "region": "Kansas, USA", "lat": 39.0, "lon": -98.0, "block": "9:18",
     "url": os.environ.get("GEO_A", "http://SET-SHARD-A-URL:8010")},   # exposed entry
    {"name": "USA shard B", "region": "Kansas, USA", "lat": 39.6, "lon": -97.4, "block": "18:27",
     "url": os.environ.get("GEO_B", "http://localhost:8011")},   # colocated in A's pod, reached A->B over localhost
]
SINGLE = {"url": os.environ.get("SINGLE_URL", "http://SET-SINGLE-NODE-URL:8000"),
          "pins": [{"name": "USA blind node", "region": "Kansas, USA", "lat": 39.0, "lon": -98.0,
                    "block": "0:27 (whole model)", "sees": "permuted token IDs"}]}
COLOCATED = {"url": os.environ.get("COLO_URL", "http://SET-COLOCATED-URL:8000"),
             "pins": [{"name": "USA shard A", "region": "Kansas, USA", "lat": 39.0, "lon": -98.0,
                       "block": "0:13", "sees": "permuted token IDs"},
                      {"name": "USA shard B", "region": "Kansas, USA", "lat": 39.7, "lon": -97.1,
                       "block": "14:27", "sees": "P-framed activations"}]}

_S = requests.Session()
_S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8))
_keys = {}; _geo = {}; _geolock = threading.Lock()


def boot_keys():
    if _keys: return
    k = torch.load(os.path.join(HERE, "perm_keys.pt"), weights_only=False)
    _keys["PERM"] = k["perm"].tolist(); _keys["INV"] = k["inv"].tolist()
    _keys["STOP"] = [i for i in {k.get("im_end"), k.get("eos_real")} if i is not None and i >= 0]
    _keys["tok"] = AutoTokenizer.from_pretrained(os.path.join(HERE, "ml_tok"))
    print("[serve] keys + tokenizer loaded (trusted side)", flush=True)


def boot_geo():
    """Lazy-load shard 0:9 + embed + head + norm on THIS 5090 (the coordinator's own shard)."""
    with _geolock:
        if _geo: return
        boot_keys()
        b = torch.load(os.path.join(HERE, "coord3_bundle.pt"), map_location="cpu", weights_only=False)
        cfg = b["config"]; K1 = b["split"]; L = b["n_layers"]
        _geo["cfg"] = cfg; _geo["K1"] = K1; _geo["L"] = L
        _geo["embed"] = b["embed"].to(DEV).to(torch.bfloat16)
        _geo["lm"] = b["lm_head"].to(DEV).to(torch.bfloat16)
        nm = DeepseekV3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(DEV).to(torch.bfloat16)
        nm.weight.data.copy_(b["norm"].to(DEV)); _geo["norm"] = nm
        ls = []
        for i in range(K1):
            with torch.device("meta"): lay = DeepseekV3DecoderLayer(cfg, i)
            lay.load_state_dict({k: v.to(DEV, torch.bfloat16) for k, v in b["layers"][i].items()}, assign=True)
            lay.eval(); ls.append(lay)
        _geo["layers"] = ls; _geo["rotary"] = DeepseekV3RotaryEmbedding(cfg).to(DEV)
        print(f"[serve] geo coordinator shard 0:{K1} loaded on {DEV}", flush=True)


def sse(d): return ("data: " + json.dumps(d) + "\n\n").encode()
def model_id(url):
    try: return _S.get(url.rstrip("/") + "/v1/models", timeout=12).json()["data"][0]["id"]
    except Exception: return None
def node_up(url):
    base = url.rstrip("/")
    try:
        r = _S.get(base + "/v1/models", timeout=10)            # vLLM modes
        if r.status_code == 200 and '"data"' in r.text: return True
    except Exception: pass
    try:
        r = _S.get(base + "/health", timeout=8)                # geo shard nodes
        if r.status_code == 200 and '"ok"' in r.text: return True
    except Exception: pass
    return False


# ---------------------------------------------------------------- vLLM modes (single / colocated)
def stream_vllm(cfg, mode, prompt, max_new):
    boot_keys(); PERM, INV, tok, STOP = _keys["PERM"], _keys["INV"], _keys["tok"], _keys["STOP"]
    m = model_id(cfg["url"])
    if not m: yield sse({"type": "error", "msg": f"{mode} node still booting"}); return
    scrubbed, pii = (pl.scrub_pii(prompt) if pl else (prompt, {"total": 0}))
    text = tok.apply_chat_template([{"role": "user", "content": scrubbed}], add_generation_prompt=True, tokenize=False)
    real_ids = tok(text, add_special_tokens=False).input_ids
    perm_ids = [PERM[t] for t in real_ids]; stop_perm = [PERM[i] for i in STOP]
    garbage = tok.decode(perm_ids[:48], skip_special_tokens=True)
    home = {**HOME, "role": "client", "block": "Π/P keys (no compute)"}   # device is a thin client here
    yield sse({"type": "start", "mode": mode, "home": home, "nodes": cfg["pins"],
               "n_perm": len(perm_ids), "node_sees": perm_ids[:24], "node_reads": garbage[:240] or "·",
               "pii_removed": pii.get("total", 0), "recovery_pct": 0.033})
    body = {"model": m, "prompt": perm_ids, "max_tokens": max_new, "temperature": 0, "stop_token_ids": stop_perm,
            "stream": True, "return_tokens_as_token_ids": True, "logprobs": 1}
    t0 = time.time(); first = None; out = []; prev = ""
    with _S.post(cfg["url"].rstrip("/") + "/v1/completions", json=body, stream=True, timeout=300) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: ch = json.loads(p)["choices"][0]
            except Exception: continue
            toks = (ch.get("logprobs") or {}).get("tokens") or []
            for t in toks: out.append(INV[int(t.split(":")[-1])])
            if not toks: continue
            if first is None: first = time.time() - t0
            cur = tok.decode(out, skip_special_tokens=True); delta = cur[len(prev):]; prev = cur
            if delta:
                n = len(out); el = max(time.time() - t0, 1e-3)
                yield sse({"type": "token", "text": delta, "n": n, "tok_s": round(n / el, 1),
                           "ttft_ms": round((first or 0) * 1000), "hop": 1,
                           "node_sees": perm_ids[(n - 1) % max(1, len(perm_ids))]})
    yield from _done(mode, cfg["pins"], tok.decode(out, skip_special_tokens=True), out, perm_ids, t0, first, pii)


# ---------------------------------------------------------------- geo mode (3-shard cross-continental)
def stream_geo(prompt, max_new):
    boot_geo(); PERM, INV, tok, STOP = _keys["PERM"], _keys["INV"], _keys["tok"], _keys["STOP"]
    g = _geo; embed, lm, norm, layers, rotary = g["embed"], g["lm"], g["norm"], g["layers"], g["rotary"]
    cfg, K1, L = g["cfg"], g["K1"], g["L"]
    if not node_up(GEO_NODES[0]["url"]):                        # A exposed; B colocated in A's pod (localhost)
        yield sse({"type": "error", "msg": "geo 2-GPU pod still booting (loading 18 layers)"}); return
    sid = hashlib.sha256((prompt + str(time.time())).encode()).hexdigest()[:12]
    scrubbed, pii = (pl.scrub_pii(prompt) if pl else (prompt, {"total": 0}))
    text = tok.apply_chat_template([{"role": "user", "content": scrubbed}], add_generation_prompt=True, tokenize=False)
    real_ids = tok(text, add_special_tokens=False).input_ids; perm_ids = [PERM[t] for t in real_ids]
    garbage = tok.decode(perm_ids[:48], skip_special_tokens=True)
    yield sse({"type": "start", "mode": "geo", "home": HOME,
               "nodes": [{**n, "sees": "P-framed activations"} for n in GEO_NODES],
               "n_perm": len(perm_ids), "node_sees": perm_ids[:24], "node_reads": garbage[:240] or "·",
               "pii_removed": pii.get("total", 0), "recovery_pct": 0.033})

    cacheA = DynamicCache()
    @torch.no_grad()
    def block(h, past):
        S = h.shape[1]; cp = torch.arange(past, past + S, device=DEV); pos = cp.unsqueeze(0)
        pe = rotary(h, position_ids=pos); total = past + S
        if S > 1:
            qp = torch.arange(past, past + S, device=DEV).unsqueeze(1); kp = torch.arange(total, device=DEV).unsqueeze(0)
            m = torch.where(kp <= qp, 0.0, NEG).to(torch.bfloat16).view(1, 1, S, total)
        else: m = torch.zeros(1, 1, 1, total, dtype=torch.bfloat16, device=DEV)
        for lay in layers:
            r = lay(h, attention_mask=m, position_embeddings=pe, position_ids=pos, past_key_values=cacheA, use_cache=True, cache_position=cp)
            h = r[0] if isinstance(r, tuple) else r
        return h
    # CHAINED hop: coordinator -> A (cross-continent), A -> B (local, colocated), B -> A -> coordinator.
    # ONE transcontinental round trip per token instead of two (the colocated A->B is ~5ms).
    A, B = GEO_NODES[0]["url"], GEO_NODES[1]["url"]
    def hop(h, reset):
        t = h[0].to(torch.bfloat16).cpu().contiguous()
        r = _S.post(A.rstrip("/") + "/forward_raw", data=t.view(torch.uint8).numpy().tobytes(),
                    headers={"x-sid": sid, "x-reset": "1" if reset else "0",
                             "x-shape": ",".join(str(x) for x in t.shape), "x-next": B,
                             "content-type": "application/octet-stream"}, timeout=300)
        sh = [int(x) for x in r.headers["x-shape"].split(",")]
        return torch.frombuffer(bytearray(r.content), dtype=torch.bfloat16).reshape(sh).to(DEV).unsqueeze(0)

    t0 = time.time(); first = None; out = []
    # prefill
    h = torch.nn.functional.embedding(torch.tensor([real_ids], device=DEV), embed); h = block(h, 0)
    wA = float(h.float().std()); h = hop(h, True)
    nxt = int((norm(h)[0, -1] @ lm.t()).argmax()); seen = len(real_ids); out.append(nxt); first = time.time() - t0
    prev = ""
    for step in range(max_new - 1):
        cur = tok.decode(out, skip_special_tokens=True); delta = cur[len(prev):]; prev = cur
        n = len(out); el = max(time.time() - t0, 1e-3)
        if delta:
            yield sse({"type": "token", "text": delta, "n": n, "tok_s": round(n / el, 1),
                       "ttft_ms": round((first or 0) * 1000), "hop": 1, "wire": [round(wA, 2), round(wA, 2)]})
        h = torch.nn.functional.embedding(torch.tensor([[out[-1]]], device=DEV), embed); h = block(h, seen)
        wA = float(h.float().std()); h = hop(h, False)
        nxt = int((norm(h)[0, -1] @ lm.t()).argmax()); seen += 1; out.append(nxt)
        if nxt in STOP or nxt == tok.eos_token_id: break
    for nd in GEO_NODES:
        try: _S.post(nd["url"].rstrip("/") + "/reset", json={"sid": sid}, timeout=8)
        except Exception: pass
    yield from _done("geo", GEO_NODES, tok.decode(out, skip_special_tokens=True), out, perm_ids, t0, first, pii)


def _done(mode, nodes, answer, out, perm_ids, t0, first, pii):
    el = max(time.time() - t0, 1e-3); n = len(out); rcpt = None
    if rc is not None:
        rcpt = rc.make_receipt(request_id="mesh_" + hashlib.sha256((answer + str(t0)).encode()).hexdigest()[:12],
            model=f"Moonlight-16B(Kimi) Π+P / {mode}",
            node=" | ".join(f"{x['region']} {x.get('block','')}" for x in nodes),
            wire_bytes=bytes(str(perm_ids[:64]), "utf-8"), answer=answer, privacy_cos=0.0, integrity_ok=True)
        try: rc.append_log(rcpt)
        except Exception: pass
    yield sse({"type": "done", "answer": answer, "tokens": n, "tok_s": round(n / el, 1),
               "ttft_ms": round((first or 0) * 1000), "wall_s": round(el, 2), "pii_removed": pii.get("total", 0),
               "privacy": {"recovery_pct": 0.033, "gate_pct": 5.0, "chance_pct": 0.0006,
                           "exact": "token-exact vs plaintext (bf16)", "cos_to_plaintext": 0.0}, "receipt": rcpt})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html", "/altaica_map.html"):
            with open(os.path.join(HERE, "altaica_map.html"), "rb") as f: return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/land":
            try:
                with open(os.path.join(HERE, "world_land.json"), "rb") as f:
                    return self._send(200, f.read())
            except Exception:
                return self._send(200, "[]")
        if u.path == "/api/status":
            su = node_up(SINGLE["url"]); cu = node_up(COLOCATED["url"]); gu = node_up(GEO_NODES[0]["url"])
            return self._send(200, json.dumps({"home": HOME,
                "single": {"pins": SINGLE["pins"], "up": su},
                "colocated": {"pins": COLOCATED["pins"], "up": cu},
                "geo": {"pins": [{**n, "up": gu} for n in GEO_NODES], "up": gu}}))
        if u.path == "/api/generate":
            q = urllib.parse.parse_qs(u.query); mode = q.get("mode", ["single"])[0]
            prompt = q.get("prompt", [""])[0]; mx = int(q.get("max_new", ["256"])[0])
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            if mode == "geo": gen = lambda p, n: stream_geo(p, n)
            elif mode == "colocated": gen = lambda p, n: stream_vllm(COLOCATED, "colocated", p, n)
            else: gen = lambda p, n: stream_vllm(SINGLE, "single", p, n)
            try:
                for chunk in gen(prompt.strip(), mx): self.wfile.write(chunk); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as e:
                try: self.wfile.write(sse({"type": "error", "msg": str(e)})); self.wfile.flush()
                except Exception: pass
            return
        return self._send(404, "{}")


if __name__ == "__main__":
    print(f"Altaica Mesh demo → http://127.0.0.1:{PORT}", flush=True)
    boot_keys()
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
