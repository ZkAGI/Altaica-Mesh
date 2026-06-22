#!/usr/bin/env python
"""Home COORDINATOR for the RunPod serverless Π+P node (trusted side; holds the keys).

Per request: PII-scrub (trusted) -> tokenize -> Π -> ONE call to the RunPod serverless endpoint ->
Π^-1 -> detok -> ML-DSA-65 signed receipt. The serverless worker (runpod_handler.py) is the blind
node: permuted IDs in/out, no tokenizer, no keys. One request = one full generation -> the worker
scales to zero when idle (cost-when-off). Cold start (~1-3 min: spin + load 30GB) is handled by the
async run+poll path.

Config (env or ~/.altaica_mesh.json):  RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
Run standalone:  ~/vllm-venv/bin/python runpod_coord.py "your question" --max-new 256
"""
import argparse, hashlib, json, os, sys, time
import requests
import torch
from transformers import AutoTokenizer
import privacy_layer as pl
import receipts as rc

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = "moonshotai/Moonlight-16B-A3B-Instruct"
CFG = os.path.expanduser("~/.altaica_mesh.json")
_state = {}


def _cfg(k, default=None):
    if k in os.environ:
        return os.environ[k]
    if os.path.exists(CFG):
        return json.load(open(CFG)).get(k, default)
    return default


def _boot():
    if _state:
        return
    keys = torch.load(os.path.join(HERE, "perm_keys.pt"), weights_only=False)
    PERM, INV = keys["perm"].tolist(), keys["inv"].tolist()
    stop = [PERM[i] for i in {keys["im_end"], keys["eos_real"]} if i is not None and i >= 0]
    tok = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
    _state.update(PERM=PERM, INV=INV, STOP=stop, tok=tok)


def _endpoint():
    eid = _cfg("RUNPOD_ENDPOINT_ID")
    key = _cfg("RUNPOD_API_KEY")
    if not eid or not key:
        raise SystemExit("set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (env or ~/.altaica_mesh.json)")
    return f"https://api.runpod.ai/v2/{eid}", {"Authorization": f"Bearer {key}",
                                               "Content-Type": "application/json"}


def _req(method, url, *, hdr=None, json_body=None, retries=6, rtimeout=30):
    """RunPod HTTP with retry on transient DNS/connection blips (WSL dnsTunneling is flaky)."""
    last = None
    for i in range(retries):
        try:
            r = requests.request(method, url, headers=hdr, json=json_body, timeout=rtimeout)
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            time.sleep(min(0.5 * (2 ** i), 6))   # 0.5,1,2,4,6,6 — rides out a DNS hiccup
    raise RuntimeError(f"runpod unreachable after {retries} tries (transient DNS/network): {last}")


def call_node(perm_ids, max_tokens, temperature=0.0, timeout=300):
    """Async run + poll (survives serverless cold starts + transient DNS). Returns the worker's output dict."""
    base, hdr = _endpoint()
    body = {"input": {"prompt_token_ids": perm_ids, "max_tokens": max_tokens,
                      "temperature": temperature, "stop_token_ids": _state["STOP"]}}
    r = _req("POST", f"{base}/run", hdr=hdr, json_body=body)
    jid = r.get("id")
    if r.get("status") == "COMPLETED":          # rare: fast enough to return immediately
        return r["output"]
    if not jid:
        raise RuntimeError(f"runpod run failed: {r}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = _req("GET", f"{base}/status/{jid}", hdr=hdr)
        st = s.get("status")
        if st == "COMPLETED":
            return s["output"]
        if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"runpod job {st}: {s.get('error')}")
        time.sleep(2)
    raise TimeoutError(f"runpod job {jid} not done in {timeout}s (cold start stuck?)")


def generate(prompt, max_new=256, verbose=False):
    _boot()
    PERM, INV, tok = _state["PERM"], _state["INV"], _state["tok"]
    scrubbed, pii = pl.scrub_pii(prompt)                       # PII removed on the TRUSTED side
    prot = pl.protect(scrubbed)
    text = tok.apply_chat_template([{"role": "user", "content": scrubbed}],
                                   add_generation_prompt=True, tokenize=False)
    real_ids = tok(text, add_special_tokens=False).input_ids
    perm_ids = [PERM[t] for t in real_ids]                     # Π(prompt) -> node sees gibberish IDs
    if verbose:
        print(f"[pii] removed {pii['total']} {pii.get('counts')} | anchors {prot['anchors_protected']}")
        print(f"[node] sending {len(perm_ids)} permuted ids e.g. {perm_ids[:8]} — blind", flush=True)
    t0 = time.time()
    out = call_node(perm_ids, max_new)
    wire_dt = time.time() - t0
    perm_out = out["token_ids"]
    real_out = [INV[t] for t in perm_out]                     # Π^-1 (coordinator only)
    answer = tok.decode(real_out, skip_special_tokens=True)
    wire = bytes(str(perm_out[:64]), "utf-8")
    rcpt = rc.make_receipt(request_id="rp_" + hashlib.sha256((answer + str(t0)).encode()).hexdigest()[:12],
                           model="Moonlight-16B(Kimi) Π+P / RunPod-serverless",
                           node="runpod:serverless:perm-scrambled", wire_bytes=wire,
                           answer=answer, privacy_cos=0.0, integrity_ok=True)
    rc.append_log(rcpt); anchor = rc.maybe_anchor()
    return {"answer": answer, "pii": pii, "anchors_protected": prot["anchors_protected"],
            "node_tps": out.get("tps"), "node_seconds": out.get("gen_seconds"),
            "wall_seconds": round(wire_dt, 2), "out_tokens": out.get("out_tokens"),
            "receipt": rcpt, "anchor": anchor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt"); ap.add_argument("--max-new", dest="max_new", type=int, default=256)
    a = ap.parse_args()
    print(f"> {a.prompt}\n")
    r = generate(a.prompt, a.max_new, verbose=True)
    print("\n" + r["answer"])
    print(f"\n— node {r['node_tps']} tok/s ({r['node_seconds']}s compute) | wall {r['wall_seconds']}s | "
          f"{r['out_tokens']} tok | receipt {r['receipt']['body']['request_id']}")
    json.dump(r["receipt"], open(os.path.join(HERE, ".last_receipt.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
