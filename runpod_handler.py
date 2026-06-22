#!/usr/bin/env python
"""RunPod SERVERLESS worker — the untrusted blind node (Π+P obfuscation tier).

Runs the Π+P scrambled Moonlight-16B (= Kimi/DeepSeek-V3 arch) as a STOCK vLLM token->token
model. Receives PERMUTED token IDs + sampling params, returns PERMUTED output token IDs. It has
NO tokenizer, NO Π/Π^-1, NO covariant key P — it cannot reconstruct the prompt (see phase1_gate:
true-token recovery 0.033% for exactly this no-keys adversary). One request = one full generation
-> stateless -> scales to zero between requests (the cost-when-off win).

SECURITY INVARIANT: the network volume mounted here holds ONLY perm_scrambled_hf/ (scrambled
weights). NEVER place perm_keys.pt, the tokenizer custom-code, or coord state on this worker.

Deploy: build runpod_serverless.Dockerfile, attach a network volume with perm_scrambled_hf/ at
/runpod-volume, set GPU = 48GB (L40S/A40). Input/output contract:
  in : {"prompt_token_ids":[int], "max_tokens":256, "temperature":0.0, "stop_token_ids":[int]|None}
  out: {"token_ids":[int], "out_tokens":int, "gen_seconds":float, "tps":float}
"""
import os, time
import runpod
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CKPT = os.environ.get("CKPT", "/runpod-volume/perm_scrambled_hf")
llm = None
NODE = {}      # real region/gpu of THIS worker, detected once at boot (honest, not a default)


def _detect_node():
    """Where is this blind worker actually running? RunPod env first, geo-IP fallback. Cached."""
    info = {"dc": os.environ.get("RUNPOD_DC_ID") or os.environ.get("RUNPOD_DATACENTER"),
            "gpu": os.environ.get("RUNPOD_GPU_NAME") or os.environ.get("RUNPOD_GPU_TYPE"),
            "region": None, "country": None, "city": None}
    try:
        import urllib.request, json as _json
        g = _json.load(urllib.request.urlopen(
            "http://ip-api.com/json/?fields=status,country,countryCode,regionName,city", timeout=4))
        if g.get("status") == "success":
            info.update(country=g.get("country"), region=g.get("regionName"), city=g.get("city"),
                        cc=g.get("countryCode"))
    except Exception as e:
        print("[node] geo detect skipped:", e, flush=True)
    # human label, e.g. "Amsterdam, Netherlands (NL)" or the RunPod DC id
    loc = ", ".join([x for x in (info.get("city"), info.get("country")) if x]) or info.get("dc") or "unknown"
    info["label"] = loc + (f" ({info['cc']})" if info.get("cc") else "")
    print("[node] running in:", info["label"], "| gpu:", info.get("gpu"), flush=True)
    return info


def _boot():
    global llm, NODE
    NODE = _detect_node()
    print(f"[node] loading scrambled checkpoint {CKPT} (fp8) ...", flush=True)
    # NOTE: fp8 here REQUIRES native-fp8 GPUs (sm_89 Ada / sm_90 Hopper / Blackwell). On Ampere
    # (A40 sm_86, no hw fp8) vLLM's emulated fp8 corrupts the P-scrambled logits -> repetition
    # collapse / garbage. Pin the endpoint to ADA_48_PRO (see runpod_deploy --gpu-ids) or switch
    # QUANT=bf16 (16B in bf16 ~32GB, fits 48GB) for a GPU-agnostic, token-exact path.
    quant = os.environ.get("QUANT", "fp8")
    kw = dict(model=CKPT, skip_tokenizer_init=True, trust_remote_code=True, dtype="bfloat16",
              quantization=None if quant in ("bf16", "none") else quant,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.92")),
              max_model_len=int(os.environ.get("MAXLEN", "4096")))
    spec = int(os.environ.get("SPEC_NGRAM", "0"))          # permutation-invariant n-gram spec, obf-safe
    if spec > 0:
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": spec,
                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
        kw["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}
    llm = LLM(**kw)
    # warmup: trigger decode graphs/JIT before first real job (arbitrary permuted-space IDs)
    llm.generate(TokensPrompt(prompt_token_ids=[10, 20, 30, 40]),
                 SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
    print("[node] ready: blind vLLM node (permuted IDs in/out, no keys)", flush=True)


def handler(job):
    inp = job.get("input", {})
    ids = inp.get("prompt_token_ids")
    if not ids:
        return {"error": "prompt_token_ids required (permuted token IDs from the coordinator)"}
    sp = SamplingParams(temperature=float(inp.get("temperature", 0.0)),
                        max_tokens=int(inp.get("max_tokens", 256)),
                        stop_token_ids=inp.get("stop_token_ids"))
    t0 = time.time()
    out = llm.generate(TokensPrompt(prompt_token_ids=ids), sp, use_tqdm=False)
    dt = time.time() - t0
    tok_ids = list(out[0].outputs[0].token_ids)
    return {"token_ids": tok_ids, "out_tokens": len(tok_ids), "gen_seconds": round(dt, 3),
            "tps": round(len(tok_ids) / dt, 1) if dt > 0 else None,
            "node_region": NODE.get("label"), "node_gpu": NODE.get("gpu"),
            "node_cc": NODE.get("cc"), "node_city": NODE.get("city"), "node_country": NODE.get("country")}


if __name__ == "__main__":
    _boot()
    runpod.serverless.start({"handler": handler})
