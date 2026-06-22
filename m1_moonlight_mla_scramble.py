#!/usr/bin/env python
"""M1 — covariant-obfuscation loader for the DeepSeek-V3 / Kimi-K2 architecture,
validated on Moonlight-16B-A3B (model_type=deepseek_v3 == Kimi K2 architecture).

Extends the GQA loader (M0) to MLA + MoE. Scrambles REAL HF weights so the node
runs a scrambled model on the P-frame residual stream and reproduces the plaintext
tokens, while seeing only noise. THE asset that makes Kimi-K2.7-private real (same
arch, just scaled). Math proven in phase0_mla_covariance.py + phase0_mla_moe_layer.py.

Keys (one shared P across all layers; per-layer A/R/O/T):
  P  orthogonal [hidden]                  residual key
  A  orthogonal [kv_lora]   per layer     KV-latent (orthogonal: commutes w/ kv_a_layernorm)
  R  rope-commuting rot [qk_rope]/layer   decoupled-RoPE key (shared across heads)
  O  orthogonal [qk_nope]   per head      NoPE content (preserves q·k)
  T  invertible [v_head]    per head      value scramble (folds out at o_proj)
  Perm permutation [moe_inter]/expert     SwiGLU hidden (commutes w/ silu*)

Validation: token-argmax(scrambled(eP)) == token-argmax(plaintext(e)). CPU bf16 by
default (16B doesn't fit 32GB GPU in bf16). Writes ../m1_moonlight_mla_scramble.json
"""
import json, math, os, time
import torch

MODEL = os.environ.get("MOONLIGHT", "moonshotai/Moonlight-16B-A3B-Instruct")
PROMPT = "Explain why secret sharing keeps data private during AI inference."
NEW = int(os.environ.get("M1_TOKENS", "16"))
DEV = os.environ.get("M1_DEV", "cpu")          # 16B bf16 ~32GB -> CPU safest


def commuting_rot(hd, rng):
    # DeepSeek-V3 / Kimi arrange RoPE dims INTERLEAVED (planes (2i, 2i+1)), unlike
    # Qwen's half-split. Verified empirically: interleaved R̂ commutes -> token-exact.
    phi = torch.rand(hd // 2, generator=rng, dtype=torch.float64) * 6.28318530718
    M = torch.eye(hd, dtype=torch.float64)
    for i in range(hd // 2):
        c, s = phi[i].cos(), phi[i].sin(); a, b = 2 * i, 2 * i + 1
        M[a, a] = c; M[a, b] = -s; M[b, a] = s; M[b, b] = c
    return M


def orth(n, rng):
    return torch.linalg.qr(torch.randn(n, n, generator=rng, dtype=torch.float64))[0]


def invertible(n, rng):
    q1, q2 = orth(n, rng), orth(n, rng)
    d = 0.5 + 1.5 * torch.rand(n, generator=rng, dtype=torch.float64)
    return q1 @ torch.diag(d) @ q2


def f64(t): return t.detach().cpu().to(torch.float64)   # transforms on CPU, copy_ back to device


@torch.no_grad()
def scramble(model, cfg, seed=0):
    rng = torch.Generator().manual_seed(seed)
    H = cfg.hidden_size
    nope, rope_d, vd = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    nh, kvr = cfg.num_attention_heads, cfg.kv_lora_rank
    P = orth(H, rng)
    layers = model.model.layers
    for li, ly in enumerate(layers):
        at = ly.self_attn
        g_in = f64(ly.input_layernorm.weight)
        g_post = f64(ly.post_attention_layernorm.weight)
        Dgi = torch.diag(g_in)
        ly.input_layernorm.weight.fill_(1.0)        # g_in folded into q/kv -> make norm bare
        ly.post_attention_layernorm.weight.fill_(1.0)  # g_post folded into router/experts
        A = orth(kvr, rng)                                   # orthogonal (kv_a_layernorm)
        R = commuting_rot(rope_d, rng)
        O = [orth(nope, rng) for _ in range(nh)]
        T = [invertible(vd, rng) for _ in range(nh)]
        Ti = [torch.linalg.inv(t) for t in T]

        # ---- q_proj: [nh*(nope+rope), H]; per head nope@O, rope@R; fold g_in,P on in ----
        qk = nope + rope_d
        Wq = f64(at.q_proj.weight)                            # [nh*qk, H]
        Mq = Wq.t().reshape(H, nh, qk).clone()               # [H, nh, qk]
        for i in range(nh):
            Mq[:, i, :nope] = P.t() @ Dgi @ Mq[:, i, :nope] @ O[i]
            Mq[:, i, nope:] = P.t() @ Dgi @ Mq[:, i, nope:] @ R
        at.q_proj.weight.copy_(Mq.reshape(H, nh * qk).t().to(at.q_proj.weight.dtype))

        # ---- kv_a_proj_with_mqa: [kvr+rope, H]; cKV@A, k_rot@R ----
        Wkva = f64(at.kv_a_proj_with_mqa.weight)             # [kvr+rope, H]
        Mkva = Wkva.t().clone()                              # [H, kvr+rope]
        Mkva[:, :kvr] = P.t() @ Dgi @ Mkva[:, :kvr] @ A
        Mkva[:, kvr:] = P.t() @ Dgi @ Mkva[:, kvr:] @ R
        at.kv_a_proj_with_mqa.weight.copy_(Mkva.t().to(at.kv_a_proj_with_mqa.weight.dtype))

        # ---- kv_a_layernorm gamma -> fold into kv_b_proj; bare ----
        g_kva = f64(at.kv_a_layernorm.weight); Dgk = torch.diag(g_kva)
        at.kv_a_layernorm.weight.fill_(1.0)

        # ---- kv_b_proj: [nh*(nope+vd), kvr]; in: A^T diag(g_kva); out per head nope@O, v@T ----
        Wkvb = f64(at.kv_b_proj.weight)                      # [nh*(nope+vd), kvr]
        Mkvb = Wkvb.t().reshape(kvr, nh, nope + vd).clone()  # [kvr, nh, nope+vd]
        AtDg = A.t() @ Dgk                                   # [kvr,kvr]
        for i in range(nh):
            Mkvb[:, i, :nope] = AtDg @ Mkvb[:, i, :nope] @ O[i]
            Mkvb[:, i, nope:] = AtDg @ Mkvb[:, i, nope:] @ T[i]
        at.kv_b_proj.weight.copy_(Mkvb.reshape(kvr, nh * (nope + vd)).t().to(at.kv_b_proj.weight.dtype))

        # ---- o_proj: [H, nh*vd]; per head in: T^-1; out: P ----
        Wo = f64(at.o_proj.weight)                           # [H, nh*vd]
        Mo = Wo.t().reshape(nh, vd, H).clone()               # [nh, vd, H]
        for i in range(nh):
            Mo[i] = Ti[i] @ Mo[i] @ P
        at.o_proj.weight.copy_(Mo.reshape(nh * vd, H).t().to(at.o_proj.weight.dtype))

        # ---- post_attention_layernorm gamma -> fold into MLP/MoE; bare ----
        Dgp = torch.diag(g_post)
        ly.post_attention_layernorm.weight.fill_(1.0)

        mlp = ly.mlp
        if mlp.__class__.__name__.endswith("MoE"):
            # router gate: logits preserved -> weight @ diag(g_post) @ P
            gw = f64(mlp.gate.weight)                         # [n_exp, H]
            mlp.gate.weight.copy_((gw @ Dgp @ P).to(mlp.gate.weight.dtype))
            # routed experts (fused): gate_up_proj [n_exp, 2*inter, H]; down_proj [n_exp, H, inter]
            ex = mlp.experts
            gup = f64(ex.gate_up_proj); dwn = f64(ex.down_proj)
            n_exp = gup.shape[0]; inter = dwn.shape[-1]
            for e in range(n_exp):
                perm = torch.randperm(inter, generator=rng)
                Pm = torch.eye(inter, dtype=torch.float64)[perm]    # row-perm matrix
                W = gup[e]                                    # [2*inter, H]
                Wg, Wu = W[:inter], W[inter:]                 # each [inter, H]
                Wg = Pm @ (Wg @ Dgp @ P)                      # [inter,H]
                Wu = Pm @ (Wu @ Dgp @ P)
                gup[e] = torch.cat([Wg, Wu], 0)
                Wd = dwn[e]                                   # [H, inter]
                dwn[e] = P.t() @ Wd @ Pm.t()                 # [H, inter]
            ex.gate_up_proj.copy_(gup.to(ex.gate_up_proj.dtype))
            ex.down_proj.copy_(dwn.to(ex.down_proj.dtype))
            # shared expert (DeepseekV3MLP)
            scramble_mlp(mlp.shared_experts, Dgp, P, rng)
        else:
            scramble_mlp(mlp, Dgp, P, rng)                   # dense layer (first_k_dense_replace)

    # ---- embed -> E P ; final norm -> fold into lm_head ; lm_head unscramble P ----
    # chunk the 163k-vocab matmuls so fp64 temporaries stay small (RAM-safe)
    g_f = f64(model.model.norm.weight); Dgf = torch.diag(g_f)
    Pf = P.to(torch.float32)
    def chunk_rows(W, right):                                # W[vocab,H] @ right[H,H], chunked
        V = W.shape[0]; out = torch.empty_like(W)
        for s in range(0, V, 8192):
            out[s:s+8192] = (W[s:s+8192].to(torch.float32) @ right).to(W.dtype)
        return out
    ew = model.model.embed_tokens.weight
    ew.copy_(chunk_rows(ew, Pf))
    lw = model.lm_head.weight                                # [vocab, H]
    lw.copy_(chunk_rows(lw, (Dgf.to(torch.float32) @ Pf)))
    model.model.norm.weight.fill_(1.0)
    return P


@torch.no_grad()
def scramble_mlp(mlp, Dgp, P, rng):
    Wg = f64(mlp.gate_proj.weight); Wu = f64(mlp.up_proj.weight); Wd = f64(mlp.down_proj.weight)
    inter = Wg.shape[0]
    perm = torch.randperm(inter, generator=rng); Pm = torch.eye(inter, dtype=torch.float64)[perm]
    mlp.gate_proj.weight.copy_((Pm @ Wg @ Dgp @ P).to(mlp.gate_proj.weight.dtype))
    mlp.up_proj.weight.copy_((Pm @ Wu @ Dgp @ P).to(mlp.up_proj.weight.dtype))
    mlp.down_proj.weight.copy_((P.t() @ Wd @ Pm.t()).to(mlp.down_proj.weight.dtype))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"== load {MODEL} on {DEV} (bf16) ==", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cpu"}).eval()                # all on CPU: no offload, no meta tensors
    cfg = model.config
    DEV2 = torch.device("cpu")
    print(f"  loaded in {time.time()-t0:.0f}s | layers {cfg.num_hidden_layers} | "
          f"experts {cfg.n_routed_experts}+{cfg.n_shared_experts}sh | MLA kvr {cfg.kv_lora_rank}", flush=True)

    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV2)

    @torch.no_grad()
    def gen():
        return model.generate(ids, max_new_tokens=NEW, do_sample=False)[0, ids.shape[1]:]

    print("== plaintext generate ==", flush=True)
    tp = time.time(); tok_plain = gen(); print(f"  {time.time()-tp:.0f}s", flush=True)
    lg_plain = model(ids).logits[0, -1].float()

    print("== scramble real weights (MLA+MoE) ==", flush=True)
    ts = time.time(); scramble(model, cfg); print(f"  scrambled in {time.time()-ts:.0f}s", flush=True)

    print("== scrambled generate (on e P) ==", flush=True)
    tok_scr = gen()
    lg_scr = model(ids).logits[0, -1].float()

    tok_match = bool(torch.equal(tok_scr, tok_plain))
    same_top = bool(lg_scr.argmax() == lg_plain.argmax())
    rel = float((lg_scr - lg_plain).abs().max() / (lg_plain.abs().max() + 1e-6))
    res = {"model": MODEL, "arch": cfg.model_type, "layers": cfg.num_hidden_layers,
           "experts": f"{cfg.n_routed_experts}+{cfg.n_shared_experts}", "kv_lora_rank": cfg.kv_lora_rank,
           "tokens_checked": NEW, "greedy_token_exact_match": tok_match,
           "argmax_match": same_top, "logit_rel_err": round(rel, 5),
           "plain_text": tok.decode(tok_plain, skip_special_tokens=True)[:200],
           "scrambled_text": tok.decode(tok_scr, skip_special_tokens=True)[:200],
           "verdict": ("M1 PASS if token-exact: covariant obfuscation works on the DeepSeek-V3/"
                       "Kimi-K2 architecture (MLA+MoE) on REAL weights -> Kimi-K2.7-private is "
                       "the same loader scaled. Node runs scrambled model on e P, sees noise.")}
    print(json.dumps({k: v for k, v in res.items() if k not in ("plain_text", "scrambled_text")}, indent=2))
    print("  plain:", res["plain_text"][:120])
    print("  scram:", res["scrambled_text"][:120])
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "m1_moonlight_mla_scramble.json"), "w"), indent=2)
    print(f"\nGATE: token-exact={tok_match} argmax={same_top} rel-err={rel:.2e}")


if __name__ == "__main__":
    main()
