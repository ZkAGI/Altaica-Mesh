#!/usr/bin/env python
"""Export scrambled Moonlight-16B into a coordinator bundle (5090) + node-B bundle (4090).
Scrambles once (M1 loader, one shared P), then splits the 27 layers at SPLIT.

  coord_bundle.pt (stays on 5090): embed(E·P), lm_head(scrambled), norm(bare), layers[0:SPLIT]
  nodeB_bundle.pt (copy to 4090) : layers[SPLIT:L]  (scrambled; sees only P-framed noise)
"""
import os, sys, time
import torch

import m1_moonlight_mla_scramble as m1

MODEL = os.environ.get("MODEL", "moonshotai/Moonlight-16B-A3B-Instruct")
SPLIT = int(os.environ.get("SPLIT", "13"))
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"== load + scramble {MODEL} (SPLIT={SPLIT}) ==", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.save_pretrained(os.path.join(HERE, "ml_tok"))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map={"": "cpu"}).eval()
    cfg = model.config
    t = time.time(); m1.scramble(model, cfg); print(f"  scrambled {time.time()-t:.0f}s", flush=True)
    L = cfg.num_hidden_layers
    mdl = model.model

    print("== save node-B bundle (layers %d:%d) ==" % (SPLIT, L), flush=True)
    torch.save({"config": cfg, "split": SPLIT, "n_layers": L,
                "layers": {i: mdl.layers[i].state_dict() for i in range(SPLIT, L)}},
               os.path.join(HERE, "nodeB_bundle.pt"))
    print("== save coordinator bundle (embed/head/norm + layers 0:%d) ==" % SPLIT, flush=True)
    torch.save({"config": cfg, "split": SPLIT, "n_layers": L,
                "embed": mdl.embed_tokens.weight.data.clone(),
                "lm_head": model.lm_head.weight.data.clone(),
                "norm": mdl.norm.weight.data.clone(),
                "layers": {i: mdl.layers[i].state_dict() for i in range(0, SPLIT)}},
               os.path.join(HERE, "coord_bundle.pt"))
    print(f"DONE -> {HERE}/coord_bundle.pt (5090)  +  nodeB_bundle.pt (copy to 4090)", flush=True)


if __name__ == "__main__":
    main()
