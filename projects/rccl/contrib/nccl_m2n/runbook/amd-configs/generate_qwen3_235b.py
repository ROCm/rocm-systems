#!/usr/bin/env python3
"""Synthetic Qwen3-235B-A22B manifest (first N of 94 layers, 128 experts, BF16) for reshard_model_bench."""
import argparse, json, math
from pathlib import Path

def t(shape): return {"shape": shape, "dtype": "BF16", "shard": "synthetic"}

p = argparse.ArgumentParser(); p.add_argument("output", type=Path); p.add_argument("--layers", type=int, default=4)
a = p.parse_args()
hidden, moe_inter, experts, q, kv, hd, vocab = 4096, 1536, 128, 8192, 512, 128, 151936
m = {"model.embed_tokens.weight": t([vocab, hidden])}
for l in range(a.layers):
    pre = f"model.layers.{l}"
    m.update({f"{pre}.input_layernorm.weight": t([hidden]),
              f"{pre}.self_attn.q_proj.weight": t([q, hidden]), f"{pre}.self_attn.q_norm.weight": t([hd]),
              f"{pre}.self_attn.k_proj.weight": t([kv, hidden]), f"{pre}.self_attn.k_norm.weight": t([hd]),
              f"{pre}.self_attn.v_proj.weight": t([kv, hidden]), f"{pre}.self_attn.o_proj.weight": t([hidden, q]),
              f"{pre}.post_attention_layernorm.weight": t([hidden]), f"{pre}.mlp.gate.weight": t([experts, hidden])})
    for e in range(experts):
        m.update({f"{pre}.mlp.experts.{e}.gate_proj.weight": t([moe_inter, hidden]),
                  f"{pre}.mlp.experts.{e}.up_proj.weight": t([moe_inter, hidden]),
                  f"{pre}.mlp.experts.{e}.down_proj.weight": t([hidden, moe_inter])})
m["model.norm.weight"] = t([hidden]); m["lm_head.weight"] = t([vocab, hidden])
a.output.write_text(json.dumps(m, indent=2) + "\n")
print(f"wrote {len(m)} tensors, {sum(math.prod(v['shape']) for v in m.values()) * 2 / 1e9:.1f} GB BF16 -> {a.output}")
