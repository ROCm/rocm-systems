#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

"""Generate the synthetic Qwen3-32B manifest used by the reshard study."""

import argparse
import json
from pathlib import Path


def tensor(shape: list[int]) -> dict[str, object]:
    return {"shape": shape, "dtype": "BF16", "shard": "synthetic"}


def dimension_product(shape: object) -> int:
    result = 1
    for dimension in shape:  # type: ignore[union-attr]
        result *= int(dimension)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    layers = 64
    hidden = 5120
    intermediate = 25600
    query = 8192
    key_value = 1024
    head_dim = 128
    vocab = 151936

    tensors: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": tensor([vocab, hidden]),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": tensor([hidden]),
                f"{prefix}.self_attn.q_proj.weight": tensor([query, hidden]),
                f"{prefix}.self_attn.q_norm.weight": tensor([head_dim]),
                f"{prefix}.self_attn.k_proj.weight": tensor([key_value, hidden]),
                f"{prefix}.self_attn.k_norm.weight": tensor([head_dim]),
                f"{prefix}.self_attn.v_proj.weight": tensor([key_value, hidden]),
                f"{prefix}.self_attn.o_proj.weight": tensor([hidden, query]),
                f"{prefix}.post_attention_layernorm.weight": tensor([hidden]),
                f"{prefix}.mlp.gate_proj.weight": tensor([intermediate, hidden]),
                f"{prefix}.mlp.up_proj.weight": tensor([intermediate, hidden]),
                f"{prefix}.mlp.down_proj.weight": tensor([hidden, intermediate]),
            }
        )
    tensors["model.norm.weight"] = tensor([hidden])
    tensors["lm_head.weight"] = tensor([vocab, hidden])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tensors, indent=2) + "\n", encoding="utf-8")
    elements = sum(dimension_product(metadata["shape"]) for metadata in tensors.values())
    print(f"wrote {len(tensors)} tensors, {elements * 2:,} BF16 bytes -> {args.output}")


if __name__ == "__main__":
    main()
