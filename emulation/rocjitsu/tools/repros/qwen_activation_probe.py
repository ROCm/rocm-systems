#!/usr/bin/env python3
"""Save or compare Qwen activations for rocjitsu numerical triage.

The Qwen DBT pass/fail signal should not be only "finite logits".  This helper
records layer hidden states, logits, and optionally selected layer-0 module
outputs so native gfx942 execution can be compared against gfx950 guest code
translated by rocjitsu.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

DEFAULT_INPUT_IDS = [9707, 11, 419, 374, 1296]
DEFAULT_LAYER0_HOOKS = (
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
)
LAYER_HOOK_SUFFIXES = (
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.conv1d",
    "linear_attn.act",
    "linear_attn.norm",
    "linear_attn.out_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_input_ids(raw: str) -> list[int]:
    return [int(part) for part in raw.replace(",", " ").split()]


def parse_layers(raw: str | None, layer_count: int) -> list[int]:
    if not raw:
        return list(range(layer_count))

    layers: list[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    result = sorted(set(layers))
    for layer in result:
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"layer index {layer} outside [0, {layer_count})")
    return result


def first_tensor(value: Any) -> torch.Tensor | None:
    """Extract the first tensor from common module-output containers."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().contiguous()


def lookup_saved_tensor(source: dict[str, Any], key: str) -> torch.Tensor:
    """Find a saved tensor by exact key in the probe's tensor containers."""

    for container_name in ("hooks", "layer_replay_hooks"):
        container = source.get(container_name)
        if isinstance(container, dict) and key in container:
            tensor = container[key]
            if isinstance(tensor, torch.Tensor):
                return tensor

    if key.startswith("hidden_states[") and key.endswith("]"):
        index = int(key.removeprefix("hidden_states[").removesuffix("]"))
        return source["hidden_states"][index]

    if key.startswith("layer_replay_outputs[") and key.endswith("]"):
        index = int(key.removeprefix("layer_replay_outputs[").removesuffix("]"))
        return source["layer_replay_outputs"][index]

    raise KeyError(f"could not find saved tensor key {key!r}")


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    return (
        AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval().cuda()
    )


def lookup_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    module = modules.get(name)
    if module is None:
        raise KeyError(f"could not find module {name!r}")
    return module


def hook_names_for_model(
    model: torch.nn.Module, hook_layer0: bool, hook_all_layers: bool
) -> list[str]:
    if hook_all_layers:
        layers = getattr(getattr(model, "model", None), "layers", [])
        return [
            f"model.layers.{idx}.{suffix}"
            for idx in range(len(layers))
            for suffix in LAYER_HOOK_SUFFIXES
        ]
    if hook_layer0:
        return list(DEFAULT_LAYER0_HOOKS)
    return []


def save_activations(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model = load_model(args)
    input_ids = torch.tensor(
        [parse_input_ids(args.input_ids)], device="cuda", dtype=torch.long
    )

    captures: dict[str, torch.Tensor] = {}
    hooks = []
    hook_names = hook_names_for_model(model, args.hook_layer0, args.hook_all_layers)
    if hook_names:
        modules = dict(model.named_modules())
        for name in hook_names:
            module = modules.get(name)
            if module is None:
                continue

            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                capture_name: str = name,
            ) -> None:
                tensor = first_tensor(output)
                if tensor is not None:
                    captures[f"hook:{capture_name}"] = tensor_to_cpu(tensor)

            hooks.append(module.register_forward_hook(hook))

    torch.cuda.synchronize()
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    torch.cuda.synchronize()

    for hook in hooks:
        hook.remove()

    result: dict[str, Any] = {
        "model": args.model,
        "attn_implementation": args.attn_implementation or "default",
        "input_ids": input_ids.cpu(),
        "device": torch.cuda.get_device_name(0),
        "logits": tensor_to_cpu(output.logits),
        "hidden_states": [tensor_to_cpu(tensor) for tensor in output.hidden_states],
        "hooks": captures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {args.output}", flush=True)
    print(f"device {result['device']}", flush=True)
    print(f"logits_shape {tuple(result['logits'].shape)}", flush=True)
    print(f"hidden_states {len(result['hidden_states'])}", flush=True)
    print(f"hooks {len(captures)}", flush=True)


def make_qwen_causal_masks(
    model: torch.nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> dict[str, torch.Tensor | None]:
    mask_kwargs = {
        "config": model.model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if model.model.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return masks


def replay_layers(args: argparse.Namespace) -> None:
    """Replay decoder layers from native-saved hidden states.

    `save` records hidden states after each layer.  Feeding layer `i` exactly
    `hidden_states[i]` removes accumulated upstream drift and tells us whether a
    single translated layer behaves differently on the same bf16 input.
    """

    torch.manual_seed(args.seed)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    model = load_model(args)
    saved_hidden = source["hidden_states"]
    layer_count = len(model.model.layers)
    if len(saved_hidden) < layer_count + 1:
        raise ValueError(
            f"{args.input} has {len(saved_hidden)} hidden states, need {layer_count + 1}"
        )

    layer_indices = parse_layers(args.layers, layer_count)
    input_ids = source.get("input_ids")
    if input_ids is None:
        input_ids = torch.tensor([parse_input_ids(args.input_ids)], dtype=torch.long)
    input_ids = input_ids.to(device="cuda", dtype=torch.long)
    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)

    outputs: dict[int, torch.Tensor] = {}
    reference_diffs: dict[int, tuple[float, float, float]] = {}
    captures: dict[str, torch.Tensor] = {}

    torch.cuda.synchronize()
    with torch.no_grad():
        for layer_idx in layer_indices:
            layer_start = time.perf_counter()
            hidden = saved_hidden[layer_idx].to(device="cuda", dtype=torch.bfloat16)
            if hidden.shape[1] != seq_len:
                raise ValueError(
                    f"layer {layer_idx} hidden seq_len {hidden.shape[1]} != input seq_len {seq_len}"
                )

            masks = make_qwen_causal_masks(model, hidden, position_ids)
            position_embeddings = model.model.rotary_emb(hidden, position_ids)

            hooks = []
            if args.hook_modules:
                modules = dict(model.model.layers[layer_idx].named_modules())
                for suffix in LAYER_HOOK_SUFFIXES:
                    module = modules.get(suffix)
                    if module is None:
                        continue

                    def hook(
                        _module: torch.nn.Module,
                        _inputs: tuple[Any, ...],
                        output: Any,
                        *,
                        capture_name: str = f"replay:{layer_idx}:{suffix}",
                    ) -> None:
                        tensor = first_tensor(output)
                        if tensor is not None:
                            captures[capture_name] = tensor_to_cpu(tensor)

                    hooks.append(module.register_forward_hook(hook))

            output = model.model.layers[layer_idx](
                hidden,
                attention_mask=masks[model.model.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
            )

            for hook in hooks:
                hook.remove()

            output_cpu = tensor_to_cpu(output)
            outputs[layer_idx] = output_cpu
            reference = output_cpu
            if layer_idx == layer_count - 1:
                # Transformers records the final hidden state after the model
                # norm, not the raw output of the last decoder layer.  Keep
                # saving raw layer outputs for native-vs-DBT comparison, but
                # validate the replay against the same value that `save`
                # captured.
                reference = tensor_to_cpu(model.model.norm(output))
            reference_diffs[layer_idx] = tensor_diff(
                reference, saved_hidden[layer_idx + 1]
            )
            torch.cuda.synchronize()
            max_diff, mean_diff, l2_diff = reference_diffs[layer_idx]
            print(
                f"replayed layer[{layer_idx:02d}] elapsed_s={time.perf_counter() - layer_start:.3f} "
                f"reference_max={max_diff:.9g} reference_mean={mean_diff:.9g} reference_l2={l2_diff:.9g}",
                flush=True,
            )
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "model": args.model,
        "attn_implementation": args.attn_implementation or "default",
        "input_path": str(args.input),
        "input_ids": input_ids.cpu(),
        "device": torch.cuda.get_device_name(0),
        "layer_replay_outputs": outputs,
        "layer_replay_reference_diffs": reference_diffs,
        "layer_replay_hooks": captures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {args.output}", flush=True)
    print(f"device {result['device']}", flush=True)
    print(f"layers {layer_indices}", flush=True)
    print(f"layer_replay_outputs {len(outputs)}", flush=True)
    print(f"layer_replay_hooks {len(captures)}", flush=True)
    print("reference_diffs", flush=True)
    for layer_idx in layer_indices:
        max_diff, mean_diff, l2_diff = reference_diffs[layer_idx]
        print(
            f"layer[{layer_idx:02d}] max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
            flush=True,
        )


def replay_module(args: argparse.Namespace) -> None:
    """Replay one module from a saved tensor.

    Layer replay tells us whether a whole block diverges on identical input.
    This mode goes one step narrower: feed one saved tensor into one named
    module, usually a single Qwen `Linear`, so the DBT comparison maps to one
    operator dispatch rather than a full decoder layer.
    """

    torch.manual_seed(args.seed)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    model = load_model(args)
    module = lookup_module(model, args.module)
    input_tensor = lookup_saved_tensor(source, args.input_key).to(
        device="cuda", dtype=torch.bfloat16
    )
    reference_tensor = None
    if args.reference_key:
        reference_tensor = lookup_saved_tensor(source, args.reference_key)
    captures: dict[str, torch.Tensor] = {}
    hooks = []

    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = module(input_tensor)
        torch.cuda.synchronize()

        if args.hook_children:
            # Capture the exact inputs and outputs seen by child modules inside
            # this standalone replay.  For Qwen MLP triage this tells us
            # whether the DBT error is introduced by the Linear kernels or by
            # the activation/multiply tensor feeding the final projection.
            for child_name, child_module in module.named_modules():
                if not child_name:
                    continue

                def hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    capture_name: str = child_name,
                ) -> None:
                    input_value = first_tensor(_inputs)
                    if input_value is not None:
                        captures[f"module_hook_input:{args.module}.{capture_name}"] = (
                            tensor_to_cpu(input_value)
                        )
                    output_value = first_tensor(output)
                    if output_value is not None:
                        captures[f"module_hook:{args.module}.{capture_name}"] = (
                            tensor_to_cpu(output_value)
                        )

                hooks.append(child_module.register_forward_hook(hook))

        start = time.perf_counter()
        output = None
        try:
            for _ in range(args.repeat):
                output = module(input_tensor)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start
        finally:
            for hook_handle in hooks:
                hook_handle.remove()

    if output is None:
        raise ValueError("--repeat must be greater than zero")

    output_cpu = tensor_to_cpu(output)
    result: dict[str, Any] = {
        "model": args.model,
        "attn_implementation": args.attn_implementation or "default",
        "input_path": str(args.input),
        "input_key": args.input_key,
        "reference_key": args.reference_key,
        "module": args.module,
        "device": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "elapsed_s": elapsed_s,
        "module_replay_outputs": {args.module: output_cpu},
    }
    if captures:
        result["module_replay_hooks"] = captures
    if reference_tensor is not None:
        result["module_replay_reference_diff"] = tensor_diff(
            output_cpu, reference_tensor
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {args.output}", flush=True)
    print(f"device {result['device']}", flush=True)
    print(f"module {args.module}", flush=True)
    print(f"input_key {args.input_key}", flush=True)
    if args.reference_key:
        print(f"reference_key {args.reference_key}", flush=True)
    print(
        f"elapsed_s {elapsed_s:.6f} repeat {args.repeat} per_iter_s {elapsed_s / args.repeat:.6f}",
        flush=True,
    )
    if captures:
        print(f"module_replay_hooks {len(captures)}", flush=True)
    if reference_tensor is not None:
        max_diff, mean_diff, l2_diff = result["module_replay_reference_diff"]
        print(
            f"reference_diff max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
            flush=True,
        )


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float, float]:
    diff = (lhs - rhs).abs()
    return float(diff.max()), float(diff.mean()), float(torch.linalg.vector_norm(diff))


def compare_activations(args: argparse.Namespace) -> None:
    lhs = torch.load(args.lhs, map_location="cpu", weights_only=False)
    rhs = torch.load(args.rhs, map_location="cpu", weights_only=False)

    print(f"lhs_device {lhs.get('device')}", flush=True)
    print(f"rhs_device {rhs.get('device')}", flush=True)

    if "hidden_states" in lhs and "hidden_states" in rhs:
        print("hidden_state_diffs", flush=True)
        for idx, (lhs_tensor, rhs_tensor) in enumerate(
            zip(lhs["hidden_states"], rhs["hidden_states"])
        ):
            max_diff, mean_diff, l2_diff = tensor_diff(lhs_tensor, rhs_tensor)
            print(
                f"hidden[{idx:02d}] max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    if "hooks" in lhs and "hooks" in rhs:
        print("hook_diffs", flush=True)
        for name in sorted(set(lhs["hooks"]) & set(rhs["hooks"])):
            max_diff, mean_diff, l2_diff = tensor_diff(
                lhs["hooks"][name], rhs["hooks"][name]
            )
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    if "logits" in lhs and "logits" in rhs:
        max_diff, mean_diff, l2_diff = tensor_diff(lhs["logits"], rhs["logits"])
        lhs_argmax = int(lhs["logits"][0, -1].argmax())
        rhs_argmax = int(rhs["logits"][0, -1].argmax())
        print("logit_diff", flush=True)
        print(
            f"logits max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
            flush=True,
        )
        print(f"last_token_argmax lhs={lhs_argmax} rhs={rhs_argmax}", flush=True)

    if "layer_replay_outputs" in lhs and "layer_replay_outputs" in rhs:
        lhs_outputs = lhs["layer_replay_outputs"]
        rhs_outputs = rhs["layer_replay_outputs"]
        print("layer_replay_diffs", flush=True)
        for layer_idx in sorted(set(lhs_outputs) & set(rhs_outputs)):
            max_diff, mean_diff, l2_diff = tensor_diff(
                lhs_outputs[layer_idx], rhs_outputs[layer_idx]
            )
            print(
                f"layer[{layer_idx:02d}] max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    if "layer_replay_hooks" in lhs and "layer_replay_hooks" in rhs:
        print("layer_replay_hook_diffs", flush=True)
        for name in sorted(
            set(lhs["layer_replay_hooks"]) & set(rhs["layer_replay_hooks"])
        ):
            max_diff, mean_diff, l2_diff = tensor_diff(
                lhs["layer_replay_hooks"][name], rhs["layer_replay_hooks"][name]
            )
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    if "module_replay_outputs" in lhs and "module_replay_outputs" in rhs:
        print("module_replay_diffs", flush=True)
        for name in sorted(
            set(lhs["module_replay_outputs"]) & set(rhs["module_replay_outputs"])
        ):
            max_diff, mean_diff, l2_diff = tensor_diff(
                lhs["module_replay_outputs"][name],
                rhs["module_replay_outputs"][name],
            )
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    if "module_replay_hooks" in lhs and "module_replay_hooks" in rhs:
        print("module_replay_hook_diffs", flush=True)
        for name in sorted(
            set(lhs["module_replay_hooks"]) & set(rhs["module_replay_hooks"])
        ):
            max_diff, mean_diff, l2_diff = tensor_diff(
                lhs["module_replay_hooks"][name],
                rhs["module_replay_hooks"][name],
            )
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save")
    save_parser.add_argument("--output", type=Path, required=True)
    save_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    save_parser.add_argument(
        "--input-ids", default=" ".join(str(x) for x in DEFAULT_INPUT_IDS)
    )
    save_parser.add_argument("--seed", type=int, default=0)
    save_parser.add_argument("--attn-implementation")
    save_parser.add_argument("--hook-layer0", action="store_true")
    save_parser.add_argument("--hook-all-layers", action="store_true")
    save_parser.set_defaults(func=save_activations)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("lhs", type=Path)
    compare_parser.add_argument("rhs", type=Path)
    compare_parser.set_defaults(func=compare_activations)

    replay_parser = subparsers.add_parser("replay-layers")
    replay_parser.add_argument("--input", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    replay_parser.add_argument(
        "--input-ids", default=" ".join(str(x) for x in DEFAULT_INPUT_IDS)
    )
    replay_parser.add_argument("--seed", type=int, default=0)
    replay_parser.add_argument("--attn-implementation")
    replay_parser.add_argument(
        "--layers",
        help="Layer list/ranges to replay, for example '0 1 8 16 23' or '0-23'.",
    )
    replay_parser.add_argument("--hook-modules", action="store_true")
    replay_parser.set_defaults(func=replay_layers)

    module_parser = subparsers.add_parser("replay-module")
    module_parser.add_argument("--input", type=Path, required=True)
    module_parser.add_argument("--output", type=Path, required=True)
    module_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    module_parser.add_argument("--seed", type=int, default=0)
    module_parser.add_argument("--attn-implementation")
    module_parser.add_argument("--module", required=True)
    module_parser.add_argument("--input-key", required=True)
    module_parser.add_argument("--reference-key")
    module_parser.add_argument("--warmup", type=int, default=1)
    module_parser.add_argument("--repeat", type=int, default=1)
    module_parser.add_argument("--hook-children", action="store_true")
    module_parser.set_defaults(func=replay_module)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
