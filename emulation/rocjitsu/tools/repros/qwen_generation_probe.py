#!/usr/bin/env python3
"""Save and compare Qwen greedy-generation logits for DBT triage.

`qwen_suite.py` is the acceptance harness for the workload, but its JSON
comparison intentionally stays compact: it checks generated text and a forward
argmax.  This probe mirrors the suite's greedy generation path and records the
full last-token logits at each generated step so native gfx942 execution can be
compared against gfx950 guest code translated by rocjitsu.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_PROMPT = "The GPU kernel launched, synchronized, and then"
REPLAY_GLOBAL_HOOKS = (
    "model.embed_tokens",
    "model.norm",
    "lm_head",
)
LAYER_HOOK_SUFFIXES = (
    "",
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn",
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
    """Parse a comma/space separated token-id list."""

    return [int(part) for part in raw.replace(",", " ").split()]


def parse_layer_list(raw: str | None, layer_count: int) -> list[int]:
    """Parse layer indices/ranges for targeted replay hook capture."""

    if not raw:
        return []
    if raw.strip() == "all":
        return list(range(layer_count))

    layers: list[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            layers.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            layers.append(int(part))

    result = sorted(set(layers))
    for layer in result:
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"layer index {layer} outside [0, {layer_count})")
    return result


def resolve_torch_dtype(value: str) -> torch.dtype | str:
    """Translate the suite-style dtype string into a `from_pretrained` value."""

    if value == "auto":
        return "auto"
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown torch dtype: {value}")


def module_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def make_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Match the explicit-position-id path used by `/home/kunwar/resnet_pytorch/gpt2_suite.py`."""

    return (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )


def make_causal_mask(
    input_ids: torch.Tensor, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Build the Qwen full-attention mask used by the no-cache diagnostic path."""

    seq_len = input_ids.shape[1]
    mask = torch.full(
        (input_ids.shape[0], 1, seq_len, seq_len),
        torch.finfo(dtype).min,
        device=input_ids.device,
        dtype=dtype,
    )
    return {"full_attention": torch.triu(mask, diagonal=1)}


def load_inputs(args: argparse.Namespace, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Use either explicit token ids or the same prompt text as `qwen_suite.py`."""

    if args.input_ids:
        ids = torch.tensor([parse_input_ids(args.input_ids)], dtype=torch.long)
    else:
        ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    return ids.to(args.device)


def decode(tokenizer: AutoTokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=False)


def first_tensor(value: Any) -> torch.Tensor | None:
    """Extract the first tensor from common module output containers."""

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


def install_replay_hooks(
    model: torch.nn.Module, layer_spec: str | None
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    """Install hooks that identify the first divergent cached replay module.

    The cache-consumption failure starts from identical saved native KV tensors,
    so each captured tensor is expected to match native within the same strict
    `1e-3` bound.  Hooks are intentionally opt-in because capturing every layer
    materially increases the size of saved probe artifacts.
    """

    captures: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if not layer_spec:
        return captures, handles

    layer_count = len(getattr(model.model, "layers", []))
    hook_names = list(REPLAY_GLOBAL_HOOKS)
    for layer_idx in parse_layer_list(layer_spec, layer_count):
        for suffix in LAYER_HOOK_SUFFIXES:
            hook_names.append(
                f"model.layers.{layer_idx}"
                if not suffix
                else f"model.layers.{layer_idx}.{suffix}"
            )

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
            input_tensor = first_tensor(_inputs)
            if input_tensor is not None:
                captures[f"hook_input:{capture_name}"] = tensor_to_cpu(input_tensor)
            tensor = first_tensor(output)
            if tensor is not None:
                captures[f"hook:{capture_name}"] = tensor_to_cpu(tensor)

        handles.append(module.register_forward_hook(hook))
    return captures, handles


def install_sdpa_capture(
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], Any]:
    """Capture exact SDPA tensors used by cached replay.

    Qwen's `sdpa` attention path calls `torch.nn.functional` directly, so module
    hooks can see only the projection output and final attention output.  This
    temporary monkeypatch records the real query/key/value/mask tensors and the
    SDPA result for each layer without changing the call arguments.
    """

    captures: dict[str, torch.Tensor] = {}
    metadata: list[dict[str, Any]] = []
    original = torch.nn.functional.scaled_dot_product_attention
    if not enabled:
        return captures, metadata, None

    def positional_or_kwarg(
        args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str
    ):
        if len(args) > index:
            return args[index]
        return kwargs.get(name)

    def wrapped(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs
    ):
        call_idx = len(metadata)
        prefix = f"sdpa.{call_idx:03d}"
        captures[f"{prefix}.query"] = tensor_to_cpu(query)
        captures[f"{prefix}.key"] = tensor_to_cpu(key)
        captures[f"{prefix}.value"] = tensor_to_cpu(value)

        attn_mask = positional_or_kwarg(args, kwargs, 0, "attn_mask")
        if isinstance(attn_mask, torch.Tensor):
            captures[f"{prefix}.attn_mask"] = tensor_to_cpu(attn_mask)

        metadata.append(
            {
                "dropout_p": positional_or_kwarg(args, kwargs, 1, "dropout_p"),
                "is_causal": positional_or_kwarg(args, kwargs, 2, "is_causal"),
                "scale": kwargs.get("scale"),
                "enable_gqa": kwargs.get("enable_gqa", False),
                "has_attn_mask": isinstance(attn_mask, torch.Tensor),
            }
        )

        output = original(query, key, value, *args, **kwargs)
        captures[f"{prefix}.output"] = tensor_to_cpu(output)
        return output

    torch.nn.functional.scaled_dot_product_attention = wrapped
    return captures, metadata, original


def flatten_cache_tensors(value: Any, prefix: str = "past") -> dict[str, torch.Tensor]:
    """Convert a Transformers cache object into named CPU tensors.

    Recent Transformers releases return `DynamicCache` objects rather than the
    older tuple-of-tuples cache.  Hybrid models such as Qwen3.5 also store
    linear-attention recurrence in cache-layer attributes that are lost by
    `to_legacy_cache()`, so walk the cache object itself before falling back to
    legacy K/V tuples.
    """

    if isinstance(value, torch.Tensor):
        return {prefix: tensor_to_cpu(value)}

    # Transformers 4.5x `DynamicCache` exposes a public `layers` list whose
    # entries may be ordinary K/V layers or linear-attention recurrent layers.
    # Handle this before any legacy conversion so Qwen3.5 conv/recurrent state is
    # part of the first-pass cache snapshot.
    if hasattr(value, "layers") and isinstance(value.layers, list):
        result: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(value.layers):
            result.update(flatten_cache_tensors(layer, f"{prefix}.layer{idx}"))
        return result

    if hasattr(value, "keys") and hasattr(value, "values"):
        result = {}
        keys = getattr(value, "keys")
        values = getattr(value, "values")
        if isinstance(keys, torch.Tensor):
            result[f"{prefix}.keys"] = tensor_to_cpu(keys)
        if isinstance(values, torch.Tensor):
            result[f"{prefix}.values"] = tensor_to_cpu(values)
        return result

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        result = {}
        for key, item in attrs.items():
            if key.startswith("_"):
                continue
            result.update(flatten_cache_tensors(item, f"{prefix}.{key}"))
        if result:
            return result

    if isinstance(value, (tuple, list)):
        result: dict[str, torch.Tensor] = {}
        for idx, item in enumerate(value):
            result.update(flatten_cache_tensors(item, f"{prefix}.{idx}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten_cache_tensors(item, f"{prefix}.{key}"))
        return result
    if hasattr(value, "to_legacy_cache"):
        legacy = value.to_legacy_cache()
        if legacy is not value:
            return flatten_cache_tensors(legacy, prefix)
    return {}


def load_model(args: argparse.Namespace) -> tuple[AutoTokenizer, torch.nn.Module]:
    """Load the same model shape and dtype as the Qwen suite run."""

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "local_files_only": True,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    return tokenizer, model.eval().to(args.device)


def run_generation(args: argparse.Namespace) -> None:
    """Run greedy generation and save every step's full last-token logits."""

    torch.manual_seed(args.seed)
    tokenizer, model = load_model(args)
    seq = load_inputs(args, tokenizer)

    initial_ids = seq.detach().cpu().contiguous()
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    first_step_full_logits: torch.Tensor | None = None
    past_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    past_key_values = None
    cur = seq
    # The generation path is where the first KV cache is produced.  Reuse the
    # replay hook helper here so a one-token `save` run can localize first-pass
    # cache drift without a separate repro script.  For multi-token generation
    # these names naturally hold the final model forward that ran.
    hook_captures, hook_handles = install_replay_hooks(model, args.hook_layers)
    sdpa_captures, sdpa_metadata, original_sdpa = install_sdpa_capture(
        args.capture_sdpa
    )

    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    try:
        with torch.inference_mode():
            for step in range(args.tokens):
                if args.use_cache:
                    if args.explicit_position_ids:
                        position_ids = (
                            make_position_ids(seq)
                            if past_key_values is None
                            else torch.full(
                                (1, 1),
                                seq.shape[1] - 1,
                                device=args.device,
                                dtype=torch.long,
                            )
                        )
                    else:
                        position_ids = None

                    output = model(
                        input_ids=cur,
                        past_key_values=past_key_values,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                    past_key_values = output.past_key_values
                    if args.save_past:
                        past_snapshots[step] = flatten_cache_tensors(past_key_values)
                else:
                    position_ids = (
                        make_position_ids(seq) if args.explicit_position_ids else None
                    )
                    attention_mask = (
                        make_causal_mask(seq, module_dtype(model))
                        if args.explicit_causal_mask
                        else None
                    )
                    output = model(
                        input_ids=seq,
                        attention_mask=attention_mask,
                        use_cache=False,
                        position_ids=position_ids,
                    )

                if step == 0:
                    first_step_full_logits = tensor_to_cpu(output.logits)

                last_logits = tensor_to_cpu(output.logits[:, -1, :])
                step_logits.append(last_logits.squeeze(0))
                cur = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(int(cur.item()))
                seq = torch.cat([seq, cur], dim=1)

                if args.sync_steps and args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    print(f"step {step} token {generated[-1]}", flush=True)
    finally:
        if original_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for hook_handle in hook_handles:
            hook_handle.remove()

    torch.cuda.synchronize() if args.device.startswith("cuda") else None

    result: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt if not args.input_ids else None,
        "input_ids": initial_ids,
        "tokens": args.tokens,
        "seed": args.seed,
        "use_cache": args.use_cache,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "explicit_position_ids": args.explicit_position_ids,
        "explicit_causal_mask": args.explicit_causal_mask,
        "device": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else args.device
        ),
        "generated_ids": torch.tensor(generated, dtype=torch.long),
        "sequence_ids": seq.detach().cpu().contiguous(),
        "text": decode(tokenizer, seq[0]),
        "step_logits": torch.stack(step_logits).contiguous(),
        "first_step_full_logits": first_step_full_logits,
    }
    if past_snapshots:
        result["past_snapshots"] = past_snapshots
    if hook_captures:
        result["hooks"] = hook_captures
    if sdpa_captures:
        result["sdpa_captures"] = sdpa_captures
        result["sdpa_metadata"] = sdpa_metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)

    print(f"saved {args.output}", flush=True)
    print(f"device {result['device']}", flush=True)
    print(f"generated_ids {generated}", flush=True)
    print(f"text {result['text']}", flush=True)
    print(f"step_logits_shape {tuple(result['step_logits'].shape)}", flush=True)
    if past_snapshots:
        print(f"past_snapshot_steps {sorted(past_snapshots)}", flush=True)
    if hook_captures:
        print(f"hooks {len(hook_captures)}", flush=True)
    if sdpa_captures:
        print(f"sdpa_captures {len(sdpa_metadata)}", flush=True)


def rebuild_dynamic_cache(
    model: torch.nn.Module, snapshot: dict[str, torch.Tensor], device: str
) -> DynamicCache:
    """Reconstruct a Transformers `DynamicCache` from a saved probe snapshot."""

    cache = DynamicCache(config=model.config)
    restored_linear_state = False
    layer_indices = []
    for name in snapshot:
        if not name.startswith("past.layer") or not name.endswith(".keys"):
            continue
        raw_idx = name[len("past.layer") : -len(".keys")]
        if raw_idx.isdigit():
            layer_indices.append(int(raw_idx))

    # Hybrid attention models such as Qwen3.5 only materialize KV tensors for
    # their sparse full-attention layers.  Rebuild exactly the layers present in
    # the snapshot instead of assuming layer 0 starts a contiguous KV cache.
    for layer_idx in sorted(set(layer_indices)):
        key_name = f"past.layer{layer_idx}.keys"
        value_name = f"past.layer{layer_idx}.values"
        if value_name not in snapshot:
            raise KeyError(f"incomplete cache snapshot for layer {layer_idx}")
        key = snapshot[key_name].to(device=device, dtype=torch.bfloat16)
        value = snapshot[value_name].to(device=device, dtype=torch.bfloat16)
        cache.update(key, value, layer_idx)

    for name, tensor in snapshot.items():
        if not name.startswith("past.layer"):
            continue
        if name.endswith(".conv_states"):
            raw_idx = name[len("past.layer") : -len(".conv_states")]
            state_attr = "conv_states"
            init_attr = "is_conv_states_initialized"
        elif name.endswith(".recurrent_states"):
            raw_idx = name[len("past.layer") : -len(".recurrent_states")]
            state_attr = "recurrent_states"
            init_attr = "is_recurrent_states_initialized"
        else:
            continue
        if not raw_idx.isdigit():
            continue
        layer_idx = int(raw_idx)
        if layer_idx >= len(cache.layers):
            raise KeyError(f"cache snapshot references missing layer {layer_idx}")
        layer = cache.layers[layer_idx]
        setattr(layer, state_attr, tensor.to(device=device, dtype=torch.bfloat16))
        setattr(layer, init_attr, True)
        if hasattr(layer, "has_previous_state"):
            layer.has_previous_state = True
        restored_linear_state = True

    if not layer_indices and not restored_linear_state:
        raise ValueError("cache snapshot did not contain any layer tensors")
    return cache


def replay_next_token(args: argparse.Namespace) -> None:
    """Run one cached decode step from a saved cache snapshot.

    This isolates "consume an existing cache" from "produce the initial cache".
    Feeding the same native-saved cache to native and DBT should leave only the
    second cached model call under test.  Hybrid models such as Qwen3.5 need both
    ordinary K/V tensors and linear-attention conv/recurrent state restored for
    this replay to be meaningful.
    """

    torch.manual_seed(args.seed)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    tokenizer, model = load_model(args)

    past_snapshots = source.get("past_snapshots")
    if not isinstance(past_snapshots, dict) or args.step not in past_snapshots:
        raise KeyError(f"input does not contain past snapshot step {args.step}")

    generated_ids = source["generated_ids"]
    input_ids = source["input_ids"]
    cur_token = int(generated_ids[args.step])
    cur = torch.tensor([[cur_token]], device=args.device, dtype=torch.long)
    position_id = torch.tensor(
        [[int(input_ids.shape[1]) + args.step]], device=args.device, dtype=torch.long
    )
    cache = rebuild_dynamic_cache(model, past_snapshots[args.step], args.device)
    hook_captures, hook_handles = install_replay_hooks(model, args.hook_layers)
    sdpa_captures, sdpa_metadata, original_sdpa = install_sdpa_capture(
        args.capture_sdpa
    )

    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    try:
        with torch.inference_mode():
            output = model(
                input_ids=cur,
                past_key_values=cache,
                use_cache=True,
                position_ids=position_id,
            )
    finally:
        if original_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for hook_handle in hook_handles:
            hook_handle.remove()
    torch.cuda.synchronize() if args.device.startswith("cuda") else None

    logits = tensor_to_cpu(output.logits[:, -1, :])
    next_token = int(logits.argmax(dim=-1).item())
    sequence = torch.cat([input_ids, generated_ids[: args.step + 1].view(1, -1)], dim=1)
    sequence = torch.cat(
        [sequence, torch.tensor([[next_token]], dtype=torch.long)], dim=1
    )

    result: dict[str, Any] = {
        "model": args.model,
        "input_path": str(args.input),
        "replay_step": args.step,
        "input_ids": input_ids,
        "tokens": 1,
        "seed": args.seed,
        "use_cache": True,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "device": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else args.device
        ),
        "generated_ids": torch.tensor([next_token], dtype=torch.long),
        "sequence_ids": sequence.contiguous(),
        "text": decode(tokenizer, sequence[0]),
        "step_logits": logits,
        "past_snapshots": {
            args.step + 1: flatten_cache_tensors(output.past_key_values)
        },
    }
    if hook_captures:
        result["hooks"] = hook_captures
    if sdpa_captures:
        result["sdpa_captures"] = sdpa_captures
        result["sdpa_metadata"] = sdpa_metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)

    print(f"saved {args.output}", flush=True)
    print(f"device {result['device']}", flush=True)
    print(f"replay_step {args.step}", flush=True)
    print(f"input_token {cur_token}", flush=True)
    print(f"next_token {next_token}", flush=True)
    print(f"text {result['text']}", flush=True)
    print(f"step_logits_shape {tuple(result['step_logits'].shape)}", flush=True)
    if hook_captures:
        print(f"hooks {len(hook_captures)}", flush=True)
    if sdpa_captures:
        print(f"sdpa_captures {len(sdpa_metadata)}", flush=True)


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float, float]:
    diff = (lhs - rhs).abs()
    return float(diff.max()), float(diff.mean()), float(torch.linalg.vector_norm(diff))


def compare(args: argparse.Namespace) -> None:
    """Compare two saved generation runs token-by-token and logit-by-logit."""

    lhs = torch.load(args.lhs, map_location="cpu", weights_only=False)
    rhs = torch.load(args.rhs, map_location="cpu", weights_only=False)

    print(f"lhs_device {lhs.get('device')}", flush=True)
    print(f"rhs_device {rhs.get('device')}", flush=True)
    print(f"lhs_text {lhs.get('text')}", flush=True)
    print(f"rhs_text {rhs.get('text')}", flush=True)

    lhs_ids = lhs["generated_ids"].tolist()
    rhs_ids = rhs["generated_ids"].tolist()
    print(f"lhs_generated_ids {lhs_ids}", flush=True)
    print(f"rhs_generated_ids {rhs_ids}", flush=True)
    if lhs_ids != rhs_ids:
        first = next(
            idx
            for idx, (lhs_id, rhs_id) in enumerate(zip(lhs_ids, rhs_ids))
            if lhs_id != rhs_id
        )
        print(
            f"first_token_mismatch step={first} lhs={lhs_ids[first]} rhs={rhs_ids[first]}",
            flush=True,
        )
    else:
        print("generated_ids_match True", flush=True)

    lhs_logits = lhs["step_logits"]
    rhs_logits = rhs["step_logits"]
    max_diff, mean_diff, l2_diff = tensor_diff(lhs_logits, rhs_logits)
    print(
        f"step_logits_global max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
        flush=True,
    )

    print("step_logit_diffs", flush=True)
    for step in range(min(lhs_logits.shape[0], rhs_logits.shape[0])):
        step_max, step_mean, step_l2 = tensor_diff(lhs_logits[step], rhs_logits[step])
        lhs_argmax = int(lhs_logits[step].argmax())
        rhs_argmax = int(rhs_logits[step].argmax())
        print(
            f"step[{step:03d}] max={step_max:.9g} mean={step_mean:.9g} "
            f"l2={step_l2:.9g} argmax lhs={lhs_argmax} rhs={rhs_argmax}",
            flush=True,
        )

    lhs_full = lhs.get("first_step_full_logits")
    rhs_full = rhs.get("first_step_full_logits")
    if isinstance(lhs_full, torch.Tensor) and isinstance(rhs_full, torch.Tensor):
        max_diff, mean_diff, l2_diff = tensor_diff(lhs_full, rhs_full)
        print(
            f"first_step_full_logits max={max_diff:.9g} mean={mean_diff:.9g} "
            f"l2={l2_diff:.9g}",
            flush=True,
        )

    lhs_hooks = lhs.get("hooks")
    rhs_hooks = rhs.get("hooks")
    if isinstance(lhs_hooks, dict) and isinstance(rhs_hooks, dict):
        print("hook_diffs", flush=True)
        for name in sorted(set(lhs_hooks) & set(rhs_hooks)):
            max_diff, mean_diff, l2_diff = tensor_diff(lhs_hooks[name], rhs_hooks[name])
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    lhs_sdpa = lhs.get("sdpa_captures")
    rhs_sdpa = rhs.get("sdpa_captures")
    if isinstance(lhs_sdpa, dict) and isinstance(rhs_sdpa, dict):
        print("sdpa_diffs", flush=True)
        for name in sorted(set(lhs_sdpa) & set(rhs_sdpa)):
            max_diff, mean_diff, l2_diff = tensor_diff(lhs_sdpa[name], rhs_sdpa[name])
            print(
                f"{name} max={max_diff:.9g} mean={mean_diff:.9g} l2={l2_diff:.9g}",
                flush=True,
            )

    lhs_past = lhs.get("past_snapshots")
    rhs_past = rhs.get("past_snapshots")
    if isinstance(lhs_past, dict) and isinstance(rhs_past, dict):
        print("past_snapshot_diffs", flush=True)
        for step in sorted(set(lhs_past) & set(rhs_past)):
            worst_name = ""
            worst = (0.0, 0.0, 0.0)
            for name in sorted(set(lhs_past[step]) & set(rhs_past[step])):
                current = tensor_diff(lhs_past[step][name], rhs_past[step][name])
                if current[0] > worst[0]:
                    worst_name = name
                    worst = current
            print(
                f"past_step[{step:03d}] worst={worst_name} max={worst[0]:.9g} "
                f"mean={worst[1]:.9g} l2={worst[2]:.9g}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("save")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--input-ids")
    run.add_argument("--tokens", type=int, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    run.add_argument(
        "--attn-implementation", choices=("default", "eager", "sdpa"), default="default"
    )
    run.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--explicit-position-ids", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--explicit-causal-mask", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--save-past", action="store_true")
    run.add_argument("--sync-steps", action="store_true")
    run.add_argument(
        "--hook-layers",
        help=(
            "Capture hooks for the final generation forward; use --tokens 1 "
            "to localize the initial cache-producing pass."
        ),
    )
    run.add_argument("--capture-sdpa", action="store_true")
    run.set_defaults(func=run_generation)

    replay = subparsers.add_parser("replay-next")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--step", type=int, default=0)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--device", default="cuda")
    replay.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    replay.add_argument(
        "--attn-implementation", choices=("default", "eager", "sdpa"), default="default"
    )
    replay.add_argument(
        "--hook-layers",
        help="Capture replay hooks for layer indices/ranges such as '0', '0-3', or 'all'.",
    )
    replay.add_argument("--capture-sdpa", action="store_true")
    replay.set_defaults(func=replay_next_token)

    diff = subparsers.add_parser("compare")
    diff.add_argument("lhs", type=Path)
    diff.add_argument("rhs", type=Path)
    diff.set_defaults(func=compare)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
