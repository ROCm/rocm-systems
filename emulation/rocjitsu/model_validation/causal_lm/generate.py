#!/usr/bin/env python3
"""Run the fixed causal-LM smoke-test prompts for one local model snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import MAX_NEW_TOKENS, PROMPTS, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def fail(message: str) -> None:
    raise RuntimeError(message)


def normalize_tokenizer(tokenizer: Any) -> None:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None or pad_token_id < 0 or pad_token_id >= len(tokenizer):
        tokenizer.pad_token = tokenizer.eos_token


def load_model(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
    return model.eval()


def make_inputs(tokenizer: Any, *, prompt: str, device: Any) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {key: value.to(device) for key, value in inputs.items()}


def logits_payload(scores: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    import torch

    tensors = [score.detach().float().cpu().reshape(-1) for score in scores]
    for step_index, tensor in enumerate(tensors):
        if not torch.isfinite(tensor).all():
            fail(f"non-finite logits at generation step {step_index}")
    per_step = [tensor.tolist() for tensor in tensors]
    flattened = [value for step in per_step for value in step]
    if flattened:
        tensor = torch.tensor(flattened, dtype=torch.float32)
        stats = {
            "logits_step_count": len(per_step),
            "logits_vocab_size": len(per_step[0]),
            "logits_sum": tensor.sum().item(),
            "logits_mean": tensor.mean().item(),
            "logits_min": tensor.min().item(),
            "logits_max": tensor.max().item(),
            "logits_l2": torch.linalg.vector_norm(tensor).item(),
            "logits_first_16": flattened[:16],
        }
    else:
        stats = {
            "logits_step_count": 0,
            "logits_vocab_size": 0,
            "logits_sum": 0.0,
            "logits_mean": 0.0,
            "logits_min": 0.0,
            "logits_max": 0.0,
            "logits_l2": 0.0,
            "logits_first_16": [],
        }
    stats["logits"] = per_step
    return stats


def strip_case_logits(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if key != "logits"}


def summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in payload.items() if key != "cases"}
    summary["cases"] = [strip_case_logits(case) for case in payload["cases"]]
    return summary


def run_case(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    case_index: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    print(f"causal_lm_phase_start phase=tokenize_prompt case={case_index}", flush=True)
    inputs = make_inputs(tokenizer, prompt=prompt, device=device)
    input_ids = inputs["input_ids"]
    print(f"causal_lm_phase_done phase=tokenize_prompt case={case_index}", flush=True)

    with torch.inference_mode():
        print(f"causal_lm_phase_start phase=generate case={case_index}", flush=True)
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=None,
        )
        print(f"causal_lm_phase_done phase=generate case={case_index}", flush=True)
        if device.type == "cuda":
            print(f"causal_lm_phase_start phase=synchronize case={case_index}", flush=True)
            torch.cuda.synchronize()
            print(f"causal_lm_phase_done phase=synchronize case={case_index}", flush=True)

        print(f"causal_lm_phase_start phase=output_to_cpu case={case_index}", flush=True)
        sequences = generated.sequences.detach().cpu()
        score_tensors = [score.detach().cpu() for score in generated.scores]
        print(f"causal_lm_phase_done phase=output_to_cpu case={case_index}", flush=True)

    input_ids_cpu = input_ids.detach().cpu().reshape(-1).tolist()
    sequence_ids = sequences.reshape(-1).tolist()
    payload: dict[str, Any] = {
        "case_index": case_index,
        "prompt": prompt,
        "input_ids": input_ids_cpu,
        "sequence_ids": sequence_ids,
        "new_token_ids": sequence_ids[len(input_ids_cpu) :],
        "decoded_text": tokenizer.decode(sequence_ids, skip_special_tokens=False),
    }
    payload.update(logits_payload(score_tensors))
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        fail("torch.cuda.is_available() returned false")
    if not args.model_path.exists():
        fail(f"--model-path does not exist: {args.model_path}")

    torch.manual_seed(0)
    device = torch.device(args.device)

    print(f"causal_lm_phase_start phase=load_tokenizer device={device.type}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=True,
    )
    normalize_tokenizer(tokenizer)
    print("causal_lm_phase_done phase=load_tokenizer", flush=True)

    print("causal_lm_phase_start phase=load_model_cpu", flush=True)
    model = load_model(args.model_path)
    print("causal_lm_phase_done phase=load_model_cpu", flush=True)

    print("causal_lm_phase_start phase=model_to_device", flush=True)
    model = model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("causal_lm_phase_done phase=model_to_device", flush=True)

    cases = [
        run_case(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            case_index=case_index,
            device=device,
        )
        for case_index, prompt in enumerate(PROMPTS)
    ]
    payload: dict[str, Any] = {
        "workload": "causal_lm_generate",
        "model_id": args.model_id,
        "model_path": str(args.model_path),
        "device": device.type,
        "prompts": PROMPTS,
        "case_count": len(cases),
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": 0,
        "use_cache": True,
        "dtype": "float32",
        "tokenizer_class": tokenizer.__class__.__name__,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": {
            "HSA_ENABLE_SDMA": os.environ.get("HSA_ENABLE_SDMA", ""),
            "HSA_HOTSWAP_DISABLE": os.environ.get("HSA_HOTSWAP_DISABLE", ""),
            "ROCJITSU_RUNTIME_DIR": os.environ.get("ROCJITSU_RUNTIME_DIR", ""),
        },
        "cases": cases,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        payload["cuda_arch"] = getattr(props, "gcnArchName", "")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    write_json(args.output_json, payload)
    print("causal_lm_summary_json=" + json.dumps(summary_payload(payload), sort_keys=True))
    print("causal_lm_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
