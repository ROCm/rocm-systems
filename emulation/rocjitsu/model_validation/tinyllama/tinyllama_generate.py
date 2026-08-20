#!/usr/bin/env python3
"""Run a small deterministic TinyLlama generation check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_PATH = "model_cache/TinyLlama_TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The quick brown fox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    bad_pad = (
        tokenizer.pad_token_id is None
        or tokenizer.pad_token_id < 0
        or tokenizer.pad_token_id >= len(tokenizer)
    )
    if bad_pad:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


def logits_as_lists(scores: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> list[list[float]]:
    return [score.detach().float().cpu().reshape(-1).tolist() for score in scores]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned false")
    if not args.model_path.exists():
        raise RuntimeError(f"model path does not exist: {args.model_path}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=None,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    input_ids = inputs["input_ids"].detach().cpu().reshape(-1).tolist()
    sequence_ids = generated.sequences.detach().cpu().reshape(-1).tolist()
    payload: dict[str, Any] = {
        "model_id": args.model_id,
        "model_path": str(args.model_path),
        "device": device.type,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": "float32",
        "input_ids": input_ids,
        "sequence_ids": sequence_ids,
        "new_token_ids": sequence_ids[len(input_ids) :],
        "decoded_text": tokenizer.decode(sequence_ids, skip_special_tokens=False),
        "logits": logits_as_lists(generated.scores),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": __import__("transformers").__version__,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        payload["cuda_arch"] = getattr(props, "gcnArchName", "")
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in payload.items() if key != "logits"}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
