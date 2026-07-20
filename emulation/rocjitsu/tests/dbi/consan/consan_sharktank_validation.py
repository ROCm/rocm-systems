#!/usr/bin/env python3
"""Runs the Sharktank workloads used by ConSan's north-star validation.

The external iree-test-suites checkout supplies models, compiler flags, inputs,
and reference helpers. This wrapper supplies the frozen invocation/oracle and
timing policy without depending on a timestamped local artifact directory.
"""

import argparse
import importlib.util
import json
import math
import os
import pathlib
import statistics
import time


TOKEN_IDS = [
    0,
    208,
    214,
    29,
    19,
    86,
    176,
    120,
    120,
    80,
    120,
    208,
    37,
    157,
    191,
    137,
]


def load_module(path: pathlib.Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"missing iree-test-suites workload: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load workload module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_scalar(invoke, expected, tolerance, repetitions, allow_oracle_failure):
    warmup = float(invoke())
    warmup_ok = math.isfinite(warmup) and abs(warmup - expected) <= tolerance
    if not warmup_ok and not allow_oracle_failure:
        raise RuntimeError(f"warmup oracle failed: {warmup}")

    samples_ms = []
    values = []
    oracle_results = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        value = float(invoke())
        elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
        oracle_ok = math.isfinite(value) and abs(value - expected) <= tolerance
        if not oracle_ok and not allow_oracle_failure:
            raise RuntimeError(f"timed oracle failed: {value}")
        samples_ms.append(elapsed_ms)
        values.append(value)
        oracle_results.append(oracle_ok)

    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "warmup_oracle": warmup,
        "oracle_min": min(values),
        "oracle_max": max(values),
        "oracle_ok": warmup_ok and all(oracle_results),
        "samples_ms": samples_ms,
    }


def run_llama(args):
    test_path = args.suite_root / "sharktank_models" / "llama3.1" / "test_llama.py"
    module = load_module(test_path, f"consan_{args.workload}_llama")
    sharding = 1 if args.workload == "tp1" else 2
    mlir = module.llama_mlir if sharding == 1 else module.llama_tp2_mlir
    irpa = module.llama_irpa if sharding == 1 else module.llama_tp2_irpa
    compiled = module.iree.compiler.compile_file(
        mlir, extra_args=module.hip_flags(sharding)
    )

    if args.mode == "all":
        selected_modes = ("prefill", "decode", "combined")
    elif args.mode == "decode-combined":
        selected_modes = ("decode", "combined")
    else:
        selected_modes = (args.mode,)
    results = {}
    for mode in selected_modes:
        model = module.ToyLlama(
            compiled=compiled,
            device_id="hip",
            irpa=irpa,
            sharding=sharding,
        )
        if mode == "prefill":
            invoke = lambda: module.prefill_cross_entropy(model, TOKEN_IDS)
            expected, tolerance = 0.589, 0.1
        elif mode == "decode":
            invoke = lambda: module.decode_cross_entropy(model, TOKEN_IDS)
            expected, tolerance = 0.582, 0.01
        else:
            invoke = lambda: module.prefill_decode_cross_entropy(model, TOKEN_IDS)
            expected, tolerance = 0.589, 0.1
        results[mode] = measure_scalar(
            invoke,
            expected,
            tolerance,
            args.repetitions,
            args.allow_oracle_failure,
        )
    return results


def run_clip(args):
    import numpy as np

    test_path = args.suite_root / "sharktank_models" / "clip" / "test_clip.py"
    module = load_module(test_path, "consan_clip_bf16")
    compiled = module.iree.compiler.compile_file(
        module.mlir_path["bf16"], extra_args=module.compiler_args("hip")
    )
    vm_instance = module.iree.runtime.VmInstance()
    parameter_index = module.iree.runtime.ParameterIndex()
    parameter_index.load(module.parameters_path["bf16"])
    parameter_provider = parameter_index.create_provider("model")
    parameters_module = module.iree.runtime.create_io_parameters_module(
        vm_instance, parameter_provider
    )
    device = module.iree.runtime.get_device("hip")
    hal_module = module.iree.runtime.create_hal_module(
        instance=vm_instance, devices=[device]
    )
    vm_module = module.iree.runtime.VmModule.from_buffer(vm_instance, compiled)
    config = module.iree.runtime.Config(device=device)
    model = module.iree.runtime.load_vm_modules(
        hal_module, parameters_module, vm_module, config=config
    )[-1]
    model_input = module.load_tensor_from_irpa(module.function_arg0_path)
    expected = module.load_tensor_from_irpa(module.function_expected_result0)

    def invoke():
        device_result = model.forward_bs4(model_input)[0]
        return module.device_array_to_host(device_result).astype(dtype=expected.dtype)

    def oracle(result):
        cosine = module.cosine_similarity(result, expected, dim=-1)
        error = float(np.max(np.abs(cosine - np.ones_like(cosine))))
        return math.isfinite(error) and error <= module.absolute_tolerance["bf16"], error

    warmup_ok, warmup_error = oracle(invoke())
    if not warmup_ok and not args.allow_oracle_failure:
        raise RuntimeError(f"warmup oracle failed: maximum cosine error {warmup_error}")

    samples_ms = []
    oracle_errors = []
    oracle_results = []
    for _ in range(args.repetitions):
        start = time.perf_counter_ns()
        result = invoke()
        elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
        oracle_ok, oracle_error = oracle(result)
        if not oracle_ok and not args.allow_oracle_failure:
            raise RuntimeError(
                f"timed oracle failed: maximum cosine error {oracle_error}"
            )
        samples_ms.append(elapsed_ms)
        oracle_errors.append(oracle_error)
        oracle_results.append(oracle_ok)

    return {
        "clip_bf16": {
            "median_ms": statistics.median(samples_ms),
            "mean_ms": statistics.mean(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "warmup_oracle_ok": warmup_ok,
            "warmup_max_cosine_error": warmup_error,
            "oracle_ok": warmup_ok and all(oracle_results),
            "oracle_error_min": min(oracle_errors),
            "oracle_error_max": max(oracle_errors),
            "samples_ms": samples_ms,
        }
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-root",
        type=pathlib.Path,
        required=True,
        help="iree-test-suites source checkout",
    )
    parser.add_argument("--workload", choices=("tp1", "tp2", "clip-bf16"), required=True)
    parser.add_argument(
        "--mode",
        choices=("prefill", "decode", "combined", "decode-combined", "all"),
        default="all",
        help="Llama mode; ignored for CLIP",
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--label", default="unnamed")
    parser.add_argument(
        "--allow-oracle-failure",
        action="store_true",
        help="record an injected fault's oracle result instead of exiting early",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    return args


def main():
    args = parse_args()
    results = run_clip(args) if args.workload == "clip-bf16" else run_llama(args)
    document = {
        "label": args.label,
        "workload": args.workload,
        "repetitions": args.repetitions,
        **results,
    }
    print(json.dumps(document, sort_keys=True))
    result_path = os.environ.get("CONSAN_ROW_RESULT_PATH")
    if result_path:
        oracle_ok = all(
            value.get("oracle_ok", False)
            for value in results.values()
            if isinstance(value, dict)
        )
        pathlib.Path(result_path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "oracle": "pass" if oracle_ok else "fail",
                    "detail": f"Sharktank oracle results: {sorted(results)}",
                    "source_diagnostics": {
                        "expectation": "not_applicable",
                        "outcome": "not_applicable",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
