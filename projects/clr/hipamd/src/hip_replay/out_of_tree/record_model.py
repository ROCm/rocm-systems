#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License
#
# Record a MIGraphX ONNX model run with HRR.
#
# Compilation is separated from recording so the HRR trace contains only
# one clean inference pass, not compilation/tuning kernels or warmup runs.
#
# Usage (two-phase — run compile first, then record):
#
#   # Phase 1: compile (no HRR)
#   python3 record_model.py --compile model.onnx
#   # -> writes model.mxr next to the .onnx file
#
#   # Phase 2: record one inference from compiled model
#   HRR_RECORD=1 HRR_OUTPUT=model.hrr \
#     LD_PRELOAD=./libhrr_record.so \
#     python3 record_model.py --record model.onnx
#
# bench_models.sh calls both phases automatically.

import migraphx
import numpy as np
import sys
import os
import argparse


def mxr_path(onnx_path: str) -> str:
    """Return the compiled model path adjacent to the .onnx file."""
    base = os.path.splitext(onnx_path)[0]
    return base + '.mxr'


def build_inputs(prog, batch_size: int) -> dict:
    """Build random float32 inputs matching the program's parameter shapes."""
    params = prog.get_parameter_shapes()
    inputs = {}
    for name, shape in params.items():
        dims = list(shape.lens())
        if dims and dims[0] == 1:
            dims[0] = batch_size
        data = np.random.randn(*dims).astype(np.float32)
        inputs[name] = migraphx.argument(data)
    return inputs


def cmd_compile(args):
    """Compile an ONNX model to .mxr (no HRR — run without LD_PRELOAD)."""
    out = mxr_path(args.model)
    if os.path.exists(out) and not args.force:
        print(f'[record_model] {out} already exists (use --force to recompile)')
        return

    print(f'[record_model] Compiling {args.model} -> {out}')
    prog = migraphx.parse_onnx(args.model)
    if args.fp16:
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target('gpu'))
    migraphx.save(prog, out)
    print(f'[record_model] Saved compiled model: {out}')


def cmd_record(args):
    """Load compiled model and run one inference (called under HRR LD_PRELOAD)."""
    out = mxr_path(args.model)
    if not os.path.exists(out):
        print(f'[record_model] ERROR: compiled model not found: {out}')
        print(f'[record_model] Run: python3 record_model.py --compile {args.model}')
        sys.exit(1)

    print(f'[record_model] Loading compiled model {out}')
    prog = migraphx.load(out)

    inputs = build_inputs(prog, args.batch)
    for name, arg in inputs.items():
        print(f'[record_model]   input {name!r}: {list(arg.get_shape().lens())}')

    # One inference — this is what HRR will capture
    print(f'[record_model] Running one recorded inference...')
    result = prog.run(inputs)

    out_shape = list(result[0].get_shape().lens())
    print(f'[record_model] Done. Output shape: {out_shape}')


def main():
    parser = argparse.ArgumentParser(
        description='Compile and record a MIGraphX ONNX model with HRR')
    parser.add_argument('model', help='Path to ONNX model file')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the model to .mxr (run without LD_PRELOAD)')
    parser.add_argument('--record', action='store_true',
                        help='Record one inference from pre-compiled .mxr (run with LD_PRELOAD)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--fp16', action='store_true', help='Quantize to fp16')
    parser.add_argument('--force', action='store_true',
                        help='Force recompile even if .mxr exists')
    args = parser.parse_args()

    if args.compile:
        cmd_compile(args)
    elif args.record:
        cmd_record(args)
    else:
        # Default: compile then record in the same process.
        # This will capture compilation kernels in the trace — use explicit
        # --compile / --record phases to avoid that.
        print('[record_model] WARNING: running compile+record in one process.')
        print('[record_model] Compilation kernels will appear in the trace.')
        print('[record_model] Use --compile first, then --record for a clean trace.')
        cmd_compile(args)
        cmd_record(args)


if __name__ == '__main__':
    main()
