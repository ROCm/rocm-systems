#!/usr/bin/env python3
"""Two synchronized TokenSpeed operations with independent CPU references.

Input shapes and tolerances follow TokenSpeed's gfx950 kernel tests. References
run on CPU, outside the timed operation and ConSan analysis window. No pytest
runner or pytest elapsed time is used by this payload.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time

from consan_gluon_benchmark_workload import _instrumentation_control

RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="


def _prepare(config):
    import torch
    import tokenspeed_kernel as tk

    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16)

    operation = config["operation"]
    if operation == "gemm":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
        )
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            gluon_mm_a16w16_largem_gfx950,
        )
        a = rand(config["m"], config["k"]) * 0.25
        b = rand(config["n"], config["k"]) * 0.25
        reference = a.float() @ b.float().T
        a, b = a.cuda(), b.cuda()
        kernel = (gluon_mm_a16w16_mfma_lds_mediumm_gfx950
                  if config["variant"] == "mediumm" else gluon_mm_a16w16_largem_gfx950)
        return lambda: kernel(a, b, torch.bfloat16), reference, 0.02, 0.02

    if operation == "fp8-gemm":
        # CPU block quantization provides a reference independent of the GPU
        # quantization and GEMM implementation under test.
        m, n, k = (config[key] for key in ("m", "n", "k"))
        a, weight = rand(m, k) * 0.1, rand(n, k) * 0.1
        quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        scales = torch.empty(((n + 127) // 128, (k + 127) // 128))
        dequantized = torch.empty_like(weight, dtype=torch.float32)
        for i in range(scales.shape[0]):
            for j in range(scales.shape[1]):
                block = (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
                source = weight[block].float()
                scale = source.abs().max().clamp_min(1e-12) / 448.0
                scales[i, j] = scale
                quantized[block] = (source / scale).to(torch.float8_e4m3fn)
                dequantized[block] = quantized[block].float() * scale
        reference = a.float() @ dequantized.T
        a, quantized, scales = a.cuda(), quantized.cuda(), scales.cuda()
        return (lambda: tk.mm(a, quantized, B_scales=scales, out_dtype=torch.bfloat16,
                             quant="mxfp8", block_size=[128, 128],
                             override="triton_mm_fp8_blockscale"), reference, 0.05, 0.05)

    if operation == "attention-prefill":
        from tokenspeed_kernel.ops.attention.mha import mha_prefill
        lengths = [851, 914, 1053]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        q = rand(offsets[-1], 8, 64)
        k, v = rand(offsets[-1], 2, 64), rand(offsets[-1], 2, 64)
        expected = []
        for start, end in zip(offsets, offsets[1:]):
            qi = q[start:end].float().transpose(0, 1)
            ki = k[start:end].float().repeat_interleave(4, dim=1).transpose(0, 1)
            vi = v[start:end].float().repeat_interleave(4, dim=1).transpose(0, 1)
            scores = (qi @ ki.transpose(1, 2)) / 8.0
            mask = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            expected.append((scores.masked_fill(mask, -torch.inf).softmax(-1) @ vi).transpose(0, 1))
        reference = torch.cat(expected)
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        return (lambda: mha_prefill(q=q, k=k, v=v, cu_seqlens=cu,
                                   cu_seqlens_cpu=offsets, max_seqlen=max(lengths),
                                   window_left=-1, sinks=None, solution="gluon"),
                reference, 0.03, 0.03)

    if operation == "attention-decode":
        from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
        lengths = [64, 130, 18, 192]
        pages = [(length + 63) // 64 for length in lengths]
        table = torch.zeros(4, 4, dtype=torch.int32)
        k, v = torch.zeros(sum(pages), 64, 2, 64, dtype=torch.bfloat16), torch.zeros(sum(pages), 64, 2, 64, dtype=torch.bfloat16)
        q = rand(4, 8, 64)
        expected, next_page = [], 0
        for batch, (length, count) in enumerate(zip(lengths, pages)):
            table[batch, :count] = torch.arange(next_page, next_page + count)
            ki, vi = rand(length, 2, 64), rand(length, 2, 64)
            k[next_page:next_page + count].view(-1, 2, 64)[:length] = ki
            v[next_page:next_page + count].view(-1, 2, 64)[:length] = vi
            ki = ki.float().repeat_interleave(4, dim=1).transpose(0, 1)
            vi = vi.float().repeat_interleave(4, dim=1).transpose(0, 1)
            scores = (q[batch].float().unsqueeze(1) @ ki.transpose(1, 2)) / 8.0
            expected.append((scores.softmax(-1) @ vi).squeeze(1))
            next_page += count
        reference = torch.stack(expected)
        q, k, v, table = q.cuda(), k.cuda(), v.cuda(), table.cuda()
        lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        return (lambda: mha_decode_with_kvcache(q=q, k_cache=k, v_cache=v,
                                              page_table=table, cache_seqlens=lengths,
                                              max_seqlen_k=256, max_seqlen_q=1,
                                              solution="gluon"), reference, 0.03, 0.03)

    if operation == "moe":
        # DeepSeek-V3 TP=8 expert shape from the single-GPU upstream test.
        tokens, experts, hidden, intermediate, topk = 8, 256, 7168, 256, 8
        x = rand(tokens, hidden)
        w13 = rand(experts, 2 * intermediate, hidden) * 0.05
        w2 = rand(experts, hidden, intermediate) * 0.05
        logits = torch.randn(tokens, experts)
        weights, ids = logits.softmax(-1).topk(topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        reference = torch.zeros(tokens, hidden)
        for token in range(tokens):
            for slot in range(topk):
                expert = int(ids[token, slot])
                gate_up = x[token].float() @ w13[expert].float().T
                activation = torch.nn.functional.silu(gate_up[:intermediate]) * gate_up[intermediate:]
                reference[token] += weights[token, slot] * (activation @ w2[expert].float().T)
        plan = tk.moe_plan("bf16", input_dtype=torch.bfloat16, activation="swiglu",
                           ispp=intermediate, solution="gluon")
        assert plan["apply_kernel_name"] == "gluon_bf16_precomputed_moe_apply", plan
        module = torch.nn.Module()
        module.w13_weight, module.w2_weight = w13.cuda(), w2.cuda()
        module.top_k = topk
        tk.moe_process_weights(plan, module)
        x, logits, weights, ids = x.cuda(), logits.cuda(), weights.cuda(), ids.to(torch.int32).cuda()
        # Match the upstream peak-relative error bound, rather than elementwise
        # relative error around zeros in the weighted expert sum.
        atol = 0.02 * float(reference.abs().max()) + 0.01
        return (lambda: tk.moe_apply(plan, x, module, logits, topk_weights=weights,
                                    topk_ids=ids), reference, atol, 0.0)

    raise ValueError(f"TokenSpeed adapter not yet implemented: {operation}")


def _main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aorta-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config_json)
    import torch
    # TokenSpeed imports Proton/rocprofiler. Load it before HSA initialization
    # so late profiler registration cannot replace the legacy ConSan hooks.
    import tokenspeed_kernel  # noqa: F401

    torch.set_num_threads(16)
    torch.manual_seed(1234)
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx950"):
        raise RuntimeError(f"requires physical gfx950, found {props.gcnArchName}")
    query, begin, end = _instrumentation_control()
    initial = query()
    start_setup = time.perf_counter()
    operation, reference, atol, rtol = _prepare(config)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - start_setup) * 1000
    before_runs = query()
    runs = []
    with torch.inference_mode():
        for ordinal in range(2):
            if ordinal == 0:
                begin()
            before = query()
            start = time.perf_counter()
            try:
                out = operation()
                if out is None:
                    raise RuntimeError("selected kernel rejected the benchmark shape")
                torch.cuda.synchronize()
            finally:
                if ordinal == 0:
                    end()
            elapsed = (time.perf_counter() - start) * 1000
            after = query()
            actual = out.float().cpu()
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            runs.append({"index": ordinal + 1, "phase_ms": elapsed,
                         "instrumentation_ms": (after - before) / 1e6,
                         "result": {"passed": True, "metrics": {
                             "latency_ms": elapsed, "max_abs_error": float((actual - reference).abs().max()),
                             "oracle_atol": atol, "oracle_rtol": rtol}}})
    final = query()
    memory = {"allocated_bytes": torch.cuda.max_memory_allocated(),
              "reserved_bytes": torch.cuda.max_memory_reserved()}
    if max(memory.values()) > 16 * 1024**3:
        raise RuntimeError(f"workload exceeds 16 GiB envelope: {memory}")
    payload = {"result": runs[0]["result"], "runs": runs,
               "phase_ms": {"setup": setup_ms, "run": runs[0]["phase_ms"]},
               "instrumentation_ms": {"before_run": (before_runs - initial) / 1e6,
                                       "during_run": sum(run["instrumentation_ms"] for run in runs),
                                       "total": (final - initial) / 1e6},
               "peak_device_memory": memory,
               "runtime": {"torch": torch.__version__, "hip": torch.version.hip,
                           "device_name": props.name, "architecture": props.gcnArchName,
                           "tokenspeed_kernel": importlib.metadata.version("tokenspeed-kernel"),
                           "tokenspeed_triton": importlib.metadata.version("tokenspeed-triton"),
                           "config": config}}
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
