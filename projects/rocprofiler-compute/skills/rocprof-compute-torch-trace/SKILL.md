---
name: rocprof-compute-torch-trace
description: Attributes AMD GPU kernel performance counters back to the PyTorch or Triton operators that launched them, using rocprof-compute ML API tracing. Use when the user asks which PyTorch layer or operator is slow, wants to map GPU kernels to model code, profiles a training or inference script, asks about torch.compile and Inductor kernels, or wants to select user-defined ROCTx regions from a traced Python profile. Not for CUDA tools, Windows, compiled non-Python binaries, or starting and stopping a profile at ROCTx ranges.
---

# Map GPU kernels to PyTorch operators

Profiling a PyTorch model with counters alone gives kernel names, which say
little about the model. Torch tracing annotates the run so each kernel can be
traced back to the operator that launched it.

Torch and Triton tracing are experimental. Every command needs
`--experimental`, and option names can change between releases, so run
`rocprof-compute profile --experimental --help` first.

## 1. Check the requirements

Torch tracing needs PyTorch installed against a matching ROCm. PyTorch 2.13
and 2.14 use the prebuilt native collector, which records operators on every
thread, including autograd workers. Other versions, or a missing or unusable
collector, warn and use `TorchDispatchMode` with reduced coverage on the Python
thread where it is active. Confirm the version and check the loader message
before interpreting missing backward operators as an absence of GPU work.

The workload must be a Python command or Python script; the tracer cannot
annotate a compiled binary. The collector is a generic prebuilt
`torch_trace_collector.so` and needs no PyTorch or Python development packages
to build. The workload still needs Python-compatible `roctx` bindings. The
full requirement list is in the Torch trace section of
[profile mode](../../docs/how-to/profile/mode.rst).

## 2. Collect

```bash
rocprof-compute profile \
    --experimental \
    --trace-ml-api torch \
    --no-roof \
    --name <workload_name> \
    -- python <script.py> <args>
```

For Triton kernels launched through Triton's Python runtime (`@triton.jit`):

```bash
rocprof-compute profile --experimental --trace-ml-api triton --name <name> -- python <script.py>
```

`torch.compile` / Inductor fused kernels use PyTorch's static launcher, so they
do not get `|triton` ranges. Attribute them with `--trace-ml-api torch`.

The two combine as `--trace-ml-api torch triton`. User-defined ROCTx regions
are included with every `--trace-ml-api` run. A bare `--trace-ml-api` traces
all supported ML frameworks as well.

Tracing adds overhead on top of the usual counter replay. Profile a few
iterations of a steady-state model rather than a full training run, and skip
warmup. `ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=0` reduces instrumentation
overhead by dropping the device and layout wraps; see
[profile mode](../../docs/how-to/profile/mode.rst) for what that costs you.

Tracing cannot be combined with attaching to a running process, and cannot be
combined with a PC-sampling-only profile.

Profile preserves each pass's raw marker and counter CSVs with the
`ml_api_trace` prefix.

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs the roofline benchmarks.

- List what this GPU can collect with `rocprof-compute --list-metrics`, and a
  block's metrics with `rocprof-compute --list-metrics <alias>`. On ROCm 10.2
  and earlier, use `--list-blocks <gfx_arch>` and `--list-metrics <gfx_arch>`.
  Pass `-b` only the block aliases, block ids, or metric ids (for example
  `2.1.0`) the question needs. Aliases differ by architecture.
- `--set <set_name>` collects a predefined topic in one pass.
  `rocprof-compute profile --list-sets` lists the sets.
- Add `-k <kernel_name>` when the user names the kernels that matter.
- Add `--no-roof` unless the question is about roofline. A `-b` list without
  block 4 (`roof`) already skips it.
- `--iteration-multiplexing kernel` spreads counters over repeated dispatches of
  each kernel, so it only helps when every profiled kernel is dispatched many
  times. Pair it with `-k`. See Iteration multiplexing in
  [profile mode](../../docs/how-to/profile/mode.rst).
- Tell the user which analysis panels the narrower profile leaves empty.

## 3. List the captured operators

List when you need the operator names from this run; filtering does not
require a prior list. Do not pass `--list-operators` and `--operator`
on the same command; listing ignores the filter. These flags require
`--output-format stdout` (the default):

```bash
rocprof-compute analyze \
    --path ./workloads/<name>/<gpu_model> \
    --experimental --list-operators torch
```

This prints the operators as a call tree sorted by GPU duration, with
`file:line` on each node that recorded a source location, plus kernel
launch statistics, so it also shows which operators are worth opening.

The Triton equivalent is `--list-operators triton`.

`--list-operators user` lists each user ROCTx region with everything that
ran inside it.

## 4. Filter to an operator

```bash
rocprof-compute analyze \
    --path ./workloads/<name>/<gpu_model> \
    --experimental --operator 'torch:*relu*'
```

Patterns are shell-style globs, not regexes, matched against a `/`-separated
operator hierarchy. `*relu*` contains relu, `*/aten::addmm` is an addmm leaf
on the reconstructed path. Several patterns can be given. A pattern without
a prefix uses the prefix before it. Bare `--operator` or `**` matches
everything.
Quote patterns so the shell does not expand them.

`--operator 'triton:*rmsnorm_kernel*'` works the same way.

`--operator 'user:*UserDefined*'` selects matching user ROCTx regions with
everything inside them. Other patterns in the same command add operators.
They never remove anything from a selected region.

Worked examples are in the PyTorch operator analysis section of
[the analyze CLI guide](../../docs/how-to/analyze/cli.rst).

## 5. Read the result

Attribution tells you which operator owns the kernel time. It does not tell you
why that kernel is slow. Once an operator is identified, analyze its kernels
like any other: Speed-of-Light first, then the memory or roofline path.

An operator that is slow because it launches thousands of tiny kernels is a
different problem from one slow kernel, and the launch statistics in the
operator list are what distinguish the two.

## 6. ROCTx scope

rocprof-compute uses ROCTx internally: torch tracing injects ROCTx regions to
annotate operators, which is what makes attribution possible.

rocprof-compute does not start, stop, or resume counter collection at a user-authored
ROCTx range. User-defined ROCTx regions are included with every
`--trace-ml-api` run. List or filter them at analyze time with
`--list-operators user` and `--operator user:PATTERN`. That keeps the region
and nested GPU work.
To keep only some kernels or dispatches from an already collected profile,
use `-k` or `-d`.
