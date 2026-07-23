# Example 9: PyTorch Inference Debugging

## Objective

Run a small **ROCm PyTorch** model through rocjitsu (no physical GPU), confirm it
executes on the emulated device, and use per-layer hooks to **pinpoint where a
NaN originates and how it propagates**. `RJ_LOG=1` shows the dispatched GPU
kernels — including confirmation that the matmul uses the **matrix cores (MFMA)**.

> This example does **inference only**. A training step (autograd backward +
> optimizer) issues far too many kernels to emulate in practical time — even a
> tiny training loop does not complete. Use rocjitsu for small inference and
> kernel-level inspection; run full training on real hardware.

## What rocjitsu does and does not do

| rocjitsu does | rocjitsu does **not** |
|---|---|
| Run the real ROCm PyTorch workload; `device='cuda'` ops execute as emulated HIP kernels | Check tensor values — NaN/Inf detection here is plain PyTorch (`torch.isnan`) |
| Show kernel dispatches + `mfma detected` via `RJ_LOG=1` | Model training at practical speed (backward/optimizer too heavy) |

## Prerequisites

A **ROCm** PyTorch wheel (`torch+rocm`), not the CPU wheel. Verified with
`torch 2.13.0+rocm7.2`. A `torch+cpu` wheel reports `cuda_available: False` and
would not exercise the emulator.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

## Files

- `src/simple_model.py` — tiny net (16→12→8→4); clean vs corrupt forward pass with per-layer NaN localization
- `Makefile`

## Build and run

```bash
cd usage-examples/09-pytorch-model-debug
make check   # confirm a ROCm PyTorch is installed
make run     # inference in daemon mode (completes; host NaN check)
make run-log # local mode + RJ_LOG=1: kernel dispatches + 'mfma detected'
```

Under emulation a forward pass takes roughly a minute (mostly rocjitsu + PyTorch
startup).

## Expected output

`make run` (daemon mode) — verbatim from `sharkmi300x-4` (ROCm 7.2.1):

```text
PyTorch NaN-localization debugging under rocjitsu
  torch: 2.13.0+rocm7.2   cuda_available: True   device: cuda

[clean] forward pass:
  [hook] fc1 -> clean
  [hook] fc2 -> clean
  [hook] fc3 -> clean
  result: PASSED (no NaN/Inf)

[corrupt] set fc2.weight[0,0] = nan (simulates a corrupted layer):
  [hook] fc1 -> clean
  [hook] fc2 -> NaN/Inf
  [hook] fc3 -> NaN/Inf
  result: FAILED - NaN/Inf originates at 'fc2', propagates through: fc3

NaN debugging worked: clean input passed; corrupt layer was localized.
```

Reading it: the clean pass is the control (all layers clean). In the corrupt pass,
`fc1` is still clean, `fc2` is the **first** layer to report NaN/Inf (the origin —
here because its weights were corrupted), and `fc3` then also reports it (the NaN
**propagated** forward through ReLU). The ordered hook output *is* the propagation
path, and the summary line localizes the origin for you.

(A non-fatal `/sys/class/kfd/.../topology/nodes/0/name cannot be read` warning may
print during device init; PyTorch proceeds normally.)

## Inspecting the GPU kernels (`RJ_LOG=1`)

`RJ_LOG=1` prints a dispatch block per kernel and flags matrix-core use. The
dispatch log appears on **stderr in local mode** (daemon mode routes it to the
daemon process), so `make run-log` uses local mode:

```text
[rocjitsu] Kernel #1 dispatch
  entry_pc=0x5400014100  grid=[512,1,1]  wg=[512,1,1]
...
[rocjitsu] Kernel #7 dispatch
  entry_pc=0x1000042100  grid=[65536,1,1]  wg=[64,1,1]
...
[rocjitsu] mfma detected in dispatch 7
```

- The `mfma detected` line confirms the `nn.Linear` matmul ran on the **matrix
  cores** (see Example 8 for how detection works).
- **Note:** local mode is less isolated than daemon mode; with a full framework
  it may abort partway (segfault) after the kernels you want to see. That is why
  daemon mode is recommended for a clean, complete run — the trade-off is that
  daemon mode does not surface the dispatch log on the app's stderr.

## How the NaN localization works (host-side)

The detection is ordinary PyTorch in a per-layer forward hook — it works the same
with or without rocjitsu, and rocjitsu does not do it for you. Registering a hook
on each **named** layer lets you print status in execution order:

```python
def make_hook(name):
    def hook(module, inputs, output):
        bad = torch.isnan(output).any() or torch.isinf(output).any()
        print(f"  [hook] {name} -> {'NaN/Inf' if bad else 'clean'}")
    return hook

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(make_hook(name))
```

Because hooks fire in execution order, the **first** layer to report `NaN/Inf` is
where the problem originates, and every bad layer after it shows the propagation
path. This example forces the issue by corrupting `fc2`'s weights (a stand-in for
a bad checkpoint); in real code the origin is often unstable inputs, `log(0)` /
division by zero, or — during training — exploding gradients.

## Key takeaways

- rocjitsu lets you run a **ROCm PyTorch** model with **no GPU** and confirm it
  dispatches real HIP kernels (incl. MFMA) via `RJ_LOG=1`.
- **Inference is practical; training is not** under the functional emulator.
- Numerical debugging (NaN/Inf, gradients) is **host-side PyTorch**, not a
  rocjitsu sanitizer.
- Daemon mode gives a clean complete run; local mode surfaces the `RJ_LOG`
  dispatch log (but is less stable for full frameworks).

## Related

- [Example 8: GEMM debugging](../08-gemm-debugging/) — how `mfma detected` works
- [Example 6: Hook profiling](../06-memory-coalescing/) — `RJ_LOG` / profiler scope
