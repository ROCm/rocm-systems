# Example 10: Multi-GPU All-Reduce (manual, no RCCL)

## Objective

Run a multi-GPU workload under rocjitsu on the **2-GPU emulated system** and
implement an **all-reduce** by hand (per-GPU compute → host reduce → broadcast).

> **RCCL does not run under rocjitsu.** RCCL is installed, but its inter-GPU
> transport fails during communicator init under emulation (`amdgpu_bo_import`
> error → abort, even with `HSA_NO_SCRATCH_RECLAIM=1`). So this example does the
> collective manually with basic HIP, which the emulator fully supports. See
> [What works / what does not](#what-works--what-does-not).

## What works / what does not

| Works under rocjitsu | Does **not** work |
|---|---|
| 2-GPU emulation: `hipGetDeviceCount`=2, per-device `hipMalloc`/kernels/`hipMemcpy` | **RCCL collectives** (`ncclCommInitAll` crashes on inter-GPU BO import) |
| Host-mediated collectives (this example) | GPU↔GPU P2P / IPC transport |

## The manual all-reduce

An all-reduce leaves every rank holding the reduction (here, the sum) across all
ranks. Without RCCL:

1. **Per-GPU compute** — GPU `r` fills a buffer with its value `(r+1)` (a kernel per device).
2. **Host reduce** — copy each GPU's buffer to the host and sum element-wise.
3. **Broadcast** — copy the reduced result back to every GPU.
4. **Verify** — every GPU now holds the sum.

For 2 GPUs the values are `1` and `2`, so the all-reduced result is `3` on both.

## Files

- `src/multi_gpu.cpp` — hand-written all-reduce + verification
- `Makefile`

## Configuration

Uses `configs/amdgpu_cdna4_kmd_2gpu.json` (`num_gpus: 2`).

## Build and run

```bash
cd usage-examples/10-multi-gpu-collective
make
make run-2gpu      # run on the 2-GPU config (daemon mode)
make run-2gpu-log  # local mode + RJ_LOG=1: per-GPU kernel dispatches
```

## Expected output

`make run-2gpu` — verbatim from `sharkmi300x-4` (ROCm 7.2.1):

```text
Multi-GPU manual all-reduce (no RCCL)
  GPUs detected: 2   elements: 8

  GPU 0: filled with 1
  GPU 1: filled with 2

  host-reduced value: 3 (expect 3)
  GPU 0 after all-reduce: 3  OK
  GPU 1 after all-reduce: 3  OK

All-reduce PASSED (every GPU holds the sum across ranks)
```

`make run-2gpu-log` adds the kernel dispatch log (interleaved with app output).
`fill_kernel` appears as `grid=[64,1,1]`; the `grid=[512,1,1]` kernels are the HIP
runtime's memcpy helpers used by the host reduce/broadcast:

```text
[rocjitsu] Kernel #1 dispatch
  entry_pc=0x5400009900  grid=[64,1,1]  wg=[64,1,1]
[rocjitsu] Kernel #2 dispatch
  entry_pc=0x540001c100  grid=[512,1,1]  wg=[512,1,1]
...
```

(A non-fatal `/sys/class/kfd/.../topology/nodes/0/name cannot be read` warning may
print during device init.)

## How rocjitsu helps here

- Confirms a multi-GPU app **enumerates and uses both emulated GPUs** with no
  hardware (`hipGetDeviceCount`=2, per-device allocation/compute/copies).
- `RJ_LOG=1` shows the compute and copy kernels dispatched during the collective.

## Debugging tips (host-mediated collectives)

- Ensure every rank participates (loop over `hipGetDeviceCount`).
- Match buffer sizes across ranks before reduce/broadcast.
- Verify the reduced result against a host reference (as here).
- `hipSetDevice(r)` before each device's operations; sync before reading back.

## Limitations

- **No RCCL under emulation** — for real RCCL collectives (AllReduce, Broadcast,
  AllGather, ReduceScatter), run on actual multi-GPU hardware. rocjitsu is useful
  here for functional multi-device logic and per-kernel inspection, not for the
  inter-GPU transport that RCCL needs.

## Related

- [Example 9: PyTorch debugging](../09-pytorch-model-debug/) — daemon mode, `RJ_LOG`
- `configs/amdgpu_cdna4_kmd_2gpu.json` — the 2-GPU config
