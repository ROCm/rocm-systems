# Example 8: Matrix Multiplication (GEMM) Debugging

## Objective

Debug a **rocBLAS** SGEMM under rocjitsu: catch the classic **column-major**
misuse with a host golden check, and use `RJ_LOG=1` to confirm the GEMM actually
runs on the **matrix cores (MFMA)**.

## Prerequisites

rocBLAS (part of ROCm). It runs under rocjitsu — no physical GPU needed.

## The bug: rocBLAS is column-major

rocBLAS follows the BLAS/Fortran **column-major** convention, but C/C++ matrices
are usually **row-major**. Calling `rocblas_sgemm` the "natural" row-major way
compiles, runs, and returns `rocblas_status_success` — but computes the **wrong
product** (it effectively multiplies the transposes). Nothing errors; only a
numerical check reveals it.

```cpp
// BUG: pass M,N,K and A,B in the intuitive row-major order.
rocblas_sgemm(handle, none, none, M, N, K, &alpha, dA, K, dB, N, &beta, dC, N);
```

### The fix

A row-major `M×K` matrix is, read column-major, its `K×M` transpose. To get
row-major `C = A*B`, compute `Cᵀ = Bᵀ·Aᵀ` — i.e. swap the operands and use
`N, M, K`:

```cpp
// FIXED: column-major-aware call producing row-major C = A*B.
rocblas_sgemm(handle, none, none, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
```

## Files

- `src/gemm_test.cpp` — buggy vs fixed `rocblas_sgemm` + host golden verification
- `Makefile` — links rocBLAS

## Build and run

```bash
cd usage-examples/08-gemm-debugging
make
make run      # buggy vs fixed + host check
make run-log  # same, with RJ_LOG=1 (GEMM dispatch + 'mfma detected')
```

## Expected output

Verbatim from `sharkmi300x-4` (ROCm 7.2.1, `amdgpu_cdna4_kmd.json`), deterministic.

`make run`:

```text
rocBLAS SGEMM debugging: C(32x32) = A(32x32) * B(32x32)
Run under RJ_LOG=1 to see the GEMM dispatch and 'mfma detected'.

[buggy]  naive row-major call     -> FAILED (1007 mismatched elements)
         C[0,0]=-90  expected=-77
[fixed]  column-major-aware call  -> PASSED (0 mismatched elements)
         C[0,0]=-77  expected=-77

Host golden check caught the column-major bug; the fixed call is correct.
```

`make run-log` additionally shows rocjitsu detecting the matrix-core instructions
for each rocBLAS GEMM dispatch:

```text
[rocjitsu] mfma detected in dispatch 5    <- buggy GEMM
[rocjitsu] mfma detected in dispatch 9    <- fixed GEMM
```

## Understanding the `RJ_LOG=1` output

`RJ_LOG=1` enables the kernel-logging plugin, which prints two kinds of lines.

### 1. A dispatch metadata block per kernel

rocBLAS launches several kernels (data movement helpers **and** the GEMM itself),
so you will see multiple blocks. The GEMM is the one flagged with MFMA — for
example:

```text
[rocjitsu] Kernel #5 dispatch
  entry_pc=0x100002e100  grid=[65536,1,1]  wg=[64,1,1]
  wgs=1024  wfs/wg=1  sgprs=32  vgprs=24
[rocjitsu] mfma detected in dispatch 5
```

| Field | Meaning |
|---|---|
| `Kernel #5 dispatch` | 5th kernel launched in this process (rocBLAS setup + GEMM) |
| `entry_pc` | Code-object entry address of the kernel |
| `grid=[65536,1,1]` | Total work-items (rocBLAS/Tensile flattens + tiles the problem) |
| `wg=[64,1,1]` | Workgroup (block) size = 64 threads = 1 wave (wave64) |
| `wgs=1024` | Number of workgroups |
| `wfs/wg=1` | Wavefronts per workgroup |
| `sgprs=32  vgprs=24` | Registers the kernel allocates |

This block comes from the dispatch-packet hook (`onAmdgpuDispatchPacketProcessed`),
which fires when the emulator parses each kernel's AQL dispatch packet.

### 2. `mfma detected in dispatch N` — and how it is detected

This is **not** inferred from the kernel name or metadata. rocjitsu decodes and
executes the kernel's actual machine instructions, and the logging plugin
inspects **every executed instruction** via an after-instruction hook. The check
is simply whether the instruction is a matrix-multiply opcode:

```cpp
// lib/rocjitsu/src/rocjitsu/vm/plugins/logging/plugin.cpp
void KernelLoggingPlugin::onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &inst,
                                                          Wavefront &wf) {
  bool is_mfma = inst.is_mfma() || inst.mnemonic().starts_with("v_wmma_");
  if (is_mfma && mfma_printed_.insert(wf.dispatch_id()).second)
    sink().write("[rocjitsu] mfma detected in dispatch ...");
}
```

- `inst.is_mfma()` is true for CDNA matrix-core opcodes — the `MFMA` instruction
  flag set for `v_mfma_*` / `v_smfmac_*`.
- `v_wmma_*` covers the RDNA matrix instructions (Wave Matrix Multiply-Accumulate).
- The message is printed **once per dispatch** (deduplicated by `dispatch_id`), the
  first time such an instruction actually executes.

Because the signal comes from the **executed instruction stream**, it is direct
evidence that the GEMM ran on the matrix cores:

- **Line present** → rocBLAS selected an MFMA kernel; the matrix cores were used.
- **Line absent** → the GEMM ran a scalar/vector fallback path (no matrix cores)
  — a real, common performance bug you can catch here **without a physical GPU**,
  since rocjitsu emulates the actual code object.

In this example, both the buggy and fixed calls emit the line (dispatch 5 and
dispatch 9): the layout bug is a *numerical* error, not a matrix-core one, so the
kernel still uses MFMA — it just multiplies the wrong operands.

## What rocjitsu does NOT do

- **It does not verify numerics.** The column-major bug returns success and
  produces wrong values silently. The **host golden check** in this example is
  what catches it — always validate GEMM output against a reference.
- It is a functional emulator, so timings/occupancy are not modeled (see
  Examples 6–7).

## Common rocBLAS GEMM issues

| Issue | Symptom | Fix |
|---|---|---|
| Row-major vs column-major | Wrong numbers, `status_success` | Swap operands / use transpose (above) |
| Leading dims (`lda/ldb/ldc`) | `rocblas_status_invalid_size`, or wrong values | For `op=none`, `lda ≥ rows` of that matrix (column-major) |
| Transpose flags | Wrong shape / values | Match `rocblas_operation_*` to your storage |
| Alpha/beta | Result scaled or accumulated unexpectedly | `alpha=1, beta=0` for a plain `C = A*B` |

## Key takeaways

- rocBLAS is **column-major**; the #1 GEMM bug is calling it as if row-major.
- Bugs like this are **numerically silent** — a host golden check is essential.
- `RJ_LOG=1` under rocjitsu confirms **MFMA/matrix-core usage** and the dispatch,
  which is the emulator's real contribution to GEMM debugging.

## Related

- [Example 7: Occupancy analysis](../07-occupancy-analysis/) — VGPR/SGPR from `RJ_LOG`
- [Example 4: LDS data race](../04-data-race-simple/) — `RJ_RACE=1`
