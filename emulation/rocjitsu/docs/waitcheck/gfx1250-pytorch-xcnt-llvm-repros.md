# gfx1250 XCNT hazards in PyTorch kernels

## Summary

We investigated 25 distinct gfx1250 kernel entries sampled from the PyTorch
`2.11.0+rocm7.15.0a20260717` wheel. Exact source rebuilds, physical-MIR
controls, LLVM pass dumps, and final-ISA checks reduce the confirmed reports
to four root-cause patterns:

1. `SIInsertWaitcnts` relies on a branch to drain XCNT, then
   `SIPreEmitPeephole` removes the branch without preserving the drain. This
   accounts for 11 sampled kernel entries.
2. XCNT scoring includes liveness-only implicit super-register operands on
   spill pseudos. This accounts for the `warpMergeSortTopK<double>` kernel.
3. rocPRIM's RDNA4 `atomic_store(__uint128_t)` inline assembly waits for
   STORE_CNT but omits the independent XCNT source-register wait. LLVM cannot
   repair the opaque assembly.
4. `SIPreEmitPeephole::removeExeczBranch()` removes a different branch used by
   `SIInsertWaitcnts` as an XCNT drain. This is independently reproducible with
   a normal `GLOBAL_STORE_DWORDX4` physical instruction.

All four are real wait hazards according to LLVM's own gfx1250 XCNT model. The
reduced testcases make the final ISA issues reproducible, but we have not
demonstrated runtime corruption. They cannot be dismissed as irrelevant to an
XNACK-disabled execution mode: gfx1250 has XNACK always enabled and does not
support selecting an XNACK-off mode. The second ten-entry sample did not expose
a new hazard class: eight are compiler-generated variants of the existing
XCNT dependencies, and two are additional instances of the rocPRIM
inline-assembly bug. One additional true16 case in `torch.117.co`, entry
`0x33c000`, also agrees with LLVM's regunit model, but its first unsafe pass is
not yet pinned down; it remains a minimization task rather than a separately
attributed bug report.

The post-correction exhaustive checkpoint completed all 232 code objects and
41,124 kernels with zero analysis errors. It analyzed 137,101,030
instructions and 11,846,488 memory events in 11 minutes 51 seconds at `-j12`,
emitting 19,396 diagnostics across 1,163 kernels in 72 objects. Compared with
the earlier 18,772-diagnostic checkpoint, the VGPR and SGPR results are
identical; the exact increase of 624 is the newly covered explicit-EXEC
overwrites described below. The Wave32 indirect-call false positives are gone.

The attached files are compiler-only reproducers. Their generated kernels and
code objects must not be executed:

- [`repro-gfx1250-xcnt-branch-delete.ll`](repro-gfx1250-xcnt-branch-delete.ll)
- [`repro-gfx1250-xcnt-spill.ll`](repro-gfx1250-xcnt-spill.ll)
- [`repro-gfx1250-xcnt-rocprim-inline-asm.hip`](repro-gfx1250-xcnt-rocprim-inline-asm.hip)
- [`repro-gfx1250-xcnt-remove-execz.mir`](repro-gfx1250-xcnt-remove-execz.mir)

## Toolchain snapshot

- Target: `gfx1250`, wave32
- AMD clang: `23.0.0git`, ROCm LLVM commit
  `723bffa5dfbf92e452b0d4a0df674bdd849fcf12`
- PyTorch wheel: `2.11.0+rocm7.15.0a20260717`
- PyTorch source: `6f6864920e0c21f7a59eb6ba2186d08ce0ec6d49`
- Waitcheck source: `users/kuhar/waitcheck`; the initial sweep used
  `73cc0e0353aae4cc02af5016ee27a8a057681cd7`, and the explicit-EXEC reporting
  correction described below is `e6b3756631`

The commands below use the compiler from that wheel environment. A compiler
built from the named ROCm LLVM commit can be substituted.

```sh
AMDCLANG=/path/to/amdclang++
LLC=/path/to/llc
WAITCHECK=/path/to/rj_waitcheck
```

## Why XNACK does not make these reports harmless

XCNT tracks the lifetime of memory-instruction register operands until their
XNACK response has been received. On a target where XNACK can genuinely be
disabled, reusing those operands without an XCNT wait would not create this
particular replay hazard. That qualification does not suppress any of the
gfx1250 results in this report.

LLVM defines gfx1250 through `FeatureISAVersion12_50_Common`, which includes
both `FeatureXNACK` and `FeatureWaitXcnt`. It deliberately does not include
`FeatureXNACKOnOffModes`. The regression
`llvm/test/CodeGen/AMDGPU/target-id-xnack-always-on.ll` states the consequence
directly: gfx1250, gfx1251, and the gfx12.5 generic target have XNACK always on,
and their target IDs therefore omit `xnack+`/`xnack-` modifiers. Even an
explicit `-mattr=-xnack` does not create a gfx1250 XNACK-off target ID.

The absence of `xnack+` in a gfx1250 code-object target string must therefore
not be read as XNACK being disabled. All 232 PyTorch objects in this sweep have
ELF flags `0x549`, decoded by LLVM as gfx1250 with
`EF_AMDGPU_FEATURE_XNACK_ANY_V4` and `EF_AMDGPU_FEATURE_SRAMECC_ANY_V4`.
For a target without XNACK on/off modes, LLVM's `TargetID::isXnackOnOrAny()`
treats that `ANY` state as enabled. It is not an opt-out and there is no
per-kernel switch that would make these XCNT dependencies inert.

Consequently, an XNACK-based filter in waitcheck would be useful only for a
future XCNT-capable target or execution mode where XNACK can actually be off.
Applying such a filter to these gfx1250 objects would incorrectly hide the
reported hazards.

## Reproducer 1: an XCNT-draining branch is removed

Compile the reduced LLVM IR to assembly and a code object:

```sh
$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir -S repro-gfx1250-xcnt-branch-delete.ll \
  -o repro-gfx1250-xcnt-branch-delete.s

$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir repro-gfx1250-xcnt-branch-delete.ll \
  -o repro-gfx1250-xcnt-branch-delete.hsaco
```

The final assembly has a byte store immediately followed by definitions of its
data and address registers, with no `s_wait_xcnt`:

```asm
v_mov_b64_e32 v[2:3], 0
v_mov_b16_e32 v0.l, 0
global_store_b8 v[2:3], v0, off
v_cvt_f64_i32_e32 v[0:1], s1
v_mov_b32_e32 v0, 0
```

Waitcheck reports five definitions that can race the store's outstanding
address translation. The first is sufficient to show the problem:

```text
.text+0x184: missing s_wait_xcnt 0 before def of v0
  producer .text+0x178: global_store_b8 v[2:3], v0, NULL
  consumer .text+0x184: v_cvt_f64_i32_e32 v[0:1], s1
```

It can be reproduced with:

```sh
$WAITCHECK repro-gfx1250-xcnt-branch-delete.hsaco \
  --target gfx1250 --no-fail --max-diagnostics 10
```

### Pass-by-pass analysis

Capture the relevant machine pass dumps with:

```sh
$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir -S repro-gfx1250-xcnt-branch-delete.ll -o /dev/null \
  -mllvm -filter-print-funcs=repro \
  -mllvm -print-after=si-insert-waitcnts \
  -mllvm -print-after=si-pre-emit-peephole \
  2>branch-passes.txt
```

Immediately after `SIInsertWaitcnts`, the store is followed by a branch:

```text
GLOBAL_STORE_BYTE_t16 ... $vgpr2_vgpr3, ... $vgpr0_lo16 ...
$vcc_lo = S_ANDN2_B32 $exec_lo, $sgpr0
S_CBRANCH_VCCNZ %bb.16
...
$vgpr0_vgpr1 = V_CVT_F64_I32_e32 $sgpr1
```

There is no explicit XCNT wait because `SIInstrInfo::isXcntDrain()` returns
true for every branch (`SIInstrInfo.cpp:3539-3542`). `SIInsertWaitcnts` applies
`X_CNT = 0` to its internal score at that branch
(`SIInsertWaitcnts.cpp:3070-3073`). At this point LLVM's accounting is
internally consistent.

After `SIPreEmitPeephole`, the constant-false `S_CBRANCH_VCCNZ` and its
`S_ANDN2_B32` setup are gone:

```text
GLOBAL_STORE_BYTE_t16 ... $vgpr2_vgpr3, ... $vgpr0_lo16 ...
...
$vgpr0_vgpr1 = V_CVT_F64_I32_e32 $sgpr1
```

`SIPreEmitPeephole::optimizeVccBranch()` explicitly erases a `VCCNZ` branch
whose mask is zero (`SIPreEmitPeephole.cpp:322-330`). Wait insertion is not
rerun and the implicit drain is not materialized, so the final program no
longer contains the event on which the earlier analysis relied.

Possible fixes include moving this branch simplification before wait
insertion, preserving a required `S_WAIT_XCNT 0` when an XCNT-draining branch
is erased, or repairing/rerunning wait insertion afterward. The important
property is that a late pass must not remove an event used as an implicit drain
without preserving its synchronization effect.

### Original kernels with this pattern

We observed the same `SIInsertWaitcnts` then `SIPreEmitPeephole` transition in
11 independently sampled kernel entries:

| PyTorch object and kernel | Source provenance | Representative final hazard |
| --- | --- | --- |
| `torch.113.co`, `launch_masked_scatter_kernel`, entry `0x196600` | `aten/src/ATen/native/cuda/IndexKernel.cu:405-455` | `global_load_u8 v0, v[4:5]` at `0x19dd08`, then definition of address `v4` at `0x19dd14`; requires `s_wait_xcnt 0` |
| `torch.120.co`, `binary_cross_entropy_out_cuda<double>`, entry `0x113f00` | `aten/src/ATen/native/cuda/Loss.cu:71-102` | `global_store_b8 v[18:19], v0` at `0x1221e8`, then definition of data `v0` at `0x1221f4`; requires `s_wait_xcnt 0` |
| `torch.58.co`, `div_trunc_kernel_cuda<short>`, entry `0x1c4400` | `aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu:18-48` | `global_store_d16_hi_b8 v[8:9], v1` at `0x1cae2c`, then definition of `v1` at `0x1cae38`; requires `s_wait_xcnt 0` |
| `torch.165.co`, rocPRIM `reduce_by_key<unsigned char, long>`, entry `0xbdd00` | Triggered by `aten/src/ATen/native/cuda/TensorModeKernel.cu:23-81` | `s_load_b64 s[34:35], s[0:1]` at `0xbdfc0`, then definition of address `s0` at `0xbdfcc`; requires `s_wait_xcnt 0` |
| `torch.161.co`, `heaviside<BFloat16>`, entry `0x516b00` | `aten/src/ATen/native/cuda/StepKernel.cu:22` | `global_load_u16 v3, v[0:1]` at `0x51e100`, then definition of `EXEC_LO` at `0x51e13c`; requires `s_wait_xcnt 0` |
| `torch.73.co`, `copysign<float>`, entry `0x92600` | `aten/src/ATen/native/cuda/CopysignKernel.cu:23` | `global_store_b8 v[0:1], v2` at `0x93648`, then definition of `EXEC_LO` at `0x9366c`; requires `s_wait_xcnt 0` |
| `torch.165.co`, rocPRIM `merge_sort_block_merge<long, long>`, entry `0x170f00` | Triggered by `aten/src/ATen/native/cuda/TensorModeKernel.cu` | `s_load_b64 s[2:3], s[0:1]` at `0x171278`, then definition of address `s0` at `0x171290`; requires `s_wait_xcnt 0` |
| `torch.58.co`, `div_trunc_kernel_cuda<short>`, entry `0x1cca00` | `aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu:18` | `global_store_d16_hi_b8 v[6:7], v3` at `0x1d4aa4`, then definitions of payload `v3` and address `v6`; requires `s_wait_xcnt 0` |
| `torch.133.co`, `addcmul<uint8_t>`, entry `0x4a000` | `aten/src/ATen/native/cuda/PointwiseOpsKernel.cu:24` | `global_load_u8 v4, v[12:13]` at `0x5483c`, then definitions of both address halves at `0x54850` and `0x54858`; requires `s_wait_xcnt 0` |
| `torch.161.co`, `nextafter<double>`, entry `0x74e00` | `aten/src/ATen/native/cuda/StepKernel.cu:14` | Two unrolled `global_store_b8` producers are each followed by payload and address definitions; 18 reports share two producer bugs |
| `torch.120.co`, `binary_cross_entropy_out_cuda<float>`, entry `0x1cc300` | `aten/src/ATen/native/cuda/Loss.cu:71` | `global_store_b8 v[2:3], v1` at `0x1d8cf4`, then payload and address definitions, including a later packed-dual definition of `v3` |

These are ordinary HIP/CUDA or rocPRIM template instantiations compiled by
clang, not handwritten memory instructions. The common failure is in late
machine-code transformation, not in the source operation.

### Ten-entry follow-up sample

A fresh exhaustive sweep selected ten kernel entries disjoint from the first
15. Each reported producer reaches its consumer by ordinary fallthrough with
no intervening XCNT drain. LLVM's physical-register model agrees with all ten
dependencies: XCNT scores the VMEM/SMEM address, payload, and EXEC uses, and a
later definition of an overlapping regunit requires the indicated threshold.
The partial-register and packed-dual cases were also checked against LLVM's
decoded physical operands rather than inferred from disassembly spelling.

| PyTorch object and kernel | Source provenance | Representative final hazard | Classification |
| --- | --- | --- | --- |
| `torch.117.co`, `addr<signed char>`, entry `0xdbc00` | `aten/src/ATen/native/cuda/LinearAlgebra.cu:18` | `global_load_u8 v18, v[32:33]` at `0xe5bc0`, then address definition `v_add_nc_u64 v[32:33]` at `0xe5bd8` | True positive; compiler-generated VMEM address XCNT dependency |
| `torch.119.co`, rocPRIM scan for `logcumsumexp<double>`, entry `0x44a00` | Launched by `aten/src/ATen/native/cuda/LogcumsumexpKernel.cu:104` | `global_store_b128 v[10:11], v[6:9]` at `0x4d5dc`, then payload definition `v_add_nc_u32 v8` at `0x4d600` | True positive; rocPRIM uint128 inline assembly omits the XCNT wait |
| `torch.131.co`, rocPRIM partition for `nonzero`, entry `0x64700` | `aten/src/ATen/native/cuda/Nonzero.cu:170-283` | `global_store_b128 v[18:19], v[14:17]` at `0x65010`, then `s_or_b32 EXEC_LO` at `0x65020` | True positive; the same rocPRIM inline assembly also releases its implicit EXEC source too early |
| `torch.161.co`, `heaviside<BFloat16>`, entry `0x527100` | `aten/src/ATen/native/cuda/StepKernel.cu:22` | `global_load_u16 v6, v[4:5]` at `0x52ef70`, then EXEC definition at `0x52efac` | True positive; compiler-generated implicit-EXEC XCNT dependency |
| `torch.165.co`, rocPRIM `reduce_by_key<signed char,long>`, entry `0x118600` | Triggered by `aten/src/ATen/native/cuda/TensorModeKernel.cu` | `s_load_b64 s[34:35], s[0:1]` at `0x1188c0`, then address-base definition `s_mov_b32 s0` at `0x1188cc` | True positive; compiler-generated SMEM base XCNT dependency |
| `torch.170.co`, `angle<double>`, entry `0xcc00` | `aten/src/ATen/native/cuda/UnaryComplexKernels.cu:32` | `global_store_b8 v[16:17], v0` at `0x15b2c`, then payload definition `v_cndmask_b32 v0` at `0x15b60` | True positive; compiler-generated VMEM payload XCNT dependency |
| `torch.184.co`, `log2<double>`, entry `0x1d0400` | `aten/src/ATen/native/cuda/UnaryLogKernels.cu:88` | `global_store_b8 v[12:13], v0` at `0x1d8878`, then `v_frexp_mant_f64 v[0:1]` at `0x1d8884` | True positive; LLVM's 64-bit destination overlaps the locked payload `v0` |
| `torch.185.co`, `exp<BFloat16>`, entry `0x104b00` | `aten/src/ATen/native/cuda/UnaryOpsKernel.cu:38-65` | `global_store_b8 v[6:7], v1` at `0x10eb08`, then `v_mov_b16 v1.l` at `0x10eb14` | True positive; LLVM regunits make the low-half definition overlap the store payload |
| `torch.219.co`, `_fake_quant_per_channel_cachemask<float>`, entry `0x52ee00` | `aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu:169-214` | `global_store_b8 v[6:7], v3` at `0x540040`, then `v_div_scale_f32` defines `v3` at `0x54004c` | True positive; LLVM and waitcheck agree on the multi-result instruction's `v3` definition |
| `torch.36.co`, `hardsigmoid_backward<float>`, entry `0xd4700` | `aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu:44` | `global_store_b8 v[2:3], v1` at `0xdfb58`, then the Y slot of a VOPD instruction defines `v1` at `0xdfb88` | True positive; exact-source pass dumps pin this instance to `SIPreEmitPeephole::optimizeVccBranch()` |

The exact `hardsigmoid_backward` source rebuild reproduces four diagnostics at
the same entry and instruction offsets. Immediately after
`SIInsertWaitcnts`, a constant-false `S_CBRANCH_VCCNZ` drains XCNT between the
store and these definitions. `SIPreEmitPeephole::optimizeVccBranch()` removes
that branch, yielding the unwaited final sequence. This is the same
pass-to-pass failure isolated by Reproducer 1. The other compiler-generated
rows have the same final physical dependency, regardless of whether their
first unsafe transformation is minimized separately.

## Reproducer 2: spill-pseudo implicit operands corrupt XCNT scoring

The second reproducer is larger because it must retain enough register
pressure to generate the spill pseudos. `llvm-reduce` reduced it from 283 KiB
of source-derived IR to 27 KiB while requiring the final scratch-store hazard
to remain.

Compile and inspect it with:

```sh
$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir -S repro-gfx1250-xcnt-spill.ll \
  -o repro-gfx1250-xcnt-spill.s

$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir repro-gfx1250-xcnt-spill.ll \
  -o repro-gfx1250-xcnt-spill.hsaco

$WAITCHECK repro-gfx1250-xcnt-spill.hsaco \
  --target gfx1250 --no-fail --max-diagnostics 10
```

The reduced code reports:

```text
.text+0x1710: missing s_wait_xcnt 1 before def of v28
  producer .text+0x16f8: scratch_store_b128 v0, v[28:31], NULL nv
  consumer .text+0x1710: v_mov_b64_e32 v[28:29], v[52:53]
```

The corresponding final assembly is:

```asm
scratch_store_b128 off, v[0:3], off offset:396 nv
scratch_store_b128 off, v[4:7], off offset:412 nv
scratch_store_b128 off, v[8:11], off offset:428 nv
scratch_store_b128 off, v[12:15], off offset:444 nv
scratch_store_b128 off, v[16:19], off offset:460 nv
scratch_store_b128 off, v[20:23], off offset:476 nv
scratch_store_b128 off, v[24:27], off offset:492 nv
scratch_store_b128 off, v[28:31], off offset:508 nv
scratch_load_b64 v[0:1], off, off offset:384 th:TH_LOAD_LU nv
v_mov_b64_e32 v[28:29], v[52:53]
```

There is no `s_wait_xcnt 1` or stronger wait before `v28` is overwritten.

Capture the machine state around wait insertion with:

```sh
$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -nogpulib \
  -x ir -S repro-gfx1250-xcnt-spill.ll -o /dev/null \
  -mllvm -filter-print-funcs=repro_spill \
  -mllvm -print-before=si-insert-waitcnts \
  -mllvm -print-after=si-insert-waitcnts \
  2>spill-passes.txt
```

The first `SCRATCH_STORE_DWORDX4_ST` in the spill sequence has an explicit
`$vgpr0_vgpr1_vgpr2_vgpr3` source, but it also carries liveness-only implicit
use and def operands covering `$vgpr0` through `$vgpr31`. The subsequent spill
stores carry the individual explicit four-register sources. The `v28:v29`
definition remains unwaited after `SIInsertWaitcnts`.

For XCNT events, `WaitcntBrackets::updateByEvent()` scores
`Inst.all_uses()` (`SIInsertWaitcnts.cpp:1222-1223`), so these liveness-only
super-register operands participate in XCNT tracking. The generic
destination-scoring path immediately below explicitly documents why
spill-store implicit super-register operands must be excluded: they create
artificial dependencies and exist only for liveness accounting
(`SIInsertWaitcnts.cpp:1234-1245`). The same pseudo shape reaches the XCNT
source-scoring path, which does not currently apply an equivalent explicit-use
filter.

Our working diagnosis is that the artificial super-register score interferes
with the later subregister updates, allowing the real `v28:v31` scratch-store
source lock to be considered retired too early. Mirroring the existing
explicit-operand treatment for XCNT is the most direct area to investigate.

The original wheel case was `torch.167.co`,
`warpMergeSortTopK<3,3,512,1,double,unsigned,false>` at entry `0x308d00`, from
`aten/src/ATen/native/cuda/TensorTopK.cu:430` onward. Its unreduced ISA reports
three hazards:

```text
scratch_store_b128 v0, v[6:9]  -> v_mov_b64 v[8:9]   requires xcnt <= 8
scratch_store_b128 v0, v[10:13] -> v_mov_b64 v[10:11] requires xcnt <= 7
scratch_store_b128 v0, v[10:13] -> v_mov_b64 v[12:13] requires xcnt <= 7
```

## Reproducer 3: rocPRIM RDNA4 uint128 store waits the wrong domain

This is a **rocPRIM library bug**, not an LLVM pass bug. rocPRIM emits the
memory instruction as opaque inline assembly, so LLVM cannot discover its
VMEM/XCNT event. The inline assembly waits for store completion but releases
its input registers without waiting for their XCNT source lock.

[`repro-gfx1250-xcnt-rocprim-inline-asm.hip`](repro-gfx1250-xcnt-rocprim-inline-asm.hip)
contains an unsafe kernel and a fixed control. They are code-generation tests
only; do not execute them because they use address zero.

Compile and check the two kernel entries with the compiler from the gfx1250
wheel:

```sh
$AMDCLANG -x hip --offload-arch=gfx1250 --offload-device-only \
  --no-gpu-bundle-output -O3 -c \
  repro-gfx1250-xcnt-rocprim-inline-asm.hip \
  -o repro-gfx1250-xcnt-rocprim-inline-asm.co

$WAITCHECK repro-gfx1250-xcnt-rocprim-inline-asm.co \
  --target gfx1250 --kernel-entry 0x0 --no-fail
$WAITCHECK repro-gfx1250-xcnt-rocprim-inline-asm.co \
  --target gfx1250 --kernel-entry 0x100 --no-fail
```

The unsafe kernel reports:

```text
.text+0x2c: missing s_wait_xcnt 0 before def of v2
  producer .text+0x1c: global_store_b128 v[0:1], v[2:5], NULL
  consumer .text+0x2c: v_mov_b32_e32 v2, 1
```

The fixed kernel reports `diagnostics=0`. Its only material difference is the
XCNT wait:

```asm
global_store_b128 v[0:1], v[2:5], off scope:SCOPE_DEV
s_wait_storecnt 0x0
s_wait_xcnt 0x0
v_mov_b32_e32 v2, 1
```

### Root cause

The shipped rocPRIM implementation is
`rocprim/intrinsics/atomic.hpp:284-305`: the inline-assembly macro is defined
at lines 289-290, and its RDNA4 global form at lines 304-305 emits
`global_store_b128` followed only by `s_wait_storecnt 0`. LLVM keeps STORE_CNT
and X_CNT separate in `SIInsertWaitcnts.cpp:1815-1835`; it scores all source
registers for XCNT at lines 1212-1223, requires waits before physical
definitions at lines 2740-2741, and explicitly does not simplify XCNT through
a pending store at lines 1537-1543. Because the producer is hidden inside an
`INLINEASM` instruction, none of those normal MachineInstr rules can repair
the library's incomplete assembly contract.

The original reports were the rocPRIM scan kernels in `torch.29.co`, entry
`0x35700` (payload reuse), and `torch.189.co`, entry `0x1ca00` (address and EXEC
reuse). PyTorch reaches the operation through its normal scan use; the
handwritten ISA belongs to rocPRIM.

## Reproducer 4: `removeExeczBranch` deletes an XCNT drain

This is a separate **LLVM backend bug**. The direct reproducer is physical MIR
because the defect is a relation between two late machine passes; MIR keeps
the reproducer small and deterministic instead of retaining the 1.9 MiB
source-derived IR needed to preserve this exact EXECZ route.

Run the two passes independently on
[`repro-gfx1250-xcnt-remove-execz.mir`](repro-gfx1250-xcnt-remove-execz.mir):

```sh
$LLC -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
  -run-pass=si-insert-waitcnts repro-gfx1250-xcnt-remove-execz.mir \
  -o remove-execz.after-wait.mir

$LLC -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
  -run-pass=si-pre-emit-peephole remove-execz.after-wait.mir \
  -o remove-execz.after-preemit.mir
```

After `SIInsertWaitcnts`, the region is unchanged and internally safe:

```text
GLOBAL_STORE_DWORDX4 ... $vgpr2_vgpr3, $vgpr18_vgpr19_vgpr20_vgpr21 ...
S_WAIT_STORECNT 0
S_CBRANCH_EXECZ %bb.2, implicit $exec
...
$vgpr2 = V_MOV_B32_e32 0, implicit $exec
```

No explicit XCNT wait is needed at this point because the branch is itself an
XCNT drain. After `SIPreEmitPeephole`, the branch is gone:

```text
GLOBAL_STORE_DWORDX4 ... $vgpr2_vgpr3, $vgpr18_vgpr19_vgpr20_vgpr21 ...
S_WAIT_STORECNT 0
...
$vgpr2 = V_MOV_B32_e32 0, implicit $exec
```

Produce a final code object for waitcheck with:

```sh
$LLC -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
  -start-after=si-pre-emit-peephole -filetype=obj \
  remove-execz.after-preemit.mir -o remove-execz.o

$AMDCLANG --target=amdgcn-amd-amdhsa -mcpu=gfx1250 \
  -nostdlib -shared remove-execz.o -o remove-execz.hsaco

$WAITCHECK remove-execz.hsaco --target gfx1250 --no-fail
```

The final object is four instructions and reports:

```text
.text+0x10: missing s_wait_xcnt 0 before def of v2
  producer .text+0x0: global_store_b128 v[2:3], v[18:21], NULL
  consumer .text+0x10: v_mov_b32_e32 v2, 0
```

### Root cause

LLVM explicitly classifies every branch as an XCNT drain in
`SIInstrInfo.cpp:3539-3542`; `SIInsertWaitcnts` applies `X_CNT=0` to its state
for such instructions in `SIInsertWaitcnts.cpp:3070-3073`. Later,
`SIPreEmitPeephole::removeExeczBranch()` erases a legal forward
`S_CBRANCH_EXECZ` without preserving that synchronization effect
(`SIPreEmitPeephole.cpp:487-512`, dispatched at lines 790-802). Wait insertion
does not run again.

The original `torch.189.co` kernel exhibits this exact before/after
transition. Its earlier EXEC overwrites are independently unsafe because of
the rocPRIM inline-assembly defect above; the deleted branch specifically
creates the later explicit-address `v2` hazard. The native-MIR producer proves
that the compiler defect is not limited to opaque inline assembly.

## Waitcheck correction found during the audit

The two implicit-EXEC samples initially named a later `_saveexec_` consumer
instead of the first `s_or_b32 EXEC_LO` overwrite. gfx1250 operand display
names use uppercase `EXEC_LO`, while waitcheck compared only lowercase names.
Commit `e6b3756631` makes the comparison case-insensitive and adds load/store
regressions. All 285 `WaitcheckTest` tests pass, and the exact `torch.161.co`
and `torch.73.co` replays now report the earlier consumers. This was a
waitcheck false negative in diagnostic coverage, not a false positive in the
underlying LLVM hazards.

The follow-up exhaustive run found a second, unrelated waitcheck false-positive
class in `torch.62.co`. Wave32 VOPC instructions define one physical SGPR, but
the indirect-branch discovery pass retained the decoder's maximum-width SGPR
pair. A compare defining `s3` was therefore treated as defining `s[3:4]`,
which falsely killed an adjacent `s[4:5]` PC builder. Waitcheck then failed to
recover an `s_swap_pc_i64` helper call and bypassed the helper's entry
`s_wait_loadcnt_dscnt 0` in its CFG.

The correction passes the known kernel wave size into reachable CFG recovery
and normalizes VOPC and `v_div_scale` wave-mask definitions before SGPR-clobber
analysis. The exact 56,806-instruction kernel at entry `0x343400` changed from
a large false-positive stream to zero diagnostics in 1.03 seconds. Exhaustive
analysis of all 440 kernels in `torch.62.co` now completes in 14.7 seconds and
reports four diagnostics instead of 1,314,600 false producer/consumer pairs.
A focused Wave32 `s3`/`s[4:5]` regression and all 305 waitcheck/C API tests
pass. This CFG bug did not affect any of the ten entries in the table above.

## Unattributed true16 follow-up

`torch.117.co`, entry `0x33c000` (`addr<BFloat16>`), contains
`global_store_b8 v[6:7], v1` followed by `v_mov_b16 v1.l, v24.h` without an X
wait. Equivalent physical MIR makes `SIInsertWaitcnts` insert
`S_WAIT_XCNT 0` before both low- and high-half definitions: the full `v1`
store source and its subregister definitions share LLVM regunits. This rules
out a waitcheck true16 false positive, but the exact first unsafe pass in the
source-derived build is not yet stable enough to name. It remains a
minimization follow-up rather than a speculatively attributed compiler bug.

## Confidence and remaining question

The evidence is stronger than a final-ISA pattern match alone:

- The two IR reproducers start from optimized LLVM IR produced from the named
  PyTorch HIP translation units; the inline-assembly HIP and physical-MIR
  reproducers directly isolate their respective source and pass contracts.
- All four reproducers were validated with the LLVM revision used for the
  nightly wheel.
- Their final objects preserve the same hazard families as the wheel objects.
- Machine pass dumps identify the exact point where LLVM's internal XCNT model
  loses the required dependency.
- The branch case directly contradicts the earlier pass's own implicit-drain
  assumption; the spill case contradicts the neighboring code's explicit
  warning about spill-pseudo implicit super-register operands.

XNACK applicability is not an open question here: it is always enabled on
gfx1250. What remains to confirm is the narrower hardware contract: do the
producer operands remain locked until the indicated XCNT threshold, exactly as
modeled by LLVM's XCNT implementation? If yes, the compiler cases can become
LLVM regression tests after fixes are chosen, and the rocPRIM assembly must
include the missing XCNT wait.

Relevant implementation history includes Christudasan Devadasan's initial
gfx1250 `S_WAIT_XCNT` insertion change (`08b8d467d425`) and Prasoon Mishra's
implicit-drain optimization (`48a1ee7bc8e1`). Pierre van Houtryve and Sameer
Sahasrabuddhe have also recently worked in the same AMDGPU waitcnt and gfx12.5
code.
