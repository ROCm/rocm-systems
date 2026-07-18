# gfx1250 XCNT hazards in LLVM-generated PyTorch kernels

## Summary

We found five gfx1250 XCNT diagnostics while checking the PyTorch
`2.11.0+rocm7.15.0a20260717` wheel. Source rebuilds and LLVM pass dumps reduce
them to two LLVM code-generation patterns:

1. `SIInsertWaitcnts` relies on a branch to drain XCNT, then
   `SIPreEmitPeephole` removes the branch without preserving the drain. This
   accounts for four of the five sampled kernels.
2. XCNT scoring includes liveness-only implicit super-register operands on
   spill pseudos. This accounts for the `warpMergeSortTopK<double>` kernel.

We believe both are real wait hazards according to LLVM's own gfx1250 XCNT
model. The reduced testcases make the final ISA issue reproducible, but we have
not demonstrated runtime corruption. We would like AMDGPU codegen maintainers
to confirm the hardware requirement before bugs are filed more broadly.

The attached IR files are compiler-only reproducers and must not be executed:

- [`repro-gfx1250-xcnt-branch-delete.ll`](repro-gfx1250-xcnt-branch-delete.ll)
- [`repro-gfx1250-xcnt-spill.ll`](repro-gfx1250-xcnt-spill.ll)

## Toolchain snapshot

- Target: `gfx1250`, wave32
- AMD clang: `23.0.0git`, ROCm LLVM commit
  `723bffa5dfbf92e452b0d4a0df674bdd849fcf12`
- PyTorch wheel: `2.11.0+rocm7.15.0a20260717`
- PyTorch source: `6f6864920e0c21f7a59eb6ba2186d08ce0ec6d49`
- Waitcheck source: `users/kuhar/waitcheck` at
  `73cc0e0353aae4cc02af5016ee27a8a057681cd7`

The commands below use the compiler from that wheel environment. A compiler
built from the named ROCm LLVM commit can be substituted.

```sh
AMDCLANG=/path/to/amdclang++
WAITCHECK=/path/to/rj_waitcheck
```

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
four independently sampled kernels:

| PyTorch object and kernel | Source provenance | Representative final hazard |
| --- | --- | --- |
| `torch.113.co`, `launch_masked_scatter_kernel`, entry `0x196600` | `aten/src/ATen/native/cuda/IndexKernel.cu:405-455` | `global_load_u8 v0, v[4:5]` at `0x19dd08`, then definition of address `v4` at `0x19dd14`; requires `s_wait_xcnt 0` |
| `torch.120.co`, `binary_cross_entropy_out_cuda<double>`, entry `0x113f00` | `aten/src/ATen/native/cuda/Loss.cu:71-102` | `global_store_b8 v[18:19], v0` at `0x1221e8`, then definition of data `v0` at `0x1221f4`; requires `s_wait_xcnt 0` |
| `torch.58.co`, `div_trunc_kernel_cuda<short>`, entry `0x1c4400` | `aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu:18-48` | `global_store_d16_hi_b8 v[8:9], v1` at `0x1cae2c`, then definition of `v1` at `0x1cae38`; requires `s_wait_xcnt 0` |
| `torch.165.co`, rocPRIM `reduce_by_key<unsigned char, long>`, entry `0xbdd00` | Triggered by `aten/src/ATen/native/cuda/TensorModeKernel.cu:23-81` | `s_load_b64 s[34:35], s[0:1]` at `0xbdfc0`, then definition of address `s0` at `0xbdfcc`; requires `s_wait_xcnt 0` |

The first three originate in HIP/CUDA source compiled by clang. The fourth is
a rocPRIM template instantiated by PyTorch's mode implementation. The common
failure is in late machine-code transformation, not in the source operation.

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

## Confidence and remaining question

The evidence is stronger than a final-ISA pattern match alone:

- Both reduced files start from optimized LLVM IR produced from the named
  PyTorch HIP translation units.
- Both compile with the same LLVM revision used for the nightly wheel.
- The reduced final objects preserve the same hazard families as the wheel
  objects.
- Machine pass dumps identify the exact point where LLVM's internal XCNT model
  loses the required dependency.
- The branch case directly contradicts the earlier pass's own implicit-drain
  assumption; the spill case contradicts the neighboring code's explicit
  warning about spill-pseudo implicit super-register operands.

What remains to confirm is the gfx1250 hardware contract: do the producer
operands remain locked until the indicated XCNT threshold, as modeled by
LLVM's XCNT implementation? If yes, these are compiler correctness bugs and
the reduced `.ll` files can be converted into LLVM regression tests after the
fixes are chosen.

Relevant implementation history includes Christudasan Devadasan's initial
gfx1250 `S_WAIT_XCNT` insertion change (`08b8d467d425`) and Prasoon Mishra's
implicit-drain optimization (`48a1ee7bc8e1`). Pierre van Houtryve and Sameer
Sahasrabuddhe have also recently worked in the same AMDGPU waitcnt and gfx12.5
code.
