# Buffer-Store Lowering — Hardware OOB vs. Software Sink

> **Status:** implemented. `BUFFER_STORE_*` SemOps lower to
> `llvm.amdgcn.raw.buffer.store.<ty>`, mirroring the load path. Fixes
> R1 (the gfx1250 → gfx942 Triton vector-add SIGSEGV) and removes a
> latent ABI coupling that silently forced scratch enablement in the
> emitted kernel descriptor.
>
> **Scope:** gfx1250 → gfx942 today; the principle (lift to the
> matching hardware-OOB intrinsic, never to a software fallback that
> introduces an `addrspace(5)` alloca) is target-agnostic.

---

## 1. Problem in one paragraph

The original `handleMUBUF` store path replaced every `BUFFER_STORE_*`
with a generic `store` against an `addrspacecast(alloca i32,
addrspace(5))` "OOB sink", selected via `select i1 oob, sink, real`.
That looked locally innocuous but was wrong on three independent
axes: a 4-byte sink could not absorb a 16-byte
`BUFFER_STORE_DWORDX4`, the surviving `addrspace(5)` alloca made the
AMDGPU backend silently set `.amdhsa_enable_private_segment 1` /
`.amdhsa_private_segment_fixed_size > 0` in the kernel descriptor we
hand to the runtime, and it diverged from the load path which already
relies on hardware OOB clamping. The cumulative effect on gfx942 was
a guaranteed launch fault — the runtime never populates
`FLAT_SCRATCH` on a KD whose `enable_sgpr_flat_scratch_init` bit is
zero (we model only the source-ISA user-SGPR set, see
`abi-translation.md` §3.4), so the first `flat_store` to the scratch
aperture from the OOB-sink path takes a SIGSEGV.

## 2. Symptom (R1)

```
$ python runner.py --modes salmon --gpu 0 \
    --script triton-tutorial-01-vector-add.py
[run] triton-tutorial-01-vector-add.py :: salmon ... CRASH (SIGSEGV)
```

The crash reproduced even with `size = N * BLOCK_SIZE` exactly (no
out-of-bounds lanes), which ruled out the sink-overflow hypothesis as
the immediate cause and pointed at the implicit scratch enablement
itself: scratch was on, FLAT_SCRATCH was undefined, any flat
instruction touching the aperture (including the OOB sink path
emitted unconditionally per BUFFER_STORE) faulted.

## 3. The three coupled root causes

| # | Surface | What was wrong | Why it matters |
|---|---|---|---|
| 1 | Sink width | Sink alloca always `i32` (4 B), but `BUFFER_STORE_DWORDX4` writes 16 B | OOB lanes in DWORDX4 stores walked 12 B past the sink, into either the next per-thread scratch slot or unmapped scratch — a fault in its own right |
| 2 | KD coupling | Any surviving `addrspace(5)` alloca makes `AMDGPUPromoteAlloca` / the backend allocate scratch and emit `enable_private_segment 1` + `private_segment_fixed_size > 0` | Salmon's KD (modelled on the source ABI) does **not** request `flat_scratch_init`, so on entry FLAT_SCRATCH is undefined; any flat instruction touching the scratch aperture faults |
| 3 | Asymmetry with loads | `handleMUBUF` already lifted `BUFFER_LOAD_*` via `llvm.amdgcn.raw.buffer.load`, relying on hardware OOB clamp | Routing stores through software select+sink had no justification once the load side proved hardware OOB works for our shape |

The historical comment block above the old code claimed flat-memory
lowering with conditional branches "breaks under -O1+ SIMT
optimisations". That justification doesn't apply when we use the
buffer-store *intrinsic* itself — the intrinsic is not a generic
flat store, it is a buffer-resource store with hardware OOB
semantics, just like the load.

## 4. Principled fix

Replace the entire OOB-sink path with a direct
`llvm.amdgcn.raw.buffer.store.<ty>` call, parameterised on the same
SRD / voffset / soffset / cachepolicy that the load already uses.
Wrap the call in `emitUnderExec` to keep IR-level lane masking
consistent with the EXEC mask the hardware already honours.

```142:209:projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/handle_mubuf.cpp
    if (isStore) {
      // Use the gfx942 buffer-store intrinsic directly, exactly
      // mirroring the load path above. The hardware's BUFFER unit
      // handles OOB silently (the write is dropped when the per-lane
      // offset is >= num_records), so no software OOB sink is needed.
      // ...
      Function *bufSt = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_raw_buffer_store, {storeTy});
      ctx.emitUnderExec([&] {
        ctx.B.CreateCall(bufSt, {val, srd, voffset, soffset, auxFlags});
      });
```

Width handling matches the load:

| Source op | `storeTy` | Intrinsic specialisation |
|---|---|---|
| `BUFFER_STORE_BYTE` / `BUFFER_STORE_SHORT` | `iN` (`N ∈ {8,16}`) | `amdgcn.raw.buffer.store.i8` / `.i16` |
| `BUFFER_STORE_DWORD` | `i32` | `amdgcn.raw.buffer.store.i32` |
| `BUFFER_STORE_DWORDX{2,3,4}` | `<{2,3,4} x i32>` | `amdgcn.raw.buffer.store.v{2,3,4}i32` |

No `addrspace(5)` allocations are introduced anywhere in the lowered
function — which is the structural property that makes the KD stay
clean.

## 5. ABI coupling lesson

The bug class here is **implicit ABI coupling through the AMDGPU
backend's allocation-lowering heuristics**. The pipeline cares about
two ABI surfaces (see `abi-translation.md` §3 for the source side and
§4 for the target side):

- The **source-side** user-SGPR set we model — chosen to mirror the
  source binary's KD bits, deliberately omitting
  `flat_scratch_init` because the source kernel didn't request it.
- The **target-side** KD that the AMDGPU backend re-synthesises from
  IR attributes, which infers `enable_private_segment` from observed
  scratch usage in the lowered function.

Anything in the IR that the backend interprets as "this kernel
needs scratch" — most directly, *any* surviving `addrspace(5)`
alloca, but also intrinsic calls like `llvm.amdgcn.dispatch.ptr` or
`llvm.amdgcn.implicitarg.ptr` — silently flips a bit on the target
KD that we never asked for. The new bit then expects user-SGPR state
the source ABI never agreed to populate, and the kernel faults on
its first scratch-touching instruction.

This is the same shape of bug that motivated the earlier `ttmp`
alloca handling (`raiser.cpp` Phase 4 + `reg_file.cpp::collectAllocas`):
unpromoted `ttmp` allocas survived `PromoteMemToReg`, the backend's
`AMDGPUPromoteAlloca` lifted them to LDS, and as a side effect
inserted an `llvm.amdgcn.dispatch.ptr` and stripped the kernel's
`amdgpu-no-dispatch-ptr` attribute — re-enabling `dispatch_ptr` in
the emitted KD and corrupting `s[0:1]`. Same coupling, different
intrinsic.

The translation rule that falls out is:

> **Never introduce IR constructs whose presence the AMDGPU backend
> interprets as a request for new user-SGPR slots or new KD
> capability bits.** When a software fallback for a hardware
> primitive seems necessary, check whether the corresponding native
> intrinsic exposes the same hardware semantics (here: hardware OOB
> clamp on `amdgcn.raw.buffer.store` matches `amdgcn.raw.buffer.load`)
> before reaching for an `addrspace(5)` alloca, an
> `addrspacecast`-driven select, or any of the other patterns that
> the backend treats as evidence of scratch / dispatch-ptr /
> implicitarg-ptr usage.

`abi-translation.md` §4.2 lists what the backend cannot infer (and
which we therefore propagate via attributes); the dual to that list
is what the backend **does** infer from IR shape, and which therefore
must stay out of the IR unless the source binary actually requested
it.

## 6. Verification status

| Check | Result |
|---|---|
| `triton-tutorial-01-vector-add.py` salmon mode | **PASS** (was reliable SIGSEGV) |
| Salmon `add_kernel.s` KD vs native gfx942 baseline | identical (`enable_private_segment 0`, `private_segment_fixed_size 0`, `dispatch_ptr 0`) |
| Salmon `add_kernel.ll` | uses `llvm.amdgcn.raw.buffer.store.v4i32`; no `oob_sink`, no `addrspace(5)` allocas, no `dispatch.ptr` calls |
| Standalone GPU tests on the changed path | `Gfx1250Gpu.Vecadd`, `IrGpu.*`, `CrossArchGpu.VecaddCrossArch` pass |
| Pre-existing baselines (`BatchRaise.AiterGfx950`, `Gfx1250Gpu.Softmax`, `MfmaGpu.Gemm*`, `Integration.VecaddAllKernels`) | confirmed unchanged by swapping in the HEAD baseline of `handle_mubuf.cpp` and re-running — **not regressions** |

Out of scope for this fix: the remaining Triton tutorials in salmon
mode (02 fused-softmax HANG, 04 low-memory-dropout FAIL, 05
layer-norm HANG, 07 extern-functions FAIL, 08 grouped-gemm FAIL).
None reproduce the R1 SIGSEGV pattern; each is a separate
translation gap in the per-axis docs (matrix, sync, TDM, …).

## 7. Where this lives in the code

- Handler change:
  `projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/handle_mubuf.cpp`
  (`handleMUBUF`, store branch).
- Adjacent `ttmp`-alloca fix that prevents the sibling
  `dispatch_ptr` re-enablement (same coupling class, different
  intrinsic):
  `transpiler/raiser.cpp` Phase 4 +
  `transpiler/reg_file.cpp::AllocaRegFile::collectAllocas`.
- ABI surface this interacts with: `docs/abi-translation.md`
  §3.4 (source-side user-SGPR layout, including the
  `flat_scratch_init` bit we deliberately do not set) and §4.2
  (target-side scratch attribute propagation).
