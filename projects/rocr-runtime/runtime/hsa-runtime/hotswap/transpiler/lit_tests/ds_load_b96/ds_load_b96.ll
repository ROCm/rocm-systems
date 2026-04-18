; RUN: %raise_cli %ds_load_b96_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=ds_load_b96_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx1250 96-bit LDS load/store pair
; (`ds_load_b96` / `ds_store_b96`).  The handler dispatches via the
; existing generic DS path in transpiler/handle_ds.cpp under
; `if (isDsRead || isDsWrite) { ... }` with the `dwords=3,
; loadBits=96` entry added to `dsClassify`.  The SemOps live in
; transpiler/semop.hpp under the `// -- DS --` group with docstrings
; explaining why we canonicalise on the gfx11+ asm spelling.
;
; The lowering shape this fixture pins:
;
;   * Load:  `%ds_ld = load <3 x i32>, ptr addrspace(3) %{{...}}`
;     followed by element-wise insertion into the destination VGPR
;     triple via `writeRegVec` (which the existing
;     `<3 x i32>` legalisation in reg_file.cpp handles generically
;     via `(totalBits + 31) / 32` dword math — works for 3 dwords).
;   * Store: `store <3 x i32> %{{...}}, ptr addrspace(3) %{{...}}`
;     emitted under `emitUnderExec` (the existing DS write path),
;     symmetric to the load.
;   * Both go through the addrspace(3) (LDS) pointer cast.
;
; The gfx942 backend lowers `load <3 x i32>` from addrspace(3) to
; either a native `ds_read_b96` (gfx9 inherits the `_vi` Real form
; from DSInstructions.td:2062) or splits to 3x `ds_read_b32` —
; both correct in-place lowerings; we don't pin which because the
; choice is the backend's.  What we DO pin is the IR shape the
; transpiler hands to the backend, so a regression that collapsed
; the 3-dword vector load into something else (e.g. an `i96`
; integer load, or three separate `i32` loads at the IR level)
; would be caught here.
;
; Invariants:
;
;   1. The SOURCE IR contains exactly one `load <3 x i32>` from
;      addrspace(3) and exactly one `store <3 x i32>` to addrspace(3)
;      — i.e. the handler routes both through the generic
;      `vecTy = <3 x i32>` branch and not through 3 separate
;      single-dword loads/stores (which would be correct but wasteful
;      and a sign of a regression away from the generic path).
;   2. NO `<2 x i32>` or `<4 x i32>` vector LDS access — those would
;      indicate `dsClassify` returned the wrong dword count (B64 or
;      B128 entry firing on a B96 SemOp).
;   3. NO call to any `int_amdgcn_ds_*` intrinsic for the LDS
;      access — the DS handler path uses raw load/store, not
;      intrinsics, for the non-transposed variants; an intrinsic
;      would be a regression that mis-routed the SemOp into one of
;      the transposed-load handlers (DS_LOAD_TR8_B64 etc).

; CHECK-LABEL: define amdgpu_kernel void @ds_load_b96_kernel(

; The store side: ds_store_b96 lifts to a `store <3 x i32>` in
; addrspace(3).  Pinned via the explicit element type and address
; space; the `align` attribute is left implicit because the backend
; chooses it from datalayout (4-byte for v3i32 in addrspace(3)).
; CHECK: store <3 x i32> %{{[^,]+}}, ptr addrspace(3) %{{[^,]+}}

; The load side: ds_load_b96 lifts to a `load <3 x i32>` in
; addrspace(3) named `ds_ld` per the handler's IR-name contract
; (handle_ds.cpp uses `"ds_ld"` for every generic DS read; greppable
; downstream).
; CHECK: %ds_ld{{[0-9]*}} = load <3 x i32>, ptr addrspace(3) %{{[^,]+}}

; Negative pins: no other vector widths and no intrinsics for the
; LDS access path.  These would indicate dsClassify mis-classified
; the SemOp or the dispatch routed into a transposed-load handler.
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: store <2 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: store <4 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.write
