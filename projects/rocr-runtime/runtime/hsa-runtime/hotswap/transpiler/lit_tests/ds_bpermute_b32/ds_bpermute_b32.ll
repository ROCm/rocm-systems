; RUN: %raise_cli %ds_bpermute_b32_co --emit-ir=ds_bpermute_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; DS_BPERMUTE_B32 must lower to `llvm.amdgcn.ds.bpermute(index, src)`,
; NOT to an identity copy of `src`. The identity-copy regression
; collapses every `__shfl_xor` / `__shfl_down` / `__shfl` pattern to
; "each lane keeps its own value" — the failure mode that hid the bug
; in compare_correctness's `lane_swap` and `block_sum_shfl` recipes
; before the fix.
;
; This lit test pins two invariants:
;
;   1. The intrinsic call is present with the right shape — `i32` as
;      both operand and result type.
;   2. The call consumes two distinct SSA values (the XOR'd lane
;      selector and the bitcast of the per-lane float). An identity
;      copy would bind the `src` and `dst` of the bpermute to the
;      same value, which is not what we want.
;
; MODREP: the handler's wave-size assumption (wave32 → wave64 lifts
; rely on `k < 32` keeping selectors within the low half of the
; target wave) is documented in `handle_ds.cpp` with a MODREP: tag;
; grep for that marker when changing the cross-wave policy. This
; test does not exercise cross-wave — see the fixture header for
; why.

; CHECK-LABEL: define amdgpu_kernel void @ds_bpermute_b32_kernel(

; The raised IR must contain exactly this intrinsic call shape:
; `i32 @llvm.amdgcn.ds.bpermute(i32 <selector>, i32 <value>)`. The
; operand bindings (`[[SEL]]`, `[[VAL]]`) are required by lit's
; variable-capture to assert they are the call's two distinct
; inputs.
; CHECK:      %bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[^,]+}}, i32 %{{[^,]+}})

; The intrinsic must be declared.
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)
