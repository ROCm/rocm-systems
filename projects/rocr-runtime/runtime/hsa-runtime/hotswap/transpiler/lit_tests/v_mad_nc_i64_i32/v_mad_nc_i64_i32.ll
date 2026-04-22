; RUN: %raise_cli %v_mad_nc_i64_i32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_mad_nc_i64_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_mad_nc_i64_i32. Pins that the gfx1250 VOP3 no-carry
; signed 64-bit multiply-add lowers to the canonical
; `add(mul(sext s0, sext s1), s2_i64)` IR.  The handler lives in
; transpiler/handle_valu.cpp under
;   `if (sop == SemOp::V_MAD_NC_I64_I32) { ... }`
; with the matching SemOp in transpiler/semop.hpp under the
; `V_MAD_NC_*` block, and the decoder mapping in
; transpiler/opcode_map.cpp at
;   `E(V_MAD_NC_I64_I32_e64, V_MAD_NC_I64_I32)`.
;
; Per AMDGPU/VOP3Instructions.td:196 and :2129 the instruction has
; profile `VOP_I32_I32_I64_DPP` (`VOPProfile<[i64, i32, i32, i64]>`)
; and no SDPattern — it's purely a TableGen pseudo.  The AMDGPU
; backend's `SelectMad64_32` in AMDGPUISelDAGToDAG.cpp:1220 matches
; the idiomatic widening-MAD IR (two sign-extended i32 factors, i64
; accumulator) and re-emits the target-specific MAD on gfx1250 /
; gfx942.  The emitted IR must therefore contain:
;   * two `sext i32 ... to i64` for the factors (NOT zext — the `i` in
;     `v_mad_nc_i64_i32` names signed widening);
;   * a `mul {{.*}}i64` of those sign-extended values (or an isa-
;     legal commuted form the backend can still match);
;   * an `add {{.*}}i64` whose second operand is the i64 accumulator
;     (the canonical breadcrumb is the `vmad_nc_i64` value-name the
;     handler sets, mirroring `vmad_co64` from `V_MAD_CO_U64_U32`).

; CHECK-LABEL: define amdgpu_kernel void @v_mad_nc_i64_i32_kernel(

; Two signed widenings of the 32-bit factors.
; CHECK: sext i32 %{{[^ ]+}} to i64
; CHECK: sext i32 %{{[^ ]+}} to i64

; Widening multiply — 64-bit product of the sign-extended factors.
; Commutativity of `mul` means CHECK can't pin operand order.
; CHECK: mul {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Canonical accumulator add emitted with the handler's `vmad_nc_i64`
; value-name.  If the handler ever renames the value, update both
; semop.hpp's `V_MAD_NC_*` comment block (which names this breadcrumb)
; and the matching sibling row in handle_valu.cpp.
; CHECK: %vmad_nc_i64 = add {{.*}}i64

; Negative checks.
;   - No zext on the factor path — that would lower to v_mad_nc_u64_u32
;     (the unsigned sibling), not v_mad_nc_i64_i32.  Pre-gate the test
;     against an accidental handler swap.
; CHECK-NOT: %vmad_nc_u64 = add {{.*}}i64
;   - No `with.overflow` intrinsic CALL-site — the "nc" in the opcode
;     name means we deliberately do NOT model the carry-out.  The
;     `call` keyword pins this to actual use-sites rather than
;     module-level `declare` lines the LLVM textual writer emits
;     even for unused intrinsics pulled in transitively.  Emitting
;     an overflow intrinsic call here would bloat the IR and miss
;     the backend's `SelectMad64_32` pattern (which matches only
;     the plain add/mul shape).
; CHECK-NOT: call {{.*}}@llvm.smul.with.overflow
; CHECK-NOT: call {{.*}}@llvm.sadd.with.overflow
;   - No narrowing mul-i24 fallback — the 32-bit tid arithmetic
;     inside the kernel still surfaces as `add i32 / mul i32`, but
;     neither of those is the MAD we care about; use the
;     `vmad_nc_i64` breadcrumb above as the anchor and guard
;     against the specific i24 builtin that a mis-handler swap to
;     the narrow MAD family would introduce.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.mul.i24
;   - No saturating-add intrinsic.  The HIP `asm volatile` in the
;     .hip file doesn't set the VOP3 clamp bit, so this fixture
;     exercises the handler's `clamp = 0` fast-path (plain
;     `add i64`).  If a future fixture encodes `clamp = 1` (or a
;     corpus producer surfaces and we graduate the handler to
;     emit saturation per the block comment in
;     `handle_valu.cpp`'s V_MAD_NC_* arm), a sibling fixture will
;     cover `llvm.sadd.sat.i64` emission.  Until then, any
;     appearance here means the handler silently promoted to
;     saturation and we need to investigate.
; CHECK-NOT: call {{.*}}@llvm.sadd.sat.i64
