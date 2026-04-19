; RUN: %raise_cli %s_lshr_b64_imm_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_lshr_b64_imm_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx9+/gfx1250 SOP2 64-bit logical right shift
; (`s_lshr_b64`) with an immediate shift count. See SemOp::S_LSHR_B64
; in transpiler/semop.hpp; the matching handler block in
; transpiler/handle_sop2.cpp under `if (sop == SemOp::S_LSHR_B64)`;
; and the SOP2 mapping in transpiler/opcode_map.cpp.
;
; INVARIANTS PINNED:
;
;   1. The kernel signature carries the i64 source-operand argument
;      directly (`i64 %arg1`); no by_value-style decomposition. This
;      pins the typical `s_lshr_b64` corpus shape: shifting a 64-bit
;      kernarg-derived pointer/integer by an immediate.
;
;   2. The lift emits a single `lshr i64 ..., 16`. The handler
;      builds `CreateZExt(op.src(1), i64Ty)` followed by
;      `CreateLShr(...)`, which constant-folds the zext away when
;      src1 is the immediate `16` (the corpus shape). A regression
;      that emitted a 32-bit shift, masked the count with `urem 64`,
;      or routed the immediate through the wrong width would change
;      this exact instruction shape.
;
;   3. NO `zext i32 16 to i64` (the literal would be a sign that the
;      handler is leaving an unfolded zext on a constant — harmless
;      but indicates the immediate-folding path regressed).
;
;   4. NO 32-bit shift on the source value — the .td definition
;      `SOP2_64_32` makes the source 64-bit; a regression to a
;      pair-of-i32 lift would surface as `lshr i32`.

; CHECK-LABEL: define amdgpu_kernel void @s_lshr_b64_imm_kernel(
; CHECK-SAME: ptr addrspace(1) %arg0
; CHECK-SAME: i64 %arg1

; The s_lshr_b64 lift. The `lshr64` value-name is the canonical
; breadcrumb the handler emits (mirrors `shl64` for S_LSHL_B64 and
; `ashr64` for S_ASHR_I64). The shift count is the literal `16`
; from the inline asm.
; CHECK: %lshr64 = lshr i64 %{{[^,]+}}, 16

; Negative pin: no leftover zext-of-constant shape (the immediate
; `16` must constant-fold through the i32→i64 widen).
; CHECK-NOT: zext i32 16 to i64

; Negative pin: the shift must be 64-bit. A regression that lowered
; src0 to a pair of i32s would emit `lshr i32 ...` instead.
; CHECK-NOT: %lshr64 = lshr i32

; Negative pin: no defensive shift-count masking (the no-fallback
; rule rejects `urem` on the shift amount; if the source binary
; supplies an out-of-range count, the lifted IR's poison is the
; correct surfacing of a real source bug).
; CHECK-NOT: urem i64 {{.*}}, 64
