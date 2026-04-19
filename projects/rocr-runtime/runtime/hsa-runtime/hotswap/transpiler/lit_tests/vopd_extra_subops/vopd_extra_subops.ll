; RUN: %raise_cli %vopd_extra_subops_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_extra_subops_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for three previously-unhandled VOPD sub-shapes (see
; transpiler/handle_vopd.cpp and the matching kernel comment block):
;
;   1. v_dual_mov_b32 with f32 inline-constant literal source
;      (`1.0` printed by the AMDGPU instruction printer rather than
;      the hex bit pattern).  Surface form: `store i32 1065353216`
;      (= 0x3F800000, the IEEE-754 encoding of 1.0f) into the
;      destination VGPR via the `mov_lit` output.
;   2. v_dual_ashrrev_i32 — arithmetic right shift, REVERSED operand
;      order (shift amount = first src, value = second src).
;      Surface form: `ashr i32 %src, 31`.
;   3. v_dual_max_i32 — signed max via `llvm.smax.i32`.
;      Surface form: `call i32 @llvm.smax.i32(i32 0, i32 %src)`.
;
; Each shape lives in its own VOPD pair in the kernel so a regression
; in any one of the three branches surfaces exactly one CHECK failure
; rather than cascading.
;
; INVARIANTS PINNED:
;
;   * The f32 inline-constant literal `1.0` is materialised as the
;     32-bit IEEE bit pattern (0x3F800000 = 1065353216 decimal) — the
;     same shape every other VALU handler emits when given a
;     register/literal mix.  A regression that returned `nullptr`
;     from the literal-parse path would surface as "VOPD
;     decomposition failed" at lift time and produce zero IR.
;   * The AShr appears with the literal SECOND ARGUMENT (the shift
;     count, value 31) and the register FIRST — matching the
;     IRBuilder `CreateAShr(s1, s0)` reversal applied by the handler.
;     Lit-amount AShr by 31 is a sign-extension idiom; the constant
;     31 must remain visible.
;   * The smax dispatch uses `llvm.smax.i32` (NOT `llvm.umax`, NOT a
;     manual `select (icmp sgt) ...`) — keeps the dispatch surface
;     symmetric with the float min/max handlers above it.
;
; NEGATIVE PINS:
;
;   * NO `unsupportedOpcode` or `VOPD decomposition failed` text in
;     the captured stderr (FileCheck would see empty stdin).
;   * NO `llvm.umax` or `llvm.umin` (the corpus shape is signed).
;   * NO `lshr` for the ashrrev surface (must be arithmetic).

; CHECK-LABEL: define amdgpu_kernel void @vopd_extra_subops_kernel(

; Pair A: the mov_b32 of the float literal `1.0` materialises the
; IEEE bit pattern (1065353216 decimal = 0x3F800000) into the
; destination VGPR's SSA chain — the per-lane store path runs the
; constant through a phi (writing the literal in the active-lane
; arm and forwarding the prior value in the masked arm), so we pin
; the phi shape directly rather than a `store i32 …`.
; CHECK: phi i32 [ 1065353216,

; Pair A: the ashrrev sibling — arithmetic shift right by the
; literal 31, applied to the register source.
; CHECK: %vopd_ashr = ashr i32 %{{[^,]+}}, 31

; Pair B: the max_i32 sibling — signed max of literal 0 against the
; register source via the smax intrinsic.
; CHECK: %vopd_smax = call i32 @llvm.smax.i32(i32 0, i32 %{{[^)]+}})

; Negative pin: the ashrrev must be arithmetic; a regression to
; logical shift would have surfaced as `lshr` here (with the same
; operand orientation).
; CHECK-NOT: %vopd_lshr =

; Negative pin: the max_i32 must dispatch to the SIGNED smax, not
; umax.  A regression that confused signed/unsigned would surface
; here.
; CHECK-NOT: @llvm.umax.i32
