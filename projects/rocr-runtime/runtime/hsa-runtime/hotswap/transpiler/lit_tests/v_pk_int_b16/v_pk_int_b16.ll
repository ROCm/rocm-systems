; RUN: %raise_cli %v_pk_int_b16_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=v_pk_int_b16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the VOP3P packed-pair `<2 x i16>` int family
; (V_PK_LSHLREV_B16 + V_PK_ADD_U16). Pins both:
;
;   * The shared operand-decode shape — `bitcast i32 -> <2 x i16>`
;     for each source (lo i16 = bits[15:0], hi i16 = bits[31:16]),
;     followed by an `extractelement` + `insertelement` round-trip
;     that pins the op_sel / op_sel_hi default permutation
;     (lo->lo, hi->hi). The kernel uses default modifiers on every
;     source so the round-trip should fold to the identity vector
;     in any subsequent IR pass — the explicit insert/extract pair
;     is what the handler emits literally.
;
;   * The per-opcode IR dispatch — `shl <2 x i16>` (after an
;     explicit `and <2 x i16> ..., <i16 15, i16 15>` shift-count
;     mask matching the AMDGPU clshl_rev_16 hardware clamp) for
;     V_PK_LSHLREV_B16, and `add <2 x i16>` for V_PK_ADD_U16.
;
; Same-target lift (gfx1250 -> gfx1250) because the V_PK packed-int
; family is identical-on-the-wire across gfx9+ (no cross-arch
; emulation needed); the lit test pins the IR shape, the round-trip
; through llc would re-emit the same opcodes verbatim.

; CHECK-LABEL: define amdgpu_kernel void @v_pk_int_b16_kernel(

; V_PK_LSHLREV_B16 with inline-literal shift count `0x60002` = `393218`.
; Shift count is src0, value is src1 — reversed-operand convention.
; The handler bitcasts both 32-bit operands to `<2 x i16>`, masks the
; shift count by `<i16 15, i16 15>`, then `shl <2 x i16>`. The named
; identifiers (`pk_lshlrev_amt`, `pk_lshlrev`, `pk_i16_pack`) are
; pinned by the handler — a future rename pattern-fails this fixture.
;
; The shift-count operand is the inline literal `393218` (= 0x60002).
; LLVM constant-folds the inline-literal-derived `<2 x i16>` shift
; count operand AND the `<i16 15, i16 15>` mask splat — the literal
; print form for the mask is `splat (i16 15)` (LLVM IR vector-splat
; constant printer); a future LLVM that re-prints it as the
; element-by-element form `<i16 15, i16 15>` would still pass the
; intent of this fixture but would need the regex relaxed. The shl's
; value operand is `%20` (the result of the op_sel decode round-trip
; on the `val` source) and the count operand is `%pk_lshlrev_amt`,
; pinning the reversed-operand convention of clshl_rev_16 (count is
; src0, value is src1).
; CHECK: %pk_lshlrev_amt{{[0-9]*}} = and <2 x i16> {{.+}}, splat (i16 15)
; CHECK: %pk_lshlrev{{[0-9]*}} = shl <2 x i16> %{{[^,]+}}, %pk_lshlrev_amt{{[0-9]*}}
; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_lshlrev{{[0-9]*}} to i32

; V_PK_ADD_U16 over the same source (val, shifted). Lane-wise i16
; add; same bitcast/insert/extract decode; `add <2 x i16>` IR opcode
; for the actual op. `pk_add_u16` is the handler-pinned name; the
; subsequent bitcast back to i32 picks up a `pk_i16_pack` name with
; an SSA-uniqueness suffix because the V_PK_LSHLREV_B16 case earlier
; in the BB already consumed the un-suffixed name (LLVM IRBuilder
; auto-renames duplicates). The `[0-9]*` glob covers both forms.
; CHECK: %pk_add_u16{{[0-9]*}} = add <2 x i16> %{{[^,]+}}, %{{[^)]+}}
; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_add_u16{{[0-9]*}} to i32

; Negative assertions: must NOT take the F32 packed path (which would
; mis-extract `<2 x f32>` lanes), must NOT use sub/mul/lshr/ashr
; (silent miscompile of the i16 add or shift), must NOT zext the i16
; result to i32 (which would lose the high lane).
; CHECK-NOT: shl <2 x f32>
; CHECK-NOT: sub <2 x i16>
; CHECK-NOT: lshr <2 x i16>
; CHECK-NOT: ashr <2 x i16>
; CHECK-NOT: zext <2 x i16> %pk_lshlrev{{[0-9]*}} to <2 x i32>
