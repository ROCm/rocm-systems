; RUN: %raise_cli %s_mulk_i32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_mulk_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lit fixture for `s_mulk_i32` (SOPK_32TIE). Dedicated regression
; guard for the mul arm of the handler in `handle_sopk.cpp` —
; sibling to `lit_tests/s_addk_i32/` which guards the add arm.
; Both arms fixed in the same commit
; (ae0a84b2ca "transpiler: fix S_ADDK_I32 / S_MULK_I32 reading
; tied SGPR as immediate") by reading `op.src(1)` instead of
; `op.src(0)` for the `$simm16` immediate. The `s_addk_i32`
; sibling is the primary regression guard (layer-norm's hang made
; the corpus-level failure observable); this fixture closes the
; coverage gap on `s_mulk_i32` where no corpus kernel exists to
; surface a regression via end-to-end output mismatch.
;
; Without this fixture, a future refactor that broke the mul arm
; specifically — e.g. a copy-paste bug that reverted just the
; mul arm to `op.src(0)` — would pass every other test in the
; repo. With it, the corruption is caught at lit time.
;
; Immediate is 0x123 (291) rather than 0x400 so a careless
; copy-paste from the sibling fixture fails loud rather than
; masquerading as a match.
;
; CHECK-LABEL: define amdgpu_kernel void @s_mulk_i32_kernel(

; Positive: the mul materialises `291` (= 0x123) as the literal
; second operand. Pre-fix shape was `mul i32 %x, %x` (squaring);
; post-fix shape is `mul i32 %x, 291`. The first operand is
; projection-independent (some SSA value derived from blockIdx.x
; in this fixture's kernel; named `%csel` in the captured IR but
; kept generic here so rename cleanups in the source don't break
; the fixture).
; CHECK: %mulk = mul i32 %{{[^,]+}}, 291

; NEGATIVE: forbid the pre-fix squaring shape `mul i32 %x, %x`
; where both operands are SSA refs.
; CHECK-NOT: %mulk = mul i32 %{{[^,]+}}, %{{[^,]+}}
