; RUN: %raise_cli %s_addk_i32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_addk_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `s_addk_i32` (renamed to `s_addk_co_i32` in
; gfx12+ assembly). Pins the correct operand-index contract for
; SOPK_32TIE-class opcodes with tied-def `$src0`: `op.src(1)` is
; the `$simm16` immediate, NOT `op.src(0)` (which aliases the
; tied SGPR).
;
; Regression this fixture guards. A prior revision of the handler
; in `handle_sopk.cpp` read `op.src(0)` expecting the immediate —
; so the add became `dst + dst` (doubling the prior value) instead
; of `dst + K`. For Triton-compiled `layer_norm` the compiler
; emits `s_addk_co_i32 sN, 0x400` as the loop-counter increment;
; the doubled-zero counter stalls the loop forever and the kernel
; hangs. The same bug was latent in `S_MULK_I32` (identical
; TableGen class `SOPK_32TIE`). After the fix, `%addk = add i32
; <prior-dst>, 1024` with the LITERAL 1024 as the second operand.
;
; The `.hip` sibling forces `s_addk_co_i32 %[x], 0x400` via inline
; asm with a `+s` constraint so hipcc cannot substitute a
; different increment (e.g. `s_add_i32 sN, sN, sM`).
;
; CHECK-LABEL: define amdgpu_kernel void @s_addk_i32_kernel(

; Positive: the lift materialises the immediate `1024` (= 0x400)
; as a plain i32 constant in the add, and the first operand is
; some SSA value (`%csel` in the captured IR; the name is
; projection-independent but kept generic here so renames in the
; source don't break the fixture).
;
; `add i32 %<prior-dst>, 1024` with the SECOND operand being the
; literal 1024 is the post-fix shape; the NEGATIVE assertions
; below forbid the pre-fix `add i32 %x, %x` shape.
; CHECK: %addk = add i32 %{{[^,]+}}, 1024

; The SCC overflow computation feeds the same (prior-dst, 1024)
; pair into `uadd_with_overflow`. This pins the SCC contract's
; operand symmetry — if a future rewrite gets the imm arg right
; on the add but wrong on the SCC overflow (or vice versa), this
; CHECK catches the drift.
;
; Note: `uadd_with_overflow` is itself a latent inaccuracy
; (`s_addk_co_i32` on gfx12+ sets SCC from SIGNED overflow;
; `sadd_with_overflow` would be more accurate). Not in scope for
; this fix — the layer-norm kernel consumes the add result via
; `icmp slt` for its loop exit, not via SCC — but documented as a
; cleanup opportunity in the handler comment. If that cleanup
; lands later, swap this CHECK to `@llvm.sadd.with.overflow.i32`.
; CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{[^,]+}}, i32 1024)

; NEGATIVE assertions.

; (a) The pre-fix shape was `add i32 %x, %x` with the same SSA
; value on both sides. Forbid that exact shape. FileCheck's
; `[[NAME]]` / same-variable-twice pattern would be ideal but lit
; regex doesn't support back-references natively; use a
; sufficient-guard by forbidding any `add i32 %<sym>, %<sym>`
; where the whole add's second operand is an SSA ref (not a
; constant). A post-fix kernel emits `add i32 %..., 1024`, so no
; `, %` appears as the second operand on any `add i32` that
; produces `%addk`.
; CHECK-NOT: %addk = add i32 %{{[^,]+}}, %{{[^,]+}}

; (b) The SCC overflow intrinsic must not receive two SSA
; operands either (it would happen in lockstep with the add bug).
; CHECK-NOT: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}})
