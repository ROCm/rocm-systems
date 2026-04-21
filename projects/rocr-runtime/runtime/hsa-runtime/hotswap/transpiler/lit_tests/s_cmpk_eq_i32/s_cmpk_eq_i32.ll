; RUN: %raise_cli %s_cmpk_eq_i32_co --isa=gfx950 --target-isa=gfx942 \
; RUN:     --emit-ir=s_cmpk_eq_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lit fixture for the SOPK_SCC-class compare opcodes
; (`s_cmpk_{eq,lg,gt,ge,lt,le}_{i,u}32`). The handler arm in
; `handle_sopk.cpp` had the same operand-index bug as the
; SOPK_32TIE arm fixed earlier: it read `op.src(0)` expecting the
; immediate, but the SOPK_SCC operand layout is
;
;     (outs)                           ; empty — no def
;     (ins SReg_32:$sdst,              ; operand 0: sdst as SOURCE
;          s16imm:$simm16)             ; operand 1: the immediate
;
; With `getNumDefs() == 0`, the decoder's `buildSrcMap` keeps
; both operands in srcMap — `op.src(0)` returns `$sdst`, not
; `$simm16`. Pre-fix the handler then emitted `icmp eq %sdst, %sdst`
; = always true, writing a trivially-set SCC that ignored the
; real comparison against the immediate. Post-fix reads the
; immediate from `op.src(1)`.
;
; ISA gating. `s_cmpk_*_i32/u32` was dropped on gfx12; it exists
; only up through gfx11. So the bug is only reachable from
; gfx9xx-source kernels (AITER corpus path). This fixture compiles
; for gfx950 and raises with `--isa=gfx950 --target-isa=gfx942`,
; matching the canonical AITER raise direction. gfx1250-source
; kernels cannot emit `s_cmpk_*_i32` and therefore cannot exercise
; this handler arm — the latent exposure never surfaced on the
; Triton corpus for that reason.
;
; Single fixture covers the entire `S_CMPK_{EQ,LG,GT,GE,LT,LE}_
; {I,U}32` family by handler-shape identity (all twelve SemOps go
; through the same single `Value *imm = op.src(1);` read; only
; the subsequent `CreateICmp{EQ,NE,SGT,SGE,...}` differs). If a
; future change breaks the operand-index read, it breaks it for
; all twelve variants identically — one fixture catches them all.
;
; CHECK-LABEL: define amdgpu_kernel void @s_cmpk_eq_i32_kernel(

; Positive: `%scmpk = icmp eq i32 %..., 1024` with the LITERAL
; 1024 (= 0x400) as the second operand. The first operand is the
; SSA value of the `$sdst` source register; in this fixture's
; kernel it resolves to `%wg_id_x` (blockIdx.x, routed through the
; inline-asm `[x] "s"(x)` constraint) but the CHECK uses a
; generic match so renames in the kernel source don't break it.
; CHECK: %scmpk = icmp eq i32 %{{[^,]+}}, 1024

; The compare's SCC consumer — `s_cselect_b32 %[r], 1, 0` — lifts
; to a `select i1 %scmpk, i32 1, i32 0`. This pins the
; producer-to-consumer dataflow: the SCC write from %scmpk must
; thread through to the cselect's condition.
; CHECK: %csel = select i1 %scmpk, i32 1, i32 0

; NEGATIVE: forbid the pre-fix `icmp eq %x, %x` shape (same SSA
; value on both sides of the compare). Post-fix the second
; operand is a constant, so any `icmp eq i32 %a, %b` with a
; second SSA operand is the bug resurrecting.
; CHECK-NOT: %scmpk = icmp eq i32 %{{[^,]+}}, %{{[^,]+}}
