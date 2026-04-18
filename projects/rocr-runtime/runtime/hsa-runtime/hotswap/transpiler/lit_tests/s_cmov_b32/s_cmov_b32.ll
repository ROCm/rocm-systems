; RUN: %raise_cli %s_cmov_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_cmov_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_cmov_b32. Pins that the SOP1 conditional move
; lowers to an LLVM `select` keyed on SCC, with the prior dst
; value on the false leg. The handler lives in
; transpiler/handle_sop1.cpp under `if (sop == SemOp::S_CMOV_B32) { ... }`;
; the SemOp lives in transpiler/semop.hpp under SOP1.
;
; Why the explicit prior-dst test: LLVM's SOP1_32 pseudo for
; S_CMOV_B32 declares `(outs sdst), (ins src0)` without a tied
; sdst_in operand — the dst-on-SCC=0 read-modify is implicit in
; the hardware encoding rather than carried in the MachineInstr.
; If the handler ever forgets to read `regs.readReg32(op.dst())`
; on the false branch, the select would degenerate into a no-op or
; pick up `undef`, both of which would change kernel behavior.

; CHECK-LABEL: define amdgpu_kernel void @s_cmov_b32_kernel(

; The lifted IR must contain a `select i1` whose value-type is
; i32. The condition is the loaded SCC. The true value is the new
; src (an SSA `%`-named value). The false value is the prior dst
; — for this fixture, that prior is the 0xDEADBEEF sentinel folded
; through into the select as a constant on the false leg.
; LLVM IR may render the sentinel as the signed-decimal
; `-559038737` or the unsigned `3735928559`; accept either form.
; This catches:
;   * select-condition regressions (loaded SCC dropped or wrong i1)
;   * src-branch regressions (true leg not wired to the cmov src)
;   * dst-branch regressions (false leg picking up `undef` or the
;     src instead of the prior dst — would drop the sentinel).
; CHECK: select i1 %{{[^,]+}}, i32 %{{[^,]+}}, i32 {{(-559038737|3735928559)}}
