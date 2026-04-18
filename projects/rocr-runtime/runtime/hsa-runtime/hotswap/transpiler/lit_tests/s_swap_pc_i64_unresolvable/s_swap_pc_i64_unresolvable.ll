; RUN: %not %raise_cli %s_swap_pc_i64_unresolvable_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_swap_unresolvable_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for s_swap_pc_i64 — Unresolvable path. Pins the
; contractual loud-failure behaviour of the SOP1 handler in
; transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SWAP_PC_I64) { ... case Unresolvable: }`.
; The static analysis (transpiler/setpc_analysis.cpp) classifies any
; s_swap_pc_i64 site whose call-target SGPR pair was not produced by
; a complete intra-block PC chain as Unresolvable, with a refusal
; reason. The handler converts that into a
; RaiseFailure::unsupportedShape — the user rules forbid silent
; fallbacks, so a "branch to unreachable" stub would be a regression.
;
; Why this matters more than for s_set_pc_i64: the dominant
; tensilelite caller of s_swap_pc_i64 constructs the call target
; from a runtime scalar derived from a kernarg, which the current
; analysis cannot fold to a constant. Without this loud refusal a
; regression would silently miscompile the entire tensilelite F8/B8
; matmul corpus into kernels with mis-targeted branches.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`s_swap_pc_i64`) and the encoding format (`SOP1`).
;      raise_cli's failure line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>]`.

; STDERR: raise_cli: kernel 'setpc_swap_unresolvable_kernel' failed to raise:
; STDERR-SAME: s_swap_pc_i64
; STDERR-SAME: [SOP1]
