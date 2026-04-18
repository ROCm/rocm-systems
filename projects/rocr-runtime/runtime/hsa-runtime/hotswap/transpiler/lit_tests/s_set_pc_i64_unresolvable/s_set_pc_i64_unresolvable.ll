; RUN: %not %raise_cli %s_set_pc_i64_unresolvable_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_unresolvable_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for s_set_pc_i64 — Unresolvable path. Pins the
; contractual loud-failure behaviour of the SOP1 handler in
; transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case Unresolvable: }`.
; The static analysis (transpiler/setpc_analysis.cpp) classifies any
; s_set_pc_i64 site whose source SGPR pair was not produced by a
; recognized PC chain (Pattern A) and is not consumed via a recorded
; chain-terminator (Pattern B) as Unresolvable, with a refusal
; reason. The handler converts that into a
; RaiseFailure::unsupportedShape — the user rules forbid silent
; fallbacks, so a "branch to unreachable" stub would be a regression.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`s_set_pc_i64`) and the encoding format (`SOP1`).
;      raise_cli's failure line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>]`.
;
; History: this test was added together with the s_set_pc_i64
; handler implementation (see semop.hpp's S_SET_PC_I64 doc, and
; setpc_analysis.cpp). The mnemonic / format substrings are stable;
; rewordings of the human-readable refusal reason live in `detail`
; (RaiseFailure::detail) and are NOT printed by raise_cli, so this
; test does not depend on that wording.

; STDERR: raise_cli: kernel 'setpc_unresolvable_kernel' failed to raise:
; STDERR-SAME: s_set_pc_i64
; STDERR-SAME: [SOP1]
