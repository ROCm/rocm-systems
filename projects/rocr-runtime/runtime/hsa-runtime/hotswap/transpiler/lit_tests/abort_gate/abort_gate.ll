; RUN: %not %raise_cli %abort_gate_co --emit-ir=unknown_exec_writer_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; The raiser must refuse to lower a kernel that contains an
; EXEC-writing instruction whose SemOp is not on the SPE allow-list
; (`isSPEModeledExecWriter` in raiser.cpp). This prevents us from
; silently emitting IR for an opcode whose handler has not been
; audited against the per-lane predication assumption.
;
; The fixture pins `s_flbit_i32_b32` as the offending instruction:
; its `S_FLBIT_I32_B32` SemOp is deliberately excluded from the
; allow-list because no handler has been written for that shape of
; EXEC write. Routing the dst to `exec_lo` makes the EXEC-writer
; detector flag it.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the instruction and explains why,
;      matching on stable substrings. We do not pin to the full
;      sentence to keep the test resilient to future rewordings.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: 's_flbit_i32_b32'
; STDERR-SAME: writes EXEC
; STDERR-SAME: not in the SPE-modelled allow-list

; The raise_cli wrapper reports the failure once more so the kerneldex
; coverage format is preserved; the mnemonic must match.
; STDERR: raise_cli: kernel 'unknown_exec_writer_kernel' failed to raise:
; STDERR-SAME: s_flbit_i32_b32
; STDERR-SAME: SPE-unmodeled-EXEC-writer
