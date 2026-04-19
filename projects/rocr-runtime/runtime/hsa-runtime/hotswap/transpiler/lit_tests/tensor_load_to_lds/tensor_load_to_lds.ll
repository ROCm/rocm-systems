; RUN: %not %raise_cli %tensor_load_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=tensor_load_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for VIMAGE TENSOR `tensor_load_to_lds_d2`. Pins
; the contractual cross-target loud-failure behaviour of
; transpiler/handle_vimage.cpp under SemOp::TENSOR_LOAD_TO_LDS.
;
; The gfx1250 TENSOR cnt unit (`MIMGInstructions.td:2049-2113`,
; `let SubtargetPredicate = isGFX125xOnly`) has no equivalent on
; gfx942. The handler refuses with `RaiseFailure::unsupportedShape`
; carrying the `VIMAGE` format bucket; the user-rules forbid silent
; fallbacks so a "synth a global_load chain" stub would be a
; regression. The matching LLVM intrinsic
; (`int_amdgcn_tensor_load_to_lds`) is itself gated isGFX125xOnly
; in IntrinsicsAMDGPU.td:4213, so even an intrinsic-emit on a
; non-gfx1250 target would fail at codegen — the principled lift
; is the loud refusal pinned here.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`tensor_load_to_lds`) and the encoding format
;      (`VIMAGE`). raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>]`.
;
; The handler also emits an explicit `transpiler: VIMAGE: ...`
; line that names the architectural mismatch and the
; same-target intrinsic; pinning that line keeps the diagnostic
; text from drifting into something less actionable for users
; who read raise_cli's stderr directly.

; STDERR: transpiler: VIMAGE: tensor_load_to_lds
; STDERR-SAME: gfx1250 TENSORcnt unit
; STDERR-SAME: amdgcn.tensor.load.to.lds

; STDERR: raise_cli: kernel 'tensor_load_to_lds_kernel' failed to raise:
; STDERR-SAME: tensor_load_to_lds
; STDERR-SAME: [VIMAGE]
