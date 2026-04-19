; RUN: %not %raise_cli %wmma_f32_16x16x4_f32_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=wmma_f32_16x16x4_f32_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P
; opcode 0x05D). Pins the contractual cross-target loud-failure
; behaviour of transpiler/handle_valu_vop3p.cpp under
; SemOp::V_WMMA_F32_16x16x4_F32. Companion fixture to
; `wmma_f32_16x16x4_f32_same_target.ll`, which pins the same-target
; (gfx1250 -> gfx1250) intrinsic-emit path.
;
; This SemOp is structurally distinct from the K=32 (16-bit) and
; K=64 (8-bit) WMMA family covered by the parameterised
; `emitWMMAtoMFMA` helper (transpiler/wmma_lowering.cpp): the
; per-Wave32-lane A/B fragment is `<2 x f32>` (only 2 dwords) and
; the helper has no K=4 f32 codepath. Cross-target lift to gfx942
; would require a new `mfma_f32_16x16x4f32` decomposition path
; that no kernel in the current corpus exercises, so the handler
; refuses loudly via `RaiseFailure::unsupportedShape` per the
; user-rules (no silent fallbacks). The native LLVM intrinsic
; `int_amdgcn_wmma_f32_16x16x4_f32` is itself gated `isGFX125xOnly`
; in IntrinsicsAMDGPU.td:4113-4114, so even an intrinsic-emit on a
; non-gfx1250 target would fail at codegen — the principled lift
; is the loud refusal pinned here.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code —
;      the test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the offending mnemonic
;      (`v_wmma_f32_16x16x4_f32`), the encoding format (`VOP3P`),
;      and the architectural-mismatch detail. The handler routes
;      the failure through `RaiseFailure::unsupportedShape(di,
;      "VOP3P", detail)`, which raise_cli then formats as
;      `raise_cli: kernel '<name>' failed to raise: <mnemonic>
;      [<format>] @offset=0x... :: <detail>` (raise_cli.cpp). Both
;      the format bucket and the detail text are pinned so that
;      drift in either surfaces a meaningful FileCheck failure
;      pointing at the exact change.

; STDERR: raise_cli: kernel 'wmma_f32_16x16x4_f32_kernel' failed to raise:
; STDERR-SAME: v_wmma_f32_16x16x4_f32
; STDERR-SAME: [VOP3P]
; STDERR-SAME: gfx1250-only
; STDERR-SAME: int_amdgcn_wmma_f32_16x16x4_f32
; STDERR-SAME: AMDGPUWMMAIntrinsicsGFX1250
; STDERR-SAME: mfma_f32_16x16x4f32
