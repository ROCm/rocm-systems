; RUN: %not %raise_cli %wmma_scale_f32_16x16x128_f8f6f4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250
; RDNA4 VOP3PX2 opcode 0x033, ScaledWMMA family). Pins the contractual
; cross-target loud-failure behaviour of
; transpiler/handle_valu_vop3p.cpp under
; SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4. Companion fixture to
; `wmma_scale_f32_16x16x128_f8f6f4_same_target.ll`, which pins the
; same-target (gfx1250 -> gfx1250) intrinsic-emit path.
;
; gfx942 has no scaled-WMMA hardware. The closest sibling on gfx942
; is `mfma_scale_f32_16x16x128_f8f6f4` (already mapped via
; `SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4` and handled in
; `handle_mfma.cpp`), but the WMMA→MFMA lane redistribution for
; K=128 + per-matrix-fmt selection + the
; `matrix_a/b_scale_fmt × scale_src0/src1` exponent application is
; not modelled in `wmma_lowering.cpp` (only K=32 / K=64 fp16 / bf16 /
; fp8 / bf8 / iu8 paths exist there). Per the user-rules (no silent
; fallbacks) and consistent with the gfx1250-only refusal contract
; applied to `V_WMMA_F32_16x16x4_F32`, the handler refuses loudly via
; `RaiseFailure::unsupportedShape` to surface both the cross-target
; capability gap AND the missing scaled-WMMA decomposition path. The
; native LLVM intrinsic `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`
; is itself gated `isGFX125xOnly` in IntrinsicsAMDGPU.td:4138, so
; even an intrinsic-emit on a non-gfx1250 target would fail at
; codegen — the principled lift is the loud refusal pinned here.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code —
;      the test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the offending mnemonic
;      (`v_wmma_scale_f32_16x16x128_f8f6f4`), the encoding format
;      (`VOP3P`), and the architectural-mismatch detail. The handler
;      routes the failure through
;      `RaiseFailure::unsupportedShape(di, "VOP3P", detail)`, which
;      raise_cli then formats as `raise_cli: kernel '<name>' failed
;      to raise: <mnemonic> [<format>] @offset=0x... :: <detail>`
;      (raise_cli.cpp). Both the format bucket and the detail text
;      are pinned so that drift in either surfaces a meaningful
;      FileCheck failure pointing at the exact change.

; STDERR: raise_cli: kernel 'wmma_scale_f32_16x16x128_f8f6f4_kernel' failed to raise:
; STDERR-SAME: v_wmma_scale_f32_16x16x128_f8f6f4
; STDERR-SAME: [VOP3P]
; STDERR-SAME: gfx1250-only
; STDERR-SAME: int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4
; STDERR-SAME: AMDGPUWMMAIntrinsicsGFX1250
; STDERR-SAME: K=128 scaled-WMMA
