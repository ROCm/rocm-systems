; RUN: %not %raise_cli %global_load_async_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for FLAT `global_load_async_to_lds_b{8,32,64,128}`.
; Pins the contractual cross-target loud-failure behaviour of
; transpiler/handle_flat.cpp under the `GLOBAL_LOAD_ASYNC_TO_LDS_B*`
; family.
;
; The gfx1250 asynccnt unit (FLATInstructions.td:VFLAT 0x60-0x62
; reals; FLAT_Global_Load_LDS_Pseudo<IsAsync=1>) has no equivalent
; on gfx942. The matching LLVM intrinsics
; `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
; (IntrinsicsAMDGPU.td:3939-3946, all sharing the
; `AMDGPUAsyncGlobalLoadToLDS` signature on line 3904) are gated
; by `FeatureGFX1250Insts` and emit the dedicated VFLAT 0x60-0x62
; encodings — no codegen path lowers them on a non-gfx1250
; backend. The user-rules forbid silent fallbacks; a synthesised
; `global_load + ds_write` pair would alter the wave's
; memory-ordering and asynccnt observable state, which the
; gfx1250 producers (tensilelite f8 / bf16 GEMMs, triton
; block-pipelined matmul kernels) rely on for software
; pipelining. The principled lift on a non-gfx1250 target IS the
; loud refusal.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (the first async load encountered in the kernel
;      stream is the b32) and the encoding format (`FLAT`).
;      raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>] @offset=0x<offset> :: <detail>`.
;
; The handler also emits an explicit `transpiler: FLAT: ...`
; line that names the architectural mismatch and the
; same-target intrinsic; pinning that line keeps the diagnostic
; text from drifting into something less actionable for users
; who read raise_cli's stderr directly.

; STDERR: transpiler: FLAT: global_load_async_to_lds_b32
; STDERR-SAME: gfx1250 asynccnt unit
; STDERR-SAME: amdgcn.global.load.async.to.lds.b{8,32,64,128}
; STDERR-SAME: FeatureGFX1250Insts

; STDERR: raise_cli: kernel 'global_load_async_to_lds_kernel' failed to raise:
; STDERR-SAME: global_load_async_to_lds_b32
; STDERR-SAME: [FLAT]
