; RUN: %not %raise_cli %global_prefetch_b8_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=global_prefetch_b8_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for FLAT `global_prefetch_b8`.  Pins the
; contractual cross-target loud-failure behaviour of
; transpiler/handle_flat.cpp under `SemOp::GLOBAL_PREFETCH_B8`.
;
; The gfx1250 VMEM-prefetch unit (FLATInstructions.td: VFLAT 0x05D
; real `global_prefetch_b8`; `FLAT_Prefetch_Pseudo` with
; `has_vdst = 0`) has no equivalent on gfx942.  The matching LLVM
; intrinsic `int_amdgcn_global_prefetch` (IntrinsicsAMDGPU.td:3211)
; is gated by `HasVmemPrefInsts`, which is only set on gfx1250+
; (LLVM PR #150466 added both the intrinsic + the subtarget feature
; bit; the AMDGPU.td feature line on gfx1250 is `+vmem-pref-insts`,
; absent on every gfx9xx and every other gfx10/11/12 line).  No
; codegen path lowers the intrinsic on a non-gfx1250 backend.
;
; The closest sibling `int_amdgcn_s_prefetch_data` is gated on
; `HasSafeSmemPrefetch`, which is also strictly gfx12-onwards, but
; even on a hypothetical gfx12-as-target compilation we could NOT
; substitute it for `global.prefetch` here: the SMEM prefetch
; requires a UNIFORM SGPR base pointer (the lifter sees a divergent
; per-lane VGPR address derived from `vdiv = saddr + voff_zext`),
; and proving uniformity at lift time would require a divergence
; analysis pass we do not run.  The user-rules forbid silent
; fallbacks; silently dropping the hint would mask both the
; cross-target capability gap AND the resulting pipeline-stall
; regression in any software-pipelined kernel that relied on the
; prefetch overlapping a prior compute chain (the dominant Triton
; corpus shape that introduces this op).  The principled lift on a
; non-gfx1250 target IS the loud refusal.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`global_prefetch_b8`) and the encoding format
;      (`FLAT`).  raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>] @offset=0x<offset> :: <detail>`.
;
; The handler also emits an explicit `transpiler: FLAT: ...` line
; that names the architectural mismatch and the same-target
; intrinsic; pinning that line keeps the diagnostic text from
; drifting into something less actionable for users who read
; raise_cli's stderr directly.

; STDERR: transpiler: FLAT: global_prefetch_b8
; STDERR-SAME: gfx1250 VMEM-prefetch unit
; STDERR-SAME: amdgcn.global.prefetch
; STDERR-SAME: HasVmemPrefInsts

; STDERR: raise_cli: kernel 'global_prefetch_b8_kernel' failed to raise:
; STDERR-SAME: global_prefetch_b8
; STDERR-SAME: [FLAT]
