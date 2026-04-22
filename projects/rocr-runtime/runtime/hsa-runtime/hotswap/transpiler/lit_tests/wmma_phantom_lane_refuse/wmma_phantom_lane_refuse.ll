; RUN: %not %raise_cli %wmma_phantom_lane_refuse_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_phantom_lane_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Regression fence for the WMMA → MFMA refusal gate added in
; `handle_valu_vop3p.cpp`.  The fixture's launch_bounds(32) drive
; `raiser.cpp`'s phantom-lane fallback to MODREP (covered by the
; sibling `phantom_lane_modrep_fallback/` fixture), and the kernel's
; `__builtin_amdgcn_wmma_f32_16x16x4_f32` call then forces the K=4
; f32 case of `handle_valu_vop3p.cpp` to reach the
; `emitWMMAtoMFMA_F32_16x16x4` call site on a gfx942 target.  With
; the `providesFullWaveExecInvariant` gate in place, the handler
; refuses before that call is emitted.
;
; If the gate regressed (e.g. a future refactor dropped the check),
; `raise_cli` would either succeed (producing a silently-miscompiled
; HSACO that `compare_correctness`'s `matmul_fp16_16x16` caught
; empirically — all shapes WRONG numerics) or fall into the
; following `hasMFMA` decomposition with the wrong projection.  The
; `%not` wrapper + FileCheck anchors below catch the regression in
; either direction: `%not` demands non-zero exit, and the CHECKs
; demand the specific attribution chain.

; Two pieces of diagnostic state the fixture pins, in the order
; `raise_cli` emits them at handler-time refusal:
;
;   (1) The phantom-lane → MODREP fallback log from `raiser.cpp`
;       (proves the fallback fired, i.e. the WMMA gate below
;       fired under MODREP — not WaveNative).
;   (2) The `raise_cli` outer-tool failure line that carries the
;       opcode attribution, the invariant name, the intrinsic
;       reference, and the phantom-lane pointer.
;
; Both are on the same `stderr`; the `%not` prefix on the RUN line
; demands raise_cli exits non-zero (the handler-time refusal
; path).  Handler-time refusals do NOT emit the
; `transpiler: pre-translation abort:` banner that
; c5-predicate-chain and other Phase-6 classifier refusals use;
; that banner is only emitted by classifier refusals (see
; `raiser.cpp`'s Phase-6 output block).  The chain below is the
; attribution shape for handler-time `unsupportedShape` refusals.

; (1) Fallback log — same emission point as
;     `phantom_lane_modrep_fallback.ll`'s CHECK block, expected
;     here because we're in the same phantom-lane regime.
; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK-SAME: falling back to ModuloReplicationProjection

; (2) Outer-tool failure line: `raise_cli` writes a single
;     consolidated diagnostic that names the kernel, the opcode,
;     the DecodedInst class (VOP3P), and the attribution chain.
;     The `CHECK-SAME` ordering below matches the actual string
;     assembly in `handle_valu_vop3p.cpp`'s `RaiseFailure::
;     unsupportedShape` call site so a future reword that
;     drops any of the attribution anchors is caught.
; CHECK: raise_cli: kernel 'wmma_phantom_lane_refuse_kernel' failed to raise:
; CHECK-SAME: v_wmma_f32_16x16x4_f32
; CHECK-SAME: VOP3P
; CHECK-SAME: full-wave EXEC invariant
; CHECK-SAME: init_whole_wave
; CHECK-SAME: phantom-lane fallback
