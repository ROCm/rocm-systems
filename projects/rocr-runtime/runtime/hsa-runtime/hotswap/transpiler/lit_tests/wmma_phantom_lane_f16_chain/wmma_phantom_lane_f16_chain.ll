; RUN: %not %raise_cli %wmma_phantom_lane_f16_chain_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_f16_chain_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Staged fixture for the 2-WMMA-chain K=32 f16 phantom-lane regime.
; Kernel: `__launch_bounds__(32)` with two back-to-back
; `__builtin_amdgcn_wmma_f32_16x16x32_f16` calls chaining through the
; accumulator.  Under MODREP (phantom-lane fallback), the K=32 / K=64
; WMMA dispatch in `handle_valu_vop3p.cpp` currently refuses (see
; the gate in that file + matrix-translation.md §12.4 for the
; residual-divergence rationale on the larger `matmul_fp16` kernel;
; the `matmul_fp16_16x16` single-WMMA case is VERIFIED working
; in-session post the `ttmp7` init fix of §12.3, but the gate is
; shape-scoped, not kernel-scoped, so it still refuses).
;
; Today's CHECKs pin the refusal diagnostic's attribution anchors so
; a regression that silently drops the gate surfaces here.  Once the
; `matmul_fp16` multi-WMMA residual is pinned and the gate flips,
; this fixture will be rewritten to CHECK the IR shape of the
; lowering (strict.wwm markers, single-pass runGroupPass output,
; MFMA chain) via a non-%not RUN line with affirmative CHECKs.
; CHECK: raise_cli: kernel 'wmma_f16_chain_kernel' failed to raise:
; CHECK-SAME: v_wmma_f32_16x16x32_f16
; CHECK-SAME: VOP3P
; CHECK-SAME: ModuloReplicationProjection
