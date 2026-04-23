; RUN: %not %raise_cli %wmma_phantom_lane_f16_chain_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_f16_chain_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Staged fixture for the 2-WMMA-chain K=32 f16 phantom-lane regime.
; Kernel: `__launch_bounds__(32)` with two back-to-back
; `__builtin_amdgcn_wmma_f32_16x16x32_f16` calls chaining through the
; accumulator.  Under MODREP (phantom-lane fallback), the K=32 / K=64
; WMMA dispatch in `handle_valu_vop3p.cpp` currently refuses (see the
; gate in that file for the rationale — the staged strict.wwm-scoped
; lowering is verified correct for minimal repros but has an
; unexplained residual divergence on Triton `matmul_fp16_16x16` at
; M>=32).
;
; Today's CHECKs pin the refusal diagnostic's attribution anchors so
; a regression that silently drops the gate surfaces here.  Once the
; matmul_fp16_16x16 residual is pinned and the gate flips, this
; fixture will be rewritten to CHECK the IR shape of the lowering
; (strict.wwm markers, single-pass runGroupPass output, MFMA chain)
; via a non-%not RUN line with affirmative CHECKs.
; CHECK: raise_cli: kernel 'wmma_f16_chain_kernel' failed to raise:
; CHECK-SAME: v_wmma_f32_16x16x32_f16
; CHECK-SAME: VOP3P
; CHECK-SAME: ModuloReplicationProjection
; CHECK-SAME: matmul_fp16_16x16
