# mfma_enablement

**Bottleneck:** compute
**Technique:** MFMA matrix intrinsics (v_mfma_*)
**Source:** AMD Matrix Instruction Calculator + MI300X ISA manual

## Baseline DB (`mfma_enablement.baseline.db`)
- 1 kernel `gemm_vectors`, duration 5.0 s, VGPR/thread = 96
- SQ_INSTS_VALU = 10M (vector ALU), SQ_INSTS_MFMA = 0 (no matrix ops)

## Optimized DB (`mfma_enablement.optimized.db`)
- Kernel `gemm_mfma`, duration 0.8 s (6.25× faster)
- SQ_INSTS_VALU reduced to 1M, SQ_INSTS_MFMA = 9M (matrix pipeline active)
- Same VGPR/thread

## Expected agentic behaviour
1. Detects compute bottleneck with GEMM-like signature
2. Observes zero MFMA utilization in baseline
3. Recommends MFMA intrinsic enablement on CDNA3/4 targets
4. Speedup 4.0-10.0× validated; all gates pass
