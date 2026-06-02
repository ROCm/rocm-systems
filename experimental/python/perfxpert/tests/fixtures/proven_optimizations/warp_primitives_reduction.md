# warp_primitives_reduction

**Bottleneck:** memory_transfer
**Technique:** `__shfl_sync` warp-scoped reduction (replaces LDS)
**Source:** CDNA Programming Guide + RDNA ISA manual

## Baseline DB (`warp_primitives_reduction.baseline.db`)
- 1 kernel `lds_reduction`, duration 1.0 s, LDS = 4 KB
- SQ_LDS_BANK_CONFLICT = 1M (many bank conflicts in reduction)

## Optimized DB (`warp_primitives_reduction.optimized.db`)
- Kernel `shfl_reduction`, duration 0.6 s (40% faster), LDS = 0
- SQ_LDS_BANK_CONFLICT = 0 (no LDS traffic, uses VGPR shuffle)

## Expected agentic behaviour
1. Detects memory_transfer bottleneck with LDS traffic signature
2. Observes high LDS bank conflicts and small LDS pool
3. Recommends `__shfl_sync` for warp-scoped reductions
4. Speedup 1.30-1.80× confirmed; all gates pass
