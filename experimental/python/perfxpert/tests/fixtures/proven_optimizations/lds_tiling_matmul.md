# lds_tiling_matmul

**Bottleneck:** memory_transfer
**Technique:** LDS tile-and-reuse for matmul
**Source:** AMD CDNA Optimization Guide

## Baseline DB (`lds_tiling_matmul.baseline.db`)
- 1 kernel `matmul_global`, duration 3.0 s, VGPR/thread = 96, LDS = 0
- FETCH_SIZE = 5M (global memory reads every iteration)
- HBM-bound matmul with poor reuse

## Optimized DB (`lds_tiling_matmul.optimized.db`)
- Kernel `matmul_lds`, duration 0.9 s (3.33× faster)
- VGPR/thread = 96, LDS = 16 KB (tile buffer)
- FETCH_SIZE = 600k (reads cached in LDS, massive reduction)

## Expected agentic behaviour
1. Detects memory_transfer bottleneck with GEMM signature
2. Observes high HBM bandwidth, small LDS allocation
3. Recommends LDS tiling to convert to compute-bound
4. Speedup 2.5-6.0× confirmed; all gates pass
