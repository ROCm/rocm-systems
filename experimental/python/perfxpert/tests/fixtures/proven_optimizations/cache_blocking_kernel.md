# cache_blocking_kernel

**Bottleneck:** memory_transfer
**Technique:** block-size tuning for L2 cache reuse
**Source:** AMD CDNA3/4 ISA Guide

## Baseline DB (`cache_blocking_kernel.baseline.db`)
- 1 kernel `stencil_bad_block`, duration 1.5 s
- TCC_HIT_sum = 300k, TCC_MISS_sum = 700k (30% L2 hit rate, poor reuse)

## Optimized DB (`cache_blocking_kernel.optimized.db`)
- Kernel `stencil_tuned_block`, duration 0.9 s (40% faster)
- TCC_HIT_sum = 800k, TCC_MISS_sum = 200k (80% L2 hit rate, excellent reuse)
- Block size tuned to fit working set in L2

## Expected agentic behaviour
1. Classifies baseline as memory_transfer bottleneck
2. Observes low L2 hit rate and high HBM utilization
3. Recommends block-size sweep for cache-resident working set
4. Speedup 1.20-2.00× confirmed; all gates pass
