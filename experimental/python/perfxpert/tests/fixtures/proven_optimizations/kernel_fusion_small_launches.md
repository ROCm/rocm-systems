# kernel_fusion_small_launches

**Bottleneck:** latency
**Technique:** fuse pointwise kernels into one launch
**Source:** Meta KernelEvolve blog + standard GPU fusion patterns

## Baseline DB (`kernel_fusion_small_launches.baseline.db`)
- 2000 kernels `pw_op_0` to `pw_op_1999`, each 8 µs, total 16 ms
- 2000 hipLaunchKernel API calls at 2 µs each = 4 ms overhead
- Total: ~20 ms, API overhead significant

## Optimized DB (`kernel_fusion_small_launches.optimized.db`)
- 1 kernel `fused_pw`, duration 8 ms (all 2000 ops fused)
- 1 hipLaunchKernel API call at 2 µs
- Total: ~8 ms, 2.5× faster due to eliminated launch overhead

## Expected agentic behaviour
1. Detects latency bottleneck with high API overhead pct
2. Observes many short kernels (2000 × 8 µs each)
3. Recommends kernel fusion for pointwise chains
4. Speedup 2.0-6.0× confirmed; gates validate structural fusion
