# memory_coalescing_stride_fix

**Bottleneck:** memory_transfer
**Technique:** data layout transpose for coalesced HBM access
**Source:** AMD CDNA3 Optimization Guide

## Baseline DB (`memory_coalescing_stride_fix.baseline.db`)
- 1 kernel `strided_copy`, duration 1.0 s, VGPR/thread = 48
- High FETCH_SIZE (800k), high TCP read reqs (900k) — uncoalesced access pattern

## Optimized DB (`memory_coalescing_stride_fix.optimized.db`)
- Same `coalesced_copy` kernel, duration 0.25 s (4× faster)
- FETCH_SIZE reduced to 200k, TCP read reqs to 200k — coalesced layout
- Same VGPR/thread

## Expected agentic behaviour
1. Classifies baseline as `memory_transfer` bottleneck
2. Detects high L1 miss rate and low HBM utilization
3. Recommends data layout restructuring citing proven case
4. All gates pass: speedup 3.0-5.0× confirmed, no regressions
