# hip_stream_overlap

**Bottleneck:** latency
**Technique:** async memcpy + multi-stream kernel overlap
**Source:** HIP API best practices + AMD optimization guide

## Baseline DB (`hip_stream_overlap.baseline.db`)
- 1 kernel `compute_stage`, duration 0.5 s, starts at 0.5 s (blocked by memcpy)
- H2D memcpy: 0.5 s, 64 MB — serialized with kernel

## Optimized DB (`hip_stream_overlap.optimized.db`)
- Same `compute_stage`, duration 0.5 s, starts at 0.0 s (overlapped with H2D)
- H2D memcpy: 0.5 s, 64 MB — concurrent execution via hipMemcpyAsync + streams
- Total time: 0.5 s instead of 1.0 s (50% speedup from overlap)

## Expected agentic behaviour
1. Classifies as latency bottleneck (memcpy dominating)
2. Detects serialization: kernel start at 500ms > H2D end
3. Recommends stream-based async memcpy
4. Speedup 1.30-1.70× validated; overlap confirmed in optimized fixture
