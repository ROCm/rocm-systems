# device_sync_removal

**Bottleneck:** latency
**Technique:** remove spurious hipDeviceSynchronize
**Source:** HIP API guide + common source-analysis anti-pattern

## Baseline DB (`device_sync_removal.baseline.db`)
- 20 kernels `k`, each 10 ms, total 200 ms
- 20 hipDeviceSynchronize calls, each 50 ms (blocking), total 1000 ms
- Total: ~1200 ms, sync overhead dominates

## Optimized DB (`device_sync_removal.optimized.db`)
- Same 20 kernels `k`, each 10 ms, total 200 ms
- 2 hipDeviceSynchronize calls (only at boundaries), each 50 ms = 100 ms
- Total: ~300 ms (kernel + 2 syncs), 4× faster

## Expected agentic behaviour
1. Detects latency bottleneck with high GPU idle time
2. Observes many hipDeviceSynchronize calls vs kernel time
3. Recommends removing unnecessary syncs, keeping boundary only
4. Speedup 1.20-1.80× validated; regression gate checks for data races
