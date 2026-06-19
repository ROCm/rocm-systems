# vgpr_reduction_compute_bound

**Bottleneck:** compute
**Technique:** `__launch_bounds__` VGPR reduction
**Source:** GEAK paper + AMD CDNA3 Waves-per-EU VGPR occupancy table

## Baseline DB (`vgpr_reduction_compute_bound.baseline.db`)
- 1 kernel `matmul_kernel`, duration 1.0 s, VGPR/thread = 128
- Counter values placing arithmetic intensity above the ridge point (compute-bound)
- Waves/EU = 4 (low occupancy for a compute-bound kernel)

## Optimized DB (`vgpr_reduction_compute_bound.optimized.db`)
- Same `matmul_kernel`, duration 0.70 s (30% faster)
- VGPR/thread = 64 via `__launch_bounds__(256, 2)`
- Waves/EU = 8 (2× occupancy)

## Expected agentic behaviour
1. Analysis agent classifies baseline as `compute` bottleneck via `bottleneck.classify_from_metrics`
2. Recommendation agent hands off to `compute_specialist`
3. `compute_specialist` cites `proven_optimizations.vgpr_reduction_compute_bound` precedent
4. Correctness gates 1-5 pass on the optimized fixture:
   - Gate 1 (Claims): speedup 1.25-1.40× range matches measurement
   - Gate 2 (Sakana): counter values unchanged, speedup attributable to occupancy
   - Gate 3 (Schema): all output fields populated
   - Gate 4 (Regression): no kernel regressed > 5%
   - Gate 5 (Correctness): kernel name unchanged, semantics preserved (structural)
