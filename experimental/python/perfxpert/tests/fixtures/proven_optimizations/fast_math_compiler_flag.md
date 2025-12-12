# fast_math_compiler_flag

**Bottleneck:** compute
**Technique:** `-ffast-math` compiler flag
**Source:** LLVM fast-math documentation + ROCm compiler reference

## Baseline DB (`fast_math_compiler_flag.baseline.db`)
- 1 kernel `transcendental_heavy`, duration 1.0 s, VGPR/thread = 64
- Many transcendental operations (sqrt, exp, log) in default IEEE mode

## Optimized DB (`fast_math_compiler_flag.optimized.db`)
- Same `transcendental_heavy` kernel, duration 0.8 s (20% faster)
- VGPR/thread unchanged
- Compiled with `-ffast-math` enabling unsafe transforms

## Expected agentic behaviour
1. Classifies as compute bottleneck with transcendental-heavy signature
2. Recommends `-ffast-math` compiler flag with correctness gate
3. Speedup 1.10-1.40× confirmed in fixture
4. Gate 5 validates numerical stability via output comparison
