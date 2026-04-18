// ARCH: gfx1250
//
// Fixture for the Class-3 "inter-replica communication via shared
// state" obstruction (lit_tests/c3_atomic_cas).
//
// SPE_DESIGN.md §3 Class 3: atomics with non-commutative semantics
// (cmpxchg / xchg) at an address derived from lane-local state
// create a new race under modulo-replication — target lanes i and
// i + W_s collide on the same address, producing an outcome the
// source program never expressed (e.g. exactly one cmpxchg wins,
// but the source intended both to win in their respective
// single-wave executions).
//
// Commutative atomics (add / max / min) have a rewrite in §4's
// table (per-replica gate); non-commutative ones do not. The
// classifier must therefore refuse at C3_NonCommutativeAtomic.
//
// NOTE. gpt-oss-derisking.md §4 observes that GPT-OSS / hipBLASLt /
// Gluon contain ZERO non-commutative atomics in 170 kernels, so
// this class is purely a guard: no corpus kernel triggers it today,
// but if one ever does we want a loud abort rather than a silent
// miscompile.

#include <hip/hip_runtime.h>

extern "C" __global__ void c3_atomic_cas_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int expected = 0;
  int desired = tid + 1;
  // atomicCAS is the HIP-level name for atomic cmpxchg; this keeps
  // the source portable across gfx942 / gfx1250 so the fixture
  // builds under both arches. The per-lane `expected` / `desired`
  // ensure the address seen by lanes i and i + W_s is the same
  // (pointer to out[0]) but the racer value is lane-distinct,
  // reproducing the Class-3 structural hazard.
  atomicCAS(out, expected, desired);
}
