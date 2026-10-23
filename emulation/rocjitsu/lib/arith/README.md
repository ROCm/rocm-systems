# Rocjitsu Arithmetic Library

## Current implementation

Dense WMMA supports FP16/BF16, packed accumulators, gfx12 FP8/BF8 and
integer formats, in both wave sizes. Sparse WMMA adds 2:4 metadata selection
with tied accumulators. Fused arithmetic includes FP16/FP32, packed halves,
literals and mixed precision. Add, subtract, multiply and exponent operations
are also available. FP64 LDEXP supports exact results; selection and other basic FP64 arithmetic
currently support loose results only. Division helpers cover SCALE, FMAS and FIXUP;
Dot products cover integer, FP16/BF16 and FP8/BF8 formats.
Family headers describe exact-result and FP-status support.
FP status is available for gfx12 FP16 FMA under default guest rounding and denormals,
and for gfx12 FP32 MUL under nearest-even rounding, preserved denormals, and no OMOD/CLAMP.
WMMA has no FP-status output; mode 3 has the same numeric requirements as mode 2.
Source/output modifiers and FMA DPP routing are supported. Compatibility selects
the established instruction policy. CPU implementations are scalar, AVX2, AVX-512 and native BF16.

Family headers state each entry point's supported controls.

The Rocjitsu Arithmetic Library (`rj-arith`) provides a synchronous C interface to GPU arithmetic. Each call executes one GPU instruction for an entire wave on the
calling CPU thread.

[rocjitsu/docs/arith-design.md](../../docs/arith-design.md) describes the API,
numerical semantics, implementation principles, and testing policy.

The semantics field selects loose or empirical bit-exact arithmetic. rj-arith is not a GPU runtime or instruction decoder.

See [coverage per architecture](COVERAGE.md) for instruction-family support.

Rocjitsu continues to use its existing instruction handlers.

## Build and run

Build as part of Rocjitsu, using its build prerequisites. Run the following
from `emulation/rocjitsu/`. Implementation code uses Rocjitsu's C++ standard;
public headers support C23 and C++17.

```sh
cmake -S . -B ../rocjitsu-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../rocjitsu-build --parallel
ctest --test-dir ../rocjitsu-build --parallel --output-on-failure -R '^rj_arith\.'
../rocjitsu-build/lib/arith/tests/cpuinfo/rj_arith_cpuinfo
```

Build directories must be outside the source tree. Link the `rj_arith` CMake
target to use the library and its public include path.

## API example

Include [`rj_arith/rj_arith.h`](include/rj_arith/rj_arith.h). The example uses
this [RDNA4 Wave32 WMMA entry point](include/rj_arith/detail/rj_arith_wmma.h):

```c
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(
    uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
```

It computes a 16×16 matrix product with FP16 inputs and FP32 accumulation,
`D = A × B + C`. Each operand is an array of pointers to VGPRs; each VGPR
contains 32 `uint32_t` lane words. A and B each use four VGPRs, packing two
FP16 values per word; C and D each use eight VGPRs of FP32 bit patterns.
D may overlap the input registers. The call reads inputs and writes the whole
output wave synchronously; WMMA ignores EXEC and has no execution-mask parameter
or FP status output.

`flags` combines CPU capabilities from `rj_arith_init_cpu_flags()` with the
semantics selection. Cache the CPU flags when making repeated calls.
`instruction_flags = 0` selects unmodified operands. This WMMA entry point
accepts modes 0–3. With no FP-status output, mode 3 has the same numeric contract as mode 2.
The host FP environment must be in its default state; the library never explicitly
changes it. The function returns `RJ_ARITH_SUCCESS` (zero) on success and leaves
outputs unchanged on error.

The following C/C++ program uses loose semantics. It fills A and B with
FP16 ones and C with zeros, then prints the resulting 16×16 FP32 matrix (every element is 16):

```c
#include "rj_arith/rj_arith.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  // [32] is the lane count; [4] and [8] are the VGPR counts for each operand.
  uint32_t a_words[4][32] = {0}, b_words[4][32] = {0};
  uint32_t c_words[8][32] = {0}, d_words[8][32] = {0};
  const uint32_t *a[4], *b[4], *c[8];
  uint32_t *d[8];

  // Construct the test inputs and bind operand pointers to their VGPR storage.
  for (int i = 0; i < 4; ++i) {
    a[i] = a_words[i];
    b[i] = b_words[i];
    for (int lane = 0; lane < 32; ++lane) {
      // Two FP16 values of 1.0 in each lane word.
      a_words[i][lane] = b_words[i][lane] = 0x3c003c00U;
    }
  }
  for (int i = 0; i < 8; ++i) {
    c[i] = c_words[i];
    d[i] = d_words[i];
  }
  int error = rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(
      rj_arith_init_cpu_flags() | RJ_ARITH_SEMANTICS_LOOSE,
      0, d, a, b, c);
  if (error != RJ_ARITH_SUCCESS) {
    fprintf(stderr, "WMMA failed: %d\n", error);
    return 1;
  }
  // Print the test output as a matrix, decoding its VGPR and lane layout.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      // Decode the output register layout, then reinterpret the FP32 bits.
      uint32_t bits = d_words[row % 8][col + 16 * (row / 8)];
      float value;
      memcpy(&value, &bits, sizeof(value));
      printf("%6.1f%s", (double)value, col == 15 ? "\n" : " ");
    }
  }
  return 0;
}
```

The name ends in `_w32_gfx12`: Wave32, then the gfx12 (RDNA4) architecture
suffix. See [API naming](../../docs/arith-design.md#c-interface-and-instruction-names) for the general convention.

See the [WMMA declarations](include/rj_arith/detail/rj_arith_wmma.h)
and [WMMA tests](tests/wmma/v_wmma_f32_16x16x16_f16_w32_gfx12_test.cpp)
for layouts, modifiers, and nonzero examples.

Licensed under the [MIT license](../../LICENSE.md).
