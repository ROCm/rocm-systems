// Standalone unit tests for ROCM_DIM3_PACK. Compile with:
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
//       projects/clr/hipamd/test/lttng/test_dim3_pack.cpp -o test_dim3_pack
// The header is included via the relative path below; no -I needed for it.
#include "../../src/lttng/rocm_dim3_pack.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>  /* for dim3 */

#define EXPECT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::printf("FAIL: %s:%d  %s != %s  (got 0x%016lx, want 0x%016lx)\n", \
                    __FILE__, __LINE__, #a, #b, \
                    (unsigned long)(a), (unsigned long)(b)); \
        return 1; \
    } \
} while (0)

int main() {
    // Coverage of the lane-encoding contract documented in rocm_dim3_pack.h.

    // test_dim3_packed_normal_range
    EXPECT_EQ(ROCM_DIM3_PACK(dim3(1, 2, 3)),
              uint64_t(0x0003000200000001ULL));

    // test_dim3_packed_x_full_32bit
    EXPECT_EQ(ROCM_DIM3_PACK(dim3(0xFFFFFFFFu, 1, 1)),
              uint64_t(0x00010001FFFFFFFFULL));

    // test_dim3_packed_y_overflow: y=0x10000 saturates to 0xFFFF, bit 63 set
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 0x10000u, 1));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 32) & 0xFFFFu, uint64_t(0xFFFFu));      // y lane saturated
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(1));            // z untouched
        EXPECT_EQ(v & 0xFFFFFFFFu, uint64_t(1));                // x lane survived
    }

    // test_dim3_packed_z_overflow: z=0x10000 saturates to 0x7FFF, bit 63 set
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x10000u));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));      // z lane saturated
        EXPECT_EQ(v & 0xFFFFFFFFu, uint64_t(1));                // x lane survived
    }

    // test_dim3_packed_z_high_bit_overflow: z=0x8000 ALSO overflow (15-bit lane)
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x8000u));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, ROCM_DIM3_OVERFLOW_BIT);
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));
    }

    // test_dim3_packed_z_max_no_false_overflow: z=0x7FFF is the non-overflow max
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(1, 1, 0x7FFFu));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, uint64_t(0));
        EXPECT_EQ((v >> 48) & 0x7FFFu, uint64_t(0x7FFFu));
    }

    // test_dim3_packed_x_max_no_false_overflow: x is full 32 bits, never overflows
    {
        uint64_t v = ROCM_DIM3_PACK(dim3(0xFFFFFFFFu, 1, 1));
        EXPECT_EQ(v & ROCM_DIM3_OVERFLOW_BIT, uint64_t(0));
    }

    std::printf("PASS: all dim3_packed encoding tests\n");
    return 0;
}
