/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "casting_common.hh"
#include <hip/amd_detail/amd_hip_bf16.h>


/**
 * @addtogroup bfloat16_casting bfloat16_casting
 * @{
 * @ingroup MathTest
 */

TEST_CASE("Unit_hip_bfloat16_UnsignedLongLongCasting") {
  SECTION("Zero value") {
    unsigned long long input = 0ULL;
    __hip_bfloat16 bf16(input);
    __hip_bfloat16_raw raw = bf16;
    REQUIRE(raw.x == 0x0000);
  }

  SECTION("Small integer values") {
    unsigned long long input[] = {1ULL, 2ULL, 3ULL, 4ULL, 8ULL, 16ULL, 32ULL, 64ULL, 128ULL};
    unsigned short expected_raw[] = {0x3F80, 0x4000, 0x4040, 0x4080, 0x4100,
                                     0x4180, 0x4200, 0x4280, 0x4300};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }

  SECTION("Powers of 2 - exact representations") {
    unsigned long long input[] = {256ULL,   512ULL,   1024ULL,  2048ULL,    4096ULL,    8192ULL,
                                  16384ULL, 32768ULL, 65536ULL, 1048576ULL, 16777216ULL};

    unsigned short expected_raw[] = {0x4380, 0x4400, 0x4480, 0x4500, 0x4580, 0x4600,
                                     0x4680, 0x4700, 0x4780, 0x4980, 0x4B80};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }

  SECTION("Boundary values around msb_pos == 7") {
    unsigned long long input[] = {127ULL, 128ULL, 129ULL, 255ULL, 256ULL, 257ULL};

    unsigned short expected_raw[] = {0x42FE, 0x4300, 0x4301, 0x437F, 0x4380, 0x4380};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }

  SECTION("Rounding test cases") {
    unsigned long long input[] = {384ULL, 448ULL, 320ULL, 352ULL, 768ULL, 896ULL, 1536ULL, 1792ULL};

    unsigned short expected_raw[] = {0x43C0, 0x43E0, 0x43A0, 0x43B0,
                                     0x4440, 0x4460, 0x44C0, 0x44E0};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }

  SECTION("Maximum values and overflow") {
    unsigned long long input[] = {ULLONG_MAX, ULLONG_MAX - 1, (1ULL << 63) - 1};

    unsigned short expected_raw[] = {0x7F80, 0x7F80, 0x7F80};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }

  SECTION("Specific mantissa edge cases") {
    unsigned long long input[] = {0x80ULL,  0x81ULL,  0xFFULL, 0x100ULL,
                                  0x101ULL, 0x180ULL, 0x1FFULL};

    unsigned short expected_raw[] = {0x4300, 0x4301, 0x437F, 0x4380, 0x4380, 0x43C0, 0x43FF};

    for (int i = 0; i < sizeof(input) / sizeof(input[0]); i++) {
      __hip_bfloat16 bf16(input[i]);
      __hip_bfloat16_raw raw = bf16;
      REQUIRE(raw.x == expected_raw[i]);
    }
  }
}

/**
 * @}
 */
