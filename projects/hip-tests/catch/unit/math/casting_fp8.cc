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
#include <hip/amd_detail/amd_hip_fp8.h>


/**
 * @addtogroup fp8_casting fp8_casting
 * @{
 * @ingroup MathTest
 */


TEST_CASE("Unit_hip_fp8_e4m3_fnuz_UnsignedLongLongCasting") {
  SECTION("Zero value") {
    unsigned long long input = 0ULL;
    __hip_fp8_e4m3_fnuz fp8(input);
    REQUIRE(fp8.__x == 0x00);
  }

  SECTION("Small integer values") {
    unsigned long long input[] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 7ULL, 8ULL};

    unsigned short expected_raw[] = {0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E, 0x50};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Powers of 2 - exact representations") {
    unsigned long long input[] = {8ULL, 16ULL, 32ULL, 64ULL, 128ULL, 256ULL};

    unsigned short expected_raw[] = {0x50, 0x58, 0x60, 0x68, 0x70, 0x78};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Mantissa boundary values") {
    unsigned long long input[] = {7ULL, 8ULL, 9ULL, 15ULL, 16ULL, 17ULL};

    unsigned short expected_raw[] = {0x4E, 0x50, 0x51, 0x57, 0x58, 0x59};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Rounding test cases") {
    unsigned long long input[] = {10ULL, 11ULL, 12ULL, 13ULL, 14ULL, 20ULL, 24ULL, 28ULL};

    unsigned short expected_raw[] = {0x52, 0x53, 0x54, 0x55, 0x56, 0x5A, 0x5C, 0x5E};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Large values and overflow") {
    unsigned long long input[] = {(1ULL << 10), (1ULL << 12), (1ULL << 15),
                                  (1ULL << 20), (1ULL << 32), ULLONG_MAX};

    unsigned short expected_raw[] = {0x7C, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Specific mantissa edge cases") {
    unsigned long long input[] = {0x8ULL, 0x9ULL, 0xFULL, 0x10ULL, 0x11ULL, 0x18ULL, 0x1FULL};

    unsigned short expected_raw[] = {0x50, 0x51, 0x57, 0x58, 0x59, 0x5C, 0x5F};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e4m3_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }
}

TEST_CASE("Unit_hip_fp8_e5m2_fnuz_UnsignedLongLongCasting") {
  SECTION("Zero value") {
    unsigned long long input = 0ULL;
    __hip_fp8_e5m2_fnuz fp8(input);
    REQUIRE(fp8.__x == 0x00);
  }

  SECTION("Small integer values") {
    unsigned long long input[] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 7ULL, 8ULL};

    unsigned short expected_raw[] = {0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Powers of 2 - exact representations") {
    unsigned long long input[] = {4ULL, 8ULL, 16ULL, 32ULL, 64ULL, 128ULL, 256ULL, 512ULL};

    unsigned short expected_raw[] = {0x48, 0x58, 0x68, 0x78, 0x88, 0x98, 0xA8, 0xB8};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Mantissa boundary values") {
    unsigned long long input[] = {3ULL, 4ULL, 5ULL, 7ULL, 8ULL, 9ULL};

    unsigned short expected_raw[] = {0x44, 0x48, 0x4C, 0x54, 0x58, 0x5C};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Rounding test cases") {
    unsigned long long input[] = {5ULL, 6ULL, 7ULL, 10ULL, 12ULL, 14ULL};

    unsigned short expected_raw[] = {0x4C, 0x50, 0x54, 0x60, 0x68, 0x70};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Large values and overflow") {
    unsigned long long input[] = {(1ULL << 15), (1ULL << 20), (1ULL << 25),
                                  (1ULL << 32), (1ULL << 48), ULLONG_MAX};

    unsigned short expected_raw[] = {0xF4, 0xFC, 0x7E, 0x7E, 0x7E, 0x7E};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Specific mantissa edge cases") {
    unsigned long long input[] = {0x4ULL, 0x5ULL, 0x7ULL, 0x8ULL, 0x9ULL, 0xCULL, 0xFULL};

    unsigned short expected_raw[] = {0x48, 0x4C, 0x54, 0x58, 0x5C, 0x68, 0x74};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }

  SECTION("Extended range values") {
    unsigned long long input[] = {1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL, 32768ULL};

    unsigned short expected_raw[] = {0xC8, 0xD8, 0xE8, 0xF8, 0xFC, 0x7E};

    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
      __hip_fp8_e5m2_fnuz fp8(input[i]);
      REQUIRE(fp8.__x == expected_raw[i]);
    }
  }
}


/**
 * @}
 */
