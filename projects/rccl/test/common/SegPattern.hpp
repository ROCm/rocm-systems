/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_SEG_PATTERN_HPP
#define RCCL_TEST_SEG_PATTERN_HPP

#include <cstddef>
#include <cstdint>

namespace RCCLTestHelpers
{

// Multi-segment payload byte i. The period is prime, so data moved by any power-of-two distance (a segment, a
// page) reads differently; with a period of 256 a slice taken from the wrong segment would still match.
constexpr size_t  kSegPatternPeriod   = 251;
// Above every pattern byte, so a byte that was never written cannot pass for payload.
constexpr uint8_t kSegPatternSentinel = 0xFE;

constexpr uint8_t SegPatternByte(int seed, size_t i)
{
    return static_cast<uint8_t>((static_cast<size_t>(seed) + i) % kSegPatternPeriod);
}

} // namespace RCCLTestHelpers

#endif // RCCL_TEST_SEG_PATTERN_HPP
