// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Pure-logic unit test for rocprofiler::agent::parse_gfx_target_version(). This
// helper turns a "gfx<NNN>" name into the KFD-style packed encoding
// (major*10000 + minor*100 + step) and must reject anything that is not "gfx"
// followed by >= 3 decimal digits. It has no GPU/HSA dependency, so these cases
// exercise the parser directly.

#include "lib/rocprofiler-sdk/agent.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace agent = ::rocprofiler::agent;

TEST(parse_gfx_target_version, valid_public_targets)
{
    // Uses only already-public gfx targets. Encoding: last digit = step,
    // second-to-last = minor, leading digits = major.
    EXPECT_EQ(agent::parse_gfx_target_version("gfx900"), std::optional<uint32_t>{90000});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx906"), std::optional<uint32_t>{90006});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx908"), std::optional<uint32_t>{90008});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx1030"), std::optional<uint32_t>{100300});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx1100"), std::optional<uint32_t>{110000});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx1102"), std::optional<uint32_t>{110002});
}

TEST(parse_gfx_target_version, packing_decomposition)
{
    // The packed value must decompose back into major/minor/step so a malformed
    // name can never silently fold into a plausible-looking version.
    const auto packed = agent::parse_gfx_target_version("gfx1100");
    ASSERT_TRUE(packed.has_value());

    const uint32_t value = *packed;
    EXPECT_EQ(value / 10000, 11u);       // major
    EXPECT_EQ((value / 100) % 100, 0u);  // minor
    EXPECT_EQ(value % 100, 0u);          // step
}

TEST(parse_gfx_target_version, rejects_malformed)
{
    // Anything that is not "gfx" + >= 3 decimal digits must yield nullopt.
    EXPECT_EQ(agent::parse_gfx_target_version(""), std::nullopt);         // empty
    EXPECT_EQ(agent::parse_gfx_target_version("gfx"), std::nullopt);      // no digits
    EXPECT_EQ(agent::parse_gfx_target_version("gfx11"), std::nullopt);    // too few digits
    EXPECT_EQ(agent::parse_gfx_target_version("foo"), std::nullopt);      // wrong prefix
    EXPECT_EQ(agent::parse_gfx_target_version("gfx11xy"), std::nullopt);  // non-digit tail
    // Hex-suffixed names (e.g. the public gfx90a) are intentionally not handled
    // by this purely-numeric parser and must be rejected rather than misparsed.
    EXPECT_EQ(agent::parse_gfx_target_version("gfx90a"), std::nullopt);
}
