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
// followed by >= 3 digits, the last of which may be a hex step. It has no
// GPU/HSA dependency, so these cases exercise the parser directly.

#include "lib/rocprofiler-sdk/agent.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
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

TEST(parse_gfx_target_version, hex_stepping)
{
    // A step of 10 or more is spelled as a hex digit, and KFD packs it by its
    // numeric value: gfx90a is step 10, so gfx_target_version is 90010.
    EXPECT_EQ(agent::parse_gfx_target_version("gfx90a"), std::optional<uint32_t>{90010});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx90c"), std::optional<uint32_t>{90012});

    // The hex spelling is what keeps the encoding unambiguous. Spelling step 10
    // in decimal would produce "gfx9010", which is a different target entirely
    // (major 90, minor 1, step 0), so the two must not pack to the same value.
    EXPECT_EQ(agent::parse_gfx_target_version("gfx9010"), std::optional<uint32_t>{900100});
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

    const auto hex_packed = agent::parse_gfx_target_version("gfx90a");
    ASSERT_TRUE(hex_packed.has_value());

    const uint32_t hex_value = *hex_packed;
    EXPECT_EQ(hex_value / 10000, 9u);        // major
    EXPECT_EQ((hex_value / 100) % 100, 0u);  // minor
    EXPECT_EQ(hex_value % 100, 10u);         // step
}

TEST(parse_gfx_target_version, packed_uint32_boundary)
{
    EXPECT_EQ(agent::parse_gfx_target_version("gfx4294969f"), std::optional<uint32_t>{4294960915u});
    EXPECT_EQ(agent::parse_gfx_target_version("gfx42949700"), std::nullopt);
}

TEST(parse_gfx_target_version, rejects_very_long_overflow)
{
    const auto target = std::string{"gfx"} + std::string(4096, '9');
    EXPECT_EQ(agent::parse_gfx_target_version(target), std::nullopt);
}

TEST(parse_gfx_target_version, rejects_malformed)
{
    // Anything that is not "gfx" + >= 3 digits, hex allowed only in the final
    // step position, must yield nullopt.
    EXPECT_EQ(agent::parse_gfx_target_version(""), std::nullopt);         // empty
    EXPECT_EQ(agent::parse_gfx_target_version("gfx"), std::nullopt);      // no digits
    EXPECT_EQ(agent::parse_gfx_target_version("gfx11"), std::nullopt);    // too few digits
    EXPECT_EQ(agent::parse_gfx_target_version("foo"), std::nullopt);      // wrong prefix
    EXPECT_EQ(agent::parse_gfx_target_version("gfx11xy"), std::nullopt);  // non-digit tail
    EXPECT_EQ(agent::parse_gfx_target_version("gfx90z"), std::nullopt);   // step out of hex range
    EXPECT_EQ(agent::parse_gfx_target_version("gfx9a0"), std::nullopt);   // hex outside the step
}
