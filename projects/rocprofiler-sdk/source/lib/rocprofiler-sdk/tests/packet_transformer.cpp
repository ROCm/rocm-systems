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

#include "lib/rocprofiler-sdk/hsa/packet_transformer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
TEST(PacketTransformer, ScopedWriterInstallsAndRestores)
{
    std::vector<uint8_t> bytes;

    {
        ScopedWriter guard{[&bytes](const void* pkts, uint64_t count) {
            const auto* src = static_cast<const uint8_t*>(pkts);
            for(uint64_t i = 0; i < count; ++i)
            {
                bytes.push_back(src[i]);
            }
        }};

        const uint8_t first[]  = {0x01, 0x02, 0x03};
        const uint8_t second[] = {0xA0, 0xB0};

        packet_writer_trampoline(first, 3);
        EXPECT_EQ(bytes.size(), 3u);

        packet_writer_trampoline(second, 2);
        ASSERT_EQ(bytes.size(), 5u);

        EXPECT_EQ(bytes[0], 0x01u);
        EXPECT_EQ(bytes[1], 0x02u);
        EXPECT_EQ(bytes[2], 0x03u);
        EXPECT_EQ(bytes[3], 0xA0u);
        EXPECT_EQ(bytes[4], 0xB0u);
    }

    // ScopedWriter scope exited; trampoline must be a no-op now.
    const uint8_t post[] = {0xFF, 0xFE};
    packet_writer_trampoline(post, 2);
    EXPECT_EQ(bytes.size(), 5u);  // unchanged
}

TEST(PacketTransformer, ScopedWriterNestedSavesAndRestores)
{
    std::string outer;
    std::string inner;

    {
        ScopedWriter outer_guard{[&outer](const void* pkts, uint64_t count) {
            const auto* src = static_cast<const char*>(pkts);
            outer.append(src, static_cast<size_t>(count));
        }};

        {
            ScopedWriter inner_guard{[&inner](const void* pkts, uint64_t count) {
                const auto* src = static_cast<const char*>(pkts);
                inner.append(src, static_cast<size_t>(count));
            }};

            // Inner installed: trampoline goes to inner.
            const char msg_inner[] = "I";
            packet_writer_trampoline(msg_inner, 1);
            EXPECT_EQ(inner, "I");
            EXPECT_EQ(outer, "");
        }  // inner scope exits, outer restored

        // Back to outer writer.
        const char msg_outer[] = "O";
        packet_writer_trampoline(msg_outer, 1);
        EXPECT_EQ(inner, "I");
        EXPECT_EQ(outer, "O");
    }  // outer scope exits, no writer installed

    // Trampoline is now a no-op.
    const char msg_post[] = "X";
    packet_writer_trampoline(msg_post, 1);
    EXPECT_EQ(outer, "O");
    EXPECT_EQ(inner, "I");
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
