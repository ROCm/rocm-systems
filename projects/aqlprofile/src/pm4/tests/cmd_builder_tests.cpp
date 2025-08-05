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

#include <gtest/gtest.h>
#include "pm4/cmd_builder.h"

using pm4_builder::CmdBuffer;

struct DummyPacket {
    uint32_t a;
    uint32_t b;
};

TEST(CmdBufferTest, AppendSinglePacket) {
    CmdBuffer buf;
    DummyPacket pkt{0x12345678, 0x9abcdef0};
    buf.Append(pkt);

    // Should have 2 dwords
    EXPECT_EQ(buf.DwSize(), 2u);

    // Data should match what we appended
    const uint32_t* data = static_cast<const uint32_t*>(buf.Data());
    EXPECT_EQ(data[0], 0x12345678u);
    EXPECT_EQ(data[1], 0x9abcdef0u);
}

TEST(CmdBufferTest, AppendMultiplePackets) {
    CmdBuffer buf;
    DummyPacket pkt1{1, 2};
    DummyPacket pkt2{3, 4};
    buf.Append(pkt1, pkt2);

    EXPECT_EQ(buf.DwSize(), 4u);
    const uint32_t* data = static_cast<const uint32_t*>(buf.Data());
    EXPECT_EQ(data[0], 1u);
    EXPECT_EQ(data[1], 2u);
    EXPECT_EQ(data[2], 3u);
    EXPECT_EQ(data[3], 4u);
}

TEST(CmdBufferTest, AppendRawData) {
    CmdBuffer buf;
    uint32_t raw[3] = {10, 20, 30};
    buf.Append(raw, 3);

    EXPECT_EQ(buf.DwSize(), 4u);
    const uint32_t* data = static_cast<const uint32_t*>(buf.Data());
    EXPECT_EQ(data[0], 10u);
    EXPECT_EQ(data[1], 20u);
    EXPECT_EQ(data[2], 30u);
}