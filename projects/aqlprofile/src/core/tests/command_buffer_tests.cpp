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
#include "core/commandbuffermgr.hpp"
#include <climits>


using namespace aql_profile;

namespace {

struct DummyBuffer {
    std::array<char, 4096> data;
};

TEST(CommandBufferMgrTest, BasicPrefixAndSize) {
    DummyBuffer buf;
    CommandBufferMgr mgr(buf.data.data(), sizeof(buf.data));
    // Should not throw and should have a nonzero size
    EXPECT_GT(mgr.GetSize(), 0u);
}

TEST(CommandBufferMgrTest, FinalizeThrowsOnSmallData) {
    DummyBuffer buf;
    CommandBufferMgr mgr(buf.data.data(), sizeof(buf.data));
    mgr.SetPreSize(128);
    // Finalize with data_size <= precmds_size should throw
    EXPECT_THROW(mgr.Finalize(64), aql_profile_exc_msg);
}

} // namespace
