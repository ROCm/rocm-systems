/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#include "net_helpers_gtest.hpp"

#include <cstring>

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(NetHelpersTestFixture, RemoteAddrPreservesOffset) {
  RemoteRegion r{0x9000, 0x11};
  uintptr_t local_base = 0x1000, local_va = 0x1240;
  EXPECT_EQ(remote_addr(r, local_va, local_base), 0x9240u);
}

TEST_F(NetHelpersTestFixture, LaneMapBasic) {
  LaneMap lm(3 /*ctx*/, 4 /*pes*/);
  EXPECT_EQ(lm.num_lanes(), 12);
  EXPECT_EQ(lm.lane(2, 3), 11);
  EXPECT_EQ(lm.lane_begin(1), 4);
  EXPECT_EQ(lm.lane_end(1), 8);
}

TEST_F(NetHelpersTestFixture, CompletionCountersTracksOutstanding) {
  LaneMap lm(3, 4);
  CompletionCounters cc(lm.num_lanes());
  cc.on_post(11, 5);
  cc.on_complete(11, 2);
  EXPECT_EQ(cc.outstanding(11), 3u);
  EXPECT_FALSE(cc.drained(11));

  cc.on_complete(11, 3);
  EXPECT_TRUE(cc.drained(11));
}

TEST_F(NetHelpersTestFixture, AllgatherValuePopulatesAllPes) {
  struct Blob {
    uintptr_t base;
    uint64_t key;
  };
  int num_pes = 2, my_pe = 0;
  Blob mine{0xABC0, 0x22};
  auto fake_allgather = [](void *inout, size_t bpp) {
    // Pretend PE1 published {0xDEF0, 0x33}.
    auto *p = static_cast<char *>(inout);
    Blob peer{0xDEF0, 0x33};
    std::memcpy(p + 1 * bpp, &peer, sizeof(Blob));
  };

  auto all = allgather_value<Blob>(mine, num_pes, my_pe, fake_allgather);
  EXPECT_EQ(all[0].base, 0xABC0u);
  EXPECT_EQ(all[1].base, 0xDEF0u);
  EXPECT_EQ(all[1].key, 0x33u);
}
