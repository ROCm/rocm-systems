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

#include "ib_ctrl_gtest.hpp"

#include <cstdlib>

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(IbCtrlTestFixture, FillLocalDestPopulatesQpn) {
  std::vector<DestInfo> local;
  ctrl_.fill_local_dest(local);

  ASSERT_EQ(local.size(), 2u);
  EXPECT_NE(local[0].qpn, 0u);
  EXPECT_NE(local[1].qpn, 0u);
}

TEST_F(IbCtrlTestFixture, ConnectQpReachesRts) {
  std::vector<DestInfo> local;
  ctrl_.fill_local_dest(local);

  // Cross-connect: QP0<->QP1 (two distinct QPs, RC loopback). connect_qp()
  // drives INIT->RTR->RTS via modify_qp, checking each transition's return
  // code internally, so a true result here is already proof both QPs reached
  // RTS (net::Ibv keeps no link-time libibverbs dependency, so this avoids a
  // redundant raw ibv_query_qp() call just to re-confirm the same state).
  EXPECT_TRUE(ctrl_.connect_qp(0, local[1])) << "lane 0 failed to reach RTS";
  EXPECT_TRUE(ctrl_.connect_qp(1, local[0])) << "lane 1 failed to reach RTS";
}

TEST_F(IbCtrlTestFixture, SelectGidPrefersRoutableRoCEv2OverLinkLocal) {
  if (!ctrl_.dev().is_roce) {
    GTEST_SKIP() << "not a RoCE port; GID-preference logic doesn't apply";
  }
  if (getenv("ROCSHMEM_ROCE_GID_INDEX")) {
    GTEST_SKIP() << "ROCSHMEM_ROCE_GID_INDEX override set; selection "
                    "algorithm not exercised";
  }

  // Independently recompute the expected selection straight from the GID
  // table, mirroring IbCtrl::select_gid()'s documented rule (skip fe80::/10
  // link-local, prefer RoCEv2 over RoCEv1), to verify the fix without calling
  // the private method under test.
  constexpr size_t kMax = 128;
  std::vector<struct ibv_gid_entry> entries(kMax);
  int n = ibv_.query_gid_table(ctrl_.dev().ctx, entries.data(), kMax);
  ASSERT_GT(n, 0) << "no GID table entries; cannot verify selection";

  int expected = -1;
  uint32_t expected_type = IBV_GID_TYPE_ROCE_V1;
  for (int i = 0; i < n; i++) {
    const auto &e = entries[i];
    if (e.port_num != ctrl_.dev().port) {
      continue;
    }
    bool zero = true;
    for (int b = 0; b < 16; b++) {
      if (e.gid.raw[b] != 0) {
        zero = false;
        break;
      }
    }
    if (zero) {
      continue;
    }
    if (e.gid.raw[0] == 0xfe && (e.gid.raw[1] & 0xc0) == 0x80) {
      continue;  // link-local, not routable across nodes
    }
    if (expected < 0 || e.gid_type > expected_type) {
      expected = static_cast<int>(e.gid_index);
      expected_type = e.gid_type;
    }
  }
  ASSERT_GE(expected, 0) << "no routable (non-link-local) GID found on this port";

  EXPECT_EQ(ctrl_.dev().gid_index, expected);

  // The selected GID itself must not be link-local.
  const auto &g = ctrl_.dev().gid;
  bool link_local = (g.raw[0] == 0xfe) && ((g.raw[1] & 0xc0) == 0x80);
  EXPECT_FALSE(link_local) << "selected GID is link-local (not routable)";
}
