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
