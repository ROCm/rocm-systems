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

#include "rdma_loopback_gtest.hpp"

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(RdmaLoopbackTestFixture, RdmaWriteLoopbackDataMatches) {
  // Write a pattern into src region [0..64), RDMA_WRITE it to dst region
  // [2048..).
  char *h = static_cast<char *>(heap_);
  const char *msg = "ROCSHMEM-VERBS-LOOPBACK-0123456789ABCDEF";
  size_t len = strlen(msg) + 1;
  memcpy(h, msg, len);

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uintptr_t>(h);
  sge.length = len;
  sge.lkey = mr_.heap_lkey();

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0x1234;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(h) + 2048;
  wr.wr.rdma.rkey = mr_.heap_rkey();

  struct ibv_send_wr *bad = nullptr;
  ASSERT_EQ(ibv_.post_send(ctrl_.qp(0), &wr, &bad), 0);

  struct ibv_wc wc;
  int n = 0;
  for (int spins = 0; spins < 10000000 && n == 0; ++spins) {
    n = ibv_.poll_cq(ctrl_.cq(0), 1, &wc);
  }
  ASSERT_GT(n, 0) << "poll_cq timed out";
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS) << "completion status=" << wc.status;

  EXPECT_STREQ(h + 2048, msg);
}
