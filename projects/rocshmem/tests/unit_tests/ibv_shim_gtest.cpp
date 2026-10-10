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

#include "ibv_shim_gtest.hpp"

#include <cstring>

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(IbvShimTestFixture, OpenDeviceAndQueryPort) {
  ctx_ = ibv_.open_device(device_list_[0]);
  ASSERT_NE(ctx_, nullptr) << "open_device failed on an enumerated device";

  struct ibv_port_attr pa;
  std::memset(&pa, 0, sizeof(pa));
  EXPECT_EQ(ibv_.query_port(ctx_, 1, &pa), 0);
}

TEST_F(IbvShimTestFixture, AllocPdCreateCqCreateRcQp) {
  ctx_ = ibv_.open_device(device_list_[0]);
  ASSERT_NE(ctx_, nullptr) << "open_device failed on an enumerated device";

  auto *pd = ibv_.alloc_pd(ctx_);
  ASSERT_NE(pd, nullptr);

  auto *cq = ibv_.create_cq(ctx_, 64, nullptr, nullptr, 0);
  ASSERT_NE(cq, nullptr);

  struct ibv_qp_init_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.send_cq = cq;
  qa.recv_cq = cq;
  qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = 64;
  qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1;
  qa.cap.max_recv_sge = 1;
  auto *qp = ibv_.create_qp(pd, &qa);
  ASSERT_NE(qp, nullptr);

  // Data-plane fast paths are inline op-table dispatch, not dlsym'd -- confirm
  // the op-table entries are populated.
  EXPECT_NE(qp->context->ops.post_send, nullptr);
  EXPECT_NE(cq->context->ops.poll_cq, nullptr);

  ibv_.destroy_qp(qp);
  ibv_.destroy_cq(cq);
  ibv_.dealloc_pd(pd);
}
