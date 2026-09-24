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

#include "rma_engine_gtest.hpp"

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(RmaEngineTestFixture, PostWriteThenPollCompletes) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "RMAENGINE-PUT-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 0, msg, n);

  int lane = lm_.lane(/*ctx=*/0, /*pe=*/1);
  uint64_t wr_id = rma_.post_write(
      lane, reinterpret_cast<uintptr_t>(h + 0), rma_.lkey_for(h + 0),
      rma_.remote_va(1, h + 2048), rma_.rkey(1), static_cast<uint32_t>(n),
      /*inl=*/false, next_wr_id_++);
  ASSERT_NE(wr_id, 0u) << "post_write failed";

  struct ibv_wc wc = WaitForOneCompletion();
  EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_STREQ(h + 2048, msg);
}

TEST_F(RmaEngineTestFixture, PostReadThenPollCompletes) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "RMAENGINE-GET-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 2048, msg, n);

  int lane = lm_.lane(/*ctx=*/0, /*pe=*/1);
  uint64_t wr_id = rma_.post_read(
      lane, reinterpret_cast<uintptr_t>(h + 4096), rma_.lkey_for(h + 4096),
      rma_.remote_va(1, h + 2048), rma_.rkey(1), static_cast<uint32_t>(n),
      next_wr_id_++);
  ASSERT_NE(wr_id, 0u) << "post_read failed";

  struct ibv_wc wc = WaitForOneCompletion();
  EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_STREQ(h + 4096, msg);
}

TEST_F(RmaEngineTestFixture, PostFaddReturnsPreImage) {
  char *h = static_cast<char *>(heap_);
  *reinterpret_cast<uint64_t *>(h + 8192) = 100;

  int lane = lm_.lane(/*ctx=*/0, /*pe=*/1);
  uint64_t wr_id = rma_.post_fadd(lane, reinterpret_cast<uintptr_t>(scratch_),
                                  scratch_lkey_, rma_.remote_va(1, h + 8192),
                                  rma_.rkey(1), /*add=*/5, next_wr_id_++);
  ASSERT_NE(wr_id, 0u) << "post_fadd failed";

  struct ibv_wc wc = WaitForOneCompletion();
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(scratch_), 100u);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(h + 8192), 105u);
}

TEST_F(RmaEngineTestFixture, PostCasReturnsPreImage) {
  char *h = static_cast<char *>(heap_);
  *reinterpret_cast<uint64_t *>(h + 8192) = 105;

  int lane = lm_.lane(/*ctx=*/0, /*pe=*/1);
  uint64_t wr_id =
      rma_.post_cas(lane, reinterpret_cast<uintptr_t>(scratch_),
                    scratch_lkey_, rma_.remote_va(1, h + 8192), rma_.rkey(1),
                    /*cmp=*/105, /*swp=*/200, next_wr_id_++);
  ASSERT_NE(wr_id, 0u) << "post_cas failed";

  struct ibv_wc wc = WaitForOneCompletion();
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(scratch_), 105u);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(h + 8192), 200u);
}
