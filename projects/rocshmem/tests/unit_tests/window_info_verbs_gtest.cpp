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

#include "window_info_verbs_gtest.hpp"

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(WindowInfoVerbsTestFixture, PutBytesLoopback) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "WINDOWINFOVERBS-PUT-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 0, msg, n);

  win_->put(h + 2048, h + 0, n, /*pe=*/1);
  EXPECT_STREQ(h + 2048, msg);
}

TEST_F(WindowInfoVerbsTestFixture, GetBytesLoopback) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "WINDOWINFOVERBS-GET-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 2048, msg, n);

  win_->get(h + 4096, h + 2048, n, /*pe=*/1);
  EXPECT_STREQ(h + 4096, msg);
}

TEST_F(WindowInfoVerbsTestFixture, AmoFaddReturnsPreImage) {
  char *h = static_cast<char *>(heap_);
  *reinterpret_cast<uint64_t *>(h + 8192) = 100;

  uint64_t old = win_->amo_fadd(h + 8192, /*add=*/5, /*pe=*/1);
  EXPECT_EQ(old, 100u);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(h + 8192), 105u);
}

TEST_F(WindowInfoVerbsTestFixture, AmoCasReturnsPreImage) {
  char *h = static_cast<char *>(heap_);
  *reinterpret_cast<uint64_t *>(h + 8192) = 105;

  uint64_t prev = win_->amo_cas(h + 8192, /*cmp=*/105, /*swp=*/200, /*pe=*/1);
  EXPECT_EQ(prev, 105u);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(h + 8192), 200u);
}

TEST_F(WindowInfoVerbsTestFixture, PutNbiThenQuiet) {
  char *h = static_cast<char *>(heap_);
  memcpy(h + 0, "NBI", 4);

  win_->put_nbi(h + 16384, h + 0, 4, /*pe=*/1);
  win_->quiet();
  EXPECT_STREQ(h + 16384, "NBI");
}

// Burst far more non-blocking puts than the send-queue depth (64 here). Without
// the make_room() back-pressure in WindowInfoVerbs, ibv_post_send would return
// ENOMEM once the SQ filled, the dropped post would still be counted
// outstanding, and quiet() would hang forever. This must complete and land the
// last write.
TEST_F(WindowInfoVerbsTestFixture, PutNbiBurstExceedsSqDepthNoDeadlock) {
  char *h = static_cast<char *>(heap_);
  const uint32_t kPattern = 0xA5A5A5A5u;
  *reinterpret_cast<uint32_t *>(h + 0) = kPattern;

  // 256 == 4x the SQ depth configured in the fixture.
  for (int i = 0; i < 256; ++i) {
    win_->put_nbi(h + 16384, h + 0, sizeof(kPattern), /*pe=*/1);
  }
  win_->quiet();
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(h + 16384), kPattern);
}
