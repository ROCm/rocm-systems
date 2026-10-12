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

#include <chrono>
#include <thread>

#include "rocshmem/rocshmem_common.hpp"  // ROCSHMEM_SIGNAL_*, ROCSHMEM_CMP_*

using namespace rocshmem;
using namespace rocshmem::net;

TEST_F(WindowInfoVerbsTestFixture, PutBytesLoopback) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "WINDOWINFOVERBS-PUT-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 0, msg, n);

  win_->put_bytes(h + 2048, h + 0, n, /*pe=*/1);
  EXPECT_STREQ(h + 2048, msg);
}

TEST_F(WindowInfoVerbsTestFixture, GetBytesLoopback) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "WINDOWINFOVERBS-GET-CHECK-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 2048, msg, n);

  win_->get_bytes(h + 4096, h + 2048, n, /*pe=*/1);
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

// --- put-with-signal + signal query ---------------------------------------

TEST_F(WindowInfoVerbsTestFixture, PutSignalSetDeliversDataAndSignal) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "SIGNAL-SET-PAYLOAD-0123456789";
  size_t n = strlen(msg) + 1;
  memcpy(h + 0, msg, n);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);
  *sig = 0;

  win_->put_signal(h + 2048, h + 0, n, sig, /*signal=*/0xABCD,
                   ROCSHMEM_SIGNAL_SET, /*pe=*/1);

  EXPECT_STREQ(h + 2048, msg);        // payload landed
  EXPECT_EQ(*sig, 0xABCDu);           // signal set to the absolute value
}

TEST_F(WindowInfoVerbsTestFixture, PutSignalSetOverwritesPriorValue) {
  char *h = static_cast<char *>(heap_);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);
  *sig = 0xDEAD;

  win_->put_signal(h + 2048, h + 0, 8, sig, /*signal=*/7, ROCSHMEM_SIGNAL_SET,
                   /*pe=*/1);
  EXPECT_EQ(*sig, 7u);  // SET replaces, does not accumulate
}

TEST_F(WindowInfoVerbsTestFixture, PutSignalAddAccumulates) {
  char *h = static_cast<char *>(heap_);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);
  *sig = 100;

  win_->put_signal(h + 2048, h + 0, 8, sig, /*signal=*/5, ROCSHMEM_SIGNAL_ADD,
                   /*pe=*/1);
  EXPECT_EQ(*sig, 105u);

  for (int i = 0; i < 4; ++i) {
    win_->put_signal(h + 2048, h + 0, 8, sig, /*signal=*/10,
                     ROCSHMEM_SIGNAL_ADD, /*pe=*/1);
  }
  EXPECT_EQ(*sig, 145u);  // 105 + 4*10
}

// Signal-implies-data: whenever the signal is observed, the payload is already
// in place. Loop to shake out any write/signal reordering on the RC QP.
TEST_F(WindowInfoVerbsTestFixture, PutSignalOrderingSignalImpliesData) {
  char *h = static_cast<char *>(heap_);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);
  auto *payload = reinterpret_cast<uint64_t *>(h + 4096);

  for (uint64_t i = 1; i <= 256; ++i) {
    *sig = 0;
    *reinterpret_cast<uint64_t *>(h + 0) = i;  // source
    win_->put_signal(payload, h + 0, sizeof(uint64_t), sig, /*signal=*/i,
                     ROCSHMEM_SIGNAL_SET, /*pe=*/1);
    ASSERT_EQ(*sig, i);
    ASSERT_EQ(*payload, i) << "signal observed before data at iter " << i;
  }
}

TEST_F(WindowInfoVerbsTestFixture, SignalFetchReadsLocalWord) {
  char *h = static_cast<char *>(heap_);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);

  *sig = 0;
  EXPECT_EQ(win_->signal_fetch(sig), 0u);
  *sig = 0x1234;
  EXPECT_EQ(win_->signal_fetch(sig), 0x1234u);
}

TEST_F(WindowInfoVerbsTestFixture, SignalWaitUntilReturnsWhenSatisfied) {
  char *h = static_cast<char *>(heap_);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);

  // Already-satisfied predicates return immediately.
  *sig = 42;
  win_->signal_wait_until(sig, ROCSHMEM_CMP_EQ, 42);
  win_->signal_wait_until(sig, ROCSHMEM_CMP_GE, 10);
  win_->signal_wait_until(sig, ROCSHMEM_CMP_NE, 0);

  // A concurrent writer flips the word; wait_until must observe it and return.
  *sig = 0;
  std::thread writer([sig] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    *reinterpret_cast<volatile uint64_t *>(sig) = 99;
  });
  win_->signal_wait_until(sig, ROCSHMEM_CMP_EQ, 99);
  EXPECT_EQ(*sig, 99u);
  writer.join();
}

TEST_F(WindowInfoVerbsTestFixture, PutSignalNbiCompletedByQuiet) {
  char *h = static_cast<char *>(heap_);
  const char *msg = "SIGNAL-NBI";
  size_t n = strlen(msg) + 1;
  memcpy(h + 0, msg, n);
  auto *sig = reinterpret_cast<uint64_t *>(h + 8192);
  *sig = 0;

  win_->put_signal_nbi(h + 2048, h + 0, n, sig, /*signal=*/1,
                       ROCSHMEM_SIGNAL_ADD, /*pe=*/1);
  win_->quiet();
  EXPECT_STREQ(h + 2048, msg);
  EXPECT_EQ(*sig, 1u);
}
