// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression coverage for the unterminated process-name copy in
// gpuvsmi_get_pid_info(): strncpy() appends no NUL once it copies its full
// count, so a name of AMDSMI_MAX_STRING_LENGTH bytes or more left info.name
// unterminated. Reachable in production, where `name` is a readlink() of
// /proc/<pid>/exe into a PATH_MAX buffer.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "guarded_buffer.h"

using amd::smi::CopyBounded;
using amdsmi_test::GuardedBuffer;

TEST(SystemUnit, BoundedCopyShortStringCopiedWhole) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  const std::string src = "/usr/bin/rocm-smi";
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), src), src.length());
  EXPECT_EQ(std::string(gb.buf), src);
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(SystemUnit, BoundedCopyExactlyCapacityMinusOneFits) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  const std::string src(AMDSMI_MAX_STRING_LENGTH - 1, 'p');
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), src), src.length());
  EXPECT_EQ(std::strlen(gb.buf), src.length());
  EXPECT_EQ(gb.buf[AMDSMI_MAX_STRING_LENGTH - 1], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

// The exact length at which the old strncpy() stopped terminating.
TEST(SystemUnit, BoundedCopyExactlyCapacityIsTerminated) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  const std::string src(AMDSMI_MAX_STRING_LENGTH, 'q');
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), src),
            static_cast<size_t>(AMDSMI_MAX_STRING_LENGTH - 1));
  EXPECT_EQ(std::strlen(gb.buf), size_t{AMDSMI_MAX_STRING_LENGTH - 1});
  EXPECT_TRUE(gb.CanariesIntact());
}

// A PATH_MAX-scale readlink() result, the realistic production trigger.
TEST(SystemUnit, BoundedCopyPathMaxSizedSourceIsTerminated) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  const std::string src = "/" + std::string(4095, 'd');
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), src),
            static_cast<size_t>(AMDSMI_MAX_STRING_LENGTH - 1));
  EXPECT_EQ(std::strlen(gb.buf), size_t{AMDSMI_MAX_STRING_LENGTH - 1});
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(SystemUnit, BoundedCopyEmptySourceYieldsEmptyString) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), ""), 0u);
  EXPECT_EQ(gb.buf[0], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(SystemUnit, BoundedCopyZeroCapacityWritesNothing) {
  GuardedBuffer<1> gb;
  EXPECT_EQ(CopyBounded(gb.buf, 0, "anything"), 0u);
  EXPECT_TRUE(gb.CanariesIntact());
}

// Embedded NUL bytes must not shorten the copy: the source is a std::string,
// so its length, not the first NUL, bounds the write.
TEST(SystemUnit, BoundedCopyEmbeddedNulDoesNotShortenTheCopy) {
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  const std::string src = std::string("abc") + '\0' + "def";
  EXPECT_EQ(CopyBounded(gb.buf, sizeof(gb.buf), src), 7u);
  EXPECT_EQ(gb.buf[7], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}
