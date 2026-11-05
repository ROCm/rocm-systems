// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "kpack_internal.h"

namespace {

constexpr uint64_t OVERSIZED_ALLOCATION = 4ULL * 1024 * 1024 * 1024 + 1;

TEST(CompressionTest, NoopRejectsOversizedBlobBeforeAllocation) {
  kpack_archive archive{};
  archive.blobs.push_back({0, OVERSIZED_ALLOCATION});

  EXPECT_EQ(kpack::decompress_noop(&archive, 0, OVERSIZED_ALLOCATION),
            KPACK_ERROR_INVALID_FORMAT);
  EXPECT_TRUE(archive.kernel_cache.empty());
}

TEST(CompressionTest, ZstdRejectsOversizedOutputBeforeAllocation) {
  kpack_archive archive{};
  archive.zstd_frames.push_back({0, 0});

  EXPECT_EQ(kpack::decompress_zstd(&archive, 0, OVERSIZED_ALLOCATION),
            KPACK_ERROR_INVALID_FORMAT);
  EXPECT_TRUE(archive.kernel_cache.empty());
}

}  // namespace
