/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Hardware-free unit tests for the partition-redirect decision.
//
// On MI300-class accelerators split into multiple logical partitions (for
// example CPX + NPS4) the compute-/memory-partition sysfs nodes only respond on
// the primary partition (partition_id == 0). Queries against a logical
// sub-partition handle must be redirected to that primary, otherwise the tool
// reports "N/A". amdsmi.cc factors the redirect decision into the pure helper
// exercised below so it can be verified without any GPU present.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace amd::smi {

// Forward declaration of the internal helper under test (defined in
// src/amd_smi/amd_smi.cc).
int primary_partition_redirect_index(const std::vector<uint32_t>& partition_ids, size_t self_index);

}  // namespace amd::smi

namespace {

constexpr uint32_t kUnqueryable = std::numeric_limits<uint32_t>::max();

}  // namespace

// A device that exposes a single logical GPU has no separate primary partition,
// so nothing is redirected.
TEST(AmdSmiPartitionRedirectTest, SinglePartitionIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0}, 0), -1);
}

// A device with no partitions is likewise never redirected.
TEST(AmdSmiPartitionRedirectTest, EmptyDeviceIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({}, 0), -1);
}

// The primary partition already answers partition queries directly, so it is
// not redirected to itself.
TEST(AmdSmiPartitionRedirectTest, PrimaryPartitionIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 0), -1);
}

// A logical sub-partition (partition_id > 0) is redirected to the primary
// sibling's index.
TEST(AmdSmiPartitionRedirectTest, SubPartitionRedirectsToPrimary) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 3), 0);
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 1), 0);
}

// The primary need not be the first entry; its actual index is returned.
TEST(AmdSmiPartitionRedirectTest, PrimaryFoundAtNonZeroIndex) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({5, 0, 3}, 2), 1);
}

// When no sibling reports partition_id == 0 there is nothing to redirect to.
TEST(AmdSmiPartitionRedirectTest, NoPrimaryPresentIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({1, 2, 3}, 0), -1);
}

// Siblings whose partition id could not be read (UINT32_MAX) are skipped and do
// not shadow the real primary.
TEST(AmdSmiPartitionRedirectTest, UnqueryableSiblingsAreSkipped) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({kUnqueryable, 0, kUnqueryable}, 0), 1);
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({kUnqueryable, kUnqueryable}, 0), -1);
}

// The querying partition is always skipped, so a device whose only primary is
// the caller itself is not redirected.
TEST(AmdSmiPartitionRedirectTest, SelfIsSkippedWhenLocatingPrimary) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 5}, 0), -1);
}
