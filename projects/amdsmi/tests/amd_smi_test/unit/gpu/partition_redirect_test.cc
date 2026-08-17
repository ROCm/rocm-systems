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

// Hardware-free unit tests for the partition-redirect decision: which sibling
// partition a partition query must be sent to, since only the primary
// (partition_id == 0) answers on MI300-class multi-partition devices.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace amd::smi {

// Defined in src/amd_smi/amd_smi.cc. The symbol is hidden by the shared library's
// version script, so the test resolves it through the static library
// (BUILD_BOTH_LIBS, enabled with the test suite).
int primary_partition_redirect_index(const std::vector<uint32_t>& partition_ids, size_t self_index);

}  // namespace amd::smi

namespace {

constexpr uint32_t kUnqueryable = std::numeric_limits<uint32_t>::max();

}  // namespace

TEST(GpuUnit, PartitionRedirectSinglePartitionIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0}, 0), -1);
}

TEST(GpuUnit, PartitionRedirectEmptyDeviceIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({}, 0), -1);
}

TEST(GpuUnit, PartitionRedirectPrimaryPartitionIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 0), -1);
}

TEST(GpuUnit, PartitionRedirectSubPartitionRedirectsToPrimary) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 3), 0);
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 1, 2, 3}, 1), 0);
}

TEST(GpuUnit, PartitionRedirectPrimaryFoundAtNonZeroIndex) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({5, 0, 3}, 2), 1);
}

TEST(GpuUnit, PartitionRedirectNoPrimaryPresentIsNotRedirected) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({1, 2, 3}, 0), -1);
}

TEST(GpuUnit, PartitionRedirectUnqueryableSiblingsAreSkipped) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({kUnqueryable, 0, kUnqueryable}, 0), 1);
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({kUnqueryable, kUnqueryable}, 0), -1);
}

TEST(GpuUnit, PartitionRedirectSelfIsSkippedWhenLocatingPrimary) {
  EXPECT_EQ(amd::smi::primary_partition_redirect_index({0, 5}, 0), -1);
}
