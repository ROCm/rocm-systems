// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// kInvalidNumaNode is the sentinel KFDNode::Initialize() assigns when no CPU
// io_link is found, and the value topo_get_numa_node_number() compares against
// to report AMDSMI_STATUS_NOT_SUPPORTED. Keeping both sides pinned to the same
// named constant (rather than a magic UINT32_MAX/-1 literal in each place)
// stops them from silently drifting apart. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>

#include "rocm_smi/rocm_smi_kfd.h"

namespace {

TEST(SystemUnit, InvalidNumaNodeSentinelIsUint32Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNode, UINT32_MAX);
}

TEST(SystemUnit, InvalidNumaNodeWeightSentinelIsUint64Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNodeWeight, UINT64_MAX);
}

}  // namespace
