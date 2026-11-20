// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amd::smi::DiscoverGpuCpuLinks() (rocm_smi_cpu_link.cc), the
// backing layer for amdsmi_get_gpu_cpu_links(). Exercises it directly against
// a mock KFD topology via the KfdNodesPathRoot test hook, the same hook used
// by kfd_numa_node_test.cc in this directory -- no GPU or node_handle/board
// fabrication involved, so there is no WSLGPUBackend short-circuit to guard
// against here (DiscoverGpuCpuLinks and its callees have no
// ENABLE_WSL_BACKEND branch; WSLGPUBackend only intercepts the amd_smi.cc
// public-API layer, which this test does not go through).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_cpu_link.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "test_fixture_utils.h"

namespace {

// MI450-like topology: GPU node 1 has a direct XGMI io_link to CPU node 0.
TEST(SystemUnit, DiscoverGpuCpuLinksFindsDirectXgmiCpuLink) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/0, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_XGMI);
}

// GPU node 1's only io_link neighbor is peer GPU node 2 (not a CPU) -- no CPU
// link exists, which must be reported as an empty, successful result.
TEST(SystemUnit, DiscoverGpuCpuLinksEmptyWhenOnlyPeerGpuNeighbor) {
  FakeSysfsTree tree;
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);
  EXPECT_TRUE(links.empty());
}

// GPU node 1 has both a peer-GPU XGMI neighbor and a direct CPU PCIe
// neighbor -- only the CPU link must be reported.
TEST(SystemUnit, DiscoverGpuCpuLinksIgnoresNonCpuNeighbors) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/1, /*node_to=*/0, amd::smi::IOLINK_TYPE_PCIEXPRESS,
              /*weight=*/20);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

}  // namespace
