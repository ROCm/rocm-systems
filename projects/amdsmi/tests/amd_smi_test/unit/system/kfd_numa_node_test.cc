// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// kInvalidNumaNode/kInvalidNumaNodeWeight are the sentinels KFDNode::Initialize()
// assigns when no CPU io_link is found. Runs the real rsmi_init() discovery over a fake KFD
// topology and DRM tree (via the KfdNodesPathRoot and SysfsDrmRoot test hooks) and reads the result
// back through the public API, to prove the getters report no NUMA node when no CPU link exists,
// and the discovered values when one does. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "test_common.h"
#include "test_fixture_utils.h"

namespace {

amdsmi_status_t CallNumaNodeNumber(const FakeKfdTopology& topology, uint32_t node,
                                   uint32_t* numa_node, amdsmi_status_t expected) {
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_numa_node_number", "node=" + std::to_string(node), verbose);
  amdsmi_status_t status = amdsmi_topo_get_numa_node_number(topology.handle(node), numa_node);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, expected);
  return status;
}

amdsmi_status_t CallLinkWeight(const FakeKfdTopology& topology, uint32_t src_node,
                               uint32_t dst_node, uint64_t* weight, amdsmi_status_t expected) {
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API(
      "amdsmi_topo_get_link_weight",
      "src_node=" + std::to_string(src_node) + ", dst_node=" + std::to_string(dst_node), verbose);
  amdsmi_status_t status =
      amdsmi_topo_get_link_weight(topology.handle(src_node), topology.handle(dst_node), weight);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, expected);
  return status;
}

amdsmi_status_t CallKfdInfo(const FakeKfdTopology& topology, uint32_t node,
                            amdsmi_kfd_info_t* info) {
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_kfd_info", "node=" + std::to_string(node), verbose);
  amdsmi_status_t status = amdsmi_get_gpu_kfd_info(topology.handle(node), info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS);
  return status;
}

TEST(SystemUnit, KFD_TOPOLOGY__InvalidNumaNodeSentinelIsUint32Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNode, UINT32_MAX);
}

TEST(SystemUnit, KFD_TOPOLOGY__InvalidNumaNodeWeightSentinelIsUint64Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNodeWeight, UINT64_MAX);
}

// Node A has an XGMI io_link to node B and no link to a CPU node; node B has no links at all.
TEST(SystemUnit, KFD_TOPOLOGY__KfdNodeInitializeSentinelsWhenNoCpuIoLink) {
  constexpr uint32_t kNodeA = 0;
  constexpr uint32_t kNodeB = 1;
  constexpr uint32_t kCardA = 1;
  constexpr uint32_t kCardB = 0;
  constexpr uint32_t kLinkIndex = 0;
  constexpr uint64_t kGpuIdA = 12345;
  constexpr uint64_t kGpuIdB = 999;
  constexpr uint64_t kXgmiWeight = 15;

  FakeSysfsTree tree;
  WriteGpuNode(tree, kNodeA, kCardA, kGpuIdA);
  WriteIoLink(tree, kNodeA, kLinkIndex, kNodeB, amd::smi::IOLINK_TYPE_XGMI, kXgmiWeight);
  WriteGpuNode(tree, kNodeB, kCardB, kGpuIdB);
  FakeKfdTopology topology(tree);

  uint32_t numa_node = 0;
  EXPECT_EQ(CallNumaNodeNumber(topology, kNodeA, &numa_node, AMDSMI_STATUS_NOT_SUPPORTED),
            AMDSMI_STATUS_NOT_SUPPORTED);

  // B has no direct link to A and no CPU link to route through, so there is no weight to report.
  uint64_t weight = 0;
  EXPECT_EQ(CallLinkWeight(topology, kNodeB, kNodeA, &weight, AMDSMI_STATUS_NOT_SUPPORTED),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

// Nodes A and B each have a PCIe io_link to the same CPU node and no link to each other.
// The NUMA node must be the CPU node, and a weight between the GPUs is the sum of their CPU links.
TEST(SystemUnit, KFD_TOPOLOGY__KfdNodeInitializeFindsCpuIoLink) {
  constexpr uint32_t kNodeA = 2;
  constexpr uint32_t kNodeB = 3;
  constexpr uint32_t kCardA = 0;
  constexpr uint32_t kCardB = 1;
  constexpr uint32_t kCpuNode = 4;
  constexpr uint32_t kLinkIndex = 0;
  constexpr uint64_t kGpuIdA = 23456;
  constexpr uint64_t kGpuIdB = 34567;
  constexpr uint64_t kPcieWeightA = 20;
  constexpr uint64_t kPcieWeightB = 30;

  FakeSysfsTree tree;
  WriteGpuNode(tree, kNodeA, kCardA, kGpuIdA);
  WriteIoLink(tree, kNodeA, kLinkIndex, kCpuNode, amd::smi::IOLINK_TYPE_PCIEXPRESS, kPcieWeightA);
  WriteGpuNode(tree, kNodeB, kCardB, kGpuIdB);
  WriteIoLink(tree, kNodeB, kLinkIndex, kCpuNode, amd::smi::IOLINK_TYPE_PCIEXPRESS, kPcieWeightB);
  WriteCpuNode(tree, kCpuNode);
  FakeKfdTopology topology(tree);

  uint32_t numa_node = 0;
  EXPECT_EQ(CallNumaNodeNumber(topology, kNodeA, &numa_node, AMDSMI_STATUS_SUCCESS),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(numa_node, kCpuNode);

  uint64_t weight = 0;
  EXPECT_EQ(CallLinkWeight(topology, kNodeA, kNodeB, &weight, AMDSMI_STATUS_SUCCESS),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(weight, kPcieWeightA + kPcieWeightB);
}

// The KFD gpu_id and node index come from the overridden topology root.
TEST(SystemUnit, KFD_TOPOLOGY__KfdInfoUsesOverride) {
  constexpr uint32_t kNodeIndex = 5;
  constexpr uint32_t kCardIndex = 2;
  constexpr uint64_t kGpuId = 4242;

  FakeSysfsTree tree;
  WriteGpuNode(tree, kNodeIndex, kCardIndex, kGpuId);
  FakeKfdTopology topology(tree);

  amdsmi_kfd_info_t info{};
  ASSERT_EQ(CallKfdInfo(topology, kNodeIndex, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.kfd_id, kGpuId);
  EXPECT_EQ(info.node_id, kNodeIndex);
}

// amdsmi_get_pcie_info() relies on this type to tell an xGMI host link from PCIe.
TEST(SystemUnit, KfdNodeInitializeFindsXgmiCpuIoLink) {
  constexpr uint32_t kGpuNode = 1;
  constexpr uint32_t kGpuCard = 0;
  constexpr uint32_t kCpuNode = 0;
  constexpr uint32_t kLinkIndex = 0;
  constexpr uint64_t kGpuId = 23456;
  constexpr uint64_t kXgmiWeight = 15;

  FakeSysfsTree tree;
  WriteGpuNode(tree, kGpuNode, kGpuCard, kGpuId);
  WriteIoLink(tree, kGpuNode, kLinkIndex, kCpuNode, amd::smi::IOLINK_TYPE_XGMI, kXgmiWeight);
  WriteCpuNode(tree, kCpuNode);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  amd::smi::KFDNode node(kGpuNode);
  ASSERT_EQ(node.Initialize(), 0);

  EXPECT_EQ(node.numa_node_number(), kCpuNode);
  EXPECT_EQ(node.numa_node_weight(), kXgmiWeight);
  EXPECT_EQ(node.numa_node_type(), amd::smi::IOLINK_TYPE_XGMI);
}

// A link whose properties cannot be read names node 0 with weight 0, so if discovery kept it,
// it would outrank the real CPU link on node 1.
TEST(SystemUnit, KFD_TOPOLOGY__KfdNodeInitializeIgnoresUnreadableIoLink) {
  constexpr uint32_t kOtherCpuNode = 0;
  constexpr uint32_t kCpuNode = 1;
  constexpr uint32_t kGpuNode = 2;
  constexpr uint32_t kGpuCard = 0;
  constexpr uint32_t kLinkIndex = 0;
  constexpr uint32_t kUnreadableLinkIndex = 1;
  constexpr uint64_t kGpuId = 23456;
  constexpr uint64_t kPcieWeight = 20;

  FakeSysfsTree tree;
  WriteGpuNode(tree, kGpuNode, kGpuCard, kGpuId);
  WriteIoLink(tree, kGpuNode, kLinkIndex, kCpuNode, amd::smi::IOLINK_TYPE_PCIEXPRESS, kPcieWeight);
  tree.WriteFile(
      std::to_string(kGpuNode) + "/io_links/" + std::to_string(kUnreadableLinkIndex) + "/.keep",
      "");
  WriteCpuNode(tree, kOtherCpuNode);
  WriteCpuNode(tree, kCpuNode);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  amd::smi::KFDNode node(kGpuNode);
  ASSERT_EQ(node.Initialize(), 0);

  EXPECT_EQ(node.numa_node_number(), kCpuNode);
  EXPECT_EQ(node.numa_node_weight(), kPcieWeight);
  EXPECT_EQ(node.numa_node_type(), amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

// The fields a failed Initialize() never reached must read 0. Allocated on the heap like
// discovery's links, so under ASAN a field left uninitialized reads as malloc fill rather than 0.
TEST(SystemUnit, KFD_TOPOLOGY__IoLinkMissingKeysReadZero) {
  constexpr uint32_t kGpuNode = 1;
  constexpr uint32_t kLinkIndex = 0;

  FakeSysfsTree tree;
  tree.WriteFile(
      std::to_string(kGpuNode) + "/io_links/" + std::to_string(kLinkIndex) + "/properties",
      "type " + std::to_string(amd::smi::IOLINK_TYPE_XGMI) + "\nnode_from " +
          std::to_string(kGpuNode) + "\n");
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  auto link = std::make_shared<amd::smi::IOLink>(kGpuNode, kLinkIndex, amd::smi::IO_LINK_DIRECTORY);
  EXPECT_NE(link->Initialize(), 0);

  EXPECT_EQ(link->type(), amd::smi::IOLINK_TYPE_XGMI);
  EXPECT_EQ(link->node_from(), kGpuNode);
  EXPECT_EQ(link->node_to(), 0u);
  EXPECT_EQ(link->flag(), 0u);
  EXPECT_EQ(link->weight(), 0u);
  EXPECT_EQ(link->min_bandwidth(), 0u);
  EXPECT_EQ(link->max_bandwidth(), 0u);
}

}  // namespace
