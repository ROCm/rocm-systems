/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <cstdlib>
#include "topology_gtest.hpp"
#include "gda/nic_policy.hpp"

using namespace rocshmem;

// Test DeviceType helper functions
TEST_F(DeviceTypeTestFixture, IsCpuExeType) {
  EXPECT_TRUE(IsCpuExeType(EXE_CPU));
  EXPECT_FALSE(IsCpuExeType(EXE_GPU));
  EXPECT_FALSE(IsCpuExeType(EXE_NIC));
}

TEST_F(DeviceTypeTestFixture, IsGpuExeType) {
  EXPECT_FALSE(IsGpuExeType(EXE_CPU));
  EXPECT_TRUE(IsGpuExeType(EXE_GPU));
  EXPECT_FALSE(IsGpuExeType(EXE_NIC));
}

TEST_F(DeviceTypeTestFixture, IsNicExeType) {
  EXPECT_FALSE(IsNicExeType(EXE_CPU));
  EXPECT_FALSE(IsNicExeType(EXE_GPU));
  EXPECT_TRUE(IsNicExeType(EXE_NIC));
}

// Test MemType helper functions
TEST_F(DeviceTypeTestFixture, IsCpuMemType) {
  EXPECT_TRUE(IsCpuMemType(MEM_CPU));
  EXPECT_FALSE(IsCpuMemType(MEM_GPU));
}

TEST_F(DeviceTypeTestFixture, IsGpuMemType) {
  EXPECT_FALSE(IsGpuMemType(MEM_CPU));
  EXPECT_TRUE(IsGpuMemType(MEM_GPU));
}

// Test ExeDevice struct
TEST_F(DeviceTypeTestFixture, ExeDeviceComparison) {
  ExeDevice cpu0 = {EXE_CPU, 0};
  ExeDevice cpu1 = {EXE_CPU, 1};
  ExeDevice gpu0 = {EXE_GPU, 0};

  // Test less-than operator
  EXPECT_TRUE(cpu0 < cpu1);   // Same type, different index
  EXPECT_TRUE(cpu0 < gpu0);   // Different type (CPU < GPU)
  EXPECT_FALSE(cpu1 < cpu0);  // Reverse comparison
}

TEST_F(DeviceTypeTestFixture, ExeDeviceEquality) {
  ExeDevice cpu0_a = {EXE_CPU, 0};
  ExeDevice cpu0_b = {EXE_CPU, 0};

  // Two identical devices should not be less than each other
  EXPECT_FALSE(cpu0_a < cpu0_b);
  EXPECT_FALSE(cpu0_b < cpu0_a);
}

// Test MemDevice struct
TEST_F(DeviceTypeTestFixture, MemDeviceComparison) {
  MemDevice cpuMem0 = {MEM_CPU, 0};
  MemDevice cpuMem1 = {MEM_CPU, 1};
  MemDevice gpuMem0 = {MEM_GPU, 0};

  // Test less-than operator
  EXPECT_TRUE(cpuMem0 < cpuMem1);   // Same type, different index
  EXPECT_TRUE(cpuMem0 < gpuMem0);   // Different type (MEM_CPU < MEM_GPU)
  EXPECT_FALSE(cpuMem1 < cpuMem0);  // Reverse comparison
}

TEST_F(DeviceTypeTestFixture, MemDeviceEquality) {
  MemDevice cpuMem0_a = {MEM_CPU, 0};
  MemDevice cpuMem0_b = {MEM_CPU, 0};

  // Two identical devices should not be less than each other
  EXPECT_FALSE(cpuMem0_a < cpuMem0_b);
  EXPECT_FALSE(cpuMem0_b < cpuMem0_a);
}

// Test GetNumDevices function
TEST_F(TopologyTestFixture, GetNumDevicesCpu) {
  int numCpus = GetNumDevices(EXE_CPU);
  // Should have at least 1 NUMA node
  EXPECT_GT(numCpus, 0);
}

TEST_F(TopologyTestFixture, GetNumDevicesGpu) {
  int numGpus = GetNumDevices(EXE_GPU);
  // Number should be non-negative
  EXPECT_GE(numGpus, 0);
  // Should match HIP device count
  EXPECT_EQ(numGpus, num_gpus_);
}

TEST_F(TopologyTestFixture, GetNumDevicesNic) {
  int numNics = GetNumDevices(EXE_NIC);
  // Number should be non-negative
  EXPECT_GE(numNics, 0);
}

// Test GetClosestCpuNumaToGpu function
TEST_F(TopologyTestFixture, GetClosestCpuNumaToGpuInvalidIndex) {
  // Test with invalid GPU index (negative)
  int result = GetClosestCpuNumaToGpu(-1);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestCpuNumaToGpuTooLarge) {
  int numGpus = GetNumDevices(EXE_GPU);
  // Test with GPU index >= number of GPUs
  int result = GetClosestCpuNumaToGpu(numGpus + 10);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestCpuNumaToGpuValid) {
  int numGpus = GetNumDevices(EXE_GPU);
  if (numGpus > 0) {
    // Test with valid GPU index (0)
    int result = GetClosestCpuNumaToGpu(0);
    // Should return a valid NUMA node index or -1 if detection failed
    int numCpus = GetNumDevices(EXE_CPU);
    if (result >= 0) {
      EXPECT_LT(result, numCpus);
    }
  }
}

// Test GetClosestCpuNumaToNic function
TEST_F(TopologyTestFixture, GetClosestCpuNumaToNicInvalidIndex) {
  // Test with invalid NIC index (negative)
  int result = GetClosestCpuNumaToNic(-1);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestCpuNumaToNicTooLarge) {
  int numNics = GetNumDevices(EXE_NIC);
  // Test with NIC index >= number of NICs
  int result = GetClosestCpuNumaToNic(numNics + 10);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestCpuNumaToNicValid) {
  int numNics = GetNumDevices(EXE_NIC);
  if (numNics > 0) {
    // Test with valid NIC index (0)
    int result = GetClosestCpuNumaToNic(0);
    // Should return a valid NUMA node index or -1 if not set
    int numCpus = GetNumDevices(EXE_CPU);
    if (result >= 0) {
      EXPECT_LT(result, numCpus);
    }
  }
}

// Test GetClosestNicToGpu function
TEST_F(TopologyTestFixture, GetClosestNicToGpuInvalidIndex) {
  // Test with invalid GPU index (negative)
  int result = GetClosestNicToGpu(-1, nullptr, nullptr);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestNicToGpuTooLarge) {
  int numGpus = GetNumDevices(EXE_GPU);
  // Test with GPU index >= number of GPUs
  int result = GetClosestNicToGpu(numGpus + 10, nullptr, nullptr);
  EXPECT_EQ(result, -1);
}

TEST_F(TopologyTestFixture, GetClosestNicToGpuValid) {
  int numGpus = GetNumDevices(EXE_GPU);
  int numNics = GetNumDevices(EXE_NIC);

  if (numGpus > 0 && numNics > 0) {
    std::string devName;
    int result = GetClosestNicToGpu(0, nullptr, &devName);
    if (result >= 0) {
      EXPECT_LT(result, numNics);
      EXPECT_FALSE(devName.empty());
    }
  }
}

TEST_F(TopologyTestFixture, GetClosestNicToGpuWithHcaList) {
  int numGpus = GetNumDevices(EXE_GPU);
  int numNics = GetNumDevices(EXE_NIC);

  if (numGpus > 0 && numNics > 0) {
    // Test with an exclude list (^mlx5_0 excludes mlx5_0)
    const char* excludeList = "^mlx5_0";
    int result = GetClosestNicToGpu(0, excludeList, nullptr);
    // Should return valid index or -1
    EXPECT_GE(result, -1);
    if (result >= 0) {
      EXPECT_LT(result, numNics);
    }
  }
}

TEST_F(TopologyTestFixture, GetClosestNicToGpuConsistency) {
  int numGpus = GetNumDevices(EXE_GPU);
  int numNics = GetNumDevices(EXE_NIC);

  if (numGpus > 0 && numNics > 0) {
    // Call multiple times to verify consistency (static caching)
    int result1 = GetClosestNicToGpu(0, nullptr, nullptr);
    int result2 = GetClosestNicToGpu(0, nullptr, nullptr);
    EXPECT_EQ(result1, result2);
  }
}

// Test DisplayTopology function (just verify it doesn't crash)
TEST_F(TopologyTestFixture, DisplayTopologyNoCrash) {
  // This test just verifies the function can be called without crashing
  // Output is redirected to stdout, so we can't easily test it
  testing::internal::CaptureStdout();
  DisplayTopology(false);
  std::string output = testing::internal::GetCapturedStdout();
  // Should contain some expected text
  EXPECT_TRUE(output.find("Detected Topology") != std::string::npos ||
              output.find("NumCpus") != std::string::npos);
}

TEST_F(TopologyTestFixture, DisplayTopologyCSVFormat) {
  testing::internal::CaptureStdout();
  DisplayTopology(true);
  std::string output = testing::internal::GetCapturedStdout();
  // CSV format should contain NumCpus, NumGpus, NumNics
  EXPECT_TRUE(output.find("NumCpus,") != std::string::npos);
  EXPECT_TRUE(output.find("NumGpus,") != std::string::npos);
  EXPECT_TRUE(output.find("NumNics,") != std::string::npos);
}

// Test GidPriority enum ordering
TEST_F(DeviceTypeTestFixture, GidPriorityOrdering) {
  // Verify the priority ordering
  EXPECT_LT(static_cast<int>(GidPriority::UNKNOWN),
            static_cast<int>(GidPriority::ROCEV1_LINK_LOCAL));
  EXPECT_LT(static_cast<int>(GidPriority::ROCEV1_LINK_LOCAL),
            static_cast<int>(GidPriority::ROCEV2_LINK_LOCAL));
  EXPECT_LT(static_cast<int>(GidPriority::ROCEV2_LINK_LOCAL),
            static_cast<int>(GidPriority::ROCEV1_IPV6));
  EXPECT_LT(static_cast<int>(GidPriority::ROCEV1_IPV6),
            static_cast<int>(GidPriority::ROCEV2_IPV6));
  EXPECT_LT(static_cast<int>(GidPriority::ROCEV2_IPV6),
            static_cast<int>(GidPriority::ROCEV1_IPV4));
  EXPECT_LT(static_cast<int>(GidPriority::ROCEV1_IPV4),
            static_cast<int>(GidPriority::ROCEV2_IPV4));
}

// Test multiple GPUs if available
TEST_F(TopologyTestFixture, MultipleGpuTopology) {
  int numGpus = GetNumDevices(EXE_GPU);

  if (numGpus > 1) {
    // Test that each GPU can find a closest NUMA node
    for (int i = 0; i < numGpus; i++) {
      int numaNode = GetClosestCpuNumaToGpu(i);
      // Should be valid or -1
      int numCpus = GetNumDevices(EXE_CPU);
      if (numaNode >= 0) {
        EXPECT_LT(numaNode, numCpus);
      }
    }
  }
}

//==============================================================================
// PCIe Tree Tests with Theoretical Topology
//==============================================================================

// Test ExtractBusNumber function
TEST_F(PCIeTreeTestFixture, ExtractBusNumberValid) {
  // Test valid PCIe addresses
  EXPECT_EQ(ExtractBusNumber("0000:02:00.0"), 0x02);
  EXPECT_EQ(ExtractBusNumber("0000:05:00.0"), 0x05);
  EXPECT_EQ(ExtractBusNumber("0000:42:00.0"), 0x42);
  EXPECT_EQ(ExtractBusNumber("0000:45:00.0"), 0x45);
  EXPECT_EQ(ExtractBusNumber("0000:ff:1f.7"), 0xff);
}

TEST_F(PCIeTreeTestFixture, ExtractBusNumberInvalid) {
  // Test invalid PCIe addresses
  EXPECT_EQ(ExtractBusNumber("invalid"), -1);
  EXPECT_EQ(ExtractBusNumber(""), -1);
  EXPECT_EQ(ExtractBusNumber("0000:GG:00.0"), -1);
}

// Test GetBusIdDistance function
TEST_F(PCIeTreeTestFixture, GetBusIdDistanceSameSocket) {
  // GPU 0 and NIC 0 are on the same switch (bus 0x02 and 0x03)
  int distance = GetBusIdDistance(
    gpu_addresses_[0], nic_addresses_[0]);
  EXPECT_EQ(distance, 1);  // |0x02 - 0x03| = 1
}

TEST_F(PCIeTreeTestFixture, GetBusIdDistanceDifferentSocket) {
  // GPU 0 (bus 0x02) and GPU 2 (bus 0x42)
  int distance = GetBusIdDistance(
    gpu_addresses_[0], gpu_addresses_[2]);
  EXPECT_EQ(distance, 0x40);  // |0x02 - 0x42| = 0x40
}

TEST_F(PCIeTreeTestFixture, GetBusIdDistanceZero) {
  // Same address
  int distance = GetBusIdDistance(
    gpu_addresses_[0], gpu_addresses_[0]);
  EXPECT_EQ(distance, 0);
}

TEST_F(PCIeTreeTestFixture, GetBusIdDistanceInvalid) {
  // Invalid address
  int distance = GetBusIdDistance(
    "invalid", gpu_addresses_[0]);
  EXPECT_EQ(distance, -1);
}

// Test GetLcaBetweenNodes function
TEST_F(PCIeTreeTestFixture, GetLcaSameSwitch) {
  // GPU 0 and NIC 0 are on the same PCIe switch
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], nic_addresses_[0]);
  ASSERT_NE(lca, nullptr);
  // LCA should be the PCIe switch (0000:01:00.0)
  EXPECT_EQ(lca->address, "0000:01:00.0");
}

TEST_F(PCIeTreeTestFixture, GetLcaSameSocket) {
  // GPU 0 (switch 0) and GPU 1 (switch 1) are on the same socket
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], gpu_addresses_[1]);
  ASSERT_NE(lca, nullptr);
  // LCA should be the CPU socket (0000:00:01.0)
  EXPECT_EQ(lca->address, "0000:00:01.0");
}

TEST_F(PCIeTreeTestFixture, GetLcaDifferentSocket) {
  // GPU 0 (socket 0) and GPU 2 (socket 1) are on different sockets
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], gpu_addresses_[2]);
  ASSERT_NE(lca, nullptr);
  // LCA should be the root
  EXPECT_EQ(lca->address, "root");
}

TEST_F(PCIeTreeTestFixture, GetLcaSameNode) {
  // Same node - should return the node itself
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], gpu_addresses_[0]);
  ASSERT_NE(lca, nullptr);
  EXPECT_EQ(lca->address, gpu_addresses_[0]);
}

TEST_F(PCIeTreeTestFixture, GetLcaNotFound) {
  // Non-existent node
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, "0000:99:00.0", gpu_addresses_[0]);
  EXPECT_EQ(lca, nullptr);
}

// Test GetLcaDepth function
TEST_F(PCIeTreeTestFixture, GetLcaDepthRoot) {
  int depth = GetLcaDepth("root", &root_);
  EXPECT_EQ(depth, 0);
}

TEST_F(PCIeTreeTestFixture, GetLcaDepthSocket) {
  int depth = GetLcaDepth("0000:00:01.0", &root_);
  EXPECT_EQ(depth, 1);
}

TEST_F(PCIeTreeTestFixture, GetLcaDepthSwitch) {
  int depth = GetLcaDepth("0000:01:00.0", &root_);
  EXPECT_EQ(depth, 2);
}

TEST_F(PCIeTreeTestFixture, GetLcaDepthDevice) {
  int depth = GetLcaDepth(gpu_addresses_[0], &root_);
  EXPECT_EQ(depth, 3);
}

TEST_F(PCIeTreeTestFixture, GetLcaDepthNotFound) {
  int depth = GetLcaDepth("0000:99:00.0", &root_);
  EXPECT_EQ(depth, -1);
}

// Test GetNearestDevicesInTree function
TEST_F(PCIeTreeTestFixture, GetNearestDevicesSameSwitch) {
  // GPU 0 should be closest to NIC 0 (same switch)
  std::set<int> nearest = GetNearestDevicesInTree(
    gpu_addresses_[0], nic_addresses_, &root_);

  ASSERT_EQ(nearest.size(), 1);
  EXPECT_EQ(*nearest.begin(), 0);  // NIC 0
}

TEST_F(PCIeTreeTestFixture, GetNearestDevicesSameSocket) {
  // GPU 0 should prefer NICs on the same socket
  // When NIC 0 is excluded, should prefer NIC 1 (same socket) over NIC 2/3 (different socket)
  std::vector<std::string> nicsExcludingFirst = {
    "",                // NIC 0 excluded
    nic_addresses_[1], // NIC 1 (same socket)
    nic_addresses_[2], // NIC 2 (different socket)
    nic_addresses_[3]  // NIC 3 (different socket)
  };

  std::set<int> nearest = GetNearestDevicesInTree(
    gpu_addresses_[0], nicsExcludingFirst, &root_);

  ASSERT_EQ(nearest.size(), 1);
  EXPECT_EQ(*nearest.begin(), 1);  // NIC 1
}

TEST_F(PCIeTreeTestFixture, GetNearestDevicesCrossSocket) {
  // GPU 2 (socket 1) should be closest to NIC 2 or NIC 3 (both on socket 1)
  std::set<int> nearest = GetNearestDevicesInTree(
    gpu_addresses_[2], nic_addresses_, &root_);

  ASSERT_EQ(nearest.size(), 1);
  EXPECT_EQ(*nearest.begin(), 2);  // NIC 2 (same switch as GPU 2)
}

TEST_F(PCIeTreeTestFixture, GetNearestDevicesMultipleMatches) {
  // Create scenario with equidistant devices
  // GPU 0 with NICs from different sockets only
  std::vector<std::string> crossSocketNics = {
    "",                // Exclude socket 0 NICs
    "",
    nic_addresses_[2], // NIC 2 (socket 1, switch 2)
    nic_addresses_[3]  // NIC 3 (socket 1, switch 3)
  };

  std::set<int> nearest = GetNearestDevicesInTree(
    gpu_addresses_[0], crossSocketNics, &root_);

  // Should match both NIC 2 and NIC 3 with equal priority
  EXPECT_GE(nearest.size(), 1);
  // Both are equally far (different socket)
}

TEST_F(PCIeTreeTestFixture, GetNearestDevicesEmptyList) {
  std::vector<std::string> emptyList = {"", "", "", ""};

  std::set<int> nearest = GetNearestDevicesInTree(
    gpu_addresses_[0], emptyList, &root_);

  EXPECT_EQ(nearest.size(), 0);
}

// Test PCIe tree structure verification
TEST_F(PCIeTreeTestFixture, TreeStructureHasRoot) {
  EXPECT_EQ(root_.address, "root");
  EXPECT_EQ(root_.description, "PCIe Root Complex");
  EXPECT_GT(root_.children.size(), 0);
}

TEST_F(PCIeTreeTestFixture, TreeStructureHasTwoSockets) {
  // Should have 2 CPU sockets
  EXPECT_EQ(root_.children.size(), 2);
}

TEST_F(PCIeTreeTestFixture, TreeStructureSocket0) {
  // Find socket 0
  bool found = false;
  for (auto const& socket : root_.children) {
    if (socket.address == "0000:00:01.0") {
      found = true;
      // Socket 0 should have 2 switches
      EXPECT_EQ(socket.children.size(), 2);
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(PCIeTreeTestFixture, TreeStructureSocket1) {
  // Find socket 1
  bool found = false;
  for (auto const& socket : root_.children) {
    if (socket.address == "0000:40:00.0") {
      found = true;
      // Socket 1 should have 2 switches
      EXPECT_EQ(socket.children.size(), 2);
    }
  }
  EXPECT_TRUE(found);
}

// Test all GPU-NIC affinity mappings in theoretical topology
TEST_F(PCIeTreeTestFixture, GpuNicAffinityMapping) {
  // Expected optimal NIC for each GPU
  // GPU 0 -> NIC 0 (same switch)
  // GPU 1 -> NIC 1 (same switch)
  // GPU 2 -> NIC 2 (same switch)
  // GPU 3 -> NIC 3 (same switch)

  for (size_t gpu_idx = 0; gpu_idx < gpu_addresses_.size(); gpu_idx++) {
    std::set<int> nearest = GetNearestDevicesInTree(
      gpu_addresses_[gpu_idx], nic_addresses_, &root_);

    ASSERT_EQ(nearest.size(), 1) << "GPU " << gpu_idx;
    EXPECT_EQ(*nearest.begin(), gpu_idx) << "GPU " << gpu_idx;
  }
}

// Test bus distance calculations for all pairs
TEST_F(PCIeTreeTestFixture, BusDistanceMatrix) {
  // Test distances between all GPUs
  std::vector<std::vector<int>> expected_distances = {
    {0,  3,  0x40, 0x43},  // GPU 0 to GPU 0,1,2,3
    {3,  0,  0x3D, 0x40},  // GPU 1 to GPU 0,1,2,3
    {0x40, 0x3D, 0, 3},    // GPU 2 to GPU 0,1,2,3
    {0x43, 0x40, 3, 0}     // GPU 3 to GPU 0,1,2,3
  };

  for (size_t i = 0; i < gpu_addresses_.size(); i++) {
    for (size_t j = 0; j < gpu_addresses_.size(); j++) {
      int distance = GetBusIdDistance(
        gpu_addresses_[i], gpu_addresses_[j]);
      EXPECT_EQ(distance, expected_distances[i][j])
        << "Distance between GPU " << i << " and GPU " << j;
    }
  }
}

// Test LCA depth verification for known pairs
TEST_F(PCIeTreeTestFixture, LcaDepthVerification) {
  // Devices on same switch: LCA depth should be 2 (switch level)
  PCIeNode const* lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], nic_addresses_[0]);
  ASSERT_NE(lca, nullptr);
  int depth = GetLcaDepth(lca->address, &root_);
  EXPECT_EQ(depth, 2);  // Switch level

  // Devices on same socket: LCA depth should be 1 (socket level)
  lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], gpu_addresses_[1]);
  ASSERT_NE(lca, nullptr);
  depth = GetLcaDepth(lca->address, &root_);
  EXPECT_EQ(depth, 1);  // Socket level

  // Devices on different sockets: LCA depth should be 0 (root level)
  lca = GetLcaBetweenNodes(
    &root_, gpu_addresses_[0], gpu_addresses_[2]);
  ASSERT_NE(lca, nullptr);
  depth = GetLcaDepth(lca->address, &root_);
  EXPECT_EQ(depth, 0);  // Root level
}

//==============================================================================
// Multi-NIC Topology Tests
//==============================================================================

TEST_F(DeviceTypeTestFixture, NicPathTypeOrdering) {
  EXPECT_LT(NIC_PATH_PIX, NIC_PATH_PXB);
  EXPECT_LT(NIC_PATH_PXB, NIC_PATH_PHB);
  EXPECT_LT(NIC_PATH_PHB, NIC_PATH_SYS);
}

TEST_F(DeviceTypeTestFixture, ParseNicMergeLevelKnown) {
  EXPECT_EQ(ParseNicMergeLevel("PIX"), NIC_PATH_PIX);
  EXPECT_EQ(ParseNicMergeLevel("PXB"), NIC_PATH_PXB);
  EXPECT_EQ(ParseNicMergeLevel("PHB"), NIC_PATH_PHB);
  EXPECT_EQ(ParseNicMergeLevel("SYS"), NIC_PATH_SYS);
}

TEST_F(DeviceTypeTestFixture, ParseNicMergeLevelUnknown) {
  testing::internal::CaptureStderr();
  auto result = ParseNicMergeLevel("INVALID");
  std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(result, NIC_PATH_SYS);
  EXPECT_TRUE(err.find("unknown") != std::string::npos ||
              err.find("defaulting") != std::string::npos);
}

TEST_F(DeviceTypeTestFixture, ParseNicListCommaSeparated) {
  auto result = ParseNicList("rdma0,rdma1,rdma2");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "rdma0");
  EXPECT_EQ(result[1], "rdma1");
  EXPECT_EQ(result[2], "rdma2");
}

TEST_F(DeviceTypeTestFixture, ParseNicListTrimsSpaces) {
  auto result = ParseNicList("  rdma0 , rdma1 , rdma2  ");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "rdma0");
  EXPECT_EQ(result[1], "rdma1");
  EXPECT_EQ(result[2], "rdma2");
}

TEST_F(DeviceTypeTestFixture, ParseNicListSingle) {
  auto result = ParseNicList("rdma0");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "rdma0");
}

TEST_F(DeviceTypeTestFixture, ParseNicListEmpty) {
  auto result = ParseNicList("");
  EXPECT_TRUE(result.empty());
}

TEST_F(DeviceTypeTestFixture, SelectRankGroupRoundRobin) {
  std::string spec = "g0;g1;g2";

  EXPECT_EQ(SelectRankGroup(spec, 0), "g0");
  EXPECT_EQ(SelectRankGroup(spec, 1), "g1");
  EXPECT_EQ(SelectRankGroup(spec, 2), "g2");
  EXPECT_EQ(SelectRankGroup(spec, 3), "g0");
  EXPECT_EQ(SelectRankGroup(spec, 4), "g1");
  EXPECT_EQ(SelectRankGroup(spec, 5), "g2");
}

TEST_F(DeviceTypeTestFixture, SelectRankGroupSingle) {
  EXPECT_EQ(SelectRankGroup("rdma0,rdma1", 0), "rdma0,rdma1");
  EXPECT_EQ(SelectRankGroup("rdma0,rdma1", 1), "rdma0,rdma1");
}

TEST_F(DeviceTypeTestFixture, IbvDeviceNameAtIndex) {
  auto const& devList = GetIbvDeviceList();
  if (!devList.empty()) {
    EXPECT_FALSE(devList[0].name.empty());
  }
}

TEST_F(TopologyTestFixture, BuildFilteredNicAddressesNoFilter) {
  auto const& devList = GetIbvDeviceList();
  if (devList.empty()) GTEST_SKIP() << "No IB devices available";

  auto addrs = BuildFilteredNicAddresses(nullptr);
  EXPECT_EQ(addrs.size(), devList.size());
  int active_count = 0;
  for (size_t i = 0; i < devList.size(); i++) {
    if (devList[i].hasActivePort) {
      EXPECT_FALSE(addrs[i].empty());
      active_count++;
    }
  }
  if (active_count == 0) GTEST_SKIP() << "No active IB ports; skipping test";
}

TEST_F(TopologyTestFixture, BuildFilteredNicAddressesWithExclude) {
  auto const& devList = GetIbvDeviceList();
  if (devList.empty()) GTEST_SKIP() << "No IB devices available";

  std::string excl = "^" + devList[0].name;
  auto addrs = BuildFilteredNicAddresses(excl.c_str());
  EXPECT_TRUE(addrs[0].empty());
}

TEST_F(TopologyTestFixture, BuildFilteredNicAddressesWithInclude) {
  auto const& devList = GetIbvDeviceList();
  if (devList.empty()) GTEST_SKIP() << "No IB devices available";

  std::string incl = devList[0].name;
  auto addrs = BuildFilteredNicAddresses(incl.c_str());
  for (size_t i = 1; i < devList.size(); i++) {
    if (devList[i].name != devList[0].name) {
      EXPECT_TRUE(addrs[i].empty())
          << "NIC " << devList[i].name << " should be excluded by include list";
    }
  }
}

TEST_F(TopologyTestFixture, BuildFilteredNicAddressesWithMultiInclude) {
  auto const& devList = GetIbvDeviceList();
  if (devList.size() < 2) GTEST_SKIP() << "Need at least 2 IB devices";

  std::string incl = devList[0].name + "," + devList[1].name;
  auto addrs = BuildFilteredNicAddresses(incl.c_str());
  for (size_t i = 2; i < devList.size(); i++) {
    if (devList[i].name != devList[0].name && devList[i].name != devList[1].name) {
      EXPECT_TRUE(addrs[i].empty())
          << "NIC " << devList[i].name << " should be excluded by include list";
    }
  }
}

// ---------- ComputeNicIdxForQp tests ----------

TEST(NicPolicyTest, RoundRobin_SingleNic) {
  // 4 PEs, 1 NIC, 2 QP rows per PE -> all map to NIC 0
  for (int qp = 0; qp < 8; qp++) {
    EXPECT_EQ(ComputeNicIdxForQp(qp, 4, 1, 2, 1, NicPolicy::ROUND_ROBIN), 0);
  }
}

TEST(NicPolicyTest, RoundRobin_TwoNics_FourRows) {
  // 2 PEs, 2 NICs, 4 QP rows per PE default, 0 user
  // row 0 -> NIC 0, row 1 -> NIC 1, row 2 -> NIC 0, row 3 -> NIC 1
  int num_pes = 2, num_nics = 2, def_qps = 4, usr_qps = 0;
  for (int row = 0; row < 4; row++) {
    for (int pe = 0; pe < num_pes; pe++) {
      int qp_idx = row * num_pes + pe;
      int expected_nic = row % num_nics;
      EXPECT_EQ(ComputeNicIdxForQp(qp_idx, num_pes, num_nics, def_qps, usr_qps,
                                    NicPolicy::ROUND_ROBIN), expected_nic)
          << "row=" << row << " pe=" << pe;
    }
  }
}

TEST(NicPolicyTest, RoundRobin_FourNics_OneRow) {
  // 2 PEs, 4 NICs, 1 QP row per PE -> all map to NIC 0 (not enough QPs)
  int num_pes = 2, num_nics = 4, def_qps = 1, usr_qps = 1;
  for (int pe = 0; pe < num_pes; pe++) {
    EXPECT_EQ(ComputeNicIdxForQp(pe, num_pes, num_nics, def_qps, usr_qps,
                                  NicPolicy::ROUND_ROBIN), 0);
  }
}

TEST(NicPolicyTest, PerContext_TwoNics_TwoContexts) {
  // 2 PEs, 2 NICs, 2 QP rows default, 2 QP rows per user ctx
  // Layout: [default: 2*2=4 QPs] [usr_ctx_1: 2*2=4 QPs] [usr_ctx_2: 2*2=4 QPs]
  int num_pes = 2, num_nics = 2, def_qps = 2, usr_qps = 2;
  auto P = NicPolicy::PER_CONTEXT;

  // Default ctx (ctx_id=0): QPs 0..3 -> NIC 0
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 0)
        << "qp_idx=" << i << " (default ctx)";
  }
  // User ctx 1 (ctx_id=1): QPs 4..7 -> NIC 1
  for (int i = 4; i < 8; i++) {
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 1)
        << "qp_idx=" << i << " (usr ctx 1)";
  }
  // User ctx 2 (ctx_id=2): QPs 8..11 -> NIC 0 (wraps)
  for (int i = 8; i < 12; i++) {
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 0)
        << "qp_idx=" << i << " (usr ctx 2)";
  }
}

TEST(NicPolicyTest, PerContext_FourNics_FourContexts) {
  // 2 PEs, 4 NICs, 1 QP row default, 1 QP row per user ctx
  // ctx 0 -> NIC 0, ctx 1 -> NIC 1, ctx 2 -> NIC 2, ctx 3 -> NIC 3
  int num_pes = 2, num_nics = 4, def_qps = 1, usr_qps = 1;
  auto P = NicPolicy::PER_CONTEXT;

  // Default (QPs 0..1)
  for (int i = 0; i < 2; i++)
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 0);
  // User ctx 1 (QPs 2..3)
  for (int i = 2; i < 4; i++)
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 1);
  // User ctx 2 (QPs 4..5)
  for (int i = 4; i < 6; i++)
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 2);
  // User ctx 3 (QPs 6..7)
  for (int i = 6; i < 8; i++)
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 3);
}

TEST(NicPolicyTest, PerContext_SingleNic) {
  // 2 PEs, 1 NIC -> all contexts map to NIC 0
  int num_pes = 2, num_nics = 1, def_qps = 2, usr_qps = 2;
  auto P = NicPolicy::PER_CONTEXT;
  for (int i = 0; i < 12; i++) {
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 0);
  }
}

TEST(NicPolicyTest, PerContext_ZeroUsrQps) {
  // Edge case: no user context QPs (usr_qps = 0), only default
  int num_pes = 2, num_nics = 2, def_qps = 2, usr_qps = 0;
  auto P = NicPolicy::PER_CONTEXT;
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(ComputeNicIdxForQp(i, num_pes, num_nics, def_qps, usr_qps, P), 0)
        << "qp_idx=" << i;
  }
}
