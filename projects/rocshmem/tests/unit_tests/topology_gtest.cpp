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
    const char* devName = nullptr;
    int result = GetClosestNicToGpu(0, nullptr, &devName);
    // Should return a valid NIC index or -1 if detection failed
    if (result >= 0) {
      EXPECT_LT(result, numNics);
      // Device name should be set
      EXPECT_NE(devName, nullptr);
      if (devName != nullptr) {
        free(const_cast<char*>(devName));
      }
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
