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

#ifndef ROCSHMEM_TOPOLOGY_GTEST_HPP
#define ROCSHMEM_TOPOLOGY_GTEST_HPP

#include <hip/hip_runtime.h>
#include <set>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "../src/gda/topology.hpp"

namespace rocshmem {

// Helper class to build PCIe paths from vector of components
class PCIeTreeTestHelper {
public:
  // Insert path to tree from vector of path components
  static void InsertPCIePathToTree(std::vector<std::string> const& pathComponents,
                                   std::string const& description,
                                   PCIeNode& root) {
    // Build path by concatenating: /sys/bus/pci/devices/<component>/<component>/...
    // But we just need the last component as the address
    if (pathComponents.empty()) return;

    // For the test tree, we build it step by step
    PCIeNode* currNode = &root;
    for (auto const& component : pathComponents) {
      auto it = (currNode->children.insert(PCIeNode(component))).first;
      currNode = const_cast<PCIeNode*>(&(*it));
    }
    currNode->description = description;
  }
};

class TopologyTestFixture : public ::testing::Test
{
  protected:
    void SetUp() override {
      // Initialize HIP runtime
      hipError_t err = hipGetDeviceCount(&num_gpus_);
      if (err != hipSuccess) {
        num_gpus_ = 0;
      }
    }

    void TearDown() override {
      // Cleanup if needed
    }

    int num_gpus_ = 0;
};

class DeviceTypeTestFixture : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test fixture for PCIe tree testing with theoretical topology
class PCIeTreeTestFixture : public ::testing::Test
{
  protected:
    PCIeNode root_;
    std::vector<std::string> gpu_addresses_;
    std::vector<std::string> nic_addresses_;

    void SetUp() override {
      // Create a theoretical PCIe tree with 4 GPUs and 4 NICs
      // Topology:
      //   Root (root)
      //     ├── CPU Socket 0 (0000:00:01.0)
      //     │   ├── PCIe Switch 0 (0000:01:00.0)
      //     │   │   ├── GPU 0 (0000:02:00.0)
      //     │   │   └── NIC 0 (0000:03:00.0)
      //     │   └── PCIe Switch 1 (0000:04:00.0)
      //     │       ├── GPU 1 (0000:05:00.0)
      //     │       └── NIC 1 (0000:06:00.0)
      //     └── CPU Socket 1 (0000:40:00.0)
      //         ├── PCIe Switch 2 (0000:41:00.0)
      //         │   ├── GPU 2 (0000:42:00.0)
      //         │   └── NIC 2 (0000:43:00.0)
      //         └── PCIe Switch 3 (0000:44:00.0)
      //             ├── GPU 3 (0000:45:00.0)
      //             └── NIC 3 (0000:46:00.0)

      // Define GPU and NIC addresses first
      gpu_addresses_ = {
        "0000:02:00.0",  // GPU 0
        "0000:05:00.0",  // GPU 1
        "0000:42:00.0",  // GPU 2
        "0000:45:00.0"   // GPU 3
      };

      nic_addresses_ = {
        "0000:03:00.0",  // NIC 0
        "0000:06:00.0",  // NIC 1
        "0000:43:00.0",  // NIC 2
        "0000:46:00.0"   // NIC 3
      };

      // Define the PCIe paths for each device
      // Each GPU/NIC pair shares the same switch
      std::vector<std::vector<std::string>> gpu_paths = {
        {"0000:00:01.0", "0000:01:00.0", gpu_addresses_[0]},  // GPU 0
        {"0000:00:01.0", "0000:04:00.0", gpu_addresses_[1]},  // GPU 1
        {"0000:40:00.0", "0000:41:00.0", gpu_addresses_[2]},  // GPU 2
        {"0000:40:00.0", "0000:44:00.0", gpu_addresses_[3]}   // GPU 3
      };

      std::vector<std::vector<std::string>> nic_paths = {
        {"0000:00:01.0", "0000:01:00.0", nic_addresses_[0]},  // NIC 0
        {"0000:00:01.0", "0000:04:00.0", nic_addresses_[1]},  // NIC 1
        {"0000:40:00.0", "0000:41:00.0", nic_addresses_[2]},  // NIC 2
        {"0000:40:00.0", "0000:44:00.0", nic_addresses_[3]}   // NIC 3
      };

      // Build the PCIe tree
      root_ = PCIeNode("root", "PCIe Root Complex");

      // Insert GPUs into tree
      for (size_t i = 0; i < gpu_addresses_.size(); i++) {
        PCIeTreeTestHelper::InsertPCIePathToTree(
          gpu_paths[i], "GPU " + std::to_string(i), root_);
      }

      // Insert NICs into tree
      for (size_t i = 0; i < nic_addresses_.size(); i++) {
        PCIeTreeTestHelper::InsertPCIePathToTree(
          nic_paths[i], "NIC " + std::to_string(i), root_);
      }
    }
};

} // namespace rocshmem

#endif // ROCSHMEM_TOPOLOGY_GTEST_HPP
