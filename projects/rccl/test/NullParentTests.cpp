// Tests for null-guard of gpu.parent in topology path computation and search.
//
// These tests call the REAL ncclTopoSetPaths and gpuPciBw from production
// paths.cc and search.cc (compiled via paths_wrapper.cc / search_wrapper.cc)
// to verify null-parent handling in real code, not replicas.
//
// Uses ProcessIsolatedTestRunner (fork+execv) to detect segfaults from
// null-pointer dereferences without crashing the test process.
//
// This file is #included by NullParentTests_wrapper.cc which provides:
//   - Real ncclTopoNode/ncclTopoSystem from topo.h
//   - test_ncclTopoSetPaths() and test_gpuPciBw() declarations
// Do NOT compile this file directly.

#ifndef NCCL_TOPO_H_
#error "NullParentTests.cpp must be compiled via NullParentTests_wrapper.cc"
#endif

#include <gtest/gtest.h>
#include <cstring>

#include "common/ProcessIsolatedTestRunner.hpp"

class NullParentTest : public ::testing::Test {
protected:
  static void initGpuNode(ncclTopoNode* node, int devNum, int rank,
                           int64_t id, ncclTopoNode* parent) {
    memset(node, 0, sizeof(*node));
    node->type = GPU;
    node->id = id;
    node->gpu.dev = devNum;
    node->gpu.rank = rank;
    node->gpu.parent = parent;
  }

  static void initDevNode(ncclTopoNode* node, int devNum, int64_t id) {
    memset(node, 0, sizeof(*node));
    node->type = DEV;
    node->id = id;
    node->dev.dev = devNum;
  }
};

// --- gpuPciBw tests: call real production function ---

TEST_F(NullParentTest, GpuPciBw_NullParent_ReturnsNeg1) {
  // GPU with NULL parent — real gpuPciBw should return -1 without crashing
  ncclTopoNode gpu{};
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), nullptr);

  RUN_ISOLATED_TEST("gpuPciBw_NullParent", [&gpu]() {
    int bw = test_gpuPciBw(&gpu);
    EXPECT_EQ(bw, -1) << "gpuPciBw should return -1 for NULL parent";
  });
}

TEST_F(NullParentTest, GpuPciBw_WithParent_NoLinks_ReturnsNeg1) {
  ncclTopoNode devNode{};
  initDevNode(&devNode, 0, 0x100);

  ncclTopoNode gpu{};
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), &devNode);

  int bw = test_gpuPciBw(&gpu);
  EXPECT_EQ(bw, -1) << "No PCI links should return -1";
}

TEST_F(NullParentTest, GpuPciBw_WithParent_PciLink_ReturnsBw) {
  ncclTopoNode pciSwitch{};
  memset(&pciSwitch, 0, sizeof(pciSwitch));
  pciSwitch.type = PCI;
  pciSwitch.nlinks = 1;
  pciSwitch.links[0].type = LINK_PCI;
  pciSwitch.links[0].bw = 16.0;

  ncclTopoNode devNode{};
  initDevNode(&devNode, 0, 0x100);
  devNode.nlinks = 1;
  devNode.links[0].type = LINK_PCI;
  devNode.links[0].bw = 12.0;
  devNode.links[0].remNode = &pciSwitch;

  pciSwitch.links[0].remNode = &devNode;

  ncclTopoNode gpu{};
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), &devNode);

  int bw = test_gpuPciBw(&gpu);
  EXPECT_EQ(bw, 12) << "Should return min(12.0, 16.0) = 12";
}

TEST_F(NullParentTest, DetectNullParent) {
  ncclTopoNode gpu{};
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), nullptr);
  EXPECT_EQ(gpu.gpu.parent, nullptr)
    << "GPU should have NULL parent in this test";
}
