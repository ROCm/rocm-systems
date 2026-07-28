// Tests for null-guard of gpu.parent in topology path computation and search.
//
// These tests call the REAL ncclTopoSetPaths, gpuPciBw, and
// ncclTopoGetChannelFromXml from production paths.cc and search.cc
// (compiled via paths_wrapper.cc / search_wrapper.cc) to verify
// null-parent handling in real code, not replicas.
//
// Uses ProcessIsolatedTestRunner (fork+execv) to detect segfaults from
// null-pointer dereferences without crashing the test process.
//
// This file is #included by NullParentTests_wrapper.cc which provides:
//   - Real ncclTopoNode/ncclTopoSystem/ncclXmlNode from topo.h and xml.h
//   - test_ncclTopoSetPaths(), test_gpuPciBw(), test_ncclTopoGetChannelFromXml()
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

// ============================================================
// Crash site 1: gpuPciBw (search.cc)
//   gpu->gpu.parent is dereferenced without null check.
// ============================================================

TEST_F(NullParentTest, GpuPciBw_NullParent_ReturnsNeg1) {
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

// ============================================================
// Fix site 2: ncclTopoSetPaths (paths.cc)
//   baseDevNode = baseNode->gpu.parent is NULL. The unfixed code
//   compares `node != baseDevNode` (i.e. node != NULL, always true)
//   which is safe but produces wrong traversal: DEV nodes that
//   should be the GPU's own parent are not recognized, so NVB
//   path filtering is incorrect. The fix blocks DEV traversal
//   entirely when parent is NULL. Not a crash — a logic bug.
// ============================================================

TEST_F(NullParentTest, TopoSetPaths_NullParent_NoSegfault) {
  ncclTopoSystem system{};
  memset(&system, 0, sizeof(system));

  ncclTopoNode& gpu = system.nodes[GPU].nodes[0];
  system.nodes[GPU].count = 1;
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), nullptr);

  ncclTopoNode& dev = system.nodes[DEV].nodes[0];
  system.nodes[DEV].count = 1;
  initDevNode(&dev, 0, NCCL_TOPO_ID(0, 0x100));

  gpu.nlinks = 1;
  gpu.links[0].type = LINK_PCI;
  gpu.links[0].bw = 16.0;
  gpu.links[0].remNode = &dev;

  dev.nlinks = 1;
  dev.links[0].type = LINK_PCI;
  dev.links[0].bw = 16.0;
  dev.links[0].remNode = &gpu;

  RUN_ISOLATED_TEST("topoSetPaths_NullParent", [&system]() {
    ncclResult_t ret = test_ncclTopoSetPaths(
        &system.nodes[GPU].nodes[0], &system);
    EXPECT_EQ(ret, ncclSuccess)
        << "ncclTopoSetPaths should handle NULL gpu.parent without crashing";
  });
}

// ============================================================
// Crash site 3: ncclTopoGetChannelFromXml (search.cc)
//   system->nodes[GPU].nodes[g].gpu.parent->id dereferences
//   parent without null check when looking up GPU rank by dev id.
// ============================================================

TEST_F(NullParentTest, TopoGetChannelFromXml_NullParent_NoSegfault) {
  // Minimal setup: 1 GPU with null parent, XML channel with a "gpu"
  // sub-element that has a "dev" attribute but no "rank" attribute,
  // forcing the code into the for-loop that dereferences gpu.parent.
  ncclTopoSystem system{};
  memset(&system, 0, sizeof(system));

  ncclTopoNode& gpu = system.nodes[GPU].nodes[0];
  system.nodes[GPU].count = 1;
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), nullptr);

  // Build a minimal ncclTopoGraph
  ncclTopoGraph graph{};
  memset(&graph, 0, sizeof(graph));

  // Build XML: channel with one "gpu" sub that has dev="0x0" but no rank attr
  ncclXmlNode xmlChannel{};
  memset(&xmlChannel, 0, sizeof(xmlChannel));
  strncpy(xmlChannel.name, "channel", MAX_STR_LEN);

  ncclXmlNode xmlGpuSub{};
  memset(&xmlGpuSub, 0, sizeof(xmlGpuSub));
  strncpy(xmlGpuSub.name, "gpu", MAX_STR_LEN);
  strncpy(xmlGpuSub.attrs[0].key, "dev", MAX_STR_LEN);
  strncpy(xmlGpuSub.attrs[0].value, "0x0", MAX_STR_LEN);
  xmlGpuSub.nAttrs = 1;

  xmlChannel.subs[0] = &xmlGpuSub;
  xmlChannel.nSubs = 1;

  RUN_ISOLATED_TEST("topoGetChannelFromXml_NullParent",
                    [&xmlChannel, &system, &graph]() {
    ncclResult_t ret = test_ncclTopoGetChannelFromXml(
        &xmlChannel, 0, &system, &graph);
    // With the fix: returns ncclSystemError (rank not found) without crashing.
    // Without the fix: segfaults on gpu.parent->id dereference.
    (void)ret;
  });
}

TEST_F(NullParentTest, DetectNullParent) {
  ncclTopoNode gpu{};
  initGpuNode(&gpu, 0, 0, NCCL_TOPO_ID(0, 0), nullptr);
  EXPECT_EQ(gpu.gpu.parent, nullptr)
    << "GPU should have NULL parent in this test";
}
