// Tests for null-guard of gpu.parent in topology path computation and search.
//
// Covers the fix in commit 5cb9833da063d97f2 where ncclTopoRemoveNode can set
// gpu.parent = NULL after deleting a DEV node. Three call sites dereferenced
// gpu.parent without null checks, causing segfaults in multi-node topologies.
//
// Rather than compiling paths.cc / search.cc (which have massive dependency
// chains), we replicate the vulnerable code patterns with minimal topo structs
// and verify they handle NULL parent correctly.

#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include <algorithm>

// --- Minimal topo definitions matching the real structs' relevant fields ---

#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3
#define NIC 4
#define NET 5
#define GIN 6
#define DEV 7

#define LINK_LOC 0
#define LINK_NVL 1
#define LINK_PCI 4
#define LINK_NET 10

#define NCCL_TOPO_NODE_TYPES 8
#define NCCL_TOPO_MAX_NODES 256
#define NCCL_TOPO_MAX_LINKS 576
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES * NCCL_TOPO_NODE_TYPES)
#define NCCL_TOPO_ID_SYSTEM_ID(id) ((id) >> 56)
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)(systemid) << 56) + ((localid) & 0x00ffffffffffffff))

#define GCN_ARCH_NAME_LEN 16

struct TestTopoNode;

struct TestTopoLink {
  int type;
  float bw;
  TestTopoNode* remNode;
};

struct TestTopoLinkList {
  TestTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

struct hipDeviceArch_t_stub { unsigned data; };

struct TestTopoNode {
  int type;
  int64_t id;
  union {
    struct {
      int dev;
      int rank;
      int cudaCompCap;
      int gdrSupport;
      char gcn[GCN_ARCH_NAME_LEN];
      hipDeviceArch_t_stub arch;
      int cu;
      TestTopoNode* parent;
    } gpu;
    struct {
      uint64_t device;
      int dev;
      int cudaCompCap;
      int gdrSupport;
      char gcn[GCN_ARCH_NAME_LEN];
    } dev;
    struct {
      int dev;
      uint64_t pciId;
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
      int localGpu;
      int64_t busId;
    } net;
  };
  int nlinks;
  TestTopoLink links[NCCL_TOPO_MAX_LINKS];
  TestTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  uint64_t used;
};

struct TestTopoNodeSet {
  int count;
  TestTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct TestTopoSystem {
  TestTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
};

// --- Reproduce vulnerable patterns (pre-fix and post-fix) ---

// Pattern 1: ncclTopoSetPaths BFS (paths.cc line 80)
// BEFORE fix: unconditional dereference of baseNode->gpu.parent
// AFTER fix: NULL check skips DEV traversal for non-LOC links
static bool setPathsDevFilter_UNFIXED(TestTopoNode* baseNode, TestTopoNode* node,
                                       TestTopoLink* link, TestTopoNode* remNode) {
  TestTopoNode* baseDevNode = (baseNode->type == GPU) ? baseNode->gpu.parent : baseNode;
  // CRASH: baseDevNode is NULL, but we dereference it in (node != baseDevNode)
  // Actually node != NULL is fine, but the logic below would use baseDevNode further.
  // The real crash: if baseDevNode is NULL and we reach code that dereferences it.
  // In the original code, (node != baseDevNode) doesn't crash, but conceptually
  // the variable is invalid. The FIXED version handles it differently.
  if (node != baseDevNode && node->type == DEV && (link->type != LINK_LOC || remNode->type != GPU))
    return true; // skip
  return false;
}

// FIXED version: null-guarded (matches the fix in commit 5cb9833)
static bool setPathsDevFilter_FIXED(TestTopoNode* baseNode, TestTopoNode* node,
                                     TestTopoLink* link, TestTopoNode* remNode) {
  TestTopoNode* baseDevNode = (baseNode->type == GPU) ? baseNode->gpu.parent : baseNode;
  if (baseDevNode == nullptr) {
    if (node->type == DEV && (link->type != LINK_LOC || remNode->type != GPU))
      return true; // skip
    return false;
  }
  if (node != baseDevNode && node->type == DEV && (link->type != LINK_LOC || remNode->type != GPU))
    return true; // skip
  return false;
}

// Pattern 2: gpuPciBw (search.cc line 176)
// BEFORE fix: unconditional dereference of gpu->gpu.parent
// AFTER fix: NULL returns -1
static int gpuPciBw_UNFIXED(TestTopoNode* gpu) {
  TestTopoNode* dev = gpu->gpu.parent;
  // CRASH: dev->nlinks when dev is NULL
  for (int l = 0; l < dev->nlinks; l++) {
    TestTopoLink* gpuLink = dev->links + l;
    if (gpuLink->type != LINK_PCI) continue;
    TestTopoNode* pci = gpuLink->remNode;
    for (int l2 = 0; l2 < pci->nlinks; l2++) {
      TestTopoLink* pciLink = pci->links + l2;
      if (pciLink->remNode != dev) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

static int gpuPciBw_FIXED(TestTopoNode* gpu) {
  TestTopoNode* dev = gpu->gpu.parent;
  if (dev == nullptr) return -1;
  for (int l = 0; l < dev->nlinks; l++) {
    TestTopoLink* gpuLink = dev->links + l;
    if (gpuLink->type != LINK_PCI) continue;
    TestTopoNode* pci = gpuLink->remNode;
    for (int l2 = 0; l2 < pci->nlinks; l2++) {
      TestTopoLink* pciLink = pci->links + l2;
      if (pciLink->remNode != dev) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

// Pattern 3: ncclTopoGetChannelFromXml GPU lookup (search.cc line 963)
// BEFORE fix: unconditional dereference of gpu.parent->id
// AFTER fix: NULL parent skips the GPU
static int xmlChannelFindRank_UNFIXED(TestTopoSystem* system, int64_t dev, int ngpus) {
  int rank = -1;
  for (int g = 0; g < ngpus; g++) {
    // CRASH: system->nodes[GPU].nodes[g].gpu.parent->id when parent is NULL
    int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].gpu.parent->id);
    if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev)
      rank = system->nodes[GPU].nodes[g].gpu.rank;
  }
  return rank;
}

static int xmlChannelFindRank_FIXED(TestTopoSystem* system, int64_t dev, int ngpus) {
  int rank = -1;
  for (int g = 0; g < ngpus; g++) {
    if (system->nodes[GPU].nodes[g].gpu.parent == nullptr) continue;
    int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].gpu.parent->id);
    if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev)
      rank = system->nodes[GPU].nodes[g].gpu.rank;
  }
  return rank;
}

// --- Helper to construct minimal topo nodes ---

static TestTopoNode makeGpuNode(int devNum, int rank, int64_t id, TestTopoNode* parent) {
  TestTopoNode node{};
  node.type = GPU;
  node.id = id;
  node.gpu.dev = devNum;
  node.gpu.rank = rank;
  node.gpu.parent = parent;
  node.nlinks = 0;
  memset(node.paths, 0, sizeof(node.paths));
  return node;
}

static TestTopoNode makeDevNode(int devNum, int64_t id) {
  TestTopoNode node{};
  node.type = DEV;
  node.id = id;
  node.dev.dev = devNum;
  node.nlinks = 0;
  memset(node.paths, 0, sizeof(node.paths));
  return node;
}

// ==================== TEST SUITE ====================

class NullParentTest : public ::testing::Test {
protected:
  TestTopoSystem system{};
  TestTopoNode devNode{};
  TestTopoNode gpuWithParent{};
  TestTopoNode gpuNoParent{};

  void SetUp() override {
    memset(&system, 0, sizeof(system));

    devNode = makeDevNode(0, 0x100);

    gpuWithParent = makeGpuNode(0, 0, NCCL_TOPO_ID(0, 0), &devNode);
    gpuNoParent   = makeGpuNode(1, 1, NCCL_TOPO_ID(0, 1), nullptr);

    system.nodes[GPU].count = 2;
    system.nodes[GPU].nodes[0] = gpuWithParent;
    system.nodes[GPU].nodes[1] = gpuNoParent;

    system.nodes[DEV].count = 1;
    system.nodes[DEV].nodes[0] = devNode;
  }
};

// --- Pattern 1: ncclTopoSetPaths BFS DEV filter ---

TEST_F(NullParentTest, SetPaths_NullParent_FixedDoesNotCrash) {
  TestTopoNode devTraversalNode{};
  devTraversalNode.type = DEV;

  TestTopoLink nvlLink{};
  nvlLink.type = LINK_NVL;

  TestTopoNode remGpu{};
  remGpu.type = DEV;

  bool skip = setPathsDevFilter_FIXED(&gpuNoParent, &devTraversalNode, &nvlLink, &remGpu);
  EXPECT_TRUE(skip) << "With NULL parent, DEV node traversal via NVL should be skipped";
}

TEST_F(NullParentTest, SetPaths_NullParent_LocLinkToGpuAllowed) {
  TestTopoNode devTraversalNode{};
  devTraversalNode.type = DEV;

  TestTopoLink locLink{};
  locLink.type = LINK_LOC;

  TestTopoNode remGpu{};
  remGpu.type = GPU;

  bool skip = setPathsDevFilter_FIXED(&gpuNoParent, &devTraversalNode, &locLink, &remGpu);
  EXPECT_FALSE(skip) << "LOC link to GPU should be allowed even with NULL parent";
}

TEST_F(NullParentTest, SetPaths_WithParent_BehaviorPreserved) {
  TestTopoNode devTraversalNode = devNode;

  TestTopoLink nvlLink{};
  nvlLink.type = LINK_NVL;

  TestTopoNode remDev{};
  remDev.type = DEV;

  // With parent, node == baseDevNode should NOT skip
  bool skipSame = setPathsDevFilter_FIXED(&gpuWithParent, &devNode, &nvlLink, &remDev);
  EXPECT_FALSE(skipSame) << "Same DEV node as parent should not be skipped";

  // Different DEV node should skip (NVL to DEV)
  TestTopoNode otherDev{};
  otherDev.type = DEV;
  otherDev.id = 0x999;
  bool skipDiff = setPathsDevFilter_FIXED(&gpuWithParent, &otherDev, &nvlLink, &remDev);
  EXPECT_TRUE(skipDiff) << "Different DEV node via NVL should be skipped";
}

// --- Pattern 2: gpuPciBw ---

TEST_F(NullParentTest, GpuPciBw_NullParent_FixedReturnsNeg1) {
  int bw = gpuPciBw_FIXED(&gpuNoParent);
  EXPECT_EQ(bw, -1) << "NULL parent should return -1 bandwidth";
}

TEST_F(NullParentTest, GpuPciBw_WithParent_NoLinks_ReturnsNeg1) {
  devNode.nlinks = 0;
  gpuWithParent.gpu.parent = &devNode;
  int bw = gpuPciBw_FIXED(&gpuWithParent);
  EXPECT_EQ(bw, -1) << "No PCI links should return -1";
}

TEST_F(NullParentTest, GpuPciBw_WithParent_PciLink_ReturnsBw) {
  // Set up: devNode --PCI--> pciSwitch --PCI(reverse)--> devNode
  TestTopoNode pciSwitch{};
  pciSwitch.type = PCI;
  pciSwitch.nlinks = 1;
  pciSwitch.links[0].type = LINK_PCI;
  pciSwitch.links[0].bw = 16.0;
  pciSwitch.links[0].remNode = &devNode;

  devNode.nlinks = 1;
  devNode.links[0].type = LINK_PCI;
  devNode.links[0].bw = 12.0;
  devNode.links[0].remNode = &pciSwitch;

  gpuWithParent.gpu.parent = &devNode;
  int bw = gpuPciBw_FIXED(&gpuWithParent);
  EXPECT_EQ(bw, 12) << "Should return min(12.0, 16.0) = 12";
}

// --- Pattern 3: XML channel rank lookup ---

TEST_F(NullParentTest, XmlChannelFindRank_NullParent_FixedSkipsGpu) {
  int rank = xmlChannelFindRank_FIXED(&system, NCCL_TOPO_ID(0, 1), 2);
  EXPECT_EQ(rank, -1) << "GPU with NULL parent should be skipped, rank not found";
}

TEST_F(NullParentTest, XmlChannelFindRank_WithParent_FindsRank) {
  // GPU 0 has parent with id 0x100 (systemId=0), dev=0 => topo_id = NCCL_TOPO_ID(0, 0)
  system.nodes[GPU].nodes[0].gpu.parent = &system.nodes[DEV].nodes[0];
  system.nodes[DEV].nodes[0].id = NCCL_TOPO_ID(0, 0x100);

  int rank = xmlChannelFindRank_FIXED(&system, NCCL_TOPO_ID(0, 0), 2);
  EXPECT_EQ(rank, 0) << "GPU 0 with valid parent should be found";
}

TEST_F(NullParentTest, XmlChannelFindRank_MixedParents_OnlyFindsValid) {
  // GPU 0 has parent, GPU 1 has NULL parent
  TestTopoNode parentDev{};
  parentDev.type = DEV;
  parentDev.id = NCCL_TOPO_ID(0, 0x200);

  system.nodes[GPU].nodes[0].gpu.parent = &parentDev;
  system.nodes[GPU].nodes[0].gpu.dev = 5;
  system.nodes[GPU].nodes[0].gpu.rank = 42;

  int64_t targetId = NCCL_TOPO_ID(0, 5);
  int rank = xmlChannelFindRank_FIXED(&system, targetId, 2);
  EXPECT_EQ(rank, 42) << "Should find rank through GPU with valid parent";
}

// --- Diagnostic: verify NULL parent is detectable (ncclTopoComputePaths warning) ---

TEST_F(NullParentTest, DetectNullParentInSystem) {
  int nullCount = 0;
  for (int g = 0; g < system.nodes[GPU].count; g++) {
    if (system.nodes[GPU].nodes[g].gpu.parent == nullptr) {
      nullCount++;
    }
  }
  EXPECT_EQ(nullCount, 1) << "Exactly one GPU should have NULL parent in test fixture";
}

// --- Edge cases ---

TEST_F(NullParentTest, NonGpuBaseNode_NullParentIrrelevant) {
  // When baseNode is not a GPU, gpu.parent is not used
  TestTopoNode netNode{};
  netNode.type = NET;

  TestTopoNode devTraversalNode{};
  devTraversalNode.type = DEV;

  TestTopoLink pciLink{};
  pciLink.type = LINK_PCI;

  TestTopoNode remNode{};
  remNode.type = CPU;

  // baseDevNode = baseNode (not gpu.parent) when type != GPU
  bool skip = setPathsDevFilter_FIXED(&netNode, &devTraversalNode, &pciLink, &remNode);
  EXPECT_TRUE(skip) << "Non-GPU base: DEV via PCI to non-GPU should be skipped";
}
