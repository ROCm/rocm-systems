/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for topology construction internals (graph/topo.cc), driven with
// hand-built systems and XML trees. Add further topology test cases here.

#include "graph/topo.h"
#include "graph/xml.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// busIdToInt64 is an internal helper (declared in utils.h). Forward-declare it
// here so the test can compute node ids exactly the way the builder does.
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

// ncclTopoAddXGMI is a non-static internal symbol in graph/topo.cc that is not
// exposed through any header. It is only visible to tests in Debug builds (the
// rccl-UnitTestsFixturesDebug target). Declare it here, guarded identically to
// its definition.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
ncclResult_t ncclTopoAddXGMI(struct ncclXmlNode* node,
                             struct ncclTopoSystem* system,
                             const char* parentBusId, int systemId);
#endif

namespace RcclUnitTesting {

class TopoTest : public ::testing::Test {
protected:
  static constexpr int kMaxXmlNodes = 64;

  void SetUp() override {
    system =
        static_cast<struct ncclTopoSystem*>(calloc(1, sizeof(struct ncclTopoSystem)));
    ASSERT_NE(system, nullptr);
    ASSERT_EQ(xmlAlloc(&xml, kMaxXmlNodes), ncclSuccess);
    ASSERT_EQ(xmlAddNode(xml, nullptr, "system", &root), ncclSuccess);
  }

  void TearDown() override {
    free(system);
    system = nullptr;
    free(xml);
    xml = nullptr;
  }

  // Register a host so ncclGetSystemId() maps a host_hash to a deterministic
  // systemId, independent of XML traversal order.
  void registerHost(int systemId, uint64_t hostHash) {
    if (systemId >= system->nHosts) system->nHosts = systemId + 1;
    system->hostHashes[systemId] = hostHash;
  }

  // Create a GPU node whose id is namespaced by systemId, mirroring how the
  // topology builder stores GPUs.
  struct ncclTopoNode* addGpu(int systemId, const char* busId, const char* gcn,
                              int rank, int dev) {
    int64_t bus = 0;
    EXPECT_EQ(busIdToInt64(busId, &bus), ncclSuccess);
    struct ncclTopoNode* gpu = nullptr;
    EXPECT_EQ(ncclTopoCreateNode(system, &gpu, GPU, NCCL_TOPO_ID(systemId, bus)),
              ncclSuccess);
    if (gpu) {
      strncpy(gpu->gpu.gcn, gcn, GCN_ARCH_NAME_LEN - 1);
      gpu->gpu.gcn[GCN_ARCH_NAME_LEN - 1] = '\0';
      gpu->gpu.rank = rank;
      gpu->gpu.dev = dev;
    }
    return gpu;
  }

  // <cpu host_hash="0x..."> — resolves to a systemId during traversal.
  struct ncclXmlNode* addCpu(uint64_t hostHash) {
    struct ncclXmlNode* cpu = nullptr;
    EXPECT_EQ(xmlAddNode(xml, root, "cpu", &cpu), ncclSuccess);
    char hash[32];
    snprintf(hash, sizeof(hash), "0x%llx",
             static_cast<unsigned long long>(hostHash));
    EXPECT_EQ(xmlSetAttr(cpu, "host_hash", hash), ncclSuccess);
    return cpu;
  }

  // <pci busid="..."> — the parent node whose bus id identifies the local GPU
  // for any link entries nested below it.
  struct ncclXmlNode* addGpuHolder(struct ncclXmlNode* parent,
                                   const char* busId) {
    struct ncclXmlNode* pci = nullptr;
    EXPECT_EQ(xmlAddNode(xml, parent, "pci", &pci), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "busid", busId), ncclSuccess);
    return pci;
  }

  // GPU-to-GPU link entry (tclass 0x03 -> GPU) connecting to a peer GPU.
  void addGpuLink(struct ncclXmlNode* holder, const char* targetBusId,
                  int count) {
    struct ncclXmlNode* link = nullptr;
    EXPECT_EQ(xmlAddNode(xml, holder, "xgmi", &link), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(link, "target", targetBusId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(link, "tclass", "0x03"), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(link, "count", count), ncclSuccess);
  }

  // <cpu> populated with the attributes ncclTopoGetSystemFromXml() requires to
  // build a full system (numaid, host id, x86/AMD identification).
  struct ncclXmlNode* addSystemCpu(uint64_t hostHash, int numaId = 0) {
    struct ncclXmlNode* cpu = nullptr;
    EXPECT_EQ(xmlAddNode(xml, root, "cpu", &cpu), ncclSuccess);
    char hash[32];
    snprintf(hash, sizeof(hash), "0x%llx",
             static_cast<unsigned long long>(hostHash));
    EXPECT_EQ(xmlSetAttr(cpu, "host_hash", hash), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(cpu, "numaid", numaId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(cpu, "arch", "x86_64"), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(cpu, "vendor", "AuthenticAMD"), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(cpu, "familyid", 25), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(cpu, "modelid", 1), ncclSuccess);
    return cpu;
  }

  // GPU <pci> node (with nested <gpu>) under a CPU, returning the pci node so
  // callers can attach <xgmi> link entries below it.
  struct ncclXmlNode* addGpuPci(struct ncclXmlNode* parent, const char* busId,
                                const char* gcn, int rank, int dev) {
    struct ncclXmlNode* pci = nullptr;
    EXPECT_EQ(xmlAddNode(xml, parent, "pci", &pci), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "busid", busId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "class", "0x03"), ncclSuccess); // 0x03 -> GPU
    EXPECT_EQ(xmlSetAttrInt(pci, "link_width", 16), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "link_speed", "16.0 GT/s PCIe"), ncclSuccess);
    struct ncclXmlNode* gpu = nullptr;
    EXPECT_EQ(xmlAddNode(xml, pci, "gpu", &gpu), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "sm", 304), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(gpu, "gcn", gcn), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "arch", 0), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "rank", rank), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "dev", dev), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "gdr", 1), ncclSuccess);
    return pci;
  }

  static struct ncclTopoLink* findLink(struct ncclTopoNode* from,
                                       struct ncclTopoNode* to) {
    if (from == nullptr || to == nullptr) return nullptr;
    for (int i = 0; i < from->nlinks; i++) {
      if (from->links[i].type == LINK_NVL && from->links[i].remNode == to)
        return &from->links[i];
    }
    return nullptr;
  }

  struct ncclTopoSystem* system = nullptr;
  struct ncclXml* xml = nullptr;
  struct ncclXmlNode* root = nullptr;
};

// ncclTopoAddXGMI() is only built on HIP/AMD platforms, so guard these tests
// with the same macro; the #else branch keeps the suite linkable elsewhere.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

// Positive: a single gfx1250 system with two GPUs gets a bidirectional link at
// the expected bandwidth.
TEST_F(TopoTest, Gfx1250_SingleSystem_BidirectionalLink) {
  const uint64_t host = 0xa1;
  registerHost(0, host);

  struct ncclTopoNode* g0 = addGpu(0, "0000:01:00.0", "gfx1250", 0, 0);
  struct ncclTopoNode* g1 = addGpu(0, "0000:02:00.0", "gfx1250", 1, 1);
  ASSERT_NE(g0, nullptr);
  ASSERT_NE(g1, nullptr);

  struct ncclXmlNode* cpu = addCpu(host);
  addGpuLink(addGpuHolder(cpu, "0000:01:00.0"), "0000:02:00.0", 8);
  addGpuLink(addGpuHolder(cpu, "0000:02:00.0"), "0000:01:00.0", 8);

  ASSERT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclSuccess);

  struct ncclTopoLink* l01 = findLink(g0, g1);
  struct ncclTopoLink* l10 = findLink(g1, g0);
  ASSERT_NE(l01, nullptr);
  ASSERT_NE(l10, nullptr);
  EXPECT_FLOAT_EQ(l01->bw, 8 * ncclTopoXGMISpeed("gfx1250"));
  EXPECT_FLOAT_EQ(l10->bw, 8 * ncclTopoXGMISpeed("gfx1250"));
}

// Positive (regression): two heterogeneous systems whose GPUs reuse identical
// local bus ids. Links must connect GPUs within the same system only — without
// system-id namespacing, host B's links would incorrectly resolve to host A's
// GPUs.
TEST_F(TopoTest, MultiSystem_SameBusIds_LinksStayWithinSystem) {
  const uint64_t hostA = 0xaa;
  const uint64_t hostB = 0xbb;
  registerHost(0, hostA);
  registerHost(1, hostB);

  // Both systems deliberately use the same local bus ids.
  struct ncclTopoNode* a0 = addGpu(0, "0000:01:00.0", "gfx1250", 0, 0);
  struct ncclTopoNode* a1 = addGpu(0, "0000:02:00.0", "gfx1250", 1, 1);
  struct ncclTopoNode* b0 = addGpu(1, "0000:01:00.0", "gfx1250", 2, 0);
  struct ncclTopoNode* b1 = addGpu(1, "0000:02:00.0", "gfx1250", 3, 1);
  ASSERT_NE(a0, nullptr);
  ASSERT_NE(a1, nullptr);
  ASSERT_NE(b0, nullptr);
  ASSERT_NE(b1, nullptr);

  struct ncclXmlNode* cpuA = addCpu(hostA);
  addGpuLink(addGpuHolder(cpuA, "0000:01:00.0"), "0000:02:00.0", 8);
  addGpuLink(addGpuHolder(cpuA, "0000:02:00.0"), "0000:01:00.0", 8);

  struct ncclXmlNode* cpuB = addCpu(hostB);
  addGpuLink(addGpuHolder(cpuB, "0000:01:00.0"), "0000:02:00.0", 8);
  addGpuLink(addGpuHolder(cpuB, "0000:02:00.0"), "0000:01:00.0", 8);

  ASSERT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclSuccess);

  // Intra-system links exist for both hosts.
  EXPECT_NE(findLink(a0, a1), nullptr);
  EXPECT_NE(findLink(a1, a0), nullptr);
  EXPECT_NE(findLink(b0, b1), nullptr);
  EXPECT_NE(findLink(b1, b0), nullptr);

  // No cross-system links despite identical local bus ids.
  EXPECT_EQ(findLink(a0, b0), nullptr);
  EXPECT_EQ(findLink(a0, b1), nullptr);
  EXPECT_EQ(findLink(a1, b0), nullptr);
  EXPECT_EQ(findLink(a1, b1), nullptr);
  EXPECT_EQ(findLink(b0, a0), nullptr);
  EXPECT_EQ(findLink(b0, a1), nullptr);
  EXPECT_EQ(findLink(b1, a0), nullptr);
  EXPECT_EQ(findLink(b1, a1), nullptr);
}

// Positive (architecture-agnostic): the same flow works for a non-gfx1250
// architecture and uses that architecture's link speed.
TEST_F(TopoTest, GenericArch_SingleSystem_LinkCreated) {
  const uint64_t host = 0x42;
  registerHost(0, host);

  struct ncclTopoNode* g0 = addGpu(0, "0000:01:00.0", "gfx942", 0, 0);
  struct ncclTopoNode* g1 = addGpu(0, "0000:02:00.0", "gfx942", 1, 1);
  ASSERT_NE(g0, nullptr);
  ASSERT_NE(g1, nullptr);

  struct ncclXmlNode* cpu = addCpu(host);
  addGpuLink(addGpuHolder(cpu, "0000:01:00.0"), "0000:02:00.0", 4);

  ASSERT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclSuccess);

  struct ncclTopoLink* l = findLink(g0, g1);
  ASSERT_NE(l, nullptr);
  EXPECT_FLOAT_EQ(l->bw, 4 * ncclTopoXGMISpeed("gfx942"));
}

// Negative: the local (parent) GPU referenced by the link node does not exist.
// The builder must report an internal error.
TEST_F(TopoTest, Negative_MissingLocalGpu_ReturnsError) {
  const uint64_t host = 0x1;
  registerHost(0, host);

  // Only the target GPU exists; the parent (bus 0000:01:00.0) does not.
  addGpu(0, "0000:02:00.0", "gfx1250", 1, 1);

  struct ncclXmlNode* cpu = addCpu(host);
  addGpuLink(addGpuHolder(cpu, "0000:01:00.0"), "0000:02:00.0", 8);

  EXPECT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclInternalError);
}

// Negative (isolation): the link target bus id exists only in another system.
// With system-id namespacing the lookup must not cross the system boundary, so
// no link is created and traversal still succeeds.
TEST_F(TopoTest, Negative_TargetInOtherSystem_NoLink) {
  const uint64_t hostA = 0xaa;
  const uint64_t hostB = 0xbb;
  registerHost(0, hostA);
  registerHost(1, hostB);

  struct ncclTopoNode* a0 = addGpu(0, "0000:01:00.0", "gfx1250", 0, 0);
  struct ncclTopoNode* b1 = addGpu(1, "0000:02:00.0", "gfx1250", 3, 1);
  ASSERT_NE(a0, nullptr);
  ASSERT_NE(b1, nullptr);

  // System 0's GPU targets a bus id that only exists in system 1.
  struct ncclXmlNode* cpuA = addCpu(hostA);
  addGpuLink(addGpuHolder(cpuA, "0000:01:00.0"), "0000:02:00.0", 8);

  ASSERT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclSuccess);

  EXPECT_EQ(findLink(a0, b1), nullptr);
  EXPECT_EQ(a0->nlinks, 0);
}

// Negative: a required attribute (count) is missing from the link entry.
TEST_F(TopoTest, Negative_MissingCountAttr_ReturnsError) {
  const uint64_t host = 0x1;
  registerHost(0, host);

  addGpu(0, "0000:01:00.0", "gfx1250", 0, 0);
  addGpu(0, "0000:02:00.0", "gfx1250", 1, 1);

  struct ncclXmlNode* cpu = addCpu(host);
  struct ncclXmlNode* holder = addGpuHolder(cpu, "0000:01:00.0");

  struct ncclXmlNode* link = nullptr;
  ASSERT_EQ(xmlAddNode(xml, holder, "xgmi", &link), ncclSuccess);
  ASSERT_EQ(xmlSetAttr(link, "target", "0000:02:00.0"), ncclSuccess);
  ASSERT_EQ(xmlSetAttr(link, "tclass", "0x03"), ncclSuccess);
  // Intentionally omit the "count" attribute.

  EXPECT_EQ(ncclTopoAddXGMI(root, system, nullptr, 0), ncclInternalError);
}

// End-to-end: drive the full ncclTopoGetSystemFromXml() entry point with a
// complete single-node topology and verify the resulting system has the
// expected XGMI links. This exercises the ncclTopoAddXGMI() call site inside
// ncclTopoGetSystemFromXml (not just the function in isolation).
TEST_F(TopoTest, GetSystemFromXml_SingleSystem_BuildsLinks) {
  const uint64_t host = 0x1234;

  struct ncclXmlNode* cpu = addSystemCpu(host);
  struct ncclXmlNode* pci0 = addGpuPci(cpu, "0000:01:00.0", "gfx942", 0, 0);
  struct ncclXmlNode* pci1 = addGpuPci(cpu, "0000:02:00.0", "gfx942", 1, 1);
  addGpuLink(pci0, "0000:02:00.0", 8);
  addGpuLink(pci1, "0000:01:00.0", 8);

  struct ncclTopoSystem* built = nullptr;
  ASSERT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
  ASSERT_NE(built, nullptr);

  int64_t bus0 = 0, bus1 = 0;
  ASSERT_EQ(busIdToInt64("0000:01:00.0", &bus0), ncclSuccess);
  ASSERT_EQ(busIdToInt64("0000:02:00.0", &bus1), ncclSuccess);

  struct ncclTopoNode* g0 = nullptr;
  struct ncclTopoNode* g1 = nullptr;
  ASSERT_EQ(ncclTopoGetNode(built, &g0, GPU, NCCL_TOPO_ID(0, bus0)), ncclSuccess);
  ASSERT_EQ(ncclTopoGetNode(built, &g1, GPU, NCCL_TOPO_ID(0, bus1)), ncclSuccess);
  ASSERT_NE(g0, nullptr);
  ASSERT_NE(g1, nullptr);

  struct ncclTopoLink* l01 = findLink(g0, g1);
  struct ncclTopoLink* l10 = findLink(g1, g0);
  ASSERT_NE(l01, nullptr);
  ASSERT_NE(l10, nullptr);
  EXPECT_FLOAT_EQ(l01->bw, 8 * ncclTopoXGMISpeed("gfx942"));
  EXPECT_FLOAT_EQ(l10->bw, 8 * ncclTopoXGMISpeed("gfx942"));

  ncclTopoFree(built);
}

#else // !(__HIP_PLATFORM_AMD__ || __HIPCC__)

// ncclTopoAddXGMI() is not built on non-HIP platforms, so register one skipped
// test instead so the suite still exists and links there.
TEST_F(TopoTest, XgmiTestsSkippedOnNonHipBuild) {
  GTEST_SKIP() << "ncclTopoAddXGMI is only built on HIP/AMD platforms";
}

#endif // __HIP_PLATFORM_AMD__ || __HIPCC__

} // namespace RcclUnitTesting
