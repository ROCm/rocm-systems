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

#include "../common/ProcessIsolatedTestRunner.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// busIdToInt64 is an internal helper (declared in utils.h).
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

  // v2.30 attaches XGMI links to the DEV node ncclTopoAddXGMI() looks up at
  // NCCL_TOPO_ID(systemId, busId), so build DEV nodes (and set dev.gcn) directly.
  struct ncclTopoNode* addGpu(int systemId, const char* busId, const char* gcn,
                              int /*rank*/, int dev) {
    int64_t bus = 0;
    EXPECT_EQ(busIdToInt64(busId, &bus), ncclSuccess);
    struct ncclTopoNode* devNode = nullptr;
    EXPECT_EQ(ncclTopoCreateNode(system, &devNode, DEV, NCCL_TOPO_ID(systemId, bus)),
              ncclSuccess);
    if (devNode) {
      strncpy(devNode->dev.gcn, gcn, GCN_ARCH_NAME_LEN - 1);
      devNode->dev.gcn[GCN_ARCH_NAME_LEN - 1] = '\0';
      devNode->dev.dev = dev;
    }
    return devNode;
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

  // GPU-to-GPU link entry connecting to a peer GPU. Default tclass 0x03 (VGA);
  // xml.cc now also writes PCI_ACCELERATOR_CLASS ("0x120000") for AMD GPUs.
  void addGpuLink(struct ncclXmlNode* holder, const char* targetBusId,
                  int count, const char* tclass = "0x03") {
    struct ncclXmlNode* link = nullptr;
    EXPECT_EQ(xmlAddNode(xml, holder, "xgmi", &link), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(link, "target", targetBusId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(link, "tclass", tclass), ncclSuccess);
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
                                const char* gcn, int rank, int dev,
                                int mloPart = NCCL_TOPO_UNDEF) {
    struct ncclXmlNode* pci = nullptr;
    EXPECT_EQ(xmlAddNode(xml, parent, "pci", &pci), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "busid", busId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "class", "0x03"), ncclSuccess); // 0x03 -> GPU
    EXPECT_EQ(xmlSetAttrInt(pci, "link_width", 16), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "link_speed", "16.0 GT/s PCIe"), ncclSuccess);
    addGpuUnderPci(pci, gcn, rank, dev, mloPart);
    return pci;
  }

  struct ncclXmlNode* addGpuUnderPci(struct ncclXmlNode* pci, const char* gcn,
                                     int rank, int dev,
                                     int mloPart = NCCL_TOPO_UNDEF) {
    struct ncclXmlNode* gpu = nullptr;
    EXPECT_EQ(xmlAddNode(xml, pci, "gpu", &gpu), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "sm", 304), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(gpu, "gcn", gcn), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "arch", 0), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "rank", rank), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "dev", dev), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(gpu, "gdr", 1), ncclSuccess);
    if (mloPart != NCCL_TOPO_UNDEF) {
      EXPECT_EQ(xmlSetAttrInt(gpu, "mlopart", mloPart), ncclSuccess);
    }
    return gpu;
  }

  // <pci class="0x060400"> — a PCIe switch. Endpoints under two of these nested
  // below a common switch are PATH_PXB apart, the rail-local GPU/NIC shape on an
  // 8-GPU MI300X node.
  struct ncclXmlNode* addPciBridge(struct ncclXmlNode* parent, const char* busId) {
    struct ncclXmlNode* pci = nullptr;
    EXPECT_EQ(xmlAddNode(xml, parent, "pci", &pci), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "busid", busId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "class", "0x060400"), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(pci, "link_width", 16), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "link_speed", "16.0 GT/s PCIe"), ncclSuccess);
    return pci;
  }

  // <pci class="0x020700"><nic><net gdr="1"/></nic></pci> — a GDR-capable IB HCA.
  struct ncclXmlNode* addNic(struct ncclXmlNode* parent, const char* busId, int dev,
                             int gdr = 1) {
    struct ncclXmlNode* pci = nullptr;
    EXPECT_EQ(xmlAddNode(xml, parent, "pci", &pci), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "busid", busId), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "class", "0x020700"), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(pci, "link_width", 16), ncclSuccess);
    EXPECT_EQ(xmlSetAttr(pci, "link_speed", "16.0 GT/s PCIe"), ncclSuccess);
    struct ncclXmlNode* nic = nullptr;
    EXPECT_EQ(xmlAddNode(xml, pci, "nic", &nic), ncclSuccess);
    struct ncclXmlNode* net = nullptr;
    EXPECT_EQ(xmlAddNode(xml, nic, "net", &net), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(net, "dev", dev), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(net, "speed", 200000), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(net, "gdr", gdr), ncclSuccess);
    return pci;
  }

  // A node of a single rail, the shape the MLO-partition cases need.
  struct ncclTopoSystem* buildRailSystem(uint64_t host, int nGpus, bool partitioned) {
    addRail(addSystemCpu(host), /*rail=*/0, nGpus, partitioned);
    return buildSystem(host);
  }

  // ncclTopoGetSystemFromXml() leaves netGdrLevel zeroed; initTransportsRank() is what
  // arms the "use the default level" sentinel, so do the same before computing paths.
  struct ncclTopoSystem* buildSystem(uint64_t host) {
    struct ncclTopoSystem* built = nullptr;
    EXPECT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
    if (built) built->netGdrLevel = -2;
    return built;
  }

  // XGMI link entries name their peer by bus id, so a caller that links two rails needs this.
  static void railGpuBusId(int rail, char* busId, size_t len) {
    snprintf(busId, len, "0000:%02x:00.0", 0x0c + rail * 0x10);
  }

  // One rail of a rail-optimised node: a GPU carrying nGpus HIP devices and a NIC, on separate legs
  // of their own PCIe switch, which puts a GPU at PATH_PXB from the NIC of its rail and at PATH_PHB
  // from any other one. GPU and NIC of a rail share the dev index, which is what marks them a pair.
  // With partitioned=true the HIP devices are CPX partitions (mlopart 0..nGpus-1).
  // Returns the GPU's pci node, to attach <xgmi> entries below it.
  struct ncclXmlNode* addRail(struct ncclXmlNode* cpu, int rail, int nGpus = 1,
                              bool partitioned = false) {
    const int busBase = 0x0b + rail * 0x10;
    char switchBus[32], gpuLegBus[32], nicLegBus[32], gpuBus[32], nicBus[32];
    snprintf(switchBus, sizeof(switchBus), "0000:%02x:00.0", busBase);
    snprintf(gpuLegBus, sizeof(gpuLegBus), "0000:%02x:01.0", busBase);
    snprintf(nicLegBus, sizeof(nicLegBus), "0000:%02x:02.0", busBase);
    railGpuBusId(rail, gpuBus, sizeof(gpuBus));
    snprintf(nicBus, sizeof(nicBus), "0000:%02x:00.0", busBase + 2);

    struct ncclXmlNode* pciSwitch = addPciBridge(cpu, switchBus);
    struct ncclXmlNode* gpuPci =
        addGpuPci(addPciBridge(pciSwitch, gpuLegBus), gpuBus, "gfx942", /*rank=*/rail, /*dev=*/rail,
                  partitioned ? 0 : NCCL_TOPO_UNDEF);
    for (int p = 1; p < nGpus; p++) {
      addGpuUnderPci(gpuPci, "gfx942", /*rank=*/rail + p, /*dev=*/rail + p,
                     partitioned ? p : NCCL_TOPO_UNDEF);
    }
    addNic(addPciBridge(pciSwitch, nicLegBus), nicBus, /*dev=*/rail);
    return gpuPci;
  }

  struct ncclTopoSystem* buildSystemWithPaths(uint64_t host) {
    struct ncclTopoSystem* built = buildSystem(host);
    if (built == nullptr) return nullptr;
    EXPECT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
    return built;
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

// Encoding lives in headers; these run on every platform.
TEST_F(TopoTest, MloPartBusId_DoesNotClobberPciFunction) {
  int64_t busFn1 = 0;
  ASSERT_EQ(busIdToInt64("0000:01:00.1", &busFn1), ncclSuccess);
  const int64_t enc0 = NCCL_TOPO_MLOPART_BUSID(busFn1, 0);
  const int64_t enc7 = NCCL_TOPO_MLOPART_BUSID(busFn1, 7);
  EXPECT_NE(enc7, enc0);
  EXPECT_EQ(enc0 & ~NCCL_TOPO_MLOPART_MASK, busFn1);
  EXPECT_EQ(enc7 & ~NCCL_TOPO_MLOPART_MASK, busFn1);
}

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

  // XGMI links attach to DEV nodes (stored at NCCL_TOPO_ID(systemId, busId));
  // GPU rank nodes live at a separate NCCL_TOPO_GPU_LOCAL_ID.
  struct ncclTopoNode* d0 = nullptr;
  struct ncclTopoNode* d1 = nullptr;
  ASSERT_EQ(ncclTopoGetNode(built, &d0, DEV, NCCL_TOPO_ID(0, bus0)), ncclSuccess);
  ASSERT_EQ(ncclTopoGetNode(built, &d1, DEV, NCCL_TOPO_ID(0, bus1)), ncclSuccess);
  ASSERT_NE(d0, nullptr);
  ASSERT_NE(d1, nullptr);

  struct ncclTopoLink* l01 = findLink(d0, d1);
  struct ncclTopoLink* l10 = findLink(d1, d0);
  ASSERT_NE(l01, nullptr);
  ASSERT_NE(l10, nullptr);
  EXPECT_FLOAT_EQ(l01->bw, 8 * ncclTopoXGMISpeed("gfx942"));
  EXPECT_FLOAT_EQ(l10->bw, 8 * ncclTopoXGMISpeed("gfx942"));

  ncclTopoFree(built);
}

// Under the 2.30 DEV model a direct XGMI GPU->GPU path is the 3-hop
// GPU-DEV-DEV-GPU route, typed PATH_NVL — the predicate the matching fixes use.
TEST_F(TopoTest, DevModel_DirectXgmiPath_IsThreeHopNvl) {
  const uint64_t host = 0x55;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  struct ncclXmlNode* pci0 = addGpuPci(cpu, "0000:01:00.0", "gfx942", 0, 0);
  struct ncclXmlNode* pci1 = addGpuPci(cpu, "0000:02:00.0", "gfx942", 1, 1);
  addGpuLink(pci0, "0000:02:00.0", 8);
  addGpuLink(pci1, "0000:01:00.0", 8);

  struct ncclTopoSystem* built = nullptr;
  ASSERT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
  ASSERT_EQ(built->nodes[GPU].count, 2);

  // paths[GPU] is indexed by GPU node index; GPU 0 -> GPU 1 is direct.
  struct ncclTopoLinkList* p01 = built->nodes[GPU].nodes[0].paths[GPU] + 1;
  EXPECT_EQ(p01->type, PATH_NVL);
  EXPECT_EQ(p01->count, 3);

  ncclTopoFree(built);
}

// An indirect XGMI route (through a middle GPU's DEV) is the 4-hop path and must
// not be mistaken for a direct link; the count<=3 filters depend on this.
TEST_F(TopoTest, DevModel_IndirectXgmiPath_IsFourHops) {
  const uint64_t host = 0x56;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  struct ncclXmlNode* pci0 = addGpuPci(cpu, "0000:01:00.0", "gfx942", 0, 0);
  struct ncclXmlNode* pci1 = addGpuPci(cpu, "0000:02:00.0", "gfx942", 1, 1);
  struct ncclXmlNode* pci2 = addGpuPci(cpu, "0000:03:00.0", "gfx942", 2, 2);
  // Chain 0<->1 and 1<->2 are direct XGMI; 0 and 2 are not directly linked.
  addGpuLink(pci0, "0000:02:00.0", 8);
  addGpuLink(pci1, "0000:01:00.0", 8);
  addGpuLink(pci1, "0000:03:00.0", 8);
  addGpuLink(pci2, "0000:02:00.0", 8);

  struct ncclTopoSystem* built = nullptr;
  ASSERT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
  ASSERT_EQ(built->nodes[GPU].count, 3);

  auto idxByRank = [&](int rank) -> int {
    for (int i = 0; i < built->nodes[GPU].count; i++)
      if (built->nodes[GPU].nodes[i].gpu.rank == rank) return i;
    return -1;
  };
  int i0 = idxByRank(0), i2 = idxByRank(2);
  ASSERT_GE(i0, 0);
  ASSERT_GE(i2, 0);

  struct ncclTopoLinkList* p02 = built->nodes[GPU].nodes[i0].paths[GPU] + i2;
  EXPECT_EQ(p02->count, 4);
  EXPECT_GT(p02->count, 3); // excluded by the direct-XGMI filter

  ncclTopoFree(built);
}

TEST_F(TopoTest, GetSystemFromXml_MloPartOnNonzeroPhysicalFunction) {
  const uint64_t host = 0x77;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  struct ncclXmlNode* pci = addGpuPci(cpu, "0000:03:00.1", "gfx942", 0, 0, /*mloPart=*/0);
  struct ncclXmlNode* gpu0 = nullptr;
  ASSERT_EQ(xmlGetSub(pci, "gpu", &gpu0), ncclSuccess);
  struct ncclXmlNode* gpu1 = addGpuUnderPci(pci, "gfx942", 1, 1, /*mloPart=*/1);
  addGpuLink(gpu0, "0000:03:00.1", 1, PCI_ACCELERATOR_CLASS);
  addGpuLink(gpu1, "0000:03:00.0", 1, PCI_ACCELERATOR_CLASS);

  struct ncclTopoSystem* built = nullptr;
  ASSERT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[DEV].count, 2);
  EXPECT_NE(findLink(built->nodes[DEV].nodes, built->nodes[DEV].nodes + 1), nullptr);
  EXPECT_NE(findLink(built->nodes[DEV].nodes + 1, built->nodes[DEV].nodes), nullptr);
  ncclTopoFree(built);
}

// CPX: eight HIP partitions (mlopart 0-7) hang off the physical function-0 PCI
// node. Overlay bits must stay above the PCI function nibble.
TEST_F(TopoTest, GetSystemFromXml_CpxEightMlopartsUnderPhysicalPci) {
  const uint64_t host = 0xc0;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  struct ncclXmlNode* pci = addGpuPci(cpu, "0000:0c:00.0", "gfx942", 0, 0, /*mloPart=*/0);
  struct ncclXmlNode* gpu0 = nullptr;
  ASSERT_EQ(xmlGetSub(pci, "gpu", &gpu0), ncclSuccess);
  ASSERT_NE(gpu0, nullptr);
  struct ncclXmlNode* gpus[NCCL_TOPO_MLOPART_DEV_MAX] = {};
  gpus[0] = gpu0;
  for (int p = 1; p < NCCL_TOPO_MLOPART_DEV_MAX; p++) {
    gpus[p] = addGpuUnderPci(pci, "gfx942", p, p, p);
  }
  for (int p = 0; p < NCCL_TOPO_MLOPART_DEV_MAX; p++) {
    char tgt[32];
    snprintf(tgt, sizeof(tgt), "0000:0c:00.%d", (p + 1) % NCCL_TOPO_MLOPART_DEV_MAX);
    addGpuLink(gpus[p], tgt, 1, PCI_ACCELERATOR_CLASS);
  }

  struct ncclTopoSystem* built = nullptr;
  ASSERT_EQ(ncclTopoGetSystemFromXml(xml, &built, host), ncclSuccess);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[DEV].count, NCCL_TOPO_MLOPART_DEV_MAX);
  ASSERT_EQ(built->nodes[GPU].count, NCCL_TOPO_MLOPART_DEV_MAX);

  int64_t bus0 = 0;
  ASSERT_EQ(busIdToInt64("0000:0c:00.0", &bus0), ncclSuccess);
  for (int p = 0; p < NCCL_TOPO_MLOPART_DEV_MAX; p++) {
    struct ncclTopoNode* dev = nullptr;
    ASSERT_EQ(ncclTopoGetNode(built, &dev, DEV,
                              NCCL_TOPO_ID(0, NCCL_TOPO_MLOPART_BUSID(bus0, p))),
              ncclSuccess);
    ASSERT_NE(dev, nullptr);
    EXPECT_EQ(NCCL_TOPO_ID_LOCAL_ID(dev->id) & 0xf, 0);
    EXPECT_NE(findLink(dev, built->nodes[DEV].nodes + (p + 1) % NCCL_TOPO_MLOPART_DEV_MAX), nullptr);
  }

  ncclTopoFree(built);
}

// GDR for an MLOPart partition is a property of the physical GPU: every CPX partition is a HIP
// logical device behind one PCI function, so a NIC one switch away is PATH_PXB for all of them and
// GDR must be enabled for all of them. Before the rework ncclTopoCheckGdr() refused GDR to any
// partition, and ncclTopoComputePaths() then diverted those GPU<->NET paths through the local CPU,
// degrading PXB to PHB on exactly the entries parseRomeSystem() matches its gdrLevel presets on.
TEST_F(TopoTest, CheckGdr_CpxPartitionsUsePhysicalGpuDistance) {
  struct ncclTopoSystem* built = buildRailSystem(0xd0, NCCL_TOPO_MLOPART_DEV_MAX, /*partitioned=*/true);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
  ASSERT_EQ(built->nodes[GPU].count, NCCL_TOPO_MLOPART_DEV_MAX);
  ASSERT_EQ(built->nodes[NET].count, 1);

  const int64_t netId = built->nodes[NET].nodes[0].id;
  for (int p = 0; p < NCCL_TOPO_MLOPART_DEV_MAX; p++) {
    struct ncclTopoNode* gpu = built->nodes[GPU].nodes + p;
    SCOPED_TRACE(testing::Message() << "partition " << p << " mlopart " << gpu->gpu.mloPart);
    enum ncclTopoGdrMode mode = ncclTopoGdrModeDisable;
    ASSERT_EQ(ncclTopoCheckGdr(built, gpu->gpu.rank, netId, /*read=*/0, &mode), ncclSuccess);
    EXPECT_NE(mode, ncclTopoGdrModeDisable);
    // The decision must leave the rail-local hop alone rather than divert it through the CPU,
    // and it must agree with the physical device's own distance to the same NIC.
    ASSERT_NE(gpu->gpu.parent, nullptr);
    EXPECT_EQ(gpu->paths[NET][0].type, PATH_PXB);
    EXPECT_EQ(gpu->paths[NET][0].type, gpu->gpu.parent->paths[NET][0].type);
  }

  ncclTopoFree(built);
}

// A partition's own GPU->NET entry is not evidence about the hardware: ncclTopoComputePaths()
// rewrites it through the local CPU whenever some earlier decision refused GDR. The physical
// device is still one switch from the NIC, so the GDR verdict must be read from the parent DEV
// node and stay unchanged. Diverting the entry by hand stands in for that rewrite.
TEST_F(TopoTest, CheckGdr_MloPartIgnoresDivertedPartitionPath) {
  struct ncclTopoSystem* built = buildRailSystem(0xd1, NCCL_TOPO_MLOPART_DEV_MAX, /*partitioned=*/true);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
  ASSERT_EQ(built->nodes[NET].count, 1);

  const int64_t netId = built->nodes[NET].nodes[0].id;
  struct ncclTopoNode* gpu = built->nodes[GPU].nodes + 3;
  ASSERT_NE(gpu->gpu.parent, nullptr);
  ASSERT_NE(gpu->gpu.mloPart, NCCL_TOPO_UNDEF);

  enum ncclTopoGdrMode mode = ncclTopoGdrModeDisable;
  ASSERT_EQ(ncclTopoCheckGdr(built, gpu->gpu.rank, netId, /*read=*/0, &mode), ncclSuccess);
  ASSERT_NE(mode, ncclTopoGdrModeDisable) << "precondition: GDR is on before the path is diverted";

  gpu->paths[NET][0].type = PATH_PHB;
  ASSERT_EQ(ncclTopoCheckGdr(built, gpu->gpu.rank, netId, /*read=*/0, &mode), ncclSuccess);
  EXPECT_NE(mode, ncclTopoGdrModeDisable);
  EXPECT_EQ(gpu->gpu.parent->paths[NET][0].type, PATH_PXB);

  ncclTopoFree(built);
}

// The parent-DEV lookup is scoped to partitions. An ordinary GPU keeps reading its own entry, so
// a diverted path still means no GDR and the non-partitioned behaviour is left alone.
TEST_F(TopoTest, CheckGdr_NonMloPartGpuStillUsesOwnPath) {
  struct ncclTopoSystem* built = buildRailSystem(0xd2, /*nGpus=*/1, /*partitioned=*/false);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(ncclTopoComputePaths(built, nullptr), ncclSuccess);
  ASSERT_EQ(built->nodes[GPU].count, 1);
  ASSERT_EQ(built->nodes[NET].count, 1);

  const int64_t netId = built->nodes[NET].nodes[0].id;
  struct ncclTopoNode* gpu = built->nodes[GPU].nodes;
  ASSERT_EQ(gpu->gpu.mloPart, NCCL_TOPO_UNDEF);

  enum ncclTopoGdrMode mode = ncclTopoGdrModeDisable;
  ASSERT_EQ(ncclTopoCheckGdr(built, gpu->gpu.rank, netId, /*read=*/0, &mode), ncclSuccess);
  ASSERT_NE(mode, ncclTopoGdrModeDisable) << "precondition: GDR is on before the path is diverted";

  gpu->paths[NET][0].type = PATH_PHB;
  ASSERT_EQ(ncclTopoCheckGdr(built, gpu->gpu.rank, netId, /*read=*/0, &mode), ncclSuccess);
  EXPECT_EQ(mode, ncclTopoGdrModeDisable);

  ncclTopoFree(built);
}

// The ring search caps the GPU-to-NIC path type it accepts with this helper (ROCM-30906), so on a
// rail-optimised node the answer has to be the switch-local NIC and not the NIC of another rail.
TEST_F(TopoTest, GpuMaxLocalNetPath_RailSystemStopsAtTheLocalNic) {
  const uint64_t host = 0xe0;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  addRail(cpu, /*rail=*/0);
  addRail(cpu, /*rail=*/1);

  struct ncclTopoSystem* built = buildSystemWithPaths(host);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[GPU].count, 2);
  ASSERT_EQ(built->nodes[NET].count, 2);

  // Keyed on the dev index the two nodes of a rail share, so the case stays about rail identity
  // whatever order the GPUs and the NICs were appended in.
  for (int g = 0; g < built->nodes[GPU].count; g++) {
    struct ncclTopoLinkList* paths = built->nodes[GPU].nodes[g].paths[NET];
    for (int n = 0; n < built->nodes[NET].count; n++) {
      SCOPED_TRACE(testing::Message() << "gpu " << g << " net " << n);
      const bool ownRail = built->nodes[NET].nodes[n].net.dev == built->nodes[GPU].nodes[g].gpu.dev;
      EXPECT_EQ(paths[n].type, ownRail ? PATH_PXB : PATH_PHB);
    }
  }

  int maxPath = PATH_DIS;
  ASSERT_EQ(ncclTopoGetGpuMaxLocalNetPath(built, &maxPath), ncclSuccess);
  EXPECT_EQ(maxPath, PATH_PXB);

  ncclTopoFree(built);
}

// The cap holds for the whole node, so the GPU with the worst of the local NICs sets it: here one
// GPU shares a switch with a NIC and the other reaches both NICs across the CPU only.
TEST_F(TopoTest, GpuMaxLocalNetPath_GpuOffTheRailsWidensTheBound) {
  const uint64_t host = 0xe1;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  addRail(cpu, /*rail=*/0);
  addGpuPci(cpu, "0000:1c:00.0", "gfx942", /*rank=*/1, /*dev=*/1);

  struct ncclTopoSystem* built = buildSystemWithPaths(host);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[GPU].count, 2);
  ASSERT_EQ(built->nodes[NET].count, 1);

  int maxPath = PATH_DIS;
  ASSERT_EQ(ncclTopoGetGpuMaxLocalNetPath(built, &maxPath), ncclSuccess);
  EXPECT_EQ(maxPath, PATH_PHB);

  ncclTopoFree(built);
}

// A relay entry has to raise the answer even where every GPU has a nearer NIC of its own, or a
// search bounded by it would never reach the step at which relays are accepted. The relay is
// written by hand, the way CheckGdr_MloPartIgnoresDivertedPartitionPath stands in for a diverted
// path, because PXN is off by default in this tree and NCCL_PARAM caches that per process.
TEST_F(TopoTest, GpuMaxLocalNetPath_RelayRaisesTheBoundToPxn) {
  const uint64_t host = 0xe2;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  addRail(cpu, /*rail=*/0);
  addRail(cpu, /*rail=*/1);

  struct ncclTopoSystem* built = buildSystemWithPaths(host);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[GPU].count, 2);
  ASSERT_EQ(built->nodes[NET].count, 2);

  for (int g = 0; g < built->nodes[GPU].count; g++) {
    struct ncclTopoLinkList* paths = built->nodes[GPU].nodes[g].paths[NET];
    SCOPED_TRACE(testing::Message() << "gpu " << g);
    for (int n = 0; n < built->nodes[NET].count; n++)
      if (paths[n].type == PATH_PHB) paths[n].type = PATH_PXN;
    EXPECT_EQ(std::min(paths[0].type, paths[1].type), PATH_PXB);
    EXPECT_EQ(std::max(paths[0].type, paths[1].type), PATH_PXN);
  }

  int maxPath = PATH_DIS;
  ASSERT_EQ(ncclTopoGetGpuMaxLocalNetPath(built, &maxPath), ncclSuccess);
  EXPECT_EQ(maxPath, PATH_PXN);

  ncclTopoFree(built);
}

// With no NIC in the system a GPU never gets a paths[NET] array at all, since those are allocated
// as the search from each NIC reaches a node. The answer then has to be PATH_LOC, where PATH_DIS
// would leave the ring search with no bound whatsoever.
TEST_F(TopoTest, GpuMaxLocalNetPath_NodeWithoutNicsBoundsNothing) {
  const uint64_t host = 0xe3;
  struct ncclXmlNode* cpu = addSystemCpu(host);
  addGpuPci(cpu, "0000:0c:00.0", "gfx942", /*rank=*/0, /*dev=*/0);

  struct ncclTopoSystem* built = buildSystemWithPaths(host);
  ASSERT_NE(built, nullptr);
  ASSERT_EQ(built->nodes[GPU].count, 1);
  ASSERT_EQ(built->nodes[NET].count, 0);

  int maxPath = PATH_DIS;
  ASSERT_EQ(ncclTopoGetGpuMaxLocalNetPath(built, &maxPath), ncclSuccess);
  EXPECT_EQ(maxPath, PATH_LOC);

  ncclTopoFree(built);
}

// The ring graph of a two-rail node has to come out on the NIC of each GPU (ROCM-30906). The
// search widens the GPU-to-NIC path type before it tries the two-NIC form of a ring, so without a
// cap it settles on PATH_PHB, where both legs fit on one NIC and the second GPU loses GDR.
TEST_F(TopoTest, RingSearch_TwoRailNode_StaysOnTheLocalNicPathType) {
  // NCCL_CROSS_NIC decides whether the cap applies at all and NCCL_PARAM caches it for the whole
  // process, so the body runs in a child with the default value pinned.
  RUN_ISOLATED_TEST_WITH_ENV(
      "RingSearch_TwoRailNode_StaysOnTheLocalNicPathType",
      [this]() {
        const uint64_t host = 0xe4;
        struct ncclXmlNode* cpu = addSystemCpu(host);
        struct ncclXmlNode* gpu0 = addRail(cpu, /*rail=*/0);
        struct ncclXmlNode* gpu1 = addRail(cpu, /*rail=*/1);

        // A ring needs a way from GPU to GPU the cap admits as well, which XGMI is and PCIe is not.
        char gpu0Bus[32], gpu1Bus[32];
        railGpuBusId(0, gpu0Bus, sizeof(gpu0Bus));
        railGpuBusId(1, gpu1Bus, sizeof(gpu1Bus));
        addGpuLink(gpu0, gpu1Bus, /*count=*/8);
        addGpuLink(gpu1, gpu0Bus, /*count=*/8);

        struct ncclTopoSystem* built = buildSystemWithPaths(host);
        ASSERT_NE(built, nullptr);
        ASSERT_EQ(built->nodes[GPU].count, 2);
        ASSERT_EQ(built->nodes[NET].count, 2);

        // Two nodes of two GPUs, which is what makes the search look for net legs at all.
        built->nRanks = 2 * built->nodes[GPU].count;
        built->inter = 1;

        struct ncclTopoGraph ring;
        memset(&ring, 0, sizeof(ring));
        ring.id = 0;
        ring.pattern = NCCL_TOPO_PATTERN_RING;
        ring.minChannels = 1;
        ring.maxChannels = MAXCHANNELS / 2;

        ASSERT_EQ(ncclTopoSearchInit(built), ncclSuccess);
        ASSERT_EQ(ncclTopoCompute(built, &ring), ncclSuccess);
        // Both legs of the ring on the NIC of their own GPU: the only form the cap leaves, and one
        // no preset order can produce, unlike typeInter which starts at PATH_PXB.
        EXPECT_EQ(ring.crossNic, 2);
        EXPECT_EQ(ring.typeInter, PATH_PXB);

        ncclTopoFree(built);
      },
      {{"NCCL_CROSS_NIC", "2"}});
}

#else // !(__HIP_PLATFORM_AMD__ || __HIPCC__)

// ncclTopoAddXGMI() is not built on non-HIP platforms, so register one skipped
// test instead so the suite still exists and links there.
TEST_F(TopoTest, XgmiTestsSkippedOnNonHipBuild) {
  GTEST_SKIP() << "ncclTopoAddXGMI is only built on HIP/AMD platforms";
}

#endif // __HIP_PLATFORM_AMD__ || __HIPCC__

} // namespace RcclUnitTesting
