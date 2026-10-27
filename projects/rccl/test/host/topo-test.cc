/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/topo.cc: building the topology system from XML, NIC fusion,
// local NIC/GPU queries and ncclTopoGetSystem, on topologies written as XML by TopoScene.h.

#include <gtest/gtest.h>

#include <csignal>

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "../common/ScopedEnv.hpp"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include TOPO_CC_PATH

#include "TopoScene.h"

using namespace topo_scene;

// The NETDEVS policy lives in this TU's statics; the other graph TUs set it through here.
void TopoSetNetDevsPolicy(enum netDevsPolicy policy, int num) {
  int unused;
  ncclTopoGetNetDevsPolicy(nullptr, &unused);  // consume the call_once first
  netDevsPolicy = policy;
  netDevsPolicyNum = num;
}

namespace {

// ASan turns a SIGSEGV into a report plus exit(1), so KilledBySignal never matches under it.
#if RCCL_TEST_ASAN
#define DEATH_BY_SEGV ::testing::ExitedWithCode(1)
#else
#define DEATH_BY_SEGV ::testing::KilledBySignal(SIGSEGV)
#endif

constexpr int64_t kBus0 = 0x10000;  // GpuBus(0), as an int

class TopoMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetGraphFakes(); }
  void TearDown() override { ResetGraphFakes(); }

  static ncclTopoNode* Node(ncclTopoSystem* system, int type, int64_t localId) {
    ncclTopoNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, ncclTopoGetNode(system, &node, type, NCCL_TOPO_ID(0, localId)));
    return node;
  }

  static ncclTopoLink* Link(ncclTopoNode* from, ncclTopoNode* to, int type = -1) {
    for (int l = 0; l < from->nlinks; l++) {
      if (from->links[l].remNode == to && (type == -1 || from->links[l].type == type)) return from->links + l;
    }
    return nullptr;
  }

  static int64_t BusId(const std::string& busId) {
    int64_t id = 0;
    EXPECT_EQ(ncclSuccess, busIdToInt64(busId.c_str(), &id));
    return id;
  }
};

// --- Building the system from XML

TEST_F(TopoMicrotest, GetSystemFromXml_XgmiGpusHangOffTheirDevNodes) {
  SystemPtr system = BuildSystem(XgmiSystem(4));

  ASSERT_NE(nullptr, system);
  EXPECT_EQ(1, system->nodes[CPU].count);
  EXPECT_EQ(4, system->nodes[PCI].count);
  EXPECT_EQ(4, system->nodes[DEV].count);
  EXPECT_EQ(4, system->nodes[GPU].count);
  EXPECT_EQ(0, system->nodes[NVS].count);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE, system->romeTopoModelIdx);
  ncclTopoNode* gpu = system->nodes[GPU].nodes;
  EXPECT_EQ(0, gpu->gpu.rank);
  EXPECT_EQ(0, gpu->gpu.dev);
  EXPECT_EQ(1, gpu->gpu.gdrSupport);
  EXPECT_EQ(304, gpu->gpu.cu);
  EXPECT_EQ(60, gpu->gpu.cudaCompCap);
  EXPECT_STREQ("gfx942", gpu->gpu.gcn);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gpu->gpu.mloPart);
  ncclTopoNode* dev = Node(system.get(), DEV, kBus0);
  ASSERT_EQ(dev, gpu->gpu.parent);
  EXPECT_EQ(NCCL_TOPO_ID(0, NCCL_TOPO_GPU_LOCAL_ID(kBus0, 0)), gpu->id);
  EXPECT_STREQ("gfx942", dev->dev.gcn);
  EXPECT_EQ(60, dev->dev.cudaCompCap);
  EXPECT_EQ(1, dev->dev.nGpus);
  EXPECT_EQ(0x100274a1100274a1ull, dev->dev.device);
  EXPECT_FLOAT_EQ(LOC_BW, Link(dev, gpu)->bw);
  EXPECT_EQ(LINK_LOC, Link(gpu, dev)->type);
  ncclTopoLink* xgmi = Link(dev, Node(system.get(), DEV, BusId(GpuBus(3))));
  ASSERT_NE(nullptr, xgmi);
  EXPECT_EQ(LINK_NVL, xgmi->type);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, xgmi->bw);
  ncclTopoLink* pci = Link(dev, Node(system.get(), PCI, BusId(SwitchBus(0))));
  ASSERT_NE(nullptr, pci);
  EXPECT_EQ(LINK_PCI, pci->type);
  EXPECT_FLOAT_EQ(16 * 240 / 80.0, pci->bw);
}

TEST_F(TopoMicrotest, GetSystemFromXml_XgmiBandwidthFollowsTheGcnArch) {
  const std::pair<const char*, float> archs[] = {
    {"gfx90a", MI200_XGMI_WIDTH}, {"gfx942", GFX94X_XGMI_WIDTH}, {"gfx950", GFX95X_XGMI_WIDTH},
    {"gfx1250", GFX125X_XGMI_WIDTH}, {"gfx908", VEGA_XGMI_WIDTH}};
  for (const auto& arch : archs) {
    std::string links = Xgmi(GpuBus(1), 2);
    SystemPtr system = BuildSystem(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, links, "", arch.first) +
                                                  GpuPci(GpuBus(1), 1, 1, "", "", arch.first))));

    ncclTopoLink* link = Link(Node(system.get(), DEV, kBus0), Node(system.get(), DEV, BusId(GpuBus(1))));

    ASSERT_NE(nullptr, link) << arch.first;
    EXPECT_FLOAT_EQ(2 * arch.second, link->bw) << arch.first;
  }
}

TEST_F(TopoMicrotest, GetSystemFromXml_XgmiToCpuOrSwitchClassLinksBothWays) {
  std::string toCpu = Xgmi("0000:ff:00.0", 1, "0x068001");
  std::string toSwitch = Xgmi("fffffff:ffff:ff", 2, "0x068000");
  std::string missing = Xgmi("0000:99:00.0");
  SystemPtr system = BuildSystem(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, toCpu + toSwitch + missing) +
                                                GpuPci(GpuBus(1), 1, 1, toSwitch))));

  ASSERT_EQ(1, system->nodes[NVS].count);
  ncclTopoNode* nvs = system->nodes[NVS].nodes;
  ncclTopoNode* cpu = system->nodes[CPU].nodes;
  ncclTopoNode* dev0 = Node(system.get(), DEV, kBus0);
  ncclTopoNode* dev1 = Node(system.get(), DEV, BusId(GpuBus(1)));
  EXPECT_FLOAT_EQ(2 * GFX94X_XGMI_WIDTH, Link(dev0, nvs)->bw);
  EXPECT_FLOAT_EQ(2 * GFX94X_XGMI_WIDTH, Link(nvs, dev1)->bw);
  EXPECT_NE(nullptr, Link(dev0, cpu, LINK_NVL));
  EXPECT_NE(nullptr, Link(cpu, dev0, LINK_NVL));
  EXPECT_EQ(nullptr, Link(dev0, dev1));
}

TEST_F(TopoMicrotest, GetSystemFromXml_NeedsASystemNode) {
  XmlPtr xml = NewXml();
  ncclXmlNode* node = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), nullptr, "cpu", &node));

  BuildSystem(xml.get(), ncclInternalError);

  EXPECT_TRUE(Logged("system node not found"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_LocalHostHashMustBePresent) {
  BuildSystem(System(Cpu(0, "", "host_hash=\"0x5\"")), ncclInvalidArgument);

  EXPECT_TRUE(Logged("localHostHash = 0x0 not found"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_HostsGetSystemIdsAndOnlyLocalCpusConnect) {
  SystemPtr system = BuildSystem(System(Cpu(0, "") + Cpu(1, "") + Cpu(0, "", "host_hash=\"0x7\"")));

  ASSERT_EQ(3, system->nodes[CPU].count);
  EXPECT_EQ(2, system->nHosts);
  EXPECT_EQ(0u, system->hostHashes[0]);
  EXPECT_EQ(7u, system->hostHashes[1]);
  ncclTopoNode* remote = system->nodes[CPU].nodes + 2;
  EXPECT_EQ(1, NCCL_TOPO_ID_SYSTEM_ID(remote->id));
  EXPECT_EQ(0, remote->nlinks);
  ncclTopoNode* cpu0 = system->nodes[CPU].nodes;
  ASSERT_EQ(1, cpu0->nlinks);
  EXPECT_EQ(LINK_SYS, cpu0->links[0].type);
  EXPECT_FLOAT_EQ(AMD_ZEN12_BW, cpu0->links[0].bw);
}

// A rank-less GPU returns before its pci node is created, so the whole branch disappears.
TEST_F(TopoMicrotest, GetSystemFromXml_GpuWithoutRankDropsItsPciBranch) {
  std::string rankless = "<pci busid=\"0000:20:00.0\" class=\"0x120000\" link_speed=\"\" link_width=\"0\">"
                         "<gpu dev=\"1\"/></pci>";
  SystemPtr system = BuildSystem(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0)) + rankless)));

  EXPECT_EQ(1, system->nodes[GPU].count);
  EXPECT_EQ(1, system->nodes[PCI].count);
}

TEST_F(TopoMicrotest, GetSystemFromXml_ZeroWidthAndUnknownSpeedFallBack) {
  std::string pci = "<pci busid=\"0000:20:00.0\" class=\"0x060400\" link_speed=\"9 GT/s\" link_width=\"0\"/>";
  SystemPtr system = BuildSystem(System(Cpu(0, pci)));

  ncclTopoNode* sw = system->nodes[PCI].nodes;
  EXPECT_FLOAT_EQ(16 * 60 / 80.0, Link(sw, system->nodes[CPU].nodes)->bw);
  EXPECT_TRUE(Logged("could not find value of '9 GT/s'"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_NetAttributesAndDefaults) {
  std::string nets = Net("a", 0, "coll=\"1\"") +
                     "<net name=\"b\" dev=\"1\" speed=\"0\" port=\"2\" gdr=\"0\" rail=\"3\" plane=\"4\"/>";
  SystemPtr system = BuildSystem(System(Cpu(1, NicPci(NicBus(0), nets))));

  ASSERT_EQ(2, system->nodes[NET].count);
  ASSERT_EQ(1, system->nodes[NIC].count);
  ncclTopoNode* nic = system->nodes[NIC].nodes;
  EXPECT_EQ(NCCL_TOPO_ID(0, NCCL_TOPO_LOCAL_NIC_ID(1, BusId(NicBus(0)))), nic->id);
  ncclTopoNode* a = system->nodes[NET].nodes;
  ncclTopoNode* b = system->nodes[NET].nodes + 1;
  EXPECT_FLOAT_EQ(400000 / 8000.0, a->net.bw);
  EXPECT_EQ(1, a->net.port);
  EXPECT_EQ(1, a->net.gdrSupport);
  EXPECT_EQ(131072, a->net.maxChannels);
  EXPECT_EQ(1, a->net.collSupport);
  EXPECT_EQ(0x100u, a->net.asic);
  EXPECT_EQ(BusId(NicBus(0)), a->net.busId);
  EXPECT_EQ(0x15b3u, a->net.vendor);
  EXPECT_EQ(0x1021u, a->net.device);
  EXPECT_FLOAT_EQ(10000 / 8000.0, b->net.bw);  // speed 0 means unknown
  EXPECT_FLOAT_EQ(0, b->net.latency);
  EXPECT_EQ(MAXCHANNELS, b->net.maxChannels);
  EXPECT_EQ(0, b->net.collSupport);
  EXPECT_EQ((uint64_t)b->id, b->net.asic);  // no guid: the topology id stands in
  EXPECT_EQ(3, b->net.railId);
  EXPECT_EQ(4, b->net.planeId);
  EXPECT_EQ(a->net.pciId, b->net.pciId);  // both hashed from the same PCI bus id
  EXPECT_FLOAT_EQ(a->net.bw, Link(nic, a)->bw);
  EXPECT_EQ(LINK_NET, Link(a, nic)->type);
}

TEST_F(TopoMicrotest, GetSystemFromXml_CpuNicGinRmaAndDisabledNets) {
  std::string nic = "<nic>" + Net("a", 0) + Net("off", 1, "net=\"0\"") + "<net name=\"nodev\"/>" +
                    "<gin name=\"g\" dev=\"2\" speed=\"0\" rail=\"0\" plane=\"0\"/>" +
                    "<rma name=\"r\" dev=\"3\" speed=\"200000\" rail=\"0\" plane=\"0\"/><other/></nic>";
  SystemPtr system = BuildSystem(System(Cpu(0, nic + "<nic>" + Net("c", 4) + "</nic>")));

  ASSERT_EQ(1, system->nodes[NIC].count);
  ncclTopoNode* nicNode = system->nodes[NIC].nodes;
  EXPECT_EQ(NCCL_TOPO_ID(0, NCCL_TOPO_LOCAL_NIC_ID(0, 0)), nicNode->id);
  EXPECT_FLOAT_EQ(LOC_BW, Link(nicNode, system->nodes[CPU].nodes)->bw);
  ASSERT_EQ(2, system->nodes[NET].count);
  EXPECT_EQ(0, system->nodes[NET].nodes[0].net.dev);
  EXPECT_EQ(4, system->nodes[NET].nodes[1].net.dev);
  ASSERT_EQ(1, system->nodes[GIN].count);
  EXPECT_FLOAT_EQ(10000 / 8000.0, system->nodes[GIN].nodes[0].net.bw);
  ASSERT_EQ(1, system->nodes[RMA].count);
  EXPECT_FLOAT_EQ(25, system->nodes[RMA].nodes[0].net.bw);
  EXPECT_NE(system->nodes[NET].nodes[0].net.pciId, system->nodes[NET].nodes[1].net.pciId);
}

TEST_F(TopoMicrotest, GetSystemFromXml_GinAndRmaUnderAPciNicReadItsIds) {
  std::string nets = "<gin name=\"g\" dev=\"0\" speed=\"400000\" rail=\"1\" plane=\"2\"/>"
                     "<rma name=\"r\" dev=\"1\" speed=\"400000\" guid=\"0x55\" rail=\"1\" plane=\"2\"/>";
  SystemPtr system = BuildSystem(System(Cpu(0, NicPci(NicBus(0), nets))));

  ncclTopoNode* gin = system->nodes[GIN].nodes;
  ncclTopoNode* rma = system->nodes[RMA].nodes;
  EXPECT_EQ(0x15b3u, gin->net.vendor);
  EXPECT_EQ(0x1021u, rma->net.device);
  EXPECT_EQ(0x55u, rma->net.asic);
  EXPECT_EQ(1, gin->net.railId);
  EXPECT_EQ(2, rma->net.planeId);
  EXPECT_EQ(LINK_NET, Link(gin, system->nodes[NIC].nodes)->type);
}

TEST_F(TopoMicrotest, GetSystemFromXml_NetsNeedRailAndPlane) {
  for (const char* net : {"<net name=\"a\" dev=\"0\" plane=\"0\"/>", "<net name=\"a\" dev=\"0\" rail=\"0\"/>",
                          "<gin name=\"a\" dev=\"0\" plane=\"0\"/>", "<gin name=\"a\" dev=\"0\" rail=\"0\"/>",
                          "<rma name=\"a\" dev=\"0\" plane=\"0\"/>", "<rma name=\"a\" dev=\"0\" rail=\"0\"/>"}) {
    BuildSystem(System(Cpu(0, std::string("<nic>") + net + "</nic>")), ncclInternalError);
  }
}

TEST_F(TopoMicrotest, GetSystemFromXml_PcilinksJoinSwitchesWithLocLinks) {
  std::string a = "<pci busid=\"0000:01:00.0\" class=\"0x060400\" link_speed=\"\" link_width=\"16\">"
                  "<pcilink target=\"0000:02:00.0\"/><pcilink target=\"0000:09:00.0\"/></pci>";
  std::string b = "<pci busid=\"0000:02:00.0\" class=\"0x060400\" link_speed=\"\" link_width=\"16\"/>";
  SystemPtr system = BuildSystem(System(Cpu(0, a + b)));

  ncclTopoNode* swA = Node(system.get(), PCI, 0x1000);
  ncclTopoNode* swB = Node(system.get(), PCI, 0x2000);
  ASSERT_NE(nullptr, Link(swA, swB));
  EXPECT_EQ(LINK_LOC, Link(swA, swB)->type);
  EXPECT_EQ(nullptr, Link(swB, swA));
}

TEST_F(TopoMicrotest, GetSystemFromXml_PcilinkOutsideAPciSwitchFails) {
  XmlPtr xml = MustParseXml(System(Cpu(0, GpuPci(GpuBus(0), 0, 0))));
  ncclXmlNode* pci = nullptr;
  ncclXmlNode* link = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml.get(), "pci", &pci));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), pci, "pcilink", &link));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(link, "target", "0000:02:00.0"));

  BuildSystem(xml.get(), ncclInternalError);

  EXPECT_TRUE(Logged("Add PCI Link error : could not find PCI SW"));
}

// --- MLOPart partitions

std::string PartitionedPci(const std::string& gpuAttrs0, const std::string& gpuAttrs1, const std::string& links0 = "") {
  return "<pci busid=\"" + GpuBus(0) + "\" class=\"0x120000\" vendor=\"0x1002\" device=\"0x74a1\" " +
         "subsystem_vendor=\"0x1002\" subsystem_device=\"0x74a1\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">" +
         Gpu(0, 0, links0, gpuAttrs0) + Gpu(1, 1, Xgmi(GpuBus(1)), gpuAttrs1) + "</pci>";
}

TEST_F(TopoMicrotest, GetSystemFromXml_MloPartsGetSiblingDevNodes) {
  SystemPtr system = BuildSystem(System(Cpu(0, PartitionedPci("mlopart=\"0\"", "mlopart=\"1\"", Xgmi(GpuBus(1))) +
                                                GpuPci(GpuBus(1), 2, 2))));

  ASSERT_EQ(3, system->nodes[DEV].count);
  ncclTopoNode* part0 = Node(system.get(), DEV, kBus0 | NCCL_TOPO_MLOPART(0));
  ncclTopoNode* part1 = Node(system.get(), DEV, kBus0 | NCCL_TOPO_MLOPART(1));
  ASSERT_NE(nullptr, part0);
  ASSERT_NE(nullptr, part1);
  EXPECT_FLOAT_EQ(MLOPART_LOC_BW, Link(part0, part1)->bw);
  EXPECT_FLOAT_EQ(MLOPART_LOC_BW, Link(part1, part0)->bw);
  EXPECT_EQ(1, system->nodes[GPU].nodes[1].gpu.mloPart);
  ncclTopoNode* other = Node(system.get(), DEV, BusId(GpuBus(1)));
  EXPECT_NE(nullptr, Link(part0, other));
  EXPECT_NE(nullptr, Link(part1, other));  // each partition is its own primary
}

TEST_F(TopoMicrotest, GetSystemFromXml_UnsplitMloPartsShareOneDev) {
  g_params["TOPO_SPLIT_MLOPART"] = 0;
  SystemPtr system = BuildSystem(System(Cpu(0, PartitionedPci("mlopart=\"0\"", "mlopart=\"1\"", Xgmi(GpuBus(1))) +
                                                GpuPci(GpuBus(1), 2, 2))));

  ASSERT_EQ(2, system->nodes[DEV].count);
  ncclTopoNode* dev = Node(system.get(), DEV, kBus0);
  EXPECT_EQ(2, dev->dev.nGpus);
  EXPECT_EQ(NCCL_TOPO_ID(0, NCCL_TOPO_GPU_LOCAL_ID(kBus0, 1)), system->nodes[GPU].nodes[1].id);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH,
                  Link(dev, Node(system.get(), DEV, BusId(GpuBus(1))))->bw);  // second GPU not primary
}

TEST_F(TopoMicrotest, GetSystemFromXml_SameMloPartTwiceSharesTheDevAndItsLinks) {
  SystemPtr system = BuildSystem(System(Cpu(0, PartitionedPci("mlopart=\"0\"", "mlopart=\"0\"", Xgmi(GpuBus(1))) +
                                                GpuPci(GpuBus(1), 2, 2))));

  ncclTopoNode* dev = Node(system.get(), DEV, kBus0 | NCCL_TOPO_MLOPART(0));
  EXPECT_EQ(2, dev->dev.nGpus);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, Link(dev, Node(system.get(), DEV, BusId(GpuBus(1))))->bw);
}

TEST_F(TopoMicrotest, GetSystemFromXml_MloPartIndexMustFitInThreeBits) {
  BuildSystem(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, "", "mlopart=\"8\""))), ncclInternalError);

  EXPECT_TRUE(Logged("MLOPart index 8 out of range (max 7)"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_BusIdMayNotUseTheMloPartBits) {
  BuildSystem(System(Cpu(0, GpuPci("10000:10:00.0", 0, 0))), ncclInternalError);

  EXPECT_TRUE(Logged("non-zero bits in MLOPart mask"));
}

// HIP numbers partition p of a device as PCI function p, so an XGMI target at function 1 is that device's DEV for
// partition 1.
TEST_F(TopoMicrotest, GetSystemFromXml_XgmiTargetFunctionSelectsTheMloPartDev) {
  std::string links = Xgmi("0000:10:00.1") + Xgmi("0000:10:00.9");
  SystemPtr system = BuildSystem(System(Cpu(0, PartitionedPci("mlopart=\"0\"", "mlopart=\"1\"") +
                                                GpuPci(GpuBus(1), 2, 2, links))));

  ncclTopoNode* from = Node(system.get(), DEV, BusId(GpuBus(1)));
  ncclTopoNode* part1 = Node(system.get(), DEV, kBus0 | NCCL_TOPO_MLOPART(1));
  ASSERT_NE(nullptr, Link(from, part1));
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, Link(from, part1)->bw);  // the .9 target does not alias onto it
  EXPECT_EQ(nullptr, Link(from, Node(system.get(), DEV, kBus0 | NCCL_TOPO_MLOPART(0))));
  EXPECT_EQ(3, from->nlinks);  // LOC to its GPU, PCI up, and the one XGMI link
}

// --- C2C (AMD XML files cannot carry c2c, so it is added to the parsed tree)

void AddC2c(ncclXml* xml, int gpuIndex, int count, int bw) {
  ncclXmlNode* gpu = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml, "gpu", &gpu));
  for (int i = 0; i < gpuIndex; i++) ASSERT_EQ(ncclSuccess, xmlFindNextTag(xml, "gpu", gpu, &gpu));
  ncclXmlNode* c2c = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml, gpu, "c2c", &c2c));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "count", count));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "bw", bw));
}

TEST_F(TopoMicrotest, GetSystemFromXml_C2cLinksTheDevToItsCpu) {
  XmlPtr xml = MustParseXml(System(Cpu(0, GpuPci(GpuBus(0), 0, 0))));
  AddC2c(xml.get(), 0, 4, 1000);

  SystemPtr system = BuildSystem(xml.get());

  ncclTopoNode* dev = system->nodes[DEV].nodes;
  ncclTopoNode* cpu = system->nodes[CPU].nodes;
  ASSERT_NE(nullptr, Link(dev, cpu, LINK_C2C));
  EXPECT_FLOAT_EQ(4, Link(dev, cpu, LINK_C2C)->bw);
  EXPECT_FLOAT_EQ(4, Link(cpu, dev, LINK_C2C)->bw);
  EXPECT_EQ(0, system->nodes[CXB].count);
}

TEST_F(TopoMicrotest, GetSystemFromXml_SplitDevicesShareOneC2cBridge) {
  XmlPtr xml = MustParseXml(System(Cpu(0, PartitionedPci("mlopart=\"0\"", "mlopart=\"1\""))));
  AddC2c(xml.get(), 0, 2, 3000);
  AddC2c(xml.get(), 1, 2, 3000);

  SystemPtr system = BuildSystem(xml.get());

  ASSERT_EQ(1, system->nodes[CXB].count);
  ncclTopoNode* bridge = system->nodes[CXB].nodes;
  EXPECT_EQ(NCCL_TOPO_ID(0, kBus0), bridge->id);
  EXPECT_FLOAT_EQ(6, Link(bridge, system->nodes[CPU].nodes)->bw);
  EXPECT_EQ(LINK_C2C, Link(system->nodes[DEV].nodes + 1, bridge)->type);
  EXPECT_EQ(3, bridge->nlinks);
}

TEST_F(TopoMicrotest, GetSystemFromXml_C2cWithoutDevFails) {
  XmlPtr xml = MustParseXml(System(Cpu(0, "<pci busid=\"0000:20:00.0\" class=\"0x120000\" link_speed=\"\" "
                                          "link_width=\"16\"><gpu dev=\"1\"/></pci>")));
  AddC2c(xml.get(), 0, 1, 1000);

  BuildSystem(xml.get(), ncclInternalError);

  EXPECT_TRUE(Logged("Add C2c error : could not find DEV for GPU 0000:20:00.0"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_XgmiWithoutDevFails) {
  BuildSystem(System(Cpu(0, "<pci busid=\"0000:20:00.0\" class=\"0x120000\" link_speed=\"\" link_width=\"16\">"
                            "<gpu dev=\"1\">" +
                            Xgmi(GpuBus(0)) + "</gpu></pci>")),
              ncclInternalError);

  EXPECT_TRUE(Logged("Add XGMI error : could not find DEV for GPU 0000:20:00.0"));
}

TEST_F(TopoMicrotest, GetSystemFromXml_XgmiOutsideAGpuUsesThePciBusId) {
  XmlPtr xml = MustParseXml(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1))));
  ncclXmlNode* pci = nullptr;
  ncclXmlNode* xgmi = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTagKv(xml.get(), "pci", &pci, "busid", GpuBus(0).c_str()));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), pci, "xgmi", &xgmi));
  for (const auto& kv : {std::make_pair("target", GpuBus(1)), std::make_pair("count", std::string("1")),
                         std::make_pair("tclass", std::string("0x120000"))}) {
    ASSERT_EQ(ncclSuccess, xmlSetAttr(xgmi, kv.first, kv.second.c_str()));
  }

  SystemPtr system = BuildSystem(xml.get());

  EXPECT_NE(nullptr, Link(Node(system.get(), DEV, kBus0), Node(system.get(), DEV, BusId(GpuBus(1)))));
}

// --- BCM PEX switches

std::string BcmSwitch(const std::string& busId, const char* subsysDevice, const std::string& body) {
  return "<pci busid=\"" + busId + "\" class=\"0x060400\" vendor=\"0x1000\" device=\"" +
         (std::string(subsysDevice) == "0xa000" ? "0xc010" : "0xc030") + "\" subsystem_vendor=\"0x1000\" " +
         "subsystem_device=\"" + subsysDevice + "\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">" + body + "</pci>";
}

TEST_F(TopoMicrotest, GetSystemFromXml_FlattensTwoLevelBcmSwitches) {
  struct Gen {
    const char* top;
    const char* sub;
  };
  for (const Gen& gen : {Gen{"0xa000", "0xa000"}, Gen{"0x0000", "0x1000"}}) {
    std::string subs = BcmSwitch("0000:02:00.0", gen.sub, GpuPci(GpuBus(0), 0, 0)) +
                       BcmSwitch("0000:03:00.0", gen.sub, GpuPci(GpuBus(1), 1, 1));
    SystemPtr system = BuildSystem(System(Cpu(0, BcmSwitch("0000:01:00.0", gen.top, subs))));

    ASSERT_EQ(1, system->nodes[PCI].count) << gen.sub;
    ncclTopoNode* top = system->nodes[PCI].nodes;
    EXPECT_EQ(0xffffu, top->pci.device & 0xffff);
    EXPECT_EQ(3, top->nlinks);  // the CPU and both devices
    ncclTopoNode* dev = Node(system.get(), DEV, kBus0);
    EXPECT_EQ(top, Link(dev, top)->remNode);
  }
}

TEST_F(TopoMicrotest, GetSystemFromXml_KeepsBcmSubSwitchesOfAnotherGeneration) {
  std::string subs = BcmSwitch("0000:02:00.0", "0x1000", GpuPci(GpuBus(0), 0, 0));
  SystemPtr system = BuildSystem(System(Cpu(0, BcmSwitch("0000:01:00.0", "0xa000", subs))));

  EXPECT_EQ(2, system->nodes[PCI].count);
}

// --- CPUs

TEST_F(TopoMicrotest, InterCpuBw_DependsOnArchVendorAndModel) {
  struct Case {
    int arch, vendor, model;
    float bw;
  };
  const Case cases[] = {
    {NCCL_TOPO_CPU_ARCH_POWER, 0, 0, P9_BW},
    {NCCL_TOPO_CPU_ARCH_ARM, 0, 0, ARM_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_ERP, ERP_QPI_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_SRP, SRP_QPI_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_SKL, SKL_QPI_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_BDW, BDW_QPI_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ZEN5, AMD_ZEN5_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ZEN34, AMD_ZEN34_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ROME, AMD_ZEN12_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_ZHAOXIN, NCCL_TOPO_CPU_MODEL_YONGFENG, YONGFENG_ZPI_BW},
    {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_ZHAOXIN, NCCL_TOPO_UNDEF, ZPI_BW},
    {NCCL_TOPO_CPU_ARCH_UNDEF, 0, 0, LOC_BW},
  };
  for (const Case& c : cases) {
    ncclTopoNode cpu{};
    cpu.cpu.arch = c.arch;
    cpu.cpu.vendor = c.vendor;
    cpu.cpu.model = c.model;
    float bw = 0;

    ASSERT_EQ(ncclSuccess, ncclTopoGetInterCpuBw(&cpu, &bw));

    EXPECT_FLOAT_EQ(c.bw, bw) << c.arch << "/" << c.vendor << "/" << c.model;
  }
}

struct CpuModel {
  const char* attrs;
  int arch, vendor, model;
};

TEST_F(TopoMicrotest, AddCpu_DecodesArchVendorAndModel) {
  const CpuModel cases[] = {
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0xcf\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_ERP},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0xad\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_ERP},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0xae\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_ERP},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0xaf\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_ERP},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0x8f\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_SRP},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0x55\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_SKL},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"0x54\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_BDW},
    {"arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"5\" modelid=\"0xcf\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_INTEL, NCCL_TOPO_CPU_MODEL_INTEL_BDW},
    {"arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"143\" modelid=\"49\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ROME},
    {"arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"143\" modelid=\"48\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ZEN},
    {"arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"175\" modelid=\"0\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_AMD, NCCL_TOPO_CPU_MODEL_AMD_ROME},
    {"arch=\"x86_64\" vendor=\"CentaurHauls\" familyid=\"7\" modelid=\"0x5b\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_ZHAOXIN, NCCL_TOPO_CPU_MODEL_YONGFENG},
    {"arch=\"x86_64\" vendor=\"  Shanghai  \" familyid=\"7\" modelid=\"0x3b\"", NCCL_TOPO_CPU_ARCH_X86,
     NCCL_TOPO_CPU_VENDOR_ZHAOXIN, NCCL_TOPO_UNDEF},
    {"arch=\"x86_64\" vendor=\"Other\"", NCCL_TOPO_CPU_ARCH_X86, 0, NCCL_TOPO_UNDEF},
    {"arch=\"arm64\"", NCCL_TOPO_CPU_ARCH_ARM, NCCL_TOPO_UNDEF, NCCL_TOPO_UNDEF},
    {"arch=\"ppc64\"", NCCL_TOPO_CPU_ARCH_POWER, NCCL_TOPO_UNDEF, NCCL_TOPO_UNDEF},
    {"arch=\"riscv\"", NCCL_TOPO_CPU_ARCH_UNDEF, NCCL_TOPO_UNDEF, NCCL_TOPO_UNDEF},
  };
  for (const CpuModel& c : cases) {
    SystemPtr system = BuildSystem(System(std::string("<cpu numaid=\"0\" ") + c.attrs + "/>"));

    ncclTopoNode* cpu = system->nodes[CPU].nodes;
    EXPECT_EQ(c.arch, cpu->cpu.arch) << c.attrs;
    EXPECT_EQ(c.vendor, cpu->cpu.vendor) << c.attrs;
    EXPECT_EQ(c.model, cpu->cpu.model) << c.attrs;
  }
}

// The Rome/Zen classification overwrites the Zen generation, so sockets get Zen 1/2 bandwidth.
// Pins AICOMRCCL-3024; a fix flips this pin.
TEST_F(TopoMicrotest, AddCpu_KnownBug_AmdZenGenerationIsOverwrittenSoSocketsGetZen12Bandwidth) {
  const char* zen5 = "arch=\"x86_64\" vendor=\"AuthenticAMD\" familyid=\"26\" modelid=\"2\"";
  SystemPtr system =
    BuildSystem(System(std::string("<cpu numaid=\"0\" ") + zen5 + "/><cpu numaid=\"1\" " + zen5 + "/>"));

  ncclTopoNode* cpu = system->nodes[CPU].nodes;
  EXPECT_EQ(NCCL_TOPO_CPU_MODEL_AMD_ZEN, cpu->cpu.model);
  EXPECT_FLOAT_EQ(AMD_ZEN12_BW, cpu->links[0].bw);
}

TEST_F(TopoMicrotest, AddCpu_AffinityHonorsTheOffset) {
  SystemPtr system = BuildSystem(System(Cpu(0, "", "affinity_offset=\"8\"") + "<cpu numaid=\"1\" arch=\"arm64\"/>"));

  ncclAffinity& affinity = system->nodes[CPU].nodes[0].cpu.affinity;
  EXPECT_EQ(4, ncclOsCpuCount(affinity));
  EXPECT_TRUE(ncclOsCpuIsSet(affinity, 8));
  EXPECT_TRUE(ncclOsCpuIsSet(affinity, 11));
  EXPECT_EQ(0, ncclOsCpuCount(system->nodes[CPU].nodes[1].cpu.affinity));
}

// --- Node and link primitives

class TopoNodeMicrotest : public TopoMicrotest {
 protected:
  std::unique_ptr<ncclTopoSystem, FreeDeleter> system_{static_cast<ncclTopoSystem*>(calloc(1, sizeof(ncclTopoSystem)))};

  ncclTopoNode* Create(int type, int64_t id) {
    ncclTopoNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, ncclTopoCreateNode(system_.get(), &node, type, id));
    return node;
  }
};

TEST_F(TopoNodeMicrotest, CreateNode_InitializesTypeSpecificFields) {
  ncclTopoNode* gpu = Create(GPU, 1);
  ncclTopoNode* cpu = Create(CPU, 2);
  ncclTopoNode* net = Create(NET, 3);
  ncclTopoNode* gin = Create(GIN, 4);
  ncclTopoNode* rma = Create(RMA, 4);
  ncclTopoNode* dev = Create(DEV, 5);

  EXPECT_EQ(GPU, gpu->type);
  EXPECT_EQ(1, gpu->id);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gpu->gpu.dev);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gpu->gpu.rank);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gpu->gpu.cudaCompCap);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gpu->gpu.mloPart);
  EXPECT_EQ(NCCL_TOPO_UNDEF, cpu->cpu.arch);
  EXPECT_EQ(NCCL_TOPO_UNDEF, cpu->cpu.vendor);
  EXPECT_EQ(NCCL_TOPO_UNDEF, cpu->cpu.model);
  EXPECT_EQ(UINT64_MAX, net->net.vendor);
  EXPECT_EQ(UINT64_MAX, net->net.device);
  EXPECT_EQ(NCCL_TOPO_UNDEF, net->net.port);
  EXPECT_EQ(NCCL_TOPO_UNDEF, net->net.railId);
  EXPECT_EQ(NCCL_TOPO_UNDEF, net->net.planeId);
  EXPECT_EQ(NCCL_TOPO_UNDEF, gin->net.railId);
  EXPECT_EQ(NCCL_TOPO_UNDEF, rma->net.planeId);
  EXPECT_EQ(NCCL_TOPO_UNDEF, dev->dev.dev);
  EXPECT_EQ(NCCL_TOPO_UNDEF, dev->dev.cudaCompCap);
}

TEST_F(TopoNodeMicrotest, CreateNode_FailsPastMaxNodes) {
  for (int n = 0; n < NCCL_TOPO_MAX_NODES; n++) Create(PCI, n);
  ncclTopoNode* node = nullptr;

  EXPECT_EQ(ncclInternalError, ncclTopoCreateNode(system_.get(), &node, PCI, NCCL_TOPO_MAX_NODES));

  EXPECT_TRUE(Logged("tried to create too many nodes of type 1"));
}

TEST_F(TopoNodeMicrotest, ConnectNodes_AggregatesPerTypeAndSortsByBandwidth) {
  ncclTopoNode* a = Create(PCI, 0);
  ncclTopoNode* b = Create(PCI, 1);
  ncclTopoNode* c = Create(PCI, 2);

  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(a, b, LINK_PCI, 10));
  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(a, c, LINK_NVL, 20));
  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(a, b, LINK_PCI, 15));
  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(a, b, LINK_NVL, 1));

  ASSERT_EQ(3, a->nlinks);
  EXPECT_EQ(b, a->links[0].remNode);
  EXPECT_FLOAT_EQ(25, a->links[0].bw);
  EXPECT_EQ(c, a->links[1].remNode);
  EXPECT_EQ(LINK_NVL, a->links[2].type);
  EXPECT_FLOAT_EQ(1, a->links[2].bw);
}

TEST_F(TopoNodeMicrotest, ConnectNodes_FailsPastMaxLinks) {
  ncclTopoNode* hub = Create(CPU, 0);
  for (int n = 0; n < NCCL_TOPO_MAX_NODES; n++) {
    ncclTopoNode* leaf = Create(PCI, n);
    for (int type : {LINK_LOC, LINK_NVL, LINK_PCI, LINK_SYS}) {
      ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(hub, leaf, type, 1));
    }
  }
  ASSERT_EQ(NCCL_TOPO_MAX_LINKS, hub->nlinks);

  EXPECT_EQ(ncclInternalError, ncclTopoConnectNodes(hub, Create(NIC, 0), LINK_PCI, 1));

  EXPECT_TRUE(Logged("too many Topo links (max 576)"));
}

TEST_F(TopoNodeMicrotest, RemoveNode_RefusesWhilePathsExist) {
  ncclTopoNode* node = Create(PCI, 0);
  ncclTopoLinkList dummy{};
  node->paths[CPU] = &dummy;

  EXPECT_EQ(ncclInternalError, ncclTopoRemoveNode(system_.get(), PCI, 0));

  node->paths[CPU] = nullptr;
  EXPECT_TRUE(Logged("Cannot remove topology node 1/0 while paths are computed"));
  EXPECT_EQ(1, system_->nodes[PCI].count);
}

TEST_F(TopoNodeMicrotest, RemoveNode_DropsLinksAndRepointsShiftedNodes) {
  SystemPtr system = BuildSystem(XgmiSystem(3));
  ncclTopoNode* gpu1 = system->nodes[GPU].nodes + 1;
  int64_t dev2Id = system->nodes[DEV].nodes[2].id;

  ASSERT_EQ(ncclSuccess, ncclTopoRemoveNode(system.get(), DEV, 0));

  ASSERT_EQ(2, system->nodes[DEV].count);
  EXPECT_EQ(nullptr, system->nodes[GPU].nodes[0].gpu.parent);
  EXPECT_EQ(system->nodes[DEV].nodes + 0, gpu1->gpu.parent);
  EXPECT_EQ(system->nodes[DEV].nodes + 0, gpu1->links[0].remNode);
  ncclTopoNode* dev = system->nodes[DEV].nodes + 0;  // was DEV 1
  EXPECT_EQ(dev2Id, system->nodes[DEV].nodes[1].id);
  for (int l = 0; l < dev->nlinks; l++) {
    if (dev->links[l].type == LINK_NVL) EXPECT_EQ(system->nodes[DEV].nodes + 1, dev->links[l].remNode);
  }
  EXPECT_EQ(3, dev->nlinks);  // its GPU, its switch, and DEV 2
}

TEST_F(TopoNodeMicrotest, GetDevNodes_MatchesAcrossMloPartsUpToEight) {
  const int64_t base = 0x10000;
  for (int p = 0; p < NCCL_TOPO_MLOPART_DEV_MAX; p++) Create(DEV, base | NCCL_TOPO_MLOPART(p));
  Create(DEV, 0x20000);
  ncclTopoNode* nodes[NCCL_TOPO_MLOPART_DEV_MAX];
  int n = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoGetDevNodes(system_.get(), base, nodes, &n));
  EXPECT_EQ(NCCL_TOPO_MLOPART_DEV_MAX, n);
  EXPECT_EQ(system_->nodes[DEV].nodes + 7, nodes[7]);
  ASSERT_EQ(ncclSuccess, ncclTopoGetDevNodes(system_.get(), 0x20000, nullptr, &n));
  EXPECT_EQ(1, n);

  Create(DEV, base);
  EXPECT_EQ(ncclInternalError, ncclTopoGetDevNodes(system_.get(), base, nullptr, &n));
  EXPECT_TRUE(Logged("too many DEV nodes for busId 10000 (max 8)"));
}

TEST_F(TopoMicrotest, SortSystem_MovesThePciUplinkLast) {
  std::string gpu = "<pci busid=\"" + GpuBus(0) + "\" class=\"0x120000\" link_speed=\"32.0 GT/s PCIe\" "
                    "link_width=\"4\">" + Gpu(0, 0) + "</pci>";
  SystemPtr system = BuildSystem(System(Cpu(0, Pci(SwitchBus(0), gpu))));

  ncclTopoNode* sw = system->nodes[PCI].nodes;
  ASSERT_EQ(2, sw->nlinks);
  EXPECT_EQ(system->nodes[DEV].nodes, sw->links[0].remNode);
  EXPECT_EQ(system->nodes[CPU].nodes, sw->links[1].remNode);  // higher bw, but up
}

TEST_F(TopoMicrotest, Print_LogsEveryNodeAndLink) {
  std::string nic = NicPci(NicBus(0), Net("a", 0));
  std::string pcilink = "<pci busid=\"0000:f0:00.0\" class=\"0x060400\" link_speed=\"\" link_width=\"16\">"
                        "<pcilink target=\"" + SwitchBus(0) + "\"/></pci>";
  SystemPtr system = BuildSystem(System(Cpu(0, Pci(SwitchBus(0), PartitionedPci("mlopart=\"0\"", "") + nic) + pcilink) +
                                        Cpu(1, "")));

  ASSERT_EQ(ncclSuccess, ncclTopoPrint(system.get()));

  EXPECT_TRUE(Logged("=== System : maxBw"));
  EXPECT_TRUE(Logged("CPU/0-0 (1/2/5)"));
  EXPECT_TRUE(Logged("+ PCI[48.0] - DEV/0-1000010000 (100274a1100274a1)"));
  EXPECT_TRUE(Logged("+ LOC[5000.0] - GPU/0-10000"));
  EXPECT_TRUE(Logged("+ NET[50.0] - NET/0-0 (0/100/1/50.000000)"));
  EXPECT_TRUE(Logged("+ SYS[16.0] - CPU/0-1"));
  EXPECT_TRUE(Logged("+ LOC[5000.0] - PCI/0-f000"));
  EXPECT_TRUE(Logged("PCI/0-f0000 (0)"));
}

// GPUs hang off their DEV by LOC links, which the print never descends, so the GPU format needs a direct call.
TEST_F(TopoMicrotest, PrintRec_GpuShowsRankAndMloPart) {
  SystemPtr system = BuildSystem(System(Cpu(0, PartitionedPci("mlopart=\"3\"", ""))));
  char line[1024];

  ASSERT_EQ(ncclSuccess, ncclTopoPrintRec(system->nodes[GPU].nodes, nullptr, line, 0));
  ASSERT_EQ(ncclSuccess, ncclTopoPrintRec(system->nodes[GPU].nodes + 1, nullptr, line, 0));

  EXPECT_TRUE(Logged("GPU/0-7000010000 (0) [MLOPart 3]"));
  EXPECT_TRUE(Logged("GPU/0-10000 (1)"));
  EXPECT_FALSE(Logged("(1) [MLOPart"));
}

// --- Local NIC and GPU queries

// Two GPUs and one NIC per entry of `nets`, all behind one switch.
std::string SharedSwitchSystem(const std::vector<std::string>& nets) {
  std::string body = GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1);
  for (size_t n = 0; n < nets.size(); n++) body += NicPci(NicBus(n), nets[n]);
  return System(Cpu(0, Pci(SwitchBus(0), body)));
}

class TopoLocalMicrotest : public TopoMicrotest {
 protected:
  void TearDown() override {
    TopoSetNetDevsPolicy(NETDEVS_POLICY_AUTO, -1);
    TopoMicrotest::TearDown();
  }

  static void SetPolicy(enum netDevsPolicy policy, int num = -1) { TopoSetNetDevsPolicy(policy, num); }

  static int LocalNetDev(ncclTopoSystem* system, int rank, int channel, int type = NET) {
    int dev = -1;
    int64_t id = -1;
    EXPECT_EQ(ncclSuccess, ncclTopoGetLocalNetType(system, type, rank, channel, &id, &dev));
    return dev;
  }
};

TEST_F(TopoLocalMicrotest, GetLocal_ReturnsTheClosestNodesAndTheirPathType) {
  SystemPtr system = BuildWithPaths(XgmiSystem(2, 2, true));
  int locals[NCCL_TOPO_MAX_NODES];
  int count = 0, pathType = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetLocal(system.get(), GPU, 1, NET, locals, &count, &pathType));
  EXPECT_EQ(1, count);
  EXPECT_EQ(1, locals[0]);
  EXPECT_EQ(PATH_PIX, pathType);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocal(system.get(), GPU, 0, GIN, locals, &count, nullptr));
  EXPECT_EQ(0, count);  // no GIN nodes, so no paths
}

TEST_F(TopoLocalMicrotest, GetLocalNet_PowerOfTwoRailsAreMirroredOverGpuDevs) {
  SystemPtr system = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0), RailNet("net", 1, 1)}));
  SetPolicy(NETDEVS_POLICY_AUTO);

  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 1));
  EXPECT_EQ(1, LocalNetDev(system.get(), 1, 0));
  EXPECT_EQ(1, LocalNetDev(system.get(), 1, 3));
}

TEST_F(TopoLocalMicrotest, GetLocalNet_AllPolicySpreadsChannelsOverEveryNic) {
  SystemPtr system = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0), RailNet("net", 1, 1)}));
  SetPolicy(NETDEVS_POLICY_ALL);

  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(1, LocalNetDev(system.get(), 0, 1));
  EXPECT_EQ(0, LocalNetDev(system.get(), 1, 1));
}

TEST_F(TopoLocalMicrotest, GetLocalNet_MaxPolicyCapsNicsPerGpu) {
  SystemPtr system = BuildWithPaths(
    SharedSwitchSystem({RailNet("net", 0, 0), RailNet("net", 1, 1), RailNet("net", 2, 2), RailNet("net", 3, 3)}));
  SetPolicy(NETDEVS_POLICY_MAX, 3);

  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(1, LocalNetDev(system.get(), 0, 1));
  EXPECT_EQ(2, LocalNetDev(system.get(), 0, 2));
  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 3));
  EXPECT_EQ(2, LocalNetDev(system.get(), 1, 0));  // dev 1 mirrored over 4 rails starts at rail 2
}

TEST_F(TopoLocalMicrotest, GetLocalNet_OddRailCountStridesFromTheGpuDev) {
  SystemPtr system =
    BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0), RailNet("net", 1, 1), RailNet("net", 2, 2)}));
  SetPolicy(NETDEVS_POLICY_AUTO);

  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(1, LocalNetDev(system.get(), 0, 1));
  EXPECT_EQ(2, LocalNetDev(system.get(), 1, 0));
  EXPECT_EQ(0, LocalNetDev(system.get(), 1, 1));
}

TEST_F(TopoLocalMicrotest, GetLocalNet_OneRailOrdersItsNicsByPlane) {
  SystemPtr system = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 5, 1), RailNet("net", 1, 5, 0)}));
  SetPolicy(NETDEVS_POLICY_AUTO);

  EXPECT_EQ(1, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 1));
  EXPECT_EQ(1, LocalNetDev(system.get(), 1, 0));
}

TEST_F(TopoLocalMicrotest, GetLocalNet_NoLocalNicIsAnError) {
  SystemPtr system = BuildWithPaths(XgmiSystem(2));
  int64_t id;

  EXPECT_EQ(ncclInternalError, ncclTopoGetLocalNet(system.get(), 0, 0, &id, nullptr));
  EXPECT_EQ(ncclInternalError, ncclTopoGetLocalNet(system.get(), 5, 0, &id, nullptr));
  EXPECT_TRUE(Logged("ncclTopoRankToIndex could not find rank 5"));
}

TEST_F(TopoLocalMicrotest, GetLocalGinAndRma_UseTheirOwnNodes) {
  SystemPtr system = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0) + RailNet("gin", 3, 0) +
                                                          RailNet("rma", 6, 0)}));
  int64_t id = -1;
  int dev = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalGinDev(system.get(), 1, 0, &id, &dev));
  EXPECT_EQ(3, dev);
  EXPECT_EQ(system->nodes[GIN].nodes[0].id, id);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalRmaDev(system.get(), 1, 0, nullptr, &dev));
  EXPECT_EQ(6, dev);
  auto comm = std::make_unique<ncclComm>();
  comm->topo = system.get();
  comm->rank = 0;
  int devs[NCCL_TOPO_MAX_NODES];
  int count = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalGinDevs(comm.get(), devs, &count));
  EXPECT_EQ(1, count);
  EXPECT_EQ(3, devs[0]);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalRmaDevs(comm.get(), devs, &count));
  EXPECT_EQ(1, count);
  EXPECT_EQ(6, devs[0]);
}

TEST_F(TopoLocalMicrotest, GetLocalGpu_MapsANicToTheGpuThatUsesIt) {
  std::string far = NicPci(NicBus(5), RailNet("net", 1, 1));
  SystemPtr system = BuildWithPaths(
    System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), RailNet("net", 0, 0)))) + Cpu(1, far)));
  int gpu = -2;

  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalGpu(system.get(), system->nodes[NET].nodes[0].id, &gpu));
  EXPECT_EQ(0, gpu);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalGpu(system.get(), system->nodes[NET].nodes[1].id, &gpu));
  EXPECT_EQ(-1, gpu);
  EXPECT_EQ(0, system->nodes[NET].nodes[0].net.localGpu);  // precomputed by ncclTopoComputePaths
  EXPECT_EQ(ncclInternalError, ncclTopoGetLocalGpu(system.get(), 0x12345, &gpu));
}

TEST_F(TopoLocalMicrotest, GetLocalNetCountByBw_AddsNicsUntilTheGpuLinkIsCovered) {
  SystemPtr fast = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0), RailNet("net", 1, 1)}));
  SystemPtr slow = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0, 0, 200000), RailNet("net", 1, 0, 1, 200000),
                                                      RailNet("net", 2, 0, 2, 200000)}));
  SystemPtr none = BuildWithPaths(XgmiSystem(1));
  SetPolicy(NETDEVS_POLICY_ALL);
  int count = -1;
  float bw = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalNetCountByBw(fast.get(), 0, &count, &bw));
  EXPECT_EQ(1, count);
  EXPECT_FLOAT_EQ(48, bw);  // the path to the NIC is capped by the 48 GB/s switch link
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalNetCountByBw(slow.get(), 0, &count, &bw));
  EXPECT_EQ(2, count);
  EXPECT_FLOAT_EQ(50, bw);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLocalNetCountByBw(none.get(), 0, &count, &bw));
  EXPECT_EQ(0, count);
  EXPECT_FLOAT_EQ(0, bw);
}

TEST_F(TopoLocalMicrotest, GetMinNetBw_ReadsThePathToTheLocalNics) {
  SystemPtr system =
    BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 0, 0, 200000), RailNet("net", 1, 0, 1, 80000)}));
  SystemPtr none = BuildWithPaths(XgmiSystem(1));
  SetPolicy(NETDEVS_POLICY_ALL);
  float bw = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetMinNetBw(system.get(), 1, &bw));
  EXPECT_FLOAT_EQ(25, bw);  // only the faster NIC is local
  ASSERT_EQ(ncclSuccess, ncclTopoGetMinNetBw(none.get(), 0, &bw));
  EXPECT_FLOAT_EQ(0, bw);
  EXPECT_EQ(ncclInternalError, ncclTopoGetMinNetBw(system.get(), 9, &bw));
}

TEST_F(TopoLocalMicrotest, NetDevsPolicy_ParsesTheEnvironmentOnce) {
  struct Case {
    const char* env;
    enum netDevsPolicy policy;
    int num;
  };
  const Case cases[] = {{"AUTO", NETDEVS_POLICY_AUTO, -1}, {"all", NETDEVS_POLICY_ALL, -1},
                        {"MAX:2", NETDEVS_POLICY_MAX, 2}, {"MAX:0", NETDEVS_POLICY_AUTO, -1},
                        {"bogus", NETDEVS_POLICY_AUTO, -1}, {nullptr, NETDEVS_POLICY_AUTO, -1}};
  for (const Case& c : cases) {
    RcclUnitTesting::ScopedEnv policyEnv("NCCL_NETDEVS_POLICY", c.env);
    netDevsPolicy = NETDEVS_POLICY_UNDEF;
    netDevsPolicyNum = -1;

    getNetDevsPolicyOnce();

    EXPECT_EQ(c.policy, netDevsPolicy) << (c.env ? c.env : "unset");
    EXPECT_EQ(c.num, netDevsPolicyNum) << (c.env ? c.env : "unset");
  }
  EXPECT_TRUE(Logged("Unable to recognize NCCL_NETDEVS_POLICY=bogus"));
  EXPECT_TRUE(Logged("NCCL_NETDEVS_POLICY set by environment to MAX:2"));
}

TEST_F(TopoLocalMicrotest, NetDevsPolicy_MaxNeedsAPositiveCount) {
  SetPolicy(NETDEVS_POLICY_MAX, 0);
  enum netDevsPolicy policy;
  int num = 7;

  EXPECT_EQ(ncclInternalError, ncclTopoGetNetDevsPolicy(&policy, &num));

  netDevsPolicyNum = 4;
  ASSERT_EQ(ncclSuccess, ncclTopoGetNetDevsPolicy(&policy, &num));
  EXPECT_EQ(NETDEVS_POLICY_MAX, policy);
  EXPECT_EQ(4, num);
  netDevsPolicy = NETDEVS_POLICY_ALL;
  netDevsPolicyNum = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoGetNetDevsPolicy(&policy, &num));
  EXPECT_EQ(4, num);  // a negative count leaves the caller's value alone
}

// --- System queries

TEST_F(TopoMicrotest, Counts_AndCpuTypeReadTheNodeSets) {
  SystemPtr system = BuildSystem(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, Xgmi("0", 1, "0x068000")) +
                                                NicPci(NicBus(0), Net("a", 0) + Net("b", 1)))));
  int count = 0, arch, vendor, model;

  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuCount(system.get(), &count));
  EXPECT_EQ(1, count);
  ASSERT_EQ(ncclSuccess, ncclTopoGetNetCount(system.get(), &count));
  EXPECT_EQ(2, count);
  ASSERT_EQ(ncclSuccess, ncclTopoGetNvsCount(system.get(), &count));
  EXPECT_EQ(1, count);
  ASSERT_EQ(ncclSuccess, ncclTopoCpuType(system.get(), &arch, &vendor, &model));
  EXPECT_EQ(NCCL_TOPO_CPU_ARCH_X86, arch);
  EXPECT_EQ(NCCL_TOPO_CPU_VENDOR_AMD, vendor);
  EXPECT_EQ(NCCL_TOPO_CPU_MODEL_AMD_ZEN, model);
}

TEST_F(TopoMicrotest, GetCompCap_SpansTheDevNodes) {
  SystemPtr system = BuildSystem(XgmiSystem(3));
  system->nodes[DEV].nodes[1].dev.cudaCompCap = 90;
  system->nodes[DEV].nodes[2].dev.cudaCompCap = 50;
  int ccMin = 0, ccMax = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoGetCompCap(system.get(), &ccMin, &ccMax));
  EXPECT_EQ(50, ccMin);
  EXPECT_EQ(90, ccMax);
  ASSERT_EQ(ncclSuccess, ncclTopoGetCompCap(system.get(), nullptr, nullptr));
  system->nodes[DEV].count = 0;
  EXPECT_EQ(ncclInternalError, ncclTopoGetCompCap(system.get(), &ccMin, &ccMax));
}

class TopoAffinityMicrotest : public TopoMicrotest {
 protected:
  SystemPtr system_;

  void SetUp() override {
    TopoMicrotest::SetUp();
    system_ = BuildWithPaths(XgmiSystem(1, 1, false, "gfx942", "affinity_offset=\"4\""));  // CPUs 4-7
  }

  ncclAffinity Affinity() {
    ncclAffinity affinity;
    EXPECT_EQ(ncclSuccess, ncclTopoGetCpuAffinity(system_.get(), 0, &affinity));
    return affinity;
  }
};

TEST_F(TopoAffinityMicrotest, GetCpuAffinity_IntersectsTheProcessMaskWithTheGpuCpus) {
  for (int cpu : {2, 5, 6}) CPU_SET(cpu, &g_processAffinity);

  ncclAffinity affinity = Affinity();

  EXPECT_EQ(2, ncclOsCpuCount(affinity));
  EXPECT_TRUE(ncclOsCpuIsSet(affinity, 5));
  EXPECT_TRUE(Logged("Affinity for GPU 0 is 5-6. (GPU affinity = 4-7 ; CPU affinity = 2,5-6)."));
}

TEST_F(TopoAffinityMicrotest, GetCpuAffinity_DisjointMasksAreReportedEmpty) {
  CPU_SET(1, &g_processAffinity);

  EXPECT_EQ(0, ncclOsCpuCount(Affinity()));
  EXPECT_TRUE(Logged("is empty, ignoring."));
}

TEST_F(TopoAffinityMicrotest, GetCpuAffinity_IgnoreParamTakesTheGpuCpus) {
  g_params["IGNORE_CPU_AFFINITY"] = 1;
  CPU_SET(1, &g_processAffinity);

  EXPECT_EQ(4, ncclOsCpuCount(Affinity()));
  EXPECT_TRUE(Logged("(GPU affinity = 4-7)."));
}

TEST_F(TopoAffinityMicrotest, GetCpuAffinity_UnknownRankFails) {
  ncclAffinity affinity;

  EXPECT_EQ(ncclInternalError, ncclTopoGetCpuAffinity(system_.get(), 3, &affinity));
}

class TopoMultiRankMicrotest : public TopoMicrotest {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::vector<ncclPeerInfo> peers_;

  void Peers(const std::vector<std::pair<uint64_t, int64_t>>& hostBus, int rank) {
    peers_.assign(hostBus.size(), ncclPeerInfo{});
    for (size_t r = 0; r < hostBus.size(); r++) {
      peers_[r].hostHash = hostBus[r].first;
      peers_[r].busId = hostBus[r].second;
    }
    comm_->peerInfo = peers_.data();
    comm_->nRanks = peers_.size();
    comm_->rank = rank;
  }
};

TEST_F(TopoMultiRankMicrotest, CheckMultiRank_SharedGpuNeedsTheOptIn) {
  Peers({{1, 0x100}, {2, 0x100}, {1, 0x200}, {1, 0x100}}, 0);

  EXPECT_EQ(ncclInvalidUsage, ncclCheckMultiRank(comm_.get()));

  EXPECT_TRUE(comm_->isMultiRankGpu);
  EXPECT_TRUE(Logged("Detected ranks 0 and 3 sharing GPU with bus 100"));
  EXPECT_TRUE(Logged("Set NCCL_MULTI_RANK_GPU_ENABLE=1"));
}

TEST_F(TopoMultiRankMicrotest, CheckMultiRank_OptInOrOtherRanksPass) {
  Peers({{1, 0x100}, {1, 0x100}}, 1);
  EXPECT_EQ(ncclSuccess, ncclCheckMultiRank(comm_.get()));  // only the first rank on the host enforces it
  EXPECT_TRUE(comm_->isMultiRankGpu);
  EXPECT_FALSE(Logged("Detected ranks"));

  comm_->rank = 0;
  g_params["MULTI_RANK_GPU_ENABLE"] = 1;
  EXPECT_EQ(ncclSuccess, ncclCheckMultiRank(comm_.get()));

  Peers({{1, 0x100}, {2, 0x100}}, 0);
  EXPECT_EQ(ncclSuccess, ncclCheckMultiRank(comm_.get()));
  EXPECT_FALSE(comm_->isMultiRankGpu);
}

TEST_F(TopoMultiRankMicrotest, CheckMultiRank_CapsRanksPerHost) {
  std::vector<std::pair<uint64_t, int64_t>> peers(2049, {1, 0});
  for (size_t r = 0; r < peers.size(); r++) peers[r].second = r;
  Peers(peers, 0);

  EXPECT_EQ(ncclInternalError, ncclCheckMultiRank(comm_.get()));

  EXPECT_TRUE(Logged("exceeded number of local ranks 2048"));
}

// --- NIC fusion

struct FakeNetDev {
  std::string name;
  int port = 1;
  int railId = NCCL_NET_ID_UNDEF;
  int planeId = NCCL_NET_ID_UNDEF;
  uint64_t guid = 0x11;
  int ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  ncclNetVDeviceProps_t vProps{};
};

std::deque<FakeNetDev> g_netDevs;  // a deque, so the names handed out stay put when a vNIC is appended
std::vector<ncclNetVDeviceProps_t> g_vDeviceCalls;
ncclResult_t g_makeVDeviceResult = ncclSuccess;

ncclResult_t FakeGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev < 0 || dev >= (int)g_netDevs.size()) return ncclInvalidArgument;
  const FakeNetDev& d = g_netDevs[dev];
  *props = ncclNetProperties_t{};
  props->name = const_cast<char*>(d.name.c_str());
  props->pciPath = const_cast<char*>("/sys/devices/none");
  props->guid = d.guid;
  props->ptrSupport = d.ptrSupport;
  props->speed = 200000;
  props->port = d.port;
  props->latency = 2;
  props->maxComms = 64;
  props->railId = d.railId;
  props->planeId = d.planeId;
  props->vProps = d.vProps;
  return ncclSuccess;
}

ncclResult_t FakeMakeVDevice(int* dev, ncclNetVDeviceProps_t* props) {
  g_vDeviceCalls.push_back(*props);
  if (g_makeVDeviceResult != ncclSuccess) return g_makeVDeviceResult;
  FakeNetDev v;
  v.name = "v";
  for (int i = 0; i < props->ndevs; i++) v.name += "_" + g_netDevs[props->devs[i]].name;
  v.vProps = *props;
  v.guid = g_netDevs[props->devs[0]].guid;
  g_netDevs.push_back(v);
  *dev = g_netDevs.size() - 1;
  return ncclSuccess;
}

ncclResult_t FakeDevices(int* n) {
  *n = g_netDevs.size();
  return ncclSuccess;
}

// A switch with a two-port NIC (a0/a1), a NIC b0, a deeper NIC e0 behind a sub-switch and a two-function NIC
// (f0/f1); c0 hangs off the CPU and d0 off the other socket.
std::string NicDev(const std::string& busId, const std::string& nets, const std::string& width = "8") {
  std::string body;
  for (size_t p = 0; p < nets.size(); p += 2) body += "<net name=\"" + nets.substr(p, 2) + "\"/>";
  return "<pci busid=\"" + busId + "\" class=\"0x020000\" link_speed=\"32.0 GT/s PCIe\" link_width=\"" + width +
         "\"><nic>" + body + "</nic></pci>";
}
const std::string kFusionXml = System(
  Cpu(0, Pci("0000:01:00.0", NicDev("0000:02:00.0", "a0a1") + NicDev("0000:03:00.0", "b0") +
                               Pci("0000:05:00.0", NicDev("0000:06:00.0", "e0")) + NicDev("0000:07:00.0", "f0") +
                               NicDev("0000:07:00.1", "f1")) +
             NicDev("0000:04:00.0", "c0", "16")) +
  Cpu(1, NicDev("0000:80:00.0", "d0")));

class TopoFusionMicrotest : public TopoMicrotest {
 protected:
  XmlPtr xml_;
  ncclTopoNetInfo info_{};
  ncclTopoNetRailKeyList rails_;

  void SetUp() override {
    TopoMicrotest::SetUp();
    xml_ = MustParseXml(kFusionXml);
    g_netDevs.clear();
    g_vDeviceCalls.clear();
    g_makeVDeviceResult = ncclSuccess;
    info_.net = true;
    info_.name = "fake";
    info_.maxDevsPerNic = NCCL_NET_MAX_DEVS_PER_NIC;
    info_.mergeLevel = PATH_PORT;
    info_.mergePolicy = NCCL_NET_MERGE_POLICY_ALL;
    info_.getDevCount = ncclNetGetDevCount;
    info_.setVirtDevCount = ncclNetSetVirtDevCount;
    info_.getProperties = FakeGetProperties;
    info_.makeVDevice = FakeMakeVDevice;
    info_.devices = FakeDevices;
    info_.railKeyList = &rails_;
  }

  ncclXmlNode* Net(const char* name, const char* tag = "net") {
    ncclXmlNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, xmlFindTagKv(xml_.get(), tag, &node, "name", name));
    return node;
  }

  int PathBetween(std::vector<const char*> names, ncclXmlNode** parent) {
    std::vector<ncclXmlNode*> nodes;
    for (const char* n : names) nodes.push_back(Net(n));
    int path = -1;
    EXPECT_EQ(ncclSuccess, ncclTopoGetPath(nodes.data(), nodes.size(), &path, parent));
    return path;
  }

  void Devs(std::vector<std::string> names) {
    for (const std::string& n : names) {
      FakeNetDev d;
      d.name = n;
      g_netDevs.push_back(d);
    }
    nPhys_ = g_netDevs.size();
  }

  // The physical devices only; vNICs made by an earlier merge sit after them.
  std::vector<ncclNetProperties_t> Props() {
    std::vector<ncclNetProperties_t> props(nPhys_);
    for (int d = 0; d < nPhys_; d++) EXPECT_EQ(ncclSuccess, FakeGetProperties(d, &props[d]));
    return props;
  }

  std::vector<ncclXmlNode*> PhysNodes() {
    std::vector<ncclXmlNode*> nodes;
    for (int d = 0; d < nPhys_; d++) nodes.push_back(Net(g_netDevs[d].name.c_str()));
    return nodes;
  }

  int nPhys_ = 0;

  static std::vector<int> DevsOf(const ncclNetVDeviceProps_t& v) { return std::vector<int>(v.devs, v.devs + v.ndevs); }

  std::string Attr(ncclXmlNode* node, const char* key) {
    const char* value = nullptr;
    EXPECT_EQ(ncclSuccess, xmlGetAttr(node, key, &value));
    return value ? value : "<unset>";
  }
};

TEST_F(TopoFusionMicrotest, GetPath_ClassifiesTheCommonAncestor) {
  ncclXmlNode* parent = nullptr;

  EXPECT_EQ(PATH_LOC, PathBetween({"a0"}, &parent));
  EXPECT_EQ("0000:02:00.0", Attr(parent, "busid"));
  EXPECT_EQ(PATH_PORT, PathBetween({"a0", "a1"}, &parent));
  EXPECT_EQ("0000:02:00.0", Attr(parent, "busid"));
  EXPECT_EQ(PATH_PORT, PathBetween({"f0", "f1"}, &parent));  // functions of one device
  EXPECT_EQ(PATH_PIX, PathBetween({"a0", "b0"}, &parent));
  EXPECT_EQ("0000:01:00.0", Attr(parent, "busid"));
  EXPECT_EQ(PATH_PXB, PathBetween({"a0", "e0"}, &parent));
  EXPECT_EQ(PATH_PXB, PathBetween({"a0", "b0", "e0"}, &parent));
  EXPECT_EQ(PATH_PHB, PathBetween({"a0", "c0"}, &parent));
  EXPECT_STREQ("cpu", parent->name);
  EXPECT_EQ(PATH_SYS, PathBetween({"a0", "d0"}, &parent));
  EXPECT_STREQ("system", parent->name);
}

TEST_F(TopoFusionMicrotest, GetPath_PortNeedsMatchingBusIdsOnEveryPeer) {
  ncclXmlNode* sw = Net("b0")->parent->parent->parent;
  ncclXmlNode* odd = nullptr;
  ncclXmlNode* nic = nullptr;
  ncclXmlNode* net = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), sw, "pci", &odd));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(odd, "busid", "0000:03:00.10"));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), odd, "nic", &nic));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nic, "net", &net));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(net, "name", "g0"));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), sw, "nic", &nic));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nic, "net", &net));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(net, "name", "h0"));
  ncclXmlNode* parent = nullptr;

  EXPECT_EQ(PATH_PIX, PathBetween({"b0", "g0"}, &parent));  // bus ids of different lengths
  EXPECT_EQ(PATH_PXB, PathBetween({"b0", "h0"}, &parent));  // a NIC with no bus id
  EXPECT_EQ(PATH_PXB, PathBetween({"h0", "b0"}, &parent));
}

TEST_F(TopoFusionMicrotest, CheckPix_NeedsEveryNicDirectlyUnderTheSwitch) {
  ncclXmlNode* sw = Net("b0")->parent->parent->parent;
  ncclXmlNode* nodes[] = {Net("a0"), Net("b0")};
  ncclXmlNode* deep[] = {Net("a0"), Net("e0")};
  ncclXmlNode* nonNet[] = {Net("a0")->parent};

  EXPECT_EQ(1, ncclTopoCheckPix(sw, nodes, 2));
  EXPECT_EQ(0, ncclTopoCheckPix(sw, deep, 2));
  EXPECT_EQ(1, ncclTopoCheckPix(sw, nonNet, 1));
}

TEST_F(TopoFusionMicrotest, MakeUniqueBusId_SkipsBusIdsInUse) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = "0000:06:ff.f";
  ncclXmlNode* node = nullptr;
  ncclXmlNode* sw = Net("a0")->parent->parent->parent;

  ASSERT_EQ(ncclSuccess, ncclTopoMakeUniqueBusId(xml_.get(), busId, &node, sw));

  EXPECT_STREQ("0000:07:00.2", busId);  // .0 and .1 are taken
  EXPECT_EQ(sw, node->parent);
  EXPECT_EQ("0000:07:00.2", Attr(node, "busid"));
}

TEST_F(TopoFusionMicrotest, MakeUniqueBusId_GivesUpAfterAHundredTries) {
  ncclXmlNode* sw = Net("a0")->parent->parent->parent;
  for (int f = 1; f <= 100; f++) {
    char id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    ASSERT_EQ(ncclSuccess, int64ToBusId(0x90000 + f, id));
    ncclXmlNode* node = nullptr;
    ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), sw, "pci", &node));
    ASSERT_EQ(ncclSuccess, xmlSetAttr(node, "busid", id));
  }
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = "0000:90:00.0";
  ncclXmlNode* node = nullptr;

  EXPECT_EQ(ncclInternalError, ncclTopoMakeUniqueBusId(xml_.get(), busId, &node, sw));

  EXPECT_TRUE(Logged("Couldn't generate unique busId after 100 tries"));
}

TEST_F(TopoFusionMicrotest, MakePciParent_ClonesTheNicPciUnderANewBusId) {
  ncclXmlNode* parent = Net("a0")->parent->parent->parent;
  ncclXmlNode* sw = parent;

  ASSERT_EQ(ncclSuccess, ncclTopoMakePciParent(xml_.get(), &parent, Net("a0")));

  EXPECT_EQ(sw, parent->parent);
  EXPECT_EQ("0000:02:00.1", Attr(parent, "busid"));
  EXPECT_EQ("0x020000", Attr(parent, "class"));
  EXPECT_EQ("8", Attr(parent, "link_width"));
}

TEST_F(TopoFusionMicrotest, MakePciParent_NeedsAPciAboveTheNic) {
  ncclXmlNode* cpu = nullptr;
  ncclXmlNode* nic = nullptr;
  ncclXmlNode* net = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml_.get(), "cpu", &cpu));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), cpu, "nic", &nic));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nic, "net", &net));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(net, "name", "virt"));
  ncclXmlNode* orphan = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nullptr, "net", &orphan));
  ncclXmlNode* parent = cpu;

  EXPECT_EQ(ncclInternalError, ncclTopoMakePciParent(xml_.get(), &parent, net));
  EXPECT_TRUE(Logged("Can't find busId of child 0 virt"));
  EXPECT_EQ(ncclInternalError, ncclTopoMakePciParent(xml_.get(), &parent, orphan));
  EXPECT_EQ(cpu, parent);
}

TEST_F(TopoFusionMicrotest, MakeVnic_SkipsSingletonsAndReportsPluginFailures) {
  Devs({"a0", "a1"});
  ncclNetVDeviceProps_t one{1, {0}};
  ncclNetVDeviceProps_t two{2, {0, 1}};

  EXPECT_EQ(ncclSuccess, ncclTopoMakeVnic(xml_.get(), &info_, &one, nullptr));
  EXPECT_TRUE(g_vDeviceCalls.empty());
  EXPECT_EQ(ncclSuccess, ncclTopoMakeVnic(xml_.get(), &info_, &two, nullptr));
  EXPECT_TRUE(Logged("TOPO/NET : Made vNic 2"));
  g_makeVDeviceResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclTopoMakeVnic(xml_.get(), &info_, &two, nullptr));
  EXPECT_TRUE(Logged("Tried merging multiple devices together and failed. vProps={ndevs=2, devs=[0 1"));
  info_.maxDevsPerNic = 1;
  EXPECT_EQ(ncclInternalError, ncclTopoMakeVnic(xml_.get(), &info_, &two, nullptr));
  EXPECT_TRUE(Logged("Tried to merge too many NICs. 2 > 1"));
}

class TopoForceMergeMicrotest : public TopoFusionMicrotest {
 protected:
  std::vector<int> placed_;

  ncclResult_t ForceMerge(const char* spec) {
    info_.forceMerge = spec;
    placed_.assign(nPhys_, 0);
    std::vector<ncclNetProperties_t> props = Props();
    std::vector<ncclXmlNode*> nodes = PhysNodes();
    return ncclTopoForceMerge(xml_.get(), &info_, placed_.data(), props.data(), nodes.data(), nPhys_);
  }
};

TEST_F(TopoForceMergeMicrotest, ForceMerge_MergesEachGroupAndSkipsUnparsableOnes) {
  Devs({"a0", "a1", "b0", "c0"});

  ASSERT_EQ(ncclSuccess, ForceMerge("a0,c0;:;b0"));

  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 3}), DevsOf(g_vDeviceCalls[0]));
  EXPECT_EQ(std::vector<int>({1, 0, 1, 1}), placed_);
  EXPECT_TRUE(Logged("Couldn't parse substring :"));
}

TEST_F(TopoForceMergeMicrotest, ForceMerge_RejectsGroupsBeyondOneVnic) {
  Devs({"a0", "a1", "b0", "c0", "d0", "e0", "f0", "f1", "x0"});

  EXPECT_EQ(ncclInvalidUsage, ForceMerge("a0,a1,b0,c0,d0,e0,f0,f1,x0"));

  EXPECT_TRUE(Logged("requests at least 9 devices. Max 8"));
  EXPECT_TRUE(g_vDeviceCalls.empty());
}

TEST_F(TopoForceMergeMicrotest, ForceMerge_PortlessPatternMayMatchTooManyPorts) {
  Devs({"a0", "a0", "a0", "a0", "a0", "a0", "a0", "a0", "a0"});
  for (int p = 0; p < 9; p++) g_netDevs[p].port = p;

  EXPECT_EQ(ncclInvalidUsage, ForceMerge("a0"));

  EXPECT_TRUE(Logged("which matches too many devices. Max 8"));
}

TEST_F(TopoForceMergeMicrotest, ForceMerge_EveryNameMustMatch) {
  Devs({"a0", "a1"});

  EXPECT_EQ(ncclInvalidUsage, ForceMerge("a0,zz"));

  EXPECT_TRUE(Logged("Only matched 1 devices, 2 requested from a0,zz"));
}

TEST_F(TopoForceMergeMicrotest, ForceMerge_HonorsThePluginDeviceLimit) {
  Devs({"a0", "a1"});
  info_.maxDevsPerNic = 1;

  EXPECT_EQ(ncclInvalidUsage, ForceMerge("a0,a1"));

  EXPECT_TRUE(Logged("Specified fused NIC a0,a1 which has too many devices (2). Max 1"));
}

TEST_F(TopoForceMergeMicrotest, ForceMerge_PluginFailureIsInvalidUsage) {
  Devs({"a0", "a1"});
  g_makeVDeviceResult = ncclInternalError;

  EXPECT_EQ(ncclInvalidUsage, ForceMerge("a0,a1"));

  EXPECT_TRUE(Logged("Could not force merge NICs a0,a1"));
  EXPECT_EQ(std::vector<int>({0, 0}), placed_);
}

class TopoAutoMergeMicrotest : public TopoFusionMicrotest {
 protected:
  std::vector<int> placed_;

  ncclResult_t AutoMerge() {
    placed_.assign(nPhys_, 0);
    std::vector<ncclNetProperties_t> props = Props();
    std::vector<ncclXmlNode*> nodes = PhysNodes();
    return ncclTopoAutoMerge(xml_.get(), &info_, placed_.data(), props.data(), nodes.data(), nPhys_);
  }
};

TEST_F(TopoAutoMergeMicrotest, AutoMerge_FusesDevicesWithinTheMergeLevel) {
  Devs({"a0", "b0", "a1", "c0"});

  ASSERT_EQ(ncclSuccess, AutoMerge());
  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 2}), DevsOf(g_vDeviceCalls[0]));

  g_vDeviceCalls.clear();
  info_.mergeLevel = PATH_PIX;
  ASSERT_EQ(ncclSuccess, AutoMerge());
  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 1, 2}), DevsOf(g_vDeviceCalls[0]));
  EXPECT_EQ(std::vector<int>({1, 1, 1, 1}), placed_);
}

TEST_F(TopoAutoMergeMicrotest, AutoMerge_StopsAtTheDeviceLimit) {
  Devs({"a0", "a1", "b0"});
  info_.mergeLevel = PATH_PIX;
  info_.maxDevsPerNic = 2;

  ASSERT_EQ(ncclSuccess, AutoMerge());

  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 1}), DevsOf(g_vDeviceCalls[0]));
}

TEST_F(TopoAutoMergeMicrotest, AutoMerge_RailPolicyOnlyFusesTheSameRail) {
  Devs({"a0", "a1", "b0", "c0"});
  g_netDevs[0].railId = 1;
  g_netDevs[2].railId = 1;
  info_.mergeLevel = PATH_PHB;
  info_.mergePolicy = NCCL_NET_MERGE_POLICY_RAIL;

  ASSERT_EQ(ncclSuccess, AutoMerge());

  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 2}), DevsOf(g_vDeviceCalls[0]));  // a1 and c0 have no rail, so no match
}

// A failed merge restarts at i = 0 before the loop's ++i, so device 0 is never re-placed.
// Pins AICOMRCCL-3027; a fix flips this pin.
TEST_F(TopoAutoMergeMicrotest, AutoMerge_KnownBug_FailedMergeRestartsPastDeviceZero) {
  Devs({"a0", "a1", "b0"});
  info_.mergeLevel = PATH_PIX;
  info_.maxDevsPerNic = 2;
  g_makeVDeviceResult = ncclInternalError;

  ASSERT_EQ(ncclSuccess, AutoMerge());

  ASSERT_EQ(3u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({0, 1}), DevsOf(g_vDeviceCalls[0]));
  EXPECT_EQ(std::vector<int>({1, 2}), DevsOf(g_vDeviceCalls[1]));
  EXPECT_EQ(std::vector<int>({2, 0}), DevsOf(g_vDeviceCalls[2]));
  EXPECT_EQ(std::vector<int>({0, 1, 1}), placed_);
  EXPECT_TRUE(Logged("Marking physical devices as unplaced"));
}

TEST_F(TopoAutoMergeMicrotest, AutoMerge_OverfullVnicIsAnError) {
  if (kAsan) GTEST_SKIP() << "this return skips free(paths), which LeakSanitizer reports";
  Devs({"a0", "a1"});
  info_.maxDevsPerNic = 0;

  EXPECT_EQ(ncclInternalError, AutoMerge());

  EXPECT_TRUE(Logged("Tried to merge too many NICs. 2 > 0"));
}

TEST_F(TopoFusionMicrotest, FindLinkWidth_SumsTheChildrenThatLeadToTheNics) {
  ncclXmlNode* sw = Net("a0")->parent->parent->parent;
  ncclXmlNode* nets[] = {Net("a0"), Net("b0"), Net("e0")};
  int width = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoFindLinkWidth(sw, nets, 3, &width));

  EXPECT_EQ(8 + 8 + 8, width);  // e0's sub-switch has no width of its own
  EXPECT_TRUE(Logged("Did not find child net device"));
}

TEST_F(TopoFusionMicrotest, FindLinkWidth_ParentWidthCapsItsChildren) {
  ncclXmlNode* sw = Net("a0")->parent->parent->parent;
  ASSERT_EQ(ncclSuccess, xmlSetAttr(sw, "link_width", "4"));
  ncclXmlNode* nets[] = {Net("a0"), Net("b0")};
  int width = -1, found = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoFindLinkWidthRec(sw, nets, 2, &found, &width));

  EXPECT_EQ(1, found);
  EXPECT_EQ(4, width);
}

class TopoVNicParentMicrotest : public TopoFusionMicrotest {
 protected:
  ncclXmlNode* Parent(std::vector<int> devs) {
    ncclNetVDeviceProps_t v{};
    for (int d : devs) v.devs[v.ndevs++] = d;
    ncclXmlNode* parent = nullptr;
    EXPECT_EQ(ncclSuccess, ncclTopoGetVNicParent(xml_.get(), "net", FakeGetProperties, &v, &parent));
    return parent;
  }

  void SetUp() override {
    TopoFusionMicrotest::SetUp();
    Devs({"a0", "a1", "b0", "c0", "d0"});
  }
};

TEST_F(TopoVNicParentMicrotest, GetVNicParent_SwitchCommonParentGetsANewPciWithTheSummedWidth) {
  ncclXmlNode* sw = Net("a0")->parent->parent->parent;

  ncclXmlNode* parent = Parent({0, 2});

  EXPECT_EQ(sw, parent->parent);
  EXPECT_EQ("0000:02:00.1", Attr(parent, "busid"));
  EXPECT_EQ("16", Attr(parent, "link_width"));
  EXPECT_EQ("16", Attr(sw, "link_width"));
}

TEST_F(TopoVNicParentMicrotest, GetVNicParent_PortsOfOneNicKeepThatNic) {
  ncclXmlNode* parent = Parent({0, 1});

  EXPECT_EQ("0000:02:00.0", Attr(parent, "busid"));
  EXPECT_EQ("8", Attr(parent, "link_width"));
}

TEST_F(TopoVNicParentMicrotest, GetVNicParent_CpuCommonParentGetsANewPci) {
  ncclXmlNode* parent = Parent({0, 3});

  EXPECT_STREQ("cpu", parent->parent->name);
  EXPECT_EQ("0000:02:00.1", Attr(parent, "busid"));
  EXPECT_EQ("24", Attr(parent, "link_width"));  // the switch's 8 plus c0's 16
}

TEST_F(TopoVNicParentMicrotest, GetVNicParent_CrossNumaGoesUnderTheFirstNicsCpu) {
  ncclXmlNode* cpu0 = Net("c0")->parent->parent->parent;

  ncclXmlNode* parent = Parent({3, 4});

  EXPECT_EQ(cpu0, parent->parent);
}

TEST_F(TopoVNicParentMicrotest, GetVNicParent_PropertyFailureIsReturned) {
  ncclNetVDeviceProps_t v{2, {0, 9}};
  ncclXmlNode* parent = nullptr;

  EXPECT_EQ(ncclInvalidArgument, ncclTopoGetVNicParent(xml_.get(), "net", FakeGetProperties, &v, &parent));
}

// --- Importing the network plugins into the XML

class TopoProcessNetMicrotest : public TopoFusionMicrotest {
 protected:
  void SetUp() override {
    TopoFusionMicrotest::SetUp();
    g_netGetDevCount = [](int, int* nPhys, int* nVirt) {
      *nPhys = (int)std::min<size_t>(g_netDevs.size(), 2);
      *nVirt = 0;
      return ncclSuccess;
    };
  }

  ncclResult_t Process(const char* dumpXmlFile = "dump.xml") {
    return ncclTopoProcessNet(xml_.get(), dumpXmlFile, &info_);
  }
};

TEST_F(TopoProcessNetMicrotest, ProcessNet_StampsPluginPropertiesOnPhysicalNics) {
  Devs({"a0", "b0"});
  g_netDevs[1].railId = 3;
  g_netDevs[1].planeId = -2;

  ASSERT_EQ(ncclSuccess, Process());

  ncclXmlNode* a0 = Net("a0");
  EXPECT_EQ("1", Attr(a0, "keep"));
  EXPECT_EQ("0", Attr(a0, "dev"));
  EXPECT_EQ("2", Attr(a0, "latency"));
  EXPECT_EQ("200000", Attr(a0, "speed"));
  EXPECT_EQ("1", Attr(a0, "port"));
  EXPECT_EQ("0x11", Attr(a0, "guid"));
  EXPECT_EQ(std::to_string(NCCL_TOPO_UNDEF_BIT | 0), Attr(a0, "rail"));  // derived from the guid
  EXPECT_EQ(std::to_string(NCCL_TOPO_UNDEF_BIT | 1), Attr(a0, "plane"));  // derived from the port
  EXPECT_EQ("64", Attr(a0, "maxconn"));
  EXPECT_EQ("1", Attr(a0, "gdr"));
  EXPECT_EQ("1", Attr(a0, "net"));
  EXPECT_EQ("<unset>", Attr(a0, "coll"));
  ncclXmlNode* b0 = Net("b0");
  EXPECT_EQ("1", Attr(b0, "dev"));
  EXPECT_EQ("3", Attr(b0, "rail"));
  EXPECT_EQ("65534", Attr(b0, "plane"));  // plugin ids are kept to 16 bits
  EXPECT_TRUE(Logged("GPU Direct RDMA Enabled for HCA 0 'a0'"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_DmaBufCountsForGdrOnlyWhenTheCommSupportsIt) {
  Devs({"a0", "b0"});
  for (FakeNetDev& d : g_netDevs) d.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_DMABUF;
  g_netDevs[1].guid = 0x11;  // same NIC as a0, so the same derived rail

  ASSERT_EQ(ncclSuccess, Process());
  EXPECT_EQ("0", Attr(Net("a0"), "gdr"));
  EXPECT_EQ(Attr(Net("a0"), "rail"), Attr(Net("b0"), "rail"));

  xml_ = MustParseXml(kFusionXml);
  info_.dmaBufSupport = true;
  ASSERT_EQ(ncclSuccess, Process());
  EXPECT_EQ("1", Attr(Net("a0"), "gdr"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_ExistingAttributesWinOverThePlugin) {
  Devs({"a0", "b0"});
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "speed", "7"));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "dev", "5"));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "gin", "1"));

  ASSERT_EQ(ncclSuccess, Process());

  EXPECT_EQ("7", Attr(Net("a0"), "speed"));
  EXPECT_EQ("0", Attr(Net("a0"), "dev"));
  EXPECT_TRUE(Logged("Changing fake dev index from 5 to 0. This device is shared by NET GIN"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_InconsistentRailIsInvalidUsage) {
  Devs({"a0", "b0"});
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "rail", "7"));
  g_netDevs[0].railId = 3;

  EXPECT_EQ(ncclInvalidUsage, Process());

  EXPECT_TRUE(Logged("Inconsistent rail/plane for net a0: existing 7/"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_CollPluginMarksCollAndKeepsNetUnlessSet) {
  Devs({"a0", "b0"});
  info_.net = false;
  info_.coll = true;
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("b0"), "net", "1"));

  ASSERT_EQ(ncclSuccess, Process());

  EXPECT_EQ("1", Attr(Net("a0"), "coll"));
  EXPECT_EQ("0", Attr(Net("a0"), "net"));
  EXPECT_EQ("1", Attr(Net("b0"), "net"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_GinPluginMovesLegacyNetAttributesToGinNodes) {
  Devs({"a0", "b0"});
  info_.net = false;
  info_.gin = true;
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "gin", "1"));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a0"), "ginonly", "x"));

  ASSERT_EQ(ncclSuccess, Process());

  ncclXmlNode* gin = Net("a0", "gin");
  ASSERT_NE(nullptr, gin);
  EXPECT_EQ("x", Attr(gin, "ginonly"));
  EXPECT_EQ("1", Attr(gin, "gin"));
  EXPECT_EQ("<unset>", Attr(Net("a0"), "gin"));
  EXPECT_EQ("<unset>", Attr(Net("b0", "gin"), "ginonly"));
  EXPECT_STREQ("cpu", gin->parent->parent->name);  // the plugin's sysfs path is not a PCI device
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_RmaPluginGetsRmaNodes) {
  Devs({"a0"});
  info_.net = false;
  info_.rma = true;

  ASSERT_EQ(ncclSuccess, Process());

  EXPECT_EQ("1", Attr(Net("a0", "rma"), "rma"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_MergesPortsIntoAVnicOnce) {
  Devs({"a0", "a1"});
  ScopedHook netGetDevCount(g_netGetDevCount, [](int, int* nPhys, int* nVirt) {
    *nPhys = 2;
    *nVirt = NCCL_UNDEF_DEV_COUNT;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, Process(nullptr));

  EXPECT_EQ(std::vector<int>({1}), g_setVirtDevCounts);
  ncclXmlNode* vnic = Net("v_a0_a1");
  ASSERT_NE(nullptr, vnic);
  EXPECT_EQ("2", Attr(vnic, "dev"));
  EXPECT_EQ("1", Attr(vnic, "keep"));
  EXPECT_EQ(Net("a0")->parent, vnic->parent);
  EXPECT_EQ("0", Attr(Net("a0"), "net"));
  EXPECT_EQ("0", Attr(Net("a0"), "keep"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_KnownVnicCountSkipsMerging) {
  Devs({"a0", "a1"});
  FakeNetDev v;
  v.name = "a0";  // the vNIC is already in the XML under this name
  v.vProps = ncclNetVDeviceProps_t{2, {0, 1}};
  g_netDevs.push_back(v);
  ScopedHook netGetDevCount(g_netGetDevCount, [](int, int* nPhys, int* nVirt) {
    *nPhys = 2;
    *nVirt = 1;
    return ncclSuccess;
  });
  ASSERT_EQ(ncclSuccess, xmlSetAttr(Net("a1"), "gin", "1"));

  ASSERT_EQ(ncclSuccess, Process(nullptr));

  EXPECT_TRUE(g_vDeviceCalls.empty());
  EXPECT_TRUE(g_setVirtDevCounts.empty());
  EXPECT_EQ("1", Attr(Net("a1"), "keep"));  // still used by the gin plugin
  EXPECT_EQ("2", Attr(Net("a0"), "dev"));
}

TEST_F(TopoProcessNetMicrotest, ProcessNet_PluginFailuresAreReturned) {
  Devs({"a0", "a1"});
  ScopedHook netGetDevCount(g_netGetDevCount, [](int, int*, int*) { return ncclSystemError; });
  EXPECT_EQ(ncclSystemError, Process());

  ScopedHook netGetDevCount2(g_netGetDevCount, [](int, int* nPhys, int* nVirt) {
    *nPhys = 3;
    *nVirt = 0;
    return ncclSuccess;
  });
  EXPECT_EQ(ncclInvalidArgument, Process());
}

TEST_F(TopoProcessNetMicrotest, MakeVNics_NothingToDoWithoutDevices) {
  EXPECT_EQ(ncclSuccess, ncclTopoMakeVNics(xml_.get(), &info_, 0));

  Devs({"a0", "a1", "b0"});
  info_.forceMerge = "b0,a1";
  ASSERT_EQ(ncclSuccess, ncclTopoMakeVNics(xml_.get(), &info_, 3));
  ASSERT_EQ(1u, g_vDeviceCalls.size());
  EXPECT_EQ(std::vector<int>({1, 2}), DevsOf(g_vDeviceCalls[0]));

  EXPECT_EQ(ncclInvalidArgument, ncclTopoMakeVNics(xml_.get(), &info_, 9));
}

TEST_F(TopoProcessNetMicrotest, UpdateVNics_PropertyFailuresAreReturned) {
  Devs({"a0"});
  FakeNetDev v;
  v.name = "v";
  v.vProps = ncclNetVDeviceProps_t{1, {7}};
  g_netDevs.push_back(v);

  EXPECT_EQ(ncclInvalidArgument, ncclTopoUpdateVNics(xml_.get(), &info_, 1, 1));
  EXPECT_EQ(ncclInvalidArgument, ncclTopoUpdateVNics(xml_.get(), &info_, 5, 1));
}

TEST_F(TopoMicrotest, GetXmlCpuArch_TakesTheFirstCpuThatHasOne) {
  XmlPtr xml = MustParseXml(System("<cpu numaid=\"0\"/><cpu numaid=\"1\" arch=\"arm64\"/>"));
  int arch = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlCpuArch(xml.get(), &arch));

  EXPECT_EQ(NCCL_TOPO_CPU_ARCH_ARM, arch);
}

TEST_F(TopoMicrotest, Lookups_StayOnTheirSystemAndWarnOnlyWhenAsked) {
  SystemPtr system = BuildSystem(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 3))) +
                                        Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x5\"")));
  int g = GpuIndexOfRank(system.get(), 0);
  int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].id);
  int rank = 7, netDev = 7;

  ASSERT_EQ(ncclSuccess, ncclTopoDevToRank(system.get(), systemId, 0, true, &rank));
  EXPECT_EQ(0, rank);
  EXPECT_EQ(ncclInternalError,
            ncclTopoDevToRank(system.get(), systemId, 1, false, &rank));  // dev 1 is on the other host
  EXPECT_EQ(-1, rank);
  EXPECT_FALSE(Logged("could not find rank for nvml dev 1"));
  EXPECT_EQ(ncclInternalError, ncclTopoIdToNetDev(system.get(), 0x99, &netDev));
  EXPECT_EQ(-1, netDev);
  EXPECT_TRUE(Logged("Could not find NET with id 99"));
  ASSERT_EQ(ncclSuccess, ncclTopoIdToNetDev(system.get(), system->nodes[NET].nodes[0].id, &netDev));
  EXPECT_EQ(3, netDev);  // the plugin's dev, not the node index

  int index = 7;
  EXPECT_EQ(ncclInternalError, ncclTopoRankToIndex(system.get(), 9, &index, false));
  EXPECT_EQ(-1, index);
  EXPECT_FALSE(Logged("ncclTopoRankToIndex could not find rank 9"));
  EXPECT_EQ(ncclInternalError, ncclTopoRankToIndex(system.get(), 9, &index, true));
  EXPECT_TRUE(Logged("ncclTopoRankToIndex could not find rank 9"));
}

TEST_F(TopoMicrotest, GetFusionEnv_ReadsEnvOrDefaultsByPlatform) {
  int level = -1;
  const char* force = "unset";

  ASSERT_EQ(ncclSuccess, ncclTopoGetFusionEnv(&level, &force, 60, NCCL_TOPO_CPU_ARCH_X86));
  EXPECT_EQ(PATH_PORT, level);
  EXPECT_EQ(nullptr, force);
  ASSERT_EQ(ncclSuccess, ncclTopoGetFusionEnv(&level, nullptr, 107, NCCL_TOPO_CPU_ARCH_ARM));
  EXPECT_EQ(PATH_PHB, level);
  ASSERT_EQ(ncclSuccess, ncclTopoGetFusionEnv(&level, nullptr, 107, NCCL_TOPO_CPU_ARCH_X86));
  EXPECT_EQ(PATH_PORT, level);
  RcclUnitTesting::ScopedEnv mergeLevel("NCCL_NET_MERGE_LEVEL", "PIX");
  RcclUnitTesting::ScopedEnv forceMerge("NCCL_NET_FORCE_MERGE", "a,b");
  ASSERT_EQ(ncclSuccess, ncclTopoGetFusionEnv(&level, &force, 60, NCCL_TOPO_CPU_ARCH_X86));
  EXPECT_EQ(PATH_PIX, level);
  EXPECT_STREQ("a,b", force);
}

TEST_F(TopoMicrotest, GetMergePolicy_RailOnVeraRubinOrByEnv) {
  int policy = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetMergePolicy(&policy, 60, NCCL_TOPO_CPU_ARCH_ARM));
  EXPECT_EQ(NCCL_NET_MERGE_POLICY_ALL, policy);
  ASSERT_EQ(ncclSuccess, ncclTopoGetMergePolicy(&policy, 107, NCCL_TOPO_CPU_ARCH_ARM));
  EXPECT_EQ(NCCL_NET_MERGE_POLICY_RAIL, policy);
  for (const char* env : {"rail", "ALL", "both"}) {
    RcclUnitTesting::ScopedEnv policyEnv("NCCL_NET_MERGE_POLICY", env);
    ASSERT_EQ(ncclSuccess, ncclTopoGetMergePolicy(&policy, 60, NCCL_TOPO_CPU_ARCH_X86));
    EXPECT_EQ(env[0] == 'r' ? NCCL_NET_MERGE_POLICY_RAIL : NCCL_NET_MERGE_POLICY_ALL, policy) << env;
  }
  EXPECT_TRUE(Logged("NCCL_NET_MERGE_POLICY: unknown value 'both', defaulting to ALL"));
  EXPECT_FALSE(Logged("unknown value 'ALL'"));
}

TEST_F(TopoMicrotest, XmlInitAttr_OnlySetsMissingValues) {
  XmlPtr xml = NewXml();
  ncclXmlNode* node = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), nullptr, "n", &node));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(node, "f", "keep"));

  ASSERT_EQ(ncclSuccess, xmlInitAttrInt(node, "i", 3));
  ASSERT_EQ(ncclSuccess, xmlInitAttrInt(node, "i", 4));
  ASSERT_EQ(ncclSuccess, xmlInitAttrUint64(node, "u", 0x1f));
  ASSERT_EQ(ncclSuccess, xmlInitAttrFloat(node, "f", 1.5f));
  ASSERT_EQ(ncclSuccess, xmlInitAttrFloat(node, "g", 1.5f));

  EXPECT_STREQ("3", node->attrs[1].value);
  EXPECT_STREQ("0x1f", node->attrs[2].value);
  EXPECT_STREQ("keep", node->attrs[0].value);
  EXPECT_STREQ("1.500000", node->attrs[3].value);
  for (int a = node->nAttrs; a < MAX_ATTR_COUNT; a++) {
    ASSERT_EQ(ncclSuccess, xmlSetAttrInt(node, std::to_string(a).c_str(), a));
  }
  EXPECT_EQ(ncclInternalError, xmlInitAttrInt(node, "x", 1));
  EXPECT_EQ(ncclInternalError, xmlInitAttrUint64(node, "x", 1));
  EXPECT_EQ(ncclInternalError, xmlInitAttrFloat(node, "x", 1));
}

TEST_F(TopoFusionMicrotest, GetPath_NodeAndItsAncestorMeetAtTheAncestor) {
  ncclXmlNode* nodes[] = {Net("a0"), Net("a0")->parent};
  ncclXmlNode* parent = nullptr;
  int path = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetPath(nodes, 2, &path, &parent));

  EXPECT_EQ(PATH_PORT, path);
}

class TopoOrphanNicMicrotest : public TopoFusionMicrotest {
 protected:
  // A NIC hung straight off <system>, which the parser never produces.
  void OrphanNet(const char* name) {
    ncclXmlNode* nic = nullptr;
    ncclXmlNode* net = nullptr;
    ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), xml_->nodes, "nic", &nic));
    ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nic, "net", &net));
    ASSERT_EQ(ncclSuccess, xmlSetAttr(net, "name", name));
  }

  ncclResult_t Parent() {
    ncclNetVDeviceProps_t v{2, {0, 1}};
    ncclXmlNode* parent = nullptr;
    return ncclTopoGetVNicParent(xml_.get(), "net", FakeGetProperties, &v, &parent);
  }
};

TEST_F(TopoOrphanNicMicrotest, GetVNicParent_CrossNumaWithoutACpuAncestorFallsBackToTheFirstCpu) {
  OrphanNet("x0");
  Devs({"x0", "d0"});

  EXPECT_EQ(ncclInternalError, Parent());  // the first CPU has no PCI device above x0 to clone

  EXPECT_TRUE(Logged("Cannot find CPU parent for cross-NUMA vNIC, attaching to first CPU"));
  EXPECT_TRUE(Logged("Can't find busId of child 0 x0"));
}

TEST_F(TopoOrphanNicMicrotest, GetVNicParent_NoCpuAtAllFails) {
  xml_ = NewXml();
  ncclXmlNode* system = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), nullptr, "system", &system));
  OrphanNet("x0");
  OrphanNet("y0");
  Devs({"x0", "y0"});

  EXPECT_EQ(ncclInternalError, Parent());

  EXPECT_TRUE(Logged("No CPU node found in topology, cannot place cross-NUMA vNIC"));
}

TEST_F(TopoMicrotest, NetRailKeyIndex_FailsPastMaxRails) {
  ncclTopoNetRailKeyList list;
  int rail = -1;
  for (uint64_t k = 0; k < NCCL_TOPO_MAX_NODES; k++) {
    ASSERT_EQ(ncclSuccess, ncclTopoGetNetRailKeyIndex(&list, k, &rail));
  }
  EXPECT_EQ(NCCL_TOPO_MAX_NODES - 1, rail);

  ASSERT_EQ(ncclSuccess, ncclTopoGetNetRailKeyIndex(&list, 5, &rail));
  EXPECT_EQ(5, rail);
  EXPECT_EQ(ncclInternalError, ncclTopoGetNetRailKeyIndex(&list, NCCL_TOPO_MAX_NODES, &rail));
  EXPECT_TRUE(Logged("NCCL cannot use more than 144 rails"));
}

TEST_F(TopoLocalMicrotest, GetLocalNet_SameRailAndPlaneKeepsDiscoveryOrder) {
  SystemPtr system = BuildWithPaths(SharedSwitchSystem({RailNet("net", 0, 2, 1), RailNet("net", 1, 2, 1)}));
  SetPolicy(NETDEVS_POLICY_ALL);

  EXPECT_EQ(0, LocalNetDev(system.get(), 0, 0));
  EXPECT_EQ(1, LocalNetDev(system.get(), 0, 1));
}

TEST_F(TopoMicrotest, NvLinkBw_ByComputeCapability) {
  EXPECT_FLOAT_EQ(RUBIN_NVLINK_BW, ncclTopoNVLinkBw(107));
  EXPECT_FLOAT_EQ(SM100_NVLINK_BW, ncclTopoNVLinkBw(110));
  EXPECT_FLOAT_EQ(SM90_NVLINK_BW, ncclTopoNVLinkBw(90));
  EXPECT_FLOAT_EQ(SM86_NVLINK_BW, ncclTopoNVLinkBw(86));
  EXPECT_FLOAT_EQ(SM80_NVLINK_BW, ncclTopoNVLinkBw(80));
  EXPECT_FLOAT_EQ(SM70_NVLINK_BW, ncclTopoNVLinkBw(70));
  EXPECT_FLOAT_EQ(SM60_NVLINK_BW, ncclTopoNVLinkBw(60));
  EXPECT_FLOAT_EQ(SM80_NVLINK_BW, ncclTopoNVLinkBw(50));
}

// --- Network property snapshots

int g_snapshotDevs = 2;
ncclResult_t g_snapshotDevicesResult = ncclSuccess;
std::string g_longName(PATH_MAX, 'n');

ncclResult_t SnapshotDevices(int* n) {
  *n = g_snapshotDevs;
  return g_snapshotDevicesResult;
}

ncclResult_t SnapshotProperties(int dev, ncclNetProperties_t* props) {
  if (dev == 1) return ncclSystemError;
  props->name = dev == 0 ? const_cast<char*>("mlx5_0") : (dev == 2 ? &g_longName[0] : nullptr);
  props->pciPath = dev == 0 ? const_cast<char*>("/sys/x") : (dev == 2 ? &g_longName[0] : nullptr);
  props->speed = 400000 + dev;
  props->port = dev;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->vProps.ndevs = 1;
  return ncclSuccess;
}

class TopoSnapshotMicrotest : public TopoMicrotest {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  ncclNet_t net_{};
  ncclTopoNetPropertiesSnapshot* snapshots_ = nullptr;
  int count_ = -1;

  void SetUp() override {
    TopoMicrotest::SetUp();
    g_snapshotDevs = 2;
    g_snapshotDevicesResult = ncclSuccess;
    net_.devices = SnapshotDevices;
    net_.getProperties = SnapshotProperties;
    comm_->ncclNet = &net_;
  }
  void TearDown() override {
    free(snapshots_);
    TopoMicrotest::TearDown();
  }

  ncclResult_t Snapshot(bool requireInit = false) {
    return ncclTopoGetNetPropertiesSnapshot(comm_.get(), requireInit, &snapshots_, &count_);
  }
};

TEST_F(TopoSnapshotMicrotest, Snapshot_CopiesEachDeviceAndSkipsFailedOnes) {
  g_snapshotDevs = 4;

  ASSERT_EQ(ncclSuccess, Snapshot(true));

  ASSERT_EQ(4, count_);
  EXPECT_TRUE(snapshots_[0].propertiesValid);
  EXPECT_STREQ("mlx5_0", snapshots_[0].name);
  EXPECT_STREQ("/sys/x", snapshots_[0].pciPath);
  EXPECT_EQ(400000, snapshots_[0].speed);
  EXPECT_EQ(1, snapshots_[0].vProps.ndevs);
  EXPECT_FALSE(snapshots_[1].propertiesValid);
  EXPECT_TRUE(snapshots_[2].propertiesValid);
  EXPECT_STREQ("", snapshots_[2].name);  // too long to keep
  EXPECT_STREQ("", snapshots_[2].pciPath);
  EXPECT_EQ(3, snapshots_[3].port);
  EXPECT_STREQ("", snapshots_[3].name);  // NULL from the plugin
}

TEST_F(TopoSnapshotMicrotest, Snapshot_ArgumentsAndPluginStateAreChecked) {
  EXPECT_EQ(ncclInvalidArgument, ncclTopoGetNetPropertiesSnapshot(nullptr, false, &snapshots_, &count_));
  EXPECT_EQ(ncclInvalidArgument, ncclTopoGetNetPropertiesSnapshot(comm_.get(), false, nullptr, &count_));
  EXPECT_EQ(ncclInvalidArgument, ncclTopoGetNetPropertiesSnapshot(comm_.get(), false, &snapshots_, nullptr));
  EXPECT_TRUE(Logged("invalid arguments for network properties snapshot"));

  comm_->initState = ncclInProgress;
  EXPECT_EQ(ncclInProgress, Snapshot(true));
  EXPECT_EQ(ncclSuccess, Snapshot(false));
  EXPECT_EQ(2, count_);
  free(snapshots_);
  snapshots_ = nullptr;

  net_.getProperties = nullptr;
  EXPECT_EQ(ncclInvalidArgument, Snapshot());
  EXPECT_TRUE(Logged("network plugin has invalid properties callbacks"));
  net_.getProperties = SnapshotProperties;
  net_.devices = nullptr;
  EXPECT_EQ(ncclInvalidArgument, Snapshot());

  comm_->ncclNet = nullptr;
  EXPECT_EQ(ncclSuccess, Snapshot());
  EXPECT_EQ(0, count_);
  EXPECT_EQ(nullptr, snapshots_);
}

TEST_F(TopoSnapshotMicrotest, Snapshot_DeviceCountProblems) {
  g_snapshotDevicesResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, Snapshot());

  g_snapshotDevicesResult = ncclSuccess;
  g_snapshotDevs = -1;
  EXPECT_EQ(ncclInternalError, Snapshot());
  EXPECT_TRUE(Logged("network plugin returned invalid device count -1"));

  g_snapshotDevs = 0;
  EXPECT_EQ(ncclSuccess, Snapshot());
  EXPECT_EQ(0, count_);
  EXPECT_EQ(nullptr, snapshots_);
}

// --- ncclTopoGetSystem

class TopoGetSystemMicrotest : public TopoMicrotest {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> shared_ = std::make_unique<ncclSharedResources>();
  std::vector<ncclPeerInfo> peers_;
  ncclNet_t net_{};
  std::unique_ptr<TempFile> topoFile_;
  ncclTopoSystem* system_ = nullptr;
  std::vector<int> allGatherRanks_;

  void SetUp() override {
    TopoMicrotest::SetUp();
    g_netDevs.clear();
    for (int n = 0; n < 2; n++) {
      FakeNetDev d;
      d.name = "mlx5_" + std::to_string(n);
      d.railId = n;
      d.planeId = 0;
      d.guid = 0x100 + n;
      g_netDevs.push_back(d);
    }
    net_.name = "fake";
    net_.devices = FakeDevices;
    net_.getProperties = FakeGetProperties;
    comm_->ncclNet = &net_;
    comm_->ncclNetVer = 12;
    comm_->sharedRes = shared_.get();
    comm_->minDriverVersion = 7;
    comm_->cuMemGdrSupport = true;
    Ranks({getHostHash()}, 0);
    g_netGetDevCount = [](int, int* nPhys, int* nVirt) {
      *nPhys = 2;
      *nVirt = 0;
      return ncclSuccess;
    };
    // The XML dev is the SMI index; the HIP ordinal runs the other way, as with reordered device visibility.
    g_amdSmiGetDeviceIndexByPciBusId = [](const char* busId, uint32_t* index) {
      *index = std::string(busId) == GpuBus(0) ? 0 : 1;
      return ncclSuccess;
    };
    g_hipDeviceGetByPCIBusId = [](int* dev, const char* busId) {
      *dev = std::string(busId) == GpuBus(0) ? 1 : 0;
      return hipSuccess;
    };
    // Every local rank sees the same node, so the gather hands each the caller's own XML.
    g_bootstrapIntraNodeAllGather = [this](void*, int* ranks, int rank, int nranks, void* data, int size) {
      allGatherRanks_.assign(ranks, ranks + nranks);
      for (int r = 0; r < nranks; r++) {
        if (r != rank) memcpy((char*)data + (size_t)r * size, (char*)data + (size_t)rank * size, size);
      }
      return ncclSuccess;
    };
  }

  void TearDown() override {
    if (system_) ncclTopoFree(system_);
    unsetenv("NCCL_TOPO_FILE");
    TopoMicrotest::TearDown();
  }

  void Ranks(const std::vector<uint64_t>& hosts, int rank) {
    peers_.assign(hosts.size(), ncclPeerInfo{});
    for (size_t r = 0; r < hosts.size(); r++) {
      peers_[r].hostHash = hosts[r];
      peers_[r].busId = r == 0 ? 0x10000 : 0x20000;
      peers_[r].gdrSupport = 1;
      peers_[r].mloPart = NCCL_TOPO_UNDEF;
    }
    comm_->peerInfo = peers_.data();
    comm_->nRanks = hosts.size();
    comm_->rank = rank;
  }

  void TopoFile(const std::string& text) {
    topoFile_ = std::make_unique<TempFile>(text);
    setenv("NCCL_TOPO_FILE", topoFile_->path(), 1);
  }

  ncclResult_t GetSystem(const char* dumpXmlFile = nullptr) {
    return ncclTopoGetSystem(comm_.get(), dumpXmlFile ? nullptr : &system_, dumpXmlFile);
  }
};

TEST_F(TopoGetSystemMicrotest, GetSystem_KeepsThisRanksGpuAndThePluginNics) {
  TopoFile(XgmiSystem(2, 1, true));

  ASSERT_EQ(ncclSuccess, GetSystem());

  ASSERT_NE(nullptr, system_);
  ASSERT_EQ(1, system_->nodes[GPU].count);
  EXPECT_EQ(0, system_->nodes[GPU].nodes[0].gpu.rank);
  EXPECT_EQ(NCCL_TOPO_UNDEF, system_->nodes[GPU].nodes[0].gpu.mloPart);
  EXPECT_EQ(2, system_->nodes[NET].count);
  EXPECT_EQ(getHostHash(), system_->hostHashes[0]);
  EXPECT_EQ(7, system_->minDriverVersion);
  EXPECT_TRUE(system_->cuMemGdrSupport);
  EXPECT_EQ(std::vector<int>({0}), allGatherRanks_);
  EXPECT_TRUE(Logged("NCCL_TOPO_FILE set by environment to"));
}

TEST_F(TopoGetSystemMicrotest, GetSystem_GathersFromTheRanksOnThisHost) {
  TopoFile(XgmiSystem(2, 1, true));
  Ranks({getHostHash(), 1, getHostHash()}, 2);

  ASSERT_EQ(ncclSuccess, GetSystem());

  EXPECT_EQ(std::vector<int>({0, 2}), allGatherRanks_);
  EXPECT_EQ(2, system_->nodes[GPU].nodes[0].gpu.rank);
}

TEST_F(TopoGetSystemMicrotest, GetSystem_MnnvlGathersOverTheClique) {
  TopoFile(XgmiSystem(2, 1, true));
  int clique[] = {0};
  comm_->MNNVL = 1;
  comm_->clique.size = 1;
  comm_->clique.ranks = clique;
  comm_->cliqueRank = 0;

  ASSERT_EQ(ncclSuccess, GetSystem());

  EXPECT_EQ(std::vector<int>({0}), allGatherRanks_);
  EXPECT_EQ(1, system_->nodes[GPU].count);
}

TEST_F(TopoGetSystemMicrotest, GetSystem_DumpRankWritesTheXmlInsteadOfBuilding) {
  TopoFile(XgmiSystem(2, 1, true));
  TempFile dump;

  ASSERT_EQ(ncclSuccess, GetSystem(dump.path()));

  std::string xml = dump.Read();
  EXPECT_NE(std::string::npos, xml.find("name=\"mlx5_1\""));
  EXPECT_NE(std::string::npos, xml.find("rank=\"0\""));
  EXPECT_EQ(std::string::npos, xml.find("busid=\"" + GpuBus(1) + "\""));
  EXPECT_EQ(nullptr, system_);
}

TEST_F(TopoGetSystemMicrotest, GetSystem_OtherRanksDoNotDump) {
  TopoFile(XgmiSystem(2, 1, true));
  g_params["TOPO_DUMP_FILE_RANK"] = 1;
  TempFile dump;

  ASSERT_EQ(ncclSuccess, GetSystem(dump.path()));

  EXPECT_EQ("", dump.Read());
}

TEST_F(TopoGetSystemMicrotest, GetSystem_ImportsGinRmaAndCollNetDevices) {
  TopoFile(XgmiSystem(2, 1, true));
  ncclGin_t gin{};
  gin.name = "gin";
  gin.devices = FakeDevices;
  gin.getProperties = FakeGetProperties;
  shared_->ginState.supported = true;
  shared_->ginState.backends[0].ncclGin = &gin;
  ncclRma_t rma{};
  rma.name = "rma";
  rma.devices = FakeDevices;
  rma.getProperties = FakeGetProperties;
  comm_->rmaState.rmaProxyState.ncclRma = &rma;
  ncclCollNet_t coll{};
  coll.name = "coll";
  coll.devices = FakeDevices;
  coll.getProperties = FakeGetProperties;
  comm_->ncclCollNet = &coll;

  ASSERT_EQ(ncclSuccess, GetSystem());

  EXPECT_EQ(2, system_->nodes[GIN].count);
  EXPECT_EQ(2, system_->nodes[RMA].count);
  EXPECT_EQ(1, system_->nodes[NET].nodes[0].net.collSupport);
}

TEST_F(TopoGetSystemMicrotest, GetSystem_WithoutATopoFileDetectsTheGpu) {
  ScopedHook netGetDevCount(g_netGetDevCount, [](int, int* nPhys, int* nVirt) {
    *nPhys = *nVirt = 0;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, GetSystem());

  ASSERT_EQ(1, system_->nodes[GPU].count);
  EXPECT_STREQ("gfx942", system_->nodes[GPU].nodes[0].gpu.gcn);
  EXPECT_EQ(304, system_->nodes[GPU].nodes[0].gpu.cu);
  EXPECT_EQ(0, system_->nodes[NET].count);
}

TEST_F(TopoGetSystemMicrotest, GetSystem_FailuresAreReturned) {
  TopoFile(XgmiSystem(2, 1, true));
  g_amdSmiInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, GetSystem());

  g_amdSmiInitResult = ncclSuccess;
  ScopedHook allGather(g_bootstrapIntraNodeAllGather,
                       [](void*, int*, int, int, void*, int) { return ncclRemoteError; });
  EXPECT_EQ(ncclRemoteError, GetSystem());

  TopoFile("<system version=\"3\"/>");
  EXPECT_EQ(ncclInvalidUsage, GetSystem());
  EXPECT_EQ(nullptr, system_);
}

}  // namespace
