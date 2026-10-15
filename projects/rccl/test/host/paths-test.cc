/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/paths.cc: path computation over topologies built from XML, the P2P/GDR/PXN
// decisions read from those paths, system trimming and the P2P channel counts.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include PATHS_CC_PATH

#include "TopoScene.h"

using namespace topo_scene;

namespace {

class PathsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetGraphFakes();
    ncclTopoUserP2pLevel = -1;
    ncclTopoUserGdrLevel = -1;
  }
  void TearDown() override {
    ncclTopoUserP2pLevel = -1;
    ncclTopoUserGdrLevel = -1;
    unsetenv("NCCL_P2P_LEVEL");
    unsetenv("NCCL_P2P_DISABLE");
    unsetenv("NCCL_NET_GDR_LEVEL");
    ResetGraphFakes();
  }

  static ncclTopoLinkList& Path(ncclTopoSystem* system, int fromType, int from, int toType, int to) {
    return system->nodes[fromType].nodes[from].paths[toType][to];
  }
};

// Two partitions of one GPU and a NIC behind one switch.
std::string PartitionedSwitch(const std::string& gpuAttrs0, const std::string& gpuAttrs1) {
  std::string pci = "<pci busid=\"" + GpuBus(0) +
                    "\" class=\"0x120000\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">" +
                    Gpu(0, 0, "", gpuAttrs0) + Gpu(1, 1, "", gpuAttrs1) + "</pci>";
  return Pci(SwitchBus(0), pci + NicPci(NicBus(0), Net("a", 0)));
}

// Two GPUs per switch, one switch per GPU pair; no XGMI.
std::string PcieSystem() {
  return System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1)) +
                           Pci(SwitchBus(2), GpuPci(GpuBus(2), 2, 2))) +
                Cpu(1, Pci(SwitchBus(3), GpuPci(GpuBus(3), 3, 3))));
}

// --- Path computation

TEST_F(PathsMicrotest, ComputePaths_XgmiPeersAreNvlThroughTheirDevs) {
  SystemPtr system = BuildWithPaths(XgmiSystem(4, 1, true));

  ncclTopoLinkList& peer = Path(system.get(), GPU, 0, GPU, 3);
  EXPECT_EQ(PATH_NVL, peer.type);
  EXPECT_EQ(3, peer.count);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, peer.bw);
  ncclTopoLinkList& self = Path(system.get(), GPU, 1, GPU, 1);
  EXPECT_EQ(PATH_LOC, self.type);
  EXPECT_EQ(0, self.count);
  EXPECT_FLOAT_EQ(LOC_BW, self.bw);
  ncclTopoLinkList& nic = Path(system.get(), GPU, 2, NET, 2);
  EXPECT_EQ(PATH_PIX, nic.type);
  EXPECT_EQ(4, nic.count);
  EXPECT_FLOAT_EQ(48, nic.bw);
  EXPECT_EQ(PATH_PHB, Path(system.get(), GPU, 2, NET, 0).type);
  EXPECT_EQ(2, system->nodes[NET].nodes[2].net.localGpu);
}

TEST_F(PathsMicrotest, ComputePaths_PciePathsFollowTheCommonAncestor) {
  SystemPtr system = BuildWithPaths(PcieSystem());

  EXPECT_EQ(PATH_PIX, Path(system.get(), GPU, 0, GPU, 1).type);
  EXPECT_EQ(PATH_PHB, Path(system.get(), GPU, 0, GPU, 2).type);
  EXPECT_EQ(PATH_SYS, Path(system.get(), GPU, 0, GPU, 3).type);
  EXPECT_EQ(PATH_SYS, Path(system.get(), GPU, 3, GPU, 1).type);
}

TEST_F(PathsMicrotest, ComputePaths_StackedSwitchesArePxb) {
  std::string inner = Pci("0000:30:00.0", GpuPci(GpuBus(1), 1, 1));
  SystemPtr system = BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + inner))));

  EXPECT_EQ(PATH_PXB, Path(system.get(), GPU, 0, GPU, 1).type);
}

// GPU0-DEV0-DEV1-DEV2-GPU2: one hop through another device's XGMI is NVB, unless disabled.
std::string XgmiRing4() {
  std::string body;
  for (int g = 0; g < 4; g++) body += GpuPci(GpuBus(g), g, g, Xgmi(GpuBus((g + 1) % 4)) + Xgmi(GpuBus((g + 3) % 4)));
  return System(Cpu(0, body));
}

TEST_F(PathsMicrotest, ComputePaths_EqualHopRoutesKeepTheWiderOne) {
  std::string body = GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1)) + Xgmi(GpuBus(3), 2)) +
                     GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0)) + Xgmi(GpuBus(2))) +
                     GpuPci(GpuBus(2), 2, 2, Xgmi(GpuBus(1)) + Xgmi(GpuBus(3), 2)) +
                     GpuPci(GpuBus(3), 3, 3, Xgmi(GpuBus(2), 2) + Xgmi(GpuBus(0), 2));
  SystemPtr system = BuildWithPaths(System(Cpu(0, body)));

  ncclTopoLinkList& path = Path(system.get(), GPU, 0, GPU, 2);
  EXPECT_EQ(PATH_NVB, path.type);
  EXPECT_FLOAT_EQ(2 * GFX94X_XGMI_WIDTH, path.bw);  // through GPU 3, not GPU 1
}

TEST_F(PathsMicrotest, ComputePaths_TwoXgmiHopsAreNvbUnlessDisabled) {
  SystemPtr system = BuildWithPaths(XgmiRing4());
  EXPECT_EQ(PATH_NVB, Path(system.get(), GPU, 0, GPU, 2).type);
  EXPECT_EQ(4, Path(system.get(), GPU, 0, GPU, 2).count);
  EXPECT_EQ(PATH_NVL, Path(system.get(), GPU, 0, GPU, 1).type);

  g_params["NVB_DISABLE"] = 1;
  SystemPtr noNvb = BuildWithPaths(XgmiRing4());
  EXPECT_EQ(PATH_PHB, Path(noNvb.get(), GPU, 0, GPU, 2).type);
}

TEST_F(PathsMicrotest, ComputePaths_XgmiSwitchIsOneNvlHopBetweenDevs) {
  std::string toSwitch = Xgmi("fffffff:ffff:ff", 1, "0x068000");
  SystemPtr system =
    BuildWithPaths(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, toSwitch) + GpuPci(GpuBus(1), 1, 1, toSwitch))));

  ncclTopoLinkList& path = Path(system.get(), GPU, 0, GPU, 1);
  EXPECT_EQ(PATH_NVL, path.type);
  EXPECT_EQ(4, path.count);
  EXPECT_EQ(PATH_NVL, Path(system.get(), GPU, 1, NVS, 0).type);
  EXPECT_EQ(PATH_LOC, Path(system.get(), NVS, 0, NVS, 0).type);
}

TEST_F(PathsMicrotest, ComputePaths_NoGdrNicTrafficGoesThroughTheCpu) {
  std::string body = Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0, "")));
  std::string noGdr = body;
  noGdr.replace(noGdr.find("gdr=\"1\" rail"), 7, "gdr=\"0\"");
  SystemPtr system = BuildWithPaths(System(Cpu(0, noGdr)));

  ncclTopoLinkList& path = Path(system.get(), GPU, 0, NET, 0);
  EXPECT_EQ(PATH_PHB, path.type);
  EXPECT_EQ(4 + 2, path.count);  // up to the CPU and back down
  EXPECT_EQ(PATH_PHB, Path(system.get(), NET, 0, GPU, 0).type);
}

TEST_F(PathsMicrotest, ComputePaths_UserP2pLevelDivertsGpuPairsThroughTheCpu) {
  setenv("NCCL_P2P_LEVEL", "LOC", 1);

  SystemPtr system = BuildWithPaths(XgmiSystem(2));

  ncclTopoLinkList& path = Path(system.get(), GPU, 0, GPU, 1);
  EXPECT_EQ(PATH_PHB, path.type);
  EXPECT_EQ(3 + 3, path.count);
  EXPECT_TRUE(Logged("NCCL_P2P_LEVEL set by environment to LOC"));
}

TEST_F(PathsMicrotest, ComputePaths_C2cCpuWithAPhbNicBecomesP2c) {
  XmlPtr xml = MustParseXml(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0)))));
  ncclXmlNode* gpu = nullptr;
  ncclXmlNode* c2c = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml.get(), "gpu", &gpu));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), gpu, "c2c", &c2c));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "count", 10));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "bw", 10000));
  SystemPtr system = BuildSystem(xml.get());
  system->nRanks = 1;
  system->netGdrLevel = -2;

  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system.get(), nullptr));

  EXPECT_EQ(PATH_C2C, Path(system.get(), GPU, 0, CPU, 0).type);
  EXPECT_EQ(PATH_P2C, Path(system.get(), GPU, 0, NET, 0).type);
  EXPECT_EQ(PATH_P2C, Path(system.get(), NET, 0, GPU, 0).type);
}

std::string DomainNicSystem(const std::string& gcn, const std::string& nicBus) {
  return System(Cpu(0, GpuPci(GpuBus(0), 0, 0, "", "", gcn) + NicPci(nicBus, Net("a", 0))));
}

TEST_F(PathsMicrotest, ComputePaths_Gfx1250SameDomainPhbNicBecomesPxb) {
  SystemPtr same = BuildWithPaths(DomainNicSystem("gfx1250", "0000:90:00.0"));
  SystemPtr other = BuildWithPaths(DomainNicSystem("gfx1250", "0001:90:00.0"));
  SystemPtr gfx942 = BuildWithPaths(DomainNicSystem("gfx942", "0000:90:00.0"));

  EXPECT_EQ(PATH_PXB, Path(same.get(), GPU, 0, NET, 0).type);
  EXPECT_EQ(PATH_PXB, Path(same.get(), NET, 0, GPU, 0).type);
  EXPECT_EQ(PATH_PXB, Path(same.get(), DEV, 0, NET, 0).type);
  EXPECT_EQ(PATH_PHB, Path(other.get(), GPU, 0, NET, 0).type);
  EXPECT_EQ(PATH_PHB, Path(gfx942.get(), GPU, 0, NET, 0).type);
  EXPECT_TRUE(Logged("Rewrote same-domain DEV 0 -> NET 0 path 8->5 (domain 0x0000)"));
}

// Under Hyper-V, Intel hosts with gfx942/gfx950 hide the PCI tree, so NICs a fixed bus offset from a GPU are
// treated as PXB.
class PathsHyperVMicrotest : public PathsMicrotest {
 protected:
  void SetUp() override {
    PathsMicrotest::SetUp();
    g_sysfs["/sys/devices/virtual/dmi/id/bios_version"] = "Hyper-V UEFI Release v4.1";
  }

  SystemPtr Build(const std::vector<std::string>& nicBuses, int nRanks) {
    std::string body;
    for (size_t g = 0; g < nicBuses.size(); g++) {
      body += GpuPci(GpuBus(g), g, g) + NicPci(nicBuses[g], Net("n" + std::to_string(g), g));
    }
    SystemPtr system = BuildSystem(System(Cpu(0, body, "", kIntelCpu)));
    system->nRanks = nRanks;
    system->netGdrLevel = -2;
    EXPECT_EQ(ncclSuccess, ncclTopoComputePaths(system.get(), nullptr));
    return system;
  }
};

TEST_F(PathsHyperVMicrotest, ComputePaths_NicsAtTheBusOffsetOfEveryGpuArePxb) {
  for (const std::vector<std::string>& nics : {std::vector<std::string>{"0001:10:00.0", "0001:20:00.0"},
                                               std::vector<std::string>{"0000:11:00.0", "0000:21:00.0"},
                                               std::vector<std::string>{"00ff:10:00.0", "00ff:20:00.0"}}) {
    SystemPtr system = Build(nics, 4);

    EXPECT_EQ(PATH_PXB, Path(system.get(), GPU, 1, NET, 1).type) << nics[0];
    EXPECT_EQ(PATH_PHB, Path(system.get(), GPU, 1, NET, 0).type) << nics[0];
  }
}

TEST_F(PathsHyperVMicrotest, ComputePaths_NoOverrideUnlessEveryGpuHasSuchANic) {
  SystemPtr partial = Build({"0001:10:00.0", "0000:90:00.0"}, 4);
  EXPECT_EQ(PATH_PHB, Path(partial.get(), GPU, 0, NET, 0).type);

  SystemPtr singleNode = Build({"0001:10:00.0", "0001:20:00.0"}, 2);  // full node, but not 8 GPUs
  EXPECT_EQ(PATH_PHB, Path(singleNode.get(), GPU, 0, NET, 0).type);

  g_sysfs.clear();
  SystemPtr bareMetal = Build({"0001:10:00.0", "0001:20:00.0"}, 4);
  EXPECT_EQ(PATH_PHB, Path(bareMetal.get(), GPU, 0, NET, 0).type);
}

TEST_F(PathsMicrotest, ComputePaths_FailsOnALinkWithoutItsReverse) {
  SystemPtr system = BuildSystem(XgmiSystem(1));
  ncclTopoNode* gpu = system->nodes[GPU].nodes;
  ncclTopoNode* cpu = system->nodes[CPU].nodes;
  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(cpu, gpu, LINK_NVL, 1));

  EXPECT_EQ(ncclInternalError, ncclTopoComputePaths(system.get(), nullptr));

  EXPECT_TRUE(Logged("Failed to find reverse path from remNode"));
}

TEST_F(PathsMicrotest, GetPath_UnknownNodeIsAnError) {
  SystemPtr system = BuildWithPaths(XgmiSystem(1));
  ncclTopoLinkList* path = nullptr;

  EXPECT_EQ(ncclInternalError, getPath(system.get(), system->nodes[GPU].nodes, CPU, 0x99, &path));

  EXPECT_TRUE(Logged("Could not find node of type 3 id 99"));
}

TEST_F(PathsMicrotest, PathReserve_GrowsAndCapsAtTheMaxHops) {
  ncclTopoLinkList path{};

  ASSERT_EQ(ncclSuccess, ncclTopoPathReserve(&path, 3));
  EXPECT_EQ(5, path.capacity);
  ASSERT_EQ(ncclSuccess, ncclTopoPathReserve(&path, 4));
  EXPECT_EQ(5, path.capacity);
  ASSERT_EQ(ncclSuccess, ncclTopoPathReserve(&path, 6));
  EXPECT_EQ(10, path.capacity);
  ASSERT_EQ(ncclSuccess, ncclTopoPathReserve(&path, NCCL_TOPO_MAX_HOPS));
  EXPECT_EQ(NCCL_TOPO_MAX_HOPS, path.capacity);
  EXPECT_EQ(ncclInternalError, ncclTopoPathReserve(&path, NCCL_TOPO_MAX_HOPS + 1));
  EXPECT_TRUE(Logged("hops, maximum is"));
  free(path.list);
}

TEST_F(PathsMicrotest, MergePathType_PhbOverC2cIsP2c) {
  EXPECT_EQ(PATH_P2C, mergePathType(PATH_PHB, PATH_C2C));
  EXPECT_EQ(PATH_P2C, mergePathType(PATH_C2C, PATH_PHB));
  EXPECT_EQ(PATH_SYS, mergePathType(PATH_C2C, PATH_SYS));
  EXPECT_EQ(PATH_PHB, mergePathType(PATH_PIX, PATH_PHB));
}

TEST_F(PathsMicrotest, AddInterStep_ConcatenatesAndMarksGpuRelaysPxn) {
  std::string nic = NicPci(NicBus(0), RailNet("net", 0, 0, 0, 100000));
  SystemPtr system = BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + nic)) +
                                           Cpu(1, GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))))));
  ncclTopoLinkList& viaGpu = Path(system.get(), GPU, 1, NET, 0);
  int toPeer = Path(system.get(), GPU, 1, GPU, 0).count;
  int peerToNic = Path(system.get(), GPU, 0, NET, 0).count;

  ASSERT_EQ(ncclSuccess, addInterStep(system.get(), GPU, 0, GPU, 1, NET, 0));

  EXPECT_EQ(PATH_PXN, viaGpu.type);
  EXPECT_EQ(toPeer + peerToNic, viaGpu.count);
  EXPECT_FLOAT_EQ(12.5, viaGpu.bw);  // the slower of the two legs
}

TEST_F(PathsMicrotest, GetLocalCpu_NeedsAPathToACpu) {
  SystemPtr system = BuildWithPaths(XgmiSystem(1));
  int cpu = -1;
  ASSERT_EQ(ncclSuccess, ncclGetLocalCpu(system.get(), 0, &cpu));
  EXPECT_EQ(0, cpu);

  Path(system.get(), GPU, 0, CPU, 0).count = 0;
  EXPECT_EQ(ncclInternalError, ncclGetLocalCpu(system.get(), 0, &cpu));
  EXPECT_TRUE(Logged("could not find CPU close to GPU 0"));
}

TEST_F(PathsMicrotest, GetLocalCpu_PicksTheFewestHops) {
  SystemPtr system = BuildWithPaths(XgmiSystem(4, 2));
  int cpu = -1;

  ASSERT_EQ(ncclSuccess, ncclGetLocalCpu(system.get(), 3, &cpu));

  EXPECT_EQ(1, cpu);
}

TEST_F(PathsMicrotest, PrintPaths_LogsEachSourceNode) {
  SystemPtr system = BuildWithPaths(
    System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0) + RailNet("gin", 1) + RailNet("rma", 2)))));

  ASSERT_EQ(ncclSuccess, ncclTopoPrintPaths(system.get()));

  EXPECT_TRUE(Logged("GPU/0-10000 :"));
  EXPECT_TRUE(Logged("NET/0-0 :"));
  EXPECT_TRUE(Logged("GIN/0-1 :"));
  EXPECT_TRUE(Logged("RMA/0-2 :"));
  EXPECT_TRUE(Logged("CPU/0-0 (2/48.0/PHB)"));
}

TEST_F(PathsMicrotest, RemovePaths_FreesEveryPathList) {
  SystemPtr system = BuildWithPaths(XgmiSystem(2, 1, true));

  ncclTopoRemovePaths(system.get());

  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    for (int n = 0; n < system->nodes[t].count; n++) {
      for (int p = 0; p < NCCL_TOPO_NODE_TYPES; p++) EXPECT_EQ(nullptr, system->nodes[t].nodes[n].paths[p]);
    }
  }
}

// --- User levels

TEST_F(PathsMicrotest, GetLevel_ParsesNamesOldNumbersAndTheDisableVariable) {
  struct Case {
    const char* disable;
    const char* level;
    int expected;
  };
  const Case cases[] = {{"1", "SYS", PATH_LOC}, {"2", "PIX", PATH_PIX}, {nullptr, "PHB", PATH_PHB},
                        {nullptr, "2", PATH_PXB}, {nullptr, "9", PATH_SYS}, {nullptr, "bogus", -2},
                        {nullptr, nullptr, -2}};
  for (const Case& c : cases) {
    if (c.disable) setenv("NCCL_X_DISABLE", c.disable, 1);
    if (c.level) setenv("NCCL_X_LEVEL", c.level, 1);
    int level = -1;

    ASSERT_EQ(ncclSuccess, ncclGetLevel(&level, "NCCL_X_DISABLE", "NCCL_X_LEVEL"));

    EXPECT_EQ(c.expected, level) << (c.level ? c.level : "unset");
    unsetenv("NCCL_X_DISABLE");
    unsetenv("NCCL_X_LEVEL");
  }
  int level = PATH_NVL;
  setenv("NCCL_X_LEVEL", "SYS", 1);
  ASSERT_EQ(ncclSuccess, ncclGetLevel(&level, nullptr, "NCCL_X_LEVEL"));
  unsetenv("NCCL_X_LEVEL");
  EXPECT_EQ(PATH_NVL, level);  // already resolved
}

TEST_F(PathsMicrotest, GetUserP2pLevel_ReadsTheEnvironmentOnce) {
  int level = PATH_SYS;
  ASSERT_EQ(ncclSuccess, ncclGetUserP2pLevel(&level));
  EXPECT_EQ(PATH_SYS, level);  // unset leaves it alone

  setenv("NCCL_P2P_DISABLE", "1", 1);
  ASSERT_EQ(ncclSuccess, ncclGetUserP2pLevel(&level));
  EXPECT_EQ(PATH_SYS, level);

  ncclTopoUserP2pLevel = -1;
  ASSERT_EQ(ncclSuccess, ncclGetUserP2pLevel(&level));
  EXPECT_EQ(PATH_LOC, level);
}

// --- P2P checks

class PathsCommMicrotest : public PathsMicrotest {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> shared_ = std::make_unique<ncclSharedResources>();
  std::vector<ncclPeerInfo> peers_;
  SystemPtr system_;

  void SetUp() override {
    PathsMicrotest::SetUp();
    comm_->sharedRes = shared_.get();
    shared_->owner = comm_.get();
    comm_->pxnDisable = RCCL_VALUE_UNSET;
    comm_->config.nChannelsPerNetPeer = NCCL_CONFIG_UNDEF_INT;
    comm_->config.maxP2pPeers = NCCL_CONFIG_UNDEF_INT;
  }

  void Use(SystemPtr system, int rank = 0) {
    system_ = std::move(system);
    comm_->topo = system_.get();
    comm_->rank = rank;
    if (peers_.empty()) Peers(system_->nodes[GPU].count);
  }

  void Peers(int n, uint64_t host = 1) {
    peers_.assign(n, ncclPeerInfo{});
    for (int r = 0; r < n; r++) {
      peers_[r].rank = r;
      peers_[r].hostHash = host;
    }
    comm_->peerInfo = peers_.data();
    comm_->nRanks = n;
  }

  int P2p(int rank1, int rank2, int* read = nullptr, int* cudaP2p = nullptr, int* crossClique = nullptr,
          ncclComm* comm = nullptr) {
    int p2p = -1;
    EXPECT_EQ(ncclSuccess, ncclTopoCheckP2p(comm, system_.get(), rank1, rank2, &p2p, read, nullptr, cudaP2p,
                                            crossClique));
    return p2p;
  }
};

TEST_F(PathsCommMicrotest, CheckP2p_AnyPathOnAnAmdHostAndPxbOnOthers) {
  Use(BuildWithPaths(PcieSystem()));
  EXPECT_EQ(1, P2p(0, 3));

  for (const char* ids :
       {kIntelCpu, "arch=\"arm64\"", "arch=\"x86_64\" vendor=\"CentaurHauls\" familyid=\"7\" modelid=\"1\""}) {
    std::string pcie = System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1)) +
                                         GpuPci(GpuBus(2), 2, 2), "", ids));
    Use(BuildWithPaths(pcie));

    EXPECT_EQ(1, P2p(0, 1)) << ids;
    EXPECT_EQ(0, P2p(0, 2)) << ids;  // PHB
  }
}

TEST_F(PathsCommMicrotest, CheckP2p_UserLevelOverridesTheCpuDefault) {
  setenv("NCCL_P2P_LEVEL", "PIX", 1);
  Use(BuildWithPaths(PcieSystem()));

  EXPECT_EQ(1, P2p(0, 1));
  EXPECT_EQ(0, P2p(0, 2));
}

TEST_F(PathsCommMicrotest, CheckP2p_ReadOnlyForAmpereNvlink) {
  Use(BuildWithPaths(XgmiSystem(2)));
  int read = -1, cudaP2p = -1, cross = -1;

  EXPECT_EQ(1, P2p(0, 1, &read, &cudaP2p, &cross));
  EXPECT_EQ(0, read);
  EXPECT_EQ(1, cudaP2p);
  EXPECT_EQ(0, cross);
  for (int g = 0; g < 2; g++) system_->nodes[GPU].nodes[g].gpu.cudaCompCap = 80;
  EXPECT_EQ(1, P2p(0, 1, &read));
  EXPECT_EQ(1, read);
}

TEST_F(PathsCommMicrotest, CheckP2p_UnknownRanks) {
  Peers(8);
  Use(BuildWithPaths(XgmiSystem(2)));
  int p2p = -1;

  EXPECT_EQ(ncclInternalError,
            ncclTopoCheckP2p(nullptr, system_.get(), 7, 0, &p2p, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(0, P2p(0, 7));
  EXPECT_FALSE(Logged("not in local topology"));
  comm_->MNNVL = 1;
  EXPECT_EQ(0, P2p(0, 7, nullptr, nullptr, nullptr, comm_.get()));
  EXPECT_TRUE(Logged("rank7 not in local topology (MNNVL active)"));
}

TEST_F(PathsCommMicrotest, CheckP2p_OtherHostsAndOtherContainersAreNotP2p) {
  Use(BuildWithPaths(XgmiSystem(2)));
  int cudaP2p = -1;
  peers_[1].hostHash = 2;
  EXPECT_EQ(0, P2p(0, 1, nullptr, &cudaP2p, nullptr, comm_.get()));
  EXPECT_EQ(0, cudaP2p);

  peers_[1].hostHash = 1;
  peers_[1].shmDev = 5;
  EXPECT_EQ(0, P2p(0, 1, nullptr, nullptr, nullptr, comm_.get()));
  peers_[1].shmDev = 0;
  EXPECT_EQ(1, P2p(0, 1, nullptr, &cudaP2p, nullptr, comm_.get()));
  EXPECT_EQ(1, cudaP2p);
}

class PathsMnnvlMicrotest : public PathsCommMicrotest {
 protected:
  void SetUp() override {
    PathsCommMicrotest::SetUp();
    Use(BuildWithPaths(XgmiSystem(2)));
    comm_->MNNVL = 1;
    peers_[1].hostHash = 2;
    for (ncclPeerInfo& p : peers_) {
      p.fabricInfo.clusterUuid[0] = 0xab;
      p.fabricInfo.cliqueId = 3;
    }
  }
};

TEST_F(PathsMnnvlMicrotest, CheckMnnvl_SameClusterAndCliqueOrCrossCliqueOptIn) {
  int ret = -9;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(1, ret);

  peers_[1].fabricInfo.cliqueId = 4;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(0, ret);
  comm_->p2pCrossClique = true;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(-1, ret);

  peers_[1].fabricInfo.clusterUuid[0] = 0xac;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(0, ret);
  memset(peers_[1].fabricInfo.clusterUuid, 0, sizeof(peers_[1].fabricInfo.clusterUuid));
  peers_[1].fabricInfo.clusterUuid[15] = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(0, ret);
  peers_[1].fabricInfo.clusterUuid[15] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckMNNVL(comm_.get(), &peers_[0], &peers_[1], &ret));
  EXPECT_EQ(0, ret);  // no fabric info
}

TEST_F(PathsMnnvlMicrotest, CheckP2p_CliquePeerOnAnotherHostIsP2p) {
  int cudaP2p = -1;

  EXPECT_EQ(1, P2p(0, 1, nullptr, &cudaP2p, nullptr, comm_.get()));

  EXPECT_EQ(1, cudaP2p);
  peers_[1].fabricInfo.cliqueId = 4;
  EXPECT_EQ(0, P2p(0, 1, nullptr, nullptr, nullptr, comm_.get()));
}

TEST_F(PathsMnnvlMicrotest, CheckP2p_CrossCliqueUsesTheFabricUnlessTheUserLevelIsBelowNvl) {
  peers_[1].fabricInfo.cliqueId = 4;
  comm_->p2pCrossClique = true;
  int cudaP2p = -1, cross = -1;

  EXPECT_EQ(1, P2p(0, 1, nullptr, &cudaP2p, &cross, comm_.get()));
  EXPECT_EQ(1, cudaP2p);
  EXPECT_EQ(1, cross);

  setenv("NCCL_P2P_LEVEL", "LOC", 1);
  ncclTopoUserP2pLevel = -1;
  EXPECT_EQ(0, P2p(0, 1, nullptr, nullptr, nullptr, comm_.get()));
}

// --- GDR

class PathsGdrMicrotest : public PathsCommMicrotest {
 protected:
  int Gdr(int rank, int net, int read = 0) {
    enum ncclTopoGdrMode mode = ncclTopoGdrModeNum;
    EXPECT_EQ(ncclSuccess, ncclTopoCheckGdr(system_.get(), rank, system_->nodes[NET].nodes[net].id, read, &mode));
    return mode;
  }
};

TEST_F(PathsGdrMicrotest, CheckGdr_CloseEnoughNicsGetTheDefaultMode) {
  Use(BuildWithPaths(XgmiSystem(2, 2, true)));

  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0, 1));
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 1));  // across the sockets
}

TEST_F(PathsGdrMicrotest, CheckGdr_EitherEndWithoutGdrDisablesIt) {
  Use(BuildWithPaths(XgmiSystem(2, 1, true)));
  system_->nodes[NET].nodes[0].net.gdrSupport = 0;
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0));

  system_->nodes[NET].nodes[0].net.gdrSupport = 1;
  system_->nodes[GPU].nodes[0].gpu.gdrSupport = 0;
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_ReadParamZeroDisablesReadsOnly) {
  Use(BuildWithPaths(XgmiSystem(1, 1, true)));
  g_params["NET_GDR_READ"] = 0;

  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0, 1));
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_LevelFromC2cParamSystemAndUser) {
  Use(BuildWithPaths(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0))))));  // PHB
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0));

  system_->netGdrLevel = PATH_PHB;
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));

  system_->netGdrLevel = -2;
  setenv("NCCL_NET_GDR_LEVEL", "SYS", 1);
  ncclTopoUserGdrLevel = -1;
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));
  unsetenv("NCCL_NET_GDR_LEVEL");
  ncclTopoUserGdrLevel = -1;

  SystemPtr p2c = BuildWithPaths(XgmiSystem(1, 1, true));
  Use(std::move(p2c));
  g_params["NET_GDR_C2C"] = 0;
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));  // PIX is still within PXB
}

// Rome: a GPU and a NIC hanging directly off the same root complex quadrant get GDR up to PHB.
TEST_F(PathsGdrMicrotest, CheckGdr_RomeSameQuadrantAllowsPhb) {
  std::string body =
    GpuPci("0000:41:00.0", 0, 0) + NicPci("0000:42:00.0", Net("a", 0)) + NicPci("0000:81:00.0", Net("b", 1));
  Use(BuildWithPaths(System(Cpu(0, body, "", kRomeCpu))));

  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 1));
}

TEST_F(PathsGdrMicrotest, CheckGdr_RomeNeedsBothOnTheRootComplex) {
  std::string body = Pci(SwitchBus(0), GpuPci("0000:41:00.0", 0, 0)) + NicPci("0000:42:00.0", Net("a", 0));
  Use(BuildWithPaths(System(Cpu(0, body, "", kRomeCpu))));

  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_MloPartFollowsTheParamAndTheFunctionZeroCarveOut) {
  std::string body = PartitionedSwitch("mlopart=\"0\"", "mlopart=\"1\"");
  Use(BuildWithPaths(System(Cpu(0, body))));
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(1, 0));

  g_params["NET_GDR_MLOPART"] = 0;
  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));  // partition 0 at function 0 is the whole GPU on HIP
  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(1, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_PartitionZeroAtAnotherFunctionIsStillAPartition) {
  std::string pci = "<pci busid=\"0000:10:00.1\" class=\"0x120000\" link_speed=\"32.0 GT/s PCIe\" link_width=\"16\">" +
                    Gpu(0, 0, "", "mlopart=\"0\"") + "</pci>";
  Use(BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), pci + NicPci(NicBus(0), Net("a", 0)))))));
  g_params["NET_GDR_MLOPART"] = 0;

  EXPECT_EQ(ncclTopoGdrModeDisable, Gdr(0, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_MloPartReadsTheDevDistance) {
  Use(BuildWithPaths(System(Cpu(0, PartitionedSwitch("mlopart=\"0\"", "mlopart=\"1\"")))));
  system_->nodes[GPU].nodes[1].paths[NET][0].type = PATH_SYS;

  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(1, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_C2cGpuOnAPciNicForcesPciMapping) {
  XmlPtr xml =
    MustParseXml(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0))))));
  ncclXmlNode* gpu = nullptr;
  ncclXmlNode* c2c = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml.get(), "gpu", &gpu));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), gpu, "c2c", &c2c));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "count", 10));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "bw", 10000));
  SystemPtr system = BuildSystem(xml.get());
  system->nRanks = 1;
  system->netGdrLevel = -2;
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system.get(), nullptr));
  Use(std::move(system));

  EXPECT_EQ(ncclTopoGdrModePci, Gdr(0, 0));
}

TEST_F(PathsGdrMicrotest, CheckGdr_C2cNicBehindTheCpuKeepsTheDefaultMode) {
  XmlPtr xml = MustParseXml(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0)))));
  ncclXmlNode* gpu = nullptr;
  ncclXmlNode* c2c = nullptr;
  ASSERT_EQ(ncclSuccess, xmlFindTag(xml.get(), "gpu", &gpu));
  ASSERT_EQ(ncclSuccess, xmlAddNode(xml.get(), gpu, "c2c", &c2c));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "count", 10));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(c2c, "bw", 10000));
  SystemPtr system = BuildSystem(xml.get());
  system->nRanks = 1;
  system->netGdrLevel = -2;
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system.get(), nullptr));
  Use(std::move(system));

  EXPECT_EQ(ncclTopoGdrModeDefault, Gdr(0, 0));  // P2C: within the C2C level, and no PCIe mapping needed
}

TEST_F(PathsGdrMicrotest, CheckGdr_UnknownNetOrRankFails) {
  Use(BuildWithPaths(XgmiSystem(1, 1, true)));
  enum ncclTopoGdrMode mode;

  EXPECT_EQ(ncclInternalError, ncclTopoCheckGdr(system_.get(), 0, 0x77, 0, &mode));
  EXPECT_EQ(ncclInternalError, ncclTopoCheckGdr(system_.get(), 5, system_->nodes[NET].nodes[0].id, 0, &mode));
}

TEST_F(PathsGdrMicrotest, IsGdrAvail_AnyNicWithGdrInEitherDirection) {
  Use(BuildWithPaths(XgmiSystem(2, 2, true)));
  bool avail = false;

  ASSERT_EQ(ncclSuccess, ncclTopoIsGdrAvail(system_.get(), 1, &avail));
  EXPECT_TRUE(avail);

  g_params["NET_GDR_READ"] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoIsGdrAvail(system_.get(), 1, &avail));
  EXPECT_TRUE(avail);

  for (int n = 0; n < 2; n++) system_->nodes[NET].nodes[n].net.gdrSupport = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoIsGdrAvail(system_.get(), 1, &avail));
  EXPECT_FALSE(avail);
}

// --- Network flush and intra-node net

int g_forceFlush = 0;
ncclResult_t FlushProps(int, ncclNetProperties_t* props) {
  *props = ncclNetProperties_t{};
  props->forceFlush = g_forceFlush;
  return ncclSuccess;
}

TEST_F(PathsCommMicrotest, NeedFlush_ForcedOrUnlessTheNetManagesIt) {
  Use(BuildWithPaths(XgmiSystem(1, 1, true)));
  ncclNet_t net{};
  net.name = "fake";
  net.getProperties = FlushProps;
  comm_->ncclNet = &net;
  enum ncclTopoFlushType flush;
  g_forceFlush = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoNeedFlush(comm_.get(), 0, 0, 0, true, &flush));
  EXPECT_EQ(ncclTopoFlushNone, flush);
  ASSERT_EQ(ncclSuccess, ncclTopoNeedFlush(comm_.get(), 0, 0, 0, false, &flush));
  EXPECT_EQ(ncclTopoFlushAlways, flush);
  g_forceFlush = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoNeedFlush(comm_.get(), 0, 0, 0, true, &flush));
  EXPECT_EQ(ncclTopoFlushAlways, flush);
  g_forceFlush = 0;
  g_params["NET_FORCE_FLUSH"] = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoNeedFlush(comm_.get(), 0, 0, 0, true, &flush));
  EXPECT_EQ(ncclTopoFlushAlways, flush);
  g_params.clear();
  EXPECT_EQ(ncclInternalError, ncclTopoNeedFlush(comm_.get(), 0, 0, 3, true, &flush));
}

TEST_F(PathsMicrotest, CheckNet_OnlyWhenBothGpusReachFasterNicsAndIntraNetIsAllowed) {
  SystemPtr pcie =
    BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0)))) +
                          Cpu(1, Pci(SwitchBus(1), GpuPci(GpuBus(1), 1, 1) + NicPci(NicBus(1), Net("b", 1))))));
  int net = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNet(pcie.get(), 0, 1, &net));
  EXPECT_EQ(0, net);

  g_params["NET_DISABLE_INTRA"] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNet(pcie.get(), 0, 1, &net));
  EXPECT_EQ(1, net);  // 48 GB/s to a NIC beats the 16 GB/s inter-socket path
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNet(pcie.get(), 0, 9, &net));
  EXPECT_EQ(1, net);

  SystemPtr xgmi = BuildWithPaths(XgmiSystem(2, 1, true));
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNet(xgmi.get(), 0, 1, &net));
  EXPECT_EQ(0, net);
}

TEST_F(PathsMicrotest, CheckNet_GpuWithoutNicPathsLosesToP2p) {
  g_params["NET_DISABLE_INTRA"] = 0;
  SystemPtr system =
    BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0)))) +
                          Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x5\"")));
  ASSERT_EQ(nullptr, system->nodes[GPU].nodes[1].paths[NET]);
  int net = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoCheckNet(system.get(), 0, 1, &net));

  EXPECT_EQ(0, net);
}

// --- PXN

TEST_F(PathsCommMicrotest, ComputePaths_PxnRelaysThroughTheXgmiPeerNextToTheNic) {
  comm_->pxnDisable = 0;
  Use(BuildWithPaths(XgmiSystem(2, 2, true)));

  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));

  EXPECT_EQ(PATH_PXN, Path(system_.get(), GPU, 1, NET, 0).type);
  EXPECT_EQ(PATH_PXN, Path(system_.get(), GPU, 0, NET, 1).type);
  EXPECT_EQ(PATH_PIX, Path(system_.get(), GPU, 0, NET, 0).type);
  enum ncclTopoGdrMode mode;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckGdr(system_.get(), 1, system_->nodes[NET].nodes[0].id, 1, &mode));
  EXPECT_EQ(ncclTopoGdrModeDefault, mode);  // judged on the relay GPU's own PIX distance
  int rank = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoGetIntermediateRank(system_.get(), 1, system_->nodes[NET].nodes[0].id, &rank));
  EXPECT_EQ(0, rank);
  ASSERT_EQ(ncclSuccess, ncclTopoGetIntermediateRank(system_.get(), 0, system_->nodes[NET].nodes[0].id, &rank));
  EXPECT_EQ(0, rank);
  int maxPath = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxLocalNetPath(system_.get(), &maxPath));
  EXPECT_EQ(PATH_PXN, maxPath);
}

TEST_F(PathsCommMicrotest, ComputePaths_UnreachablePeersAreMarkedNet) {
  Use(BuildWithPaths(XgmiSystem(3)));
  g_p2pCanConnect = [](int* ret, ncclComm*, ncclTopoGraph*, ncclPeerInfo* src, ncclPeerInfo* dst) {
    *ret = !((src->rank == 0 && dst->rank == 2) || (src->rank == 2 && dst->rank == 0));
    return ncclSuccess;
  };
  g_shmCanConnect = [](int* ret, ncclComm*, ncclTopoGraph*, ncclPeerInfo* src, ncclPeerInfo*) {
    *ret = src->rank != 0;
    return ncclSuccess;
  };

  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));

  EXPECT_EQ(PATH_NET, Path(system_.get(), GPU, 0, GPU, 2).type);
  EXPECT_EQ(PATH_NVL, Path(system_.get(), GPU, 2, GPU, 0).type);  // P2P fails, but SHM works
  EXPECT_EQ(PATH_NVL, Path(system_.get(), GPU, 0, GPU, 1).type);
}

TEST_F(PathsCommMicrotest, GetIntermediateRank_RemoteGpuHasNoneAndAMalformedPxnPathFails) {
  Use(BuildWithPaths(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0))) +
                            Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x5\""))));
  int64_t netId = system_->nodes[NET].nodes[0].id;
  int rank = 7;

  ASSERT_EQ(ncclSuccess, ncclTopoGetIntermediateRank(system_.get(), 1, netId, &rank));
  EXPECT_EQ(-1, rank);

  Path(system_.get(), GPU, 0, NET, 0).type = PATH_PXN;  // its second hop is the CPU, not a device
  EXPECT_EQ(ncclInternalError, ncclTopoGetIntermediateRank(system_.get(), 0, netId, &rank));
  EXPECT_TRUE(Logged("Could not find intermediate GPU between GPU rank 0"));
  EXPECT_EQ(ncclInternalError, ncclTopoGetIntermediateRank(system_.get(), 0, 0x99, &rank));
}

TEST_F(PathsCommMicrotest, PxnDisable_CachedPerCommFromTheRcclOverrideOrTheParam) {
  EXPECT_EQ(1, ncclPxnDisable(nullptr));
  g_params["PXN_DISABLE"] = 0;
  EXPECT_EQ(0, ncclPxnDisable(nullptr));

  EXPECT_EQ(0, ncclPxnDisable(comm_.get()));  // no RCCL override
  g_params.clear();
  EXPECT_EQ(0, ncclPxnDisable(comm_.get()));  // cached

  comm_->pxnDisable = RCCL_VALUE_UNSET;
  g_rcclPxnValue = 0;
  EXPECT_EQ(0, ncclPxnDisable(comm_.get()));  // the RCCL override beats the param default of 1

  comm_->pxnDisable = RCCL_VALUE_UNSET;
  comm_->ncclNetVer = 4;
  EXPECT_EQ(1, ncclPxnDisable(comm_.get()));
  EXPECT_TRUE(Logged("PXN Disabled as plugin is v4"));
}

// --- Path summaries

TEST_F(PathsMicrotest, GpuMinMaxPath_SkipSelfAndMissingPaths) {
  SystemPtr system = BuildWithPaths(XgmiSystem(2, 2, true));
  int value = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMinPath(system.get(), GPU, &value));
  EXPECT_EQ(PATH_NVL, value);
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxPath(system.get(), GPU, &value));
  EXPECT_EQ(PATH_NVL, value);
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMinPath(system.get(), NET, &value));
  EXPECT_EQ(PATH_PIX, value);
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxPath(system.get(), NET, &value));
  EXPECT_EQ(PATH_SYS, value);
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMinPath(system.get(), GIN, &value));
  EXPECT_EQ(PATH_SYS, value);
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxPath(system.get(), GIN, &value));
  EXPECT_EQ(PATH_LOC, value);
}

TEST_F(PathsMicrotest, GpuMaxLocalNetPath_TakesEachGpusNearestNic) {
  SystemPtr system = BuildWithPaths(System(
    Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0))) + GpuPci(GpuBus(1), 1, 1)) +
    Cpu(1, GpuPci(GpuBus(2), 2, 2), "host_hash=\"0x5\"")));
  int value = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxLocalNetPath(system.get(), &value));
  EXPECT_EQ(PATH_PHB, value);

  for (int g = 0; g < 2; g++) Path(system.get(), GPU, g, NET, 0).type = PATH_DIS;
  ASSERT_EQ(ncclSuccess, ncclTopoGetGpuMaxLocalNetPath(system.get(), &value));
  EXPECT_EQ(PATH_LOC, value);
}

TEST_F(PathsMicrotest, NvlinkSummaries_AllNvlinkDirectAndSplit) {
  SystemPtr full = BuildWithPaths(XgmiSystem(4));
  SystemPtr ring = BuildWithPaths(XgmiRing4());
  SystemPtr pcie = BuildWithPaths(PcieSystem());
  std::string halves =
    System(Cpu(0, GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))) +
                    GpuPci(GpuBus(2), 2, 2, Xgmi(GpuBus(3))) + GpuPci(GpuBus(3), 3, 3, Xgmi(GpuBus(2)))));
  SystemPtr split = BuildWithPaths(halves);
  int all = -1, splitNvl = -1;
  bool direct = false;

  ASSERT_EQ(ncclSuccess, ncclTopoPathAllNVLink(full.get(), &all));
  EXPECT_EQ(1, all);
  ASSERT_EQ(ncclSuccess, ncclTopoPathAllNVLink(pcie.get(), &all));
  EXPECT_EQ(0, all);
  SystemPtr pix = BuildWithPaths(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1)))));
  ASSERT_EQ(ncclSuccess, ncclTopoPathAllNVLink(pix.get(), &all));
  EXPECT_EQ(0, all);
  ASSERT_EQ(ncclSuccess, ncclTopoPathAllDirectNVLink(full.get(), &direct));
  EXPECT_TRUE(direct);
  ASSERT_EQ(ncclSuccess, ncclTopoPathAllDirectNVLink(ring.get(), &direct));
  EXPECT_FALSE(direct);
  ASSERT_EQ(ncclSuccess, ncclTopoSplitNvLink(split.get(), &splitNvl));
  EXPECT_EQ(1, splitNvl);
  ASSERT_EQ(ncclSuccess, ncclTopoSplitNvLink(full.get(), &splitNvl));
  EXPECT_EQ(0, splitNvl);
  ASSERT_EQ(ncclSuccess, ncclTopoSplitNvLink(pcie.get(), &splitNvl));
  EXPECT_EQ(0, splitNvl);
}

TEST_F(PathsMicrotest, GetNvbGpus_ListsTheTwoHopPeers) {
  SystemPtr system = BuildWithPaths(XgmiRing4());
  int n = -1;
  int* ranks = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoGetNvbGpus(system.get(), 1, &n, &ranks));

  ASSERT_EQ(1, n);
  EXPECT_EQ(3, ranks[0]);
  free(ranks);
}

// --- P2P channels

TEST_F(PathsMicrotest, MaxP2pNchannels_ClampsTheParamToOneThroughMaxChannels) {
  EXPECT_EQ(MAXCHANNELS, ncclMaxP2pNchannels());
  g_params["MAX_P2P_NCHANNELS"] = 1000;
  EXPECT_EQ(MAXCHANNELS, ncclMaxP2pNchannels());
  g_params["MAX_P2P_NCHANNELS"] = 0;
  EXPECT_EQ(1, ncclMaxP2pNchannels());
  EXPECT_TRUE(Logged("User asked for a maximum of 0 P2P channels, setting it to 1"));
}

TEST_F(PathsCommMicrotest, P2pChannelsUpperBound_FullCuPoolOnlyForSingleNodeGfx1250) {
  Use(BuildWithPaths(XgmiSystem(2, 1, false, "gfx1250")));
  comm_->nNodes = 1;
  bool opted = true;

  EXPECT_EQ(256, ncclP2pChannelsUpperBound(comm_.get(), &opted));  // 304 CUs, capped
  EXPECT_FALSE(opted);
  system_->nodes[GPU].nodes[0].gpu.cu = 100;
  EXPECT_EQ(64, ncclP2pChannelsUpperBound(comm_.get(), nullptr));
  system_->nodes[GPU].nodes[0].gpu.cu = 0;
  EXPECT_EQ(MAXCHANNELS, ncclP2pChannelsUpperBound(comm_.get(), nullptr));
  comm_->nNodes = 2;
  EXPECT_EQ(4 * CHANNEL_LIMIT, ncclP2pChannelsUpperBound(comm_.get(), nullptr));
  g_params["MAX_P2P_NCHANNELS"] = 100;
  EXPECT_EQ(64, ncclP2pChannelsUpperBound(comm_.get(), &opted));
  EXPECT_TRUE(opted);
  g_params["MAX_P2P_NCHANNELS"] = 32;
  EXPECT_EQ(64, ncclP2pChannelsUpperBound(comm_.get(), &opted));
  EXPECT_FALSE(opted);
}

class PathsP2pChannelsMicrotest : public PathsCommMicrotest {
 protected:
  void Setup(SystemPtr system, int nRanks, int nNodes, int nChannels) {
    Peers(nRanks);
    Use(std::move(system));
    system_->nRanks = nRanks;
    comm_->nNodes = nNodes;
    comm_->nChannels = nChannels;
  }

  void Compute() {
    ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
    ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannels(comm_.get()));
  }
};

TEST_F(PathsP2pChannelsMicrotest, PerPeer_XgmiWidthPcieAndSelf) {
  std::string body = GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1), 3)) + GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0), 3)) +
                     GpuPci(GpuBus(2), 2, 2);
  Setup(BuildWithPaths(System(Cpu(0, body))), 2, 1, 4);

  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(4 * 3, comm_->p2pnChannelsPerPeer);  // 3 XGMI links to rank 1, the only other rank
  EXPECT_EQ(2, comm_->p2pMaxPeers);

  comm_->nRanks = 3;
  comm_->config.maxP2pPeers = 5;
  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);  // PCIe to rank 2
  EXPECT_EQ(5, comm_->p2pMaxPeers);
}

TEST_F(PathsP2pChannelsMicrotest, PerPeer_OlderArchUsesTwoPerXgmiLink) {
  Setup(BuildWithPaths(XgmiSystem(2, 1, false, "gfx90a")), 2, 1, 4);

  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));

  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, PerPeer_RemoteRanksUseTheNetOrTheMnnvlClique) {
  Setup(BuildWithPaths(XgmiSystem(1, 1, true)), 2, 2, 4);
  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(4, comm_->p2pnChannelsPerPeer);  // one NIC, 48 GB/s over 14 GB/s per channel

  comm_->config.nChannelsPerNetPeer = 3;
  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(3, comm_->p2pnChannelsPerPeer);

  int clique[] = {0, 1};
  comm_->MNNVL = 1;
  comm_->clique.size = 2;
  comm_->clique.ranks = clique;
  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(4, comm_->p2pnChannelsPerPeer);
  clique[1] = 5;
  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(3, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, PerPeer_RemoteRankWithoutNicGetsTwo) {
  Setup(BuildWithPaths(XgmiSystem(1)), 2, 2, 4);
  int clique[] = {0};
  comm_->MNNVL = 1;
  comm_->clique.size = 1;
  comm_->clique.ranks = clique;

  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));

  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_SingleNodeGfx942DoublesThePerPeerCount) {
  Setup(BuildWithPaths(XgmiSystem(4)), 4, 1, 6);

  Compute();

  EXPECT_EQ(8, comm_->p2pnChannels);
  EXPECT_EQ(8, comm_->p2pnChannelsPerPeer);
  EXPECT_EQ(std::vector<int>({6, 7}), g_initChannelCalls);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_IntelWithoutFullXgmiIsEightByEight) {
  Setup(BuildWithPaths(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1), "", kIntelCpu))), 2, 1, 4);

  Compute();

  EXPECT_EQ(8, comm_->p2pnChannels);
  EXPECT_EQ(8, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_RomeFourByTwoUsesTwoPerPeer) {
  Setup(BuildWithPaths(XgmiSystem(2)), 2, 1, 6);
  system_->type |= RCCL_TOPO_4P2H_ROME;

  Compute();

  EXPECT_EQ(8, comm_->p2pnChannels);
  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_SharedCommIsBoundByTheParent) {
  Setup(BuildWithPaths(XgmiSystem(2, 1, false, "gfx90a")), 2, 1, 16);
  auto parent = std::make_unique<ncclComm>();
  shared_->owner = parent.get();
  shared_->tpP2pNChannels = 4;

  Compute();

  EXPECT_EQ(4, comm_->p2pnChannels);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_UserMaxCapsBelowThePerPeerCount) {
  Setup(BuildWithPaths(XgmiSystem(4)), 4, 1, 6);
  g_params["MAX_P2P_NCHANNELS"] = 1;
  g_params["MIN_P2P_NCHANNELS"] = 4;

  Compute();

  EXPECT_EQ(1, comm_->p2pnChannels);
  EXPECT_EQ(1, comm_->p2pnChannelsPerPeer);
  EXPECT_TRUE(Logged("NCCL_MAX_P2P_NCHANNELS=1 overrides NCCL_MIN_P2P_NCHANNELS=4; using 1 P2P channels"));
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_Gfx950MultiNodeCaps) {
  struct Case {
    int nRanks, nNodes, cuReduce, expected;
  };
  for (const Case& c : {Case{16, 2, 0, 32}, Case{32, 4, 0, 32}, Case{8, 2, 0, 16}, Case{16, 4, 0, 16},
                        Case{64, 16, 1, 16}, Case{64, 16, 0, 64}}) {
    Setup(BuildWithPaths(XgmiSystem(4, 1, false, "gfx950")), c.nRanks, c.nNodes, 64);
    system_->nRanks = c.nRanks;
    g_params["P2P_CU_REDUCE_SCALE_ENABLE"] = c.cuReduce;

    Compute();

    EXPECT_EQ(c.expected, comm_->p2pnChannels) << c.nRanks << "/" << c.nNodes;
  }
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_SaturationTilesThePoolOverThePeers) {
  Setup(BuildWithPaths(XgmiSystem(4, 1, false, "gfx1250")), 4, 1, MAXCHANNELS);

  Compute();

  EXPECT_EQ(256, comm_->p2pnChannels);  // gfx1250 single node: the CU-bound pool
  EXPECT_EQ(64, comm_->p2pnChannelsPerPeer);  // 256 / 4 peers, on by default for gfx1250
  EXPECT_TRUE(Logged("RCCL_SATURATE_P2P_NCHANNELS: p2pnChannelsPerPeer 8 -> 64"));

  g_params["RCCL_SATURATE_P2P_NCHANNELS"] = 0;
  Compute();
  EXPECT_EQ(8, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_MultiNodeHalvesPerPeerToFitOneRound) {
  Setup(BuildWithPaths(XgmiSystem(1, 1, true)), 8, 8, 8);

  Compute();

  EXPECT_EQ(8, comm_->p2pnChannels);
  EXPECT_EQ(1, comm_->p2pnChannelsPerPeer);
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_AinicRestoresTwoChannelsPerPeerBelowScale) {
  Setup(BuildWithPaths(XgmiSystem(1, 1, true)), 8, 8, 8);
  g_useAinic = true;
  comm_->minNetCount = 1;

  Compute();
  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);

  Setup(BuildWithPaths(XgmiSystem(1, 1, true)), 16, 16, 8);
  Compute();
  EXPECT_EQ(1, comm_->p2pnChannelsPerPeer);  // at scale: one per peer
}

TEST_F(PathsP2pChannelsMicrotest, P2pChannels_InitChannelFailureIsReturned) {
  Setup(BuildWithPaths(XgmiSystem(4)), 4, 1, 2);
  g_initChannelResult = ncclSystemError;

  ASSERT_EQ(ncclSuccess, ncclTopoComputeP2pChannelsPerPeer(comm_.get()));
  EXPECT_EQ(ncclSystemError, ncclTopoComputeP2pChannels(comm_.get()));
}


TEST_F(PathsP2pChannelsMicrotest, P2pChannels_GrowCommAgreesOnTheSmallestCount) {
  Setup(BuildWithPaths(XgmiSystem(4)), 4, 1, 6);
  comm_->isGrow = true;
  g_bootstrapAllGather = [](void*, void* data, int) {
    int* v = static_cast<int*>(data);
    v[1] = v[2] = 50;
    v[3] = 2;
    return ncclSuccess;
  };

  Compute();
  EXPECT_EQ(2, comm_->p2pnChannels);
  EXPECT_EQ(2, comm_->p2pnChannelsPerPeer);

  g_bootstrapAllGather = [](void*, void*, int) { return ncclSystemError; };
  EXPECT_EQ(ncclSystemError, ncclTopoComputeP2pChannels(comm_.get()));
}

// --- Trim

class PathsTrimMicrotest : public PathsCommMicrotest {
 protected:
  ncclResult_t Trim() { return ncclTopoTrimSystem(system_.get(), comm_.get()); }

  std::vector<int> Ranks() {
    std::vector<int> ranks;
    for (int g = 0; g < system_->nodes[GPU].count; g++) ranks.push_back(system_->nodes[GPU].nodes[g].gpu.rank);
    return ranks;
  }
};

TEST_F(PathsTrimMicrotest, TrimSystem_DropsGpusOutsideMyDomain) {
  Use(BuildWithPaths(XgmiSystem(4, 1, true)), 2);
  g_p2pCanConnect = [](int* ret, ncclComm*, ncclTopoGraph*, ncclPeerInfo* src, ncclPeerInfo* dst) {
    *ret = src->rank / 2 == dst->rank / 2;
    return ncclSuccess;
  };
  g_shmCanConnect = [](int* ret, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*) {
    *ret = 0;
    return ncclSuccess;
  };
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));

  ASSERT_EQ(ncclSuccess, Trim());

  EXPECT_EQ(std::vector<int>({2, 3}), Ranks());
  EXPECT_EQ(2, comm_->localRanks);
  EXPECT_EQ(1, system_->inter);
  EXPECT_EQ(4, system_->nodes[NET].count);
  EXPECT_EQ(nullptr, system_->nodes[GPU].nodes[0].paths[GPU]);  // the caller recomputes them
}

TEST_F(PathsTrimMicrotest, TrimSystem_SingleNodeXgmiDropsTheNics) {
  Use(BuildWithPaths(XgmiSystem(2, 1, true)));

  ASSERT_EQ(ncclSuccess, Trim());

  EXPECT_EQ(0, system_->nodes[NET].count);
  EXPECT_EQ(0, system_->inter);
  EXPECT_TRUE(system_->type & RCCL_TOPO_XGMI_ALL);
  EXPECT_FALSE(system_->type & RCCL_TOPO_GDR_ALL);
}

// Two GPUs with no XGMI, each next to its own NIC.
std::string GdrPcieSystem() {
  return System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0))) +
                           Pci(SwitchBus(1), GpuPci(GpuBus(1), 1, 1) + NicPci(NicBus(1), Net("a", 1)))));
}

TEST_F(PathsTrimMicrotest, TrimSystem_GdrOnEveryGpuWithoutXgmiKeepsTheNics) {
  Use(BuildWithPaths(GdrPcieSystem()));

  ASSERT_EQ(ncclSuccess, Trim());

  EXPECT_TRUE(system_->type & RCCL_TOPO_GDR_ALL);
  EXPECT_FALSE(system_->type & RCCL_TOPO_XGMI_ALL);
  EXPECT_EQ(2, system_->nodes[NET].count);
  EXPECT_EQ(0, system_->inter);
  EXPECT_TRUE(Logged("GDR is available on all GPUs"));
}

TEST_F(PathsTrimMicrotest, TrimSystem_OneGpuWithoutGdrDropsTheNics) {
  Use(BuildWithPaths(GdrPcieSystem()));
  system_->nodes[GPU].nodes[0].gpu.gdrSupport = 0;

  ASSERT_EQ(ncclSuccess, Trim());

  EXPECT_FALSE(system_->type & RCCL_TOPO_GDR_ALL);
  EXPECT_EQ(0, system_->nodes[NET].count);
}

TEST_F(PathsTrimMicrotest, TrimSystem_EnableIntranetKeepsTheNics) {
  g_params["RCCL_ENABLE_INTRANET"] = 1;
  Use(BuildWithPaths(XgmiSystem(2, 1, true)));

  ASSERT_EQ(ncclSuccess, Trim());

  EXPECT_TRUE(system_->type & RCCL_TOPO_FORCE_INTRA);
  EXPECT_EQ(2, system_->nodes[NET].count);
  EXPECT_EQ(0, system_->inter);
}

TEST_F(PathsTrimMicrotest, TrimSystem_SlowerPortOfTheSameAsicIsDropped) {
  Peers(4);
  Use(BuildWithPaths(XgmiSystem(2, 1, true)));
  ncclTopoNode* nets = system_->nodes[NET].nodes;
  nets[1].net.asic = nets[0].net.asic;
  nets[1].net.bw = nets[0].net.bw / 2;
  int64_t fast = nets[0].id;

  ASSERT_EQ(ncclSuccess, Trim());

  ASSERT_EQ(1, system_->nodes[NET].count);
  EXPECT_EQ(fast, system_->nodes[NET].nodes[0].id);
  EXPECT_EQ(1, system_->inter);

  Use(BuildWithPaths(XgmiSystem(2, 1, true)));
  system_->nodes[NET].nodes[1].net.asic = system_->nodes[NET].nodes[0].net.asic;
  ASSERT_EQ(ncclSuccess, Trim());
  EXPECT_EQ(2, system_->nodes[NET].count);  // equally fast ports both stay
}

TEST_F(PathsTrimMicrotest, TrimSystem_MnnvlCliquePeersOnAnotherHostStay) {
  std::string twoHosts = System(Cpu(0, GpuPci(GpuBus(0), 0, 0)) + Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x5\""));
  Peers(2);
  for (ncclPeerInfo& p : peers_) {
    p.fabricInfo.clusterUuid[0] = 0xab;
    p.fabricInfo.cliqueId = 3;
  }
  peers_[1].hostHash = 2;
  Use(BuildWithPaths(twoHosts));
  ASSERT_EQ(ncclSuccess, Trim());
  EXPECT_EQ(std::vector<int>({0}), Ranks());

  comm_->MNNVL = 1;
  Use(BuildWithPaths(twoHosts));
  ASSERT_EQ(ncclSuccess, Trim());
  EXPECT_EQ(std::vector<int>({0, 1}), Ranks());
  EXPECT_TRUE(Logged("MNNVL enabled, checking clique membership for 2 ranks"));
}

TEST_F(PathsTrimMicrotest, TrimSystem_PathRebuildFailureIsReturned) {
  Use(BuildWithPaths(XgmiSystem(1)));
  ASSERT_EQ(ncclSuccess, ncclTopoConnectNodes(system_->nodes[CPU].nodes, system_->nodes[GPU].nodes, LINK_NVL, 1));

  EXPECT_EQ(ncclInternalError, Trim());
}

// --- PXN ranks

TEST_F(PathsCommMicrotest, GetPxnRanks_TheXgmiPeersNextToEachRemoteRanksNic) {
  comm_->pxnDisable = 0;
  Peers(4);
  for (int r = 0; r < 4; r++) peers_[r].nvmlDev = r % 2;
  Use(BuildWithPaths(XgmiSystem(2, 2, true)), 1);
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));
  system_->inter = 1;
  int* ranks = nullptr;
  int n = -1;

  ASSERT_EQ(ncclSuccess, ncclTopoGetPxnRanks(comm_.get(), &ranks, &n));

  ASSERT_EQ(1, n);  // remote ranks 0 and 2 both relay through local rank 0
  EXPECT_EQ(0, ranks[0]);
  free(ranks);
}

TEST_F(PathsCommMicrotest, GetPxnRanks_NoneWithoutGdrReadsOrOffNode) {
  comm_->pxnDisable = 0;
  Peers(4);
  for (int r = 0; r < 4; r++) peers_[r].nvmlDev = r % 2;
  Use(BuildWithPaths(XgmiSystem(2, 2, true)), 1);
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));
  int* ranks = nullptr;
  int n = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoGetPxnRanks(comm_.get(), &ranks, &n));
  EXPECT_EQ(0, n);  // inter == 0
  EXPECT_EQ(nullptr, ranks);

  system_->inter = 1;
  g_params["NET_GDR_READ"] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoGetPxnRanks(comm_.get(), &ranks, &n));
  EXPECT_EQ(0, n);
}

}  // namespace
