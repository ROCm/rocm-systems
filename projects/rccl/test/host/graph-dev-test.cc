/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the topology GPU id: src/graph/xml.cc, topo.cc, paths.cc, search.cc and rome_models.cc.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/amdsmi_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/os_fakes.h"

#include "comm.h"
#include "graph.h"
#include "graph/rome_models.h"
#include "graph/topo.h"
#include "graph/xml.h"

namespace {

constexpr uint64_t kHostHash = 0x5eed;
constexpr int kMaxXmlNodes = 512;

struct Gpu {
  std::string busId;
  int numa;
};

// A process: the SMI indices it can see, in HIP ordinal order, and the one its rank runs on.
struct Proc {
  std::vector<int> visible;
  int own;
  int rank;
  int mloPart;
};

class TopoGpuDevMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (struct ncclXml* xml : xmls_) {
      free(xml);
    }
    if (system_ != nullptr) {
      ncclTopoFree(system_);
    }
    ResetHipFakes();
    ResetAmdSmiFakes();
    ResetOsFakes();
  }

  // SMI enumerates every GPU of the node, and reports a 1-hop XGMI link for each pair linked() accepts.
  void InstallNode(std::vector<Gpu> gpus, std::function<bool(int, int)> linked) {
    gpus_ = std::move(gpus);
    g_amdSmiGetNumDevice = [this](uint32_t* n) {
      *n = gpus_.size();
      return ncclSuccess;
    };
    g_amdSmiGetDevicePciBusIdString = [this](uint32_t i, char* busId, size_t len) {
      snprintf(busId, len, "%s", gpus_[i].busId.c_str());
      return ncclSuccess;
    };
    g_amdSmiGetDeviceIndexByPciBusId = [this](const char* busId, uint32_t* index) {
      *index = SmiIndex(busId);
      return *index < gpus_.size() ? ncclSuccess : ncclInternalError;
    };
    g_amdSmiGetFabricDeviceInfo = [](uint32_t, struct amdsmiFabricDeviceInfo* info) {
      info->fabricSupported = false;
      return ncclSuccess;
    };
    g_amdSmiGetLinkInfo = [linked](int a, int b, amdsmi_link_type_t* type, int* hops, int* count) {
      *type = linked(a, b) ? AMDSMI_LINK_TYPE_XGMI : AMDSMI_LINK_TYPE_PCIE;
      *hops = linked(a, b) ? 1 : 2;
      *count = 1;
      return ncclSuccess;
    };
    g_pciDeviceClass = "";  // partition aliases at .1-.7 are absent from sysfs
  }

  uint32_t SmiIndex(const char* busId) const {
    uint32_t i = 0;
    while (i < gpus_.size() && gpus_[i].busId != busId) {
      i++;
    }
    return i;
  }

  static void SetAttrs(struct ncclXmlNode* node, std::initializer_list<std::string> kvs) {
    for (const std::string& kv : kvs) {
      EXPECT_EQ(xmlSetAttr(node, kv.substr(0, kv.find('=')).c_str(), kv.substr(kv.find('=') + 1).c_str()), ncclSuccess);
    }
  }

  // Runs ncclTopoFillGpu as `proc` would, over the sysfs view of its GPU, then stamps what ncclTopoGetSystem adds.
  struct ncclXml* FillAs(const Proc& proc, bool fromTopoFile = false) {
    ScopedHook count(g_hipGetDeviceCount, [&proc](int* n) {
      *n = proc.visible.size();
      return hipSuccess;
    });
    ScopedHook busIdOf(g_hipDeviceGetPCIBusId, [this, &proc](char* busId, int len, int ordinal) {
      snprintf(busId, len, "%s", gpus_[proc.visible[ordinal]].busId.c_str());
      return hipSuccess;
    });
    ScopedHook ordinalOf(g_hipDeviceGetByPCIBusId, [this, &proc](int* ordinal, const char* busId) {
      auto it = std::find(proc.visible.begin(), proc.visible.end(), (int)SmiIndex(busId));
      *ordinal = it - proc.visible.begin();
      return it == proc.visible.end() ? hipErrorInvalidValue : hipSuccess;
    });
    ScopedHook props(g_hipGetDeviceProperties, [this](hipDeviceProp_t* prop, int ordinal) {
      propsOrdinals_.push_back(ordinal);
      *prop = hipDeviceProp_t{};
      snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942:sramecc+:xnack-");
      return hipSuccess;
    });

    struct ncclXml* xml = NewXml();
    struct ncclXmlNode *top, *cpu, *pci, *gpu;
    EXPECT_EQ(xmlAddNode(xml, nullptr, "system", &top), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ncclSuccess);
    EXPECT_EQ(xmlAddNode(xml, top, "cpu", &cpu), ncclSuccess);
    EXPECT_EQ(xmlSetAttrLong(cpu, "host_hash", kHostHash), ncclSuccess);
    EXPECT_EQ(xmlSetAttrInt(cpu, "numaid", gpus_[proc.own].numa), ncclSuccess);
    SetAttrs(cpu, {"affinity=", "arch=x86_64", "vendor=AuthenticAMD", "familyid=25", "modelid=1"});
    EXPECT_EQ(xmlAddNode(xml, cpu, "pci", &pci), ncclSuccess);
    std::string physBusId = gpus_[proc.own].busId.substr(0, 11) + "0";
    SetAttrs(pci, {"class=0x03", "vendor=0x1002", "device=0x74a1", "subsystem_vendor=0x1002", "subsystem_device=0x74a1",
                   "link_speed=32.0 GT/s PCIe", "link_width=16"});
    EXPECT_EQ(xmlSetAttr(pci, "busid", physBusId.c_str()), ncclSuccess);
    if (fromTopoFile) {
      EXPECT_EQ(xmlAddNode(xml, pci, "gpu", &gpu), ncclSuccess);
      SetAttrs(gpu, {"sm=304", "gcn=gfx942", "arch=0"});
      EXPECT_EQ(xmlSetAttrInt(gpu, "dev", proc.own), ncclSuccess);
    }

    EXPECT_EQ(ncclTopoFillGpu(xml, gpus_[proc.own].busId.c_str(), &gpu), ncclSuccess);
    EXPECT_NE(gpu, nullptr);
    if (gpu != nullptr) {
      EXPECT_EQ(xmlSetAttrInt(gpu, "keep", 1), ncclSuccess);
      EXPECT_EQ(xmlSetAttrInt(gpu, "rank", proc.rank), ncclSuccess);
      EXPECT_EQ(xmlSetAttrInt(gpu, "gdr", 0), ncclSuccess);
      EXPECT_EQ(xmlSetAttrInt(gpu, "mlopart", proc.mloPart), ncclSuccess);
    }
    return xml;
  }

  // Fuses every process's XML, as the intra-node allgather in ncclTopoGetSystem does.
  struct ncclXml* FuseAll(const std::vector<Proc>& procs) {
    struct ncclXml* fused = NewXml();
    for (const Proc& proc : procs) {
      EXPECT_EQ(ncclTopoFuseXml(fused, FillAs(proc)), ncclSuccess);
    }
    return fused;
  }

  struct ncclTopoSystem* BuildSystem(struct ncclXml* xml) {
    EXPECT_EQ(ncclTopoGetSystemFromXml(xml, &system_, kHostHash), ncclSuccess);
    if (system_ != nullptr) {
      system_->netGdrLevel = -2;
      EXPECT_EQ(ncclTopoComputePaths(system_, nullptr), ncclSuccess);
    }
    return system_;
  }

  struct GpuEntry {
    int rank;
    int dev;
    int mloPart;
    std::string pciBusId;
    std::vector<std::string> xgmiTargets;
  };

  // The <gpu> nodes of an XML, ordered by rank.
  static std::vector<GpuEntry> GpusByRank(struct ncclXml* xml) {
    std::vector<GpuEntry> gpus;
    for (int n = 0; n < xml->maxIndex; n++) {
      struct ncclXmlNode* node = xml->nodes + n;
      if (strcmp(node->name, "gpu") != 0) {
        continue;
      }
      GpuEntry e{};
      const char* busId = nullptr;
      EXPECT_EQ(xmlGetAttrInt(node, "rank", &e.rank), ncclSuccess);
      EXPECT_EQ(xmlGetAttrInt(node, "dev", &e.dev), ncclSuccess);
      EXPECT_EQ(xmlGetAttrInt(node, "mlopart", &e.mloPart), ncclSuccess);
      EXPECT_EQ(xmlGetAttr(node->parent, "busid", &busId), ncclSuccess);
      e.pciBusId = busId ? busId : "";
      for (int i = 0; i < node->nSubs; i++) {
        const char* target = nullptr;
        EXPECT_EQ(xmlGetAttr(node->subs[i], "target", &target), ncclSuccess);
        e.xgmiTargets.push_back(target ? target : "");
      }
      gpus.push_back(e);
    }
    std::sort(gpus.begin(), gpus.end(), [](const GpuEntry& a, const GpuEntry& b) { return a.rank < b.rank; });
    return gpus;
  }

  // n GPUs at distinct bus ids, the first half on NUMA node 0 and the rest on node 1, as on an 8-OAM MI300X node.
  static std::vector<Gpu> SpxNode(int n) {
    std::vector<Gpu> gpus;
    for (int g = 0; g < n; g++) {
      char busId[32];
      snprintf(busId, sizeof(busId), "0000:%02x:00.0", 0x0c + 0x10 * g);
      gpus.push_back({busId, g < n / 2 ? 0 : 1});
    }
    return gpus;
  }

  // Each rank in its own process, seeing only its GPU, which is therefore HIP ordinal 0 everywhere.
  static std::vector<Proc> OneGpuPerProcess(const std::vector<int>& smiByRank, bool partitions = false) {
    std::vector<Proc> procs;
    for (int r = 0; r < (int)smiByRank.size(); r++) {
      procs.push_back({{smiByRank[r]}, smiByRank[r], r, partitions ? smiByRank[r] : NCCL_TOPO_UNDEF});
    }
    return procs;
  }

  static bool AllLinked(int, int) { return true; }

  // Ranks 0-3, one per process, on SMI 4-7 of an 8-GPU node, so no rank equals its GPU's dev.
  bool TrimUpperHalfFlagsAllXgmi(std::function<bool(int, int)> linked) {
    InstallNode(SpxNode(8), linked);
    struct ncclTopoSystem* system = BuildSystem(FuseAll(OneGpuPerProcess({4, 5, 6, 7})));
    EXPECT_NE(system, nullptr);
    if (system == nullptr) {
      return false;
    }
    auto comm = std::make_unique<ncclComm>();
    std::vector<ncclPeerInfo> peers(4);
    comm->peerInfo = peers.data();
    comm->nRanks = 4;
    comm->topo = system;
    comm->pxnDisable = 1;
    EXPECT_EQ(ncclTopoComputePaths(system, comm.get()), ncclSuccess);
    EXPECT_EQ(ncclTopoTrimSystem(system, comm.get()), ncclSuccess);
    return system->type & RCCL_TOPO_XGMI_ALL;
  }

  std::vector<int> propsOrdinals_;

 private:
  struct ncclXml* NewXml() {
    struct ncclXml* xml = nullptr;
    EXPECT_EQ(xmlAlloc(&xml, kMaxXmlNodes), ncclSuccess);
    xmls_.push_back(xml);
    return xml;
  }

  std::vector<Gpu> gpus_;
  std::vector<struct ncclXml*> xmls_;
  struct ncclTopoSystem* system_ = nullptr;
};

TEST_F(TopoGpuDevMicrotest, OneGpuPerProcess_DevIsTheSmiIndexAndHipCallsUseOrdinalZero) {
  InstallNode(SpxNode(8), AllLinked);
  std::vector<GpuEntry> gpus = GpusByRank(FuseAll(OneGpuPerProcess({0, 1, 2, 3, 4, 5, 6, 7})));

  ASSERT_EQ(gpus.size(), 8u);
  for (int r = 0; r < 8; r++) {
    EXPECT_EQ(gpus[r].dev, r) << "rank " << r;
  }
  ASSERT_FALSE(propsOrdinals_.empty());
  for (int ordinal : propsOrdinals_) {
    EXPECT_EQ(ordinal, 0);
  }
}

TEST_F(TopoGpuDevMicrotest, TopoFile_GpuNamedByItsSmiIndexIsFilledInPlace) {
  InstallNode(SpxNode(8), AllLinked);
  struct ncclXml* xml = FillAs({{5}, 5, 0, NCCL_TOPO_UNDEF}, /*fromTopoFile=*/true);

  int nGpus = 0;
  for (int n = 0; n < xml->maxIndex; n++) {
    nGpus += strcmp(xml->nodes[n].name, "gpu") == 0;
  }
  EXPECT_EQ(nGpus, 1);
  EXPECT_EQ(GpusByRank(xml).at(0).dev, 5);
}

TEST_F(TopoGpuDevMicrotest, OneGpuPerProcess_RomeModelPlacesEveryRankInEachRing) {
  InstallNode(SpxNode(8), AllLinked);
  struct ncclTopoSystem* system = BuildSystem(FuseAll(OneGpuPerProcess({0, 1, 2, 3, 4, 5, 6, 7})));
  ASSERT_NE(system, nullptr);
  auto graph = std::make_unique<struct ncclTopoGraph>();
  graph->pattern = NCCL_TOPO_PATTERN_RING;
  graph->maxChannels = MAXCHANNELS;

  ASSERT_EQ(parseA2a8P(system, graph.get(), nullptr), ncclSuccess);

  ASSERT_GT(graph->nChannels, 0);
  for (int c = 0; c < graph->nChannels; c++) {
    std::vector<int> ring(graph->intra + c * 8, graph->intra + (c + 1) * 8);
    std::sort(ring.begin(), ring.end());
    EXPECT_EQ(ring, std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7})) << "channel " << c;
  }
}

TEST_F(TopoGpuDevMicrotest, CpxOnePartitionPerProcess_PartitionsKeepDistinctDevsUnderThePhysicalGpu) {
  std::vector<Gpu> partitions;
  for (int p = 0; p < 8; p++) {
    partitions.push_back({"0000:0c:00." + std::to_string(p), 0});
  }
  InstallNode(partitions, AllLinked);
  std::vector<GpuEntry> gpus = GpusByRank(FuseAll(OneGpuPerProcess({0, 1, 2, 3, 4, 5, 6, 7}, true)));

  ASSERT_EQ(gpus.size(), 8u);
  for (int r = 0; r < 8; r++) {
    EXPECT_EQ(gpus[r].dev, r) << "rank " << r;
    EXPECT_EQ(gpus[r].mloPart, r) << "rank " << r;
    EXPECT_EQ(gpus[r].pciBusId, "0000:0c:00.0") << "rank " << r;
    EXPECT_EQ(gpus[r].xgmiTargets.size(), 7u) << "rank " << r;
    EXPECT_EQ(std::count(gpus[r].xgmiTargets.begin(), gpus[r].xgmiTargets.end(), partitions[r].busId), 0);
  }
}

TEST_F(TopoGpuDevMicrotest, CpxAllPartitionsInOneProcess_PartitionsKeepDistinctDevsUnderThePhysicalGpu) {
  std::vector<Gpu> partitions;
  std::vector<Proc> procs;
  for (int p = 0; p < 8; p++) {
    partitions.push_back({"0000:0c:00." + std::to_string(p), 0});
    procs.push_back({{0, 1, 2, 3, 4, 5, 6, 7}, p, p, p});
  }
  InstallNode(partitions, AllLinked);
  std::vector<GpuEntry> gpus = GpusByRank(FuseAll(procs));

  ASSERT_EQ(gpus.size(), 8u);
  for (int r = 0; r < 8; r++) {
    EXPECT_EQ(gpus[r].dev, r) << "rank " << r;
    EXPECT_EQ(gpus[r].mloPart, r) << "rank " << r;
    EXPECT_EQ(gpus[r].pciBusId, "0000:0c:00.0") << "rank " << r;
  }
}

TEST_F(TopoGpuDevMicrotest, LinkType_FindsGpusByRankWhenNoRankEqualsItsGpusDev) {
  // An XGMI chain over SMI 4-5-6-7; ranks 0-3 run on SMI 7, 4, 5, 6.
  InstallNode(SpxNode(8), [](int a, int b) { return a >= 4 && b >= 4 && (a - b == 1 || b - a == 1); });
  struct ncclTopoSystem* system = BuildSystem(FuseAll(OneGpuPerProcess({7, 4, 5, 6})));
  ASSERT_NE(system, nullptr);
  bool isXgmi = false;

  ASSERT_EQ(ncclTopoGetLinkType(system, 1, 2, &isXgmi, 0), ncclSuccess);
  EXPECT_TRUE(isXgmi);
  ASSERT_EQ(ncclTopoGetLinkType(system, 1, 3, &isXgmi, 0), ncclSuccess);
  EXPECT_FALSE(isXgmi);
  ASSERT_EQ(ncclTopoGetLinkType(system, 1, 3, &isXgmi, 1), ncclSuccess);
  EXPECT_TRUE(isXgmi);
  ASSERT_EQ(ncclTopoGetLinkType(system, 0, 1, &isXgmi, 1), ncclSuccess);
  EXPECT_FALSE(isXgmi);
}

TEST_F(TopoGpuDevMicrotest, TrimSystem_FlagsAllXgmiForOneGpuPerProcessOnTheUpperHalfOfTheNode) {
  EXPECT_TRUE(TrimUpperHalfFlagsAllXgmi(AllLinked));
}

TEST_F(TopoGpuDevMicrotest, TrimSystem_DoesNotFlagAllXgmiWhenTheUpperHalfIsTwoHives) {
  EXPECT_FALSE(TrimUpperHalfFlagsAllXgmi([](int a, int b) { return a / 2 == b / 2; }));
}

}  // namespace
