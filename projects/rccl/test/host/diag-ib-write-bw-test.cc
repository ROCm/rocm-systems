/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/diagnostics/ib_write_bw.cc, #include-d via DIAG_IB_WRITE_BW_CC_PATH to reach its
 * file-static helpers. Peers are simulated through the bootstrap seams; no test spawns ib_write_bw.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "diagnostics.h"
#include "diagnostics_log.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/diagnostics_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/topo_stubs.h"
#include "graph.h"
#include "os.h"
#include "transport.h"
#include "utils.h"

#include "fakes/param_redirect.h"

using RcclUnitTesting::CaptureStdout;

static uint64_t DefaultIbClockNano() {
  return clockNano();
}
static std::function<uint64_t()> g_ibClockNano = DefaultIbClockNano;

static int g_ibCallocFailAt = 0;
static int g_ibCallocCalls = 0;
static std::vector<void*> g_ibHostLive;
template <typename T>
static ncclResult_t IbCalloc(T** ptr, std::size_t nelem) {
  if (++g_ibCallocCalls == g_ibCallocFailAt) {
    return ncclSystemError;
  }
  const ncclResult_t ret = ncclCallocDebug(ptr, nelem, __FILE__, __LINE__, __func__, true);
  if (ret == ncclSuccess) {
    g_ibHostLive.push_back(*ptr);
  }
  return ret;
}

static void IbFree(void* ptr) {
  auto it = std::find(g_ibHostLive.begin(), g_ibHostLive.end(), ptr);
  if (it != g_ibHostLive.end()) {
    g_ibHostLive.erase(it);
  } else if (ptr != nullptr) {
    ADD_FAILURE() << "free of untracked host pointer " << ptr;
    return;
  }
  std::free(ptr);
}

#include "fakes/libc_seam.h"
#undef ncclCalloc
#define ncclCalloc(...) IbCalloc(__VA_ARGS__)
#define clockNano() g_ibClockNano()
#define free(ptr) IbFree(ptr)

#include DIAG_IB_WRITE_BW_CC_PATH

#undef free
#undef clockNano
#undef ncclCalloc
#include "fakes/libc_seam_undef.h"

namespace {

constexpr uint64_t kCommHash = 0x1234abcdULL;
constexpr uintptr_t kBootstrapBits = 0xb007;
constexpr int kCudaDev = 3;
constexpr int kNvmlBase = 20;
constexpr int kNetDevices = 4;
constexpr int kNetDev = 2;
constexpr int kPhysPort = 1;
constexpr double kAverage = 97.44;
constexpr char kSelfHost[] = "nodeA0";
constexpr char kHelp[] = "Usage:\n  --use_cuda=<cuda device id>\n  --use_cuda_dmabuf Use CUDA DMA-BUF\n";
constexpr char kClientOutput[] =
    "---------------------------------------------------------------------------------------\n"
    " Number of qps   : 1\t\tTransport type : IB\n"
    " #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n"
    " 65536      1000             97.52              97.44  \t\t   0.185852\n"
    "---------------------------------------------------------------------------------------\n";

std::string DiagLine(const std::string& body) {
  diagLogInit();
  return std::string(diagLogHost) + ":" + std::to_string(diagLogPid) + " " + body + "\n";
}

std::string NetInfo(const std::string& body) {
  return DiagLine("NCCL DIAG [INFO] net bw: " + body + " in comm 0x1234abcd");
}

std::string SysPath(const char* name) {
  return std::string("/sys/class/infiniband/") + name;
}

LocalInfo Info(const char* host, const char* device, bool cuda = true, bool dmabuf = true) {
  LocalInfo info{};
  info.deviceCount = 1;
  info.port = kPhysPort;
  info.cuda = cuda;
  info.dmabuf = dmabuf;
  std::snprintf(info.hostname, sizeof(info.hostname), "%s", host);
  std::snprintf(info.device, sizeof(info.device), "%s", device);
  return info;
}

std::function<ncclResult_t(int*)> g_ibNetDevices;
std::function<ncclResult_t(int, ncclNetProperties_t*)> g_ibNetGetProperties;
ncclResult_t IbNetDevices(int* count) {
  return g_ibNetDevices(count);
}
ncclResult_t IbNetGetProperties(int dev, ncclNetProperties_t* props) {
  return g_ibNetGetProperties(dev, props);
}

class DiagIbWriteBwMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    diagLogInit();
    InstallHooks();
  }

  void InstallHooks() {
    net_.name = "IB";
    net_.devices = IbNetDevices;
    net_.getProperties = IbNetGetProperties;
    g_ibNetDevices = [this](int* count) {
      *count = kNetDevices;
      return devicesResult_;
    };
    g_ibNetGetProperties = [this](int dev, ncclNetProperties_t* props) {
      propsDev_ = dev;
      props->name = const_cast<char*>(propName_);
      props->port = kPhysPort;
      return propsResult_;
    };
    g_gethostname = [this](char* name, size_t len) {
      hostnameLen_ = len;
      std::snprintf(name, len, "%s", kSelfHost);
      return 0;
    };
    g_access = [this](const char* path, int mode) {
      accessed_.push_back(path);
      EXPECT_EQ(mode, F_OK);
      if (std::find(sysfs_.begin(), sysfs_.end(), path) != sysfs_.end()) {
        return 0;
      }
      errno = ENOENT;
      return -1;
    };
    g_ncclTopoGetLocalNet = [this](ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
      EXPECT_EQ(system, comm_->topo);
      EXPECT_EQ(rank, comm_->rank);
      EXPECT_EQ(id, nullptr);
      localNetChannels_.push_back(channelId);
      *dev = localNetDev_;
      return localNetResult_;
    };
    g_ncclDiagChildRun = [this](const char* command, int timeoutSec, char* output, int outputSize, bool* truncated) {
      EXPECT_EQ(timeoutSec, IB_BW_TIMEOUT_SEC);
      EXPECT_EQ(outputSize, IB_BW_TOOL_OUTPUT_BYTES);
      EXPECT_EQ(truncated, nullptr);
      const bool probe = std::string(command) == "ib_write_bw --help";
      if (!probe) {
        clientCommands_.push_back(command);
      }
      DeliverChildOutput(probe ? help_ : clientOutput_, output, outputSize, nullptr, nullptr, truncated);
      return probe ? probeExit_ : clientExit_;
    };
  }

  void TearDown() override {
    EXPECT_TRUE(g_ibHostLive.empty()) << "host buffers leaked: " << g_ibHostLive.size();
    for (void* p : g_ibHostLive) {
      std::free(p);
    }
    g_ibHostLive.clear();
    g_ibCallocFailAt = 0;
    g_ibCallocCalls = 0;
    g_ibClockNano = DefaultIbClockNano;
    g_ibNetDevices = nullptr;
    g_ibNetGetProperties = nullptr;
    ResetDevRuntimeMicroFakes();
    ResetDiagnosticsFakes();
    ResetLibcFakes();
    ResetTopoStubs();
  }

  // nodes[n] lists node n's ranks in local-rank order; rank r has nvmlDev kNvmlBase + r.
  void BuildComm(int rank, std::vector<std::vector<int>> nodes) {
    comm_ = std::make_unique<ncclComm>();
    nodes_ = std::move(nodes);
    int nRanks = 0;
    for (const std::vector<int>& node : nodes_) {
      nRanks += static_cast<int>(node.size());
    }
    peers_.assign(nRanks, ncclPeerInfo{});
    nodeRanks_.assign(nodes_.size(), ncclNodeRanks{});
    for (std::size_t n = 0; n < nodes_.size(); n++) {
      nodeRanks_[n].localRanks = static_cast<int>(nodes_[n].size());
      nodeRanks_[n].localRankToRank = nodes_[n].data();
      for (int r : nodes_[n]) {
        peers_[r].nvmlDev = kNvmlBase + r;
        if (r == rank) {
          comm_->node = static_cast<int>(n);
        }
      }
    }
    info_.assign(nRanks, LocalInfo{});
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->nNodes = static_cast<int>(nodes_.size());
    comm_->nodeRanks = nodeRanks_.data();
    comm_->peerInfo = peers_.data();
    comm_->commHash = kCommHash;
    comm_->cudaDev = kCudaDev;
    comm_->dmaBufSupport = true;
    comm_->bootstrap = reinterpret_cast<void*>(kBootstrapBits);
    comm_->ncclNet = &net_;
    comm_->topo = reinterpret_cast<ncclTopoSystem*>(&topoStorage_);
    comm_->nChannels = 2;
  }

  void SetDevices(const std::vector<const char*>& devices) {
    for (std::size_t r = 0; r < devices.size(); r++) {
      info_[r] = Info(("host" + std::to_string(r)).c_str(), devices[r]);
    }
  }

  bool FindPair(int phase, bool cross, int* serverRank, int* clientRank) {
    *serverRank = -7;
    *clientRank = -7;
    return findPair(comm_.get(), info_.data(), phase, cross, *serverRank, *clientRank);
  }

  void ResetDiscoverScene() {
    InstallHooks();
    BuildComm(0, {{0}, {1}});
    propName_ = "mlx5_0";
    probeExit_ = 0;
    devicesResult_ = ncclSuccess;
    propsResult_ = ncclSuccess;
    localNetResult_ = ncclSuccess;
    localNetDev_ = kNetDev;
    localNetChannels_.clear();
  }

  // Moves the queried rank to `rank` on its own node.
  void PlaceRank(int rank) {
    comm_->rank = rank;
    for (std::size_t n = 0; n < nodes_.size(); n++) {
      if (std::find(nodes_[n].begin(), nodes_[n].end(), rank) != nodes_[n].end()) {
        comm_->node = static_cast<int>(n);
      }
    }
  }

  LocalInfo Discover() {
    LocalInfo local;
    std::memset(&local, 0x5a, sizeof(local));
    discoverLocal(comm_.get(), &local);
    return local;
  }

  std::unique_ptr<ncclComm> comm_;
  std::vector<std::vector<int>> nodes_;
  std::vector<ncclPeerInfo> peers_;
  std::vector<ncclNodeRanks> nodeRanks_;
  std::vector<LocalInfo> info_;
  int topoStorage_ = 0;
  ncclNet_t net_{};
  std::size_t hostnameLen_ = 0;
  const char* propName_ = "mlx5_0";
  int propsDev_ = -1;
  ncclResult_t devicesResult_ = ncclSuccess;
  ncclResult_t propsResult_ = ncclSuccess;
  int localNetDev_ = kNetDev;
  ncclResult_t localNetResult_ = ncclSuccess;
  std::vector<int> localNetChannels_;
  std::vector<std::string> sysfs_ = {SysPath(""), SysPath("mlx5_0"), SysPath("mlx5_1")};
  std::vector<std::string> accessed_;
  std::string help_ = kHelp;
  int probeExit_ = 0;
  std::string clientOutput_ = kClientOutput;
  int clientExit_ = 0;
  std::vector<std::string> clientCommands_;
};

TEST_F(DiagIbWriteBwMicrotest, NodeInventory_GroupsByFirstAppearanceInLocalRankOrder) {
  BuildComm(0, {{0, 1}, {5, 2, 7, 3, 4, 6}});
  SetDevices({"x", "x", "mlx5_0", "mlx5_2", "mlx5_1", "mlx5_1", "mlx5_0", "mlx5_1"});
  const std::vector<DeviceGroup> inventory = nodeInventory(comm_.get(), info_.data(), 1);
  ASSERT_EQ(inventory.size(), 3u);
  EXPECT_STREQ(inventory[0].name, "mlx5_1");
  EXPECT_EQ(inventory[0].ranks, (std::vector<int>{5, 7, 4}));
  EXPECT_STREQ(inventory[1].name, "mlx5_0");
  EXPECT_EQ(inventory[1].ranks, (std::vector<int>{2, 6}));
  EXPECT_STREQ(inventory[2].name, "mlx5_2");
  EXPECT_EQ(inventory[2].ranks, (std::vector<int>{3}));
  EXPECT_EQ(findGroup(inventory, "mlx5_0"), 1);
  EXPECT_EQ(findGroup(inventory, "mlx5_2"), 2);
  EXPECT_EQ(findGroup(inventory, "mlx5_3"), -1);
  EXPECT_EQ(rankPosition(inventory[0], 4), 2);
  EXPECT_EQ(rankPosition(inventory[0], 6), -1);
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_SameDevicePairsKthWithKthInBothPhases) {
  int server, client;
  BuildComm(0, {{2, 0, 3, 1}, {5, 7, 4, 6}});
  SetDevices({"A", "B", "A", "B", "B", "A", "A", "B"});
  ASSERT_TRUE(FindPair(0, false, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 6);
  ASSERT_TRUE(FindPair(1, false, &server, &client));
  EXPECT_EQ(server, 6);
  EXPECT_EQ(client, 0);
  comm_->rank = 6;
  comm_->node = 1;
  ASSERT_TRUE(FindPair(0, false, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 6);
  ASSERT_TRUE(FindPair(1, false, &server, &client));
  EXPECT_EQ(server, 6);
  EXPECT_EQ(client, 0);
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_CrossRotatesServerOntoNextClientDevice) {
  int server, client;
  BuildComm(0, {{2, 0, 3, 1}, {5, 7, 4, 6}});
  SetDevices({"A", "B", "A", "B", "B", "A", "A", "B"});
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 4);
  comm_->rank = 3;
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 3);
  EXPECT_EQ(client, 5);
  comm_->rank = 4;
  comm_->node = 1;
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 4);
  comm_->rank = 5;
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 3);
  EXPECT_EQ(client, 5);
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_CrossClientAnchorsOnPreviousDeviceOfThree) {
  int server, client;
  BuildComm(4, {{0, 1, 2}, {3, 4, 5}});
  SetDevices({"A", "B", "C", "A", "B", "C"});
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 4);
  comm_->rank = 3;
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 2);
  EXPECT_EQ(client, 3);
  PlaceRank(0);
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 4);
  PlaceRank(2);
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 2);
  EXPECT_EQ(client, 3);
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_BothEndsOfEveryPairAgree) {
  struct Topology {
    std::vector<std::vector<int>> nodes;
    std::vector<const char*> devices;
  };
  const std::vector<Topology> topologies = {
      {{{0, 1, 2}, {3, 4, 5}}, {"A", "B", "C", "C", "A", "B"}},
      {{{0, 1, 2, 3}, {4, 5, 6, 7}}, {"A", "A", "B", "C", "C", "B", "A", "A"}},
      {{{0, 1}, {2, 3}, {4, 5}}, {"A", "B", "B", "A", "A", "C"}},
      {{{3, 0}, {1, 4}, {2, 5}, {6, 7}}, {"A", "B", "A", "B", "A", "B", "B", "A"}},
  };
  for (std::size_t t = 0; t < topologies.size(); t++) {
    BuildComm(0, topologies[t].nodes);
    SetDevices(topologies[t].devices);
    const int nRanks = comm_->nRanks;
    for (int phase = 0; phase < 2; phase++) {
      for (bool cross : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "topology " << t << " phase " << phase << " cross " << cross);
        std::vector<int> servers(nRanks), clients(nRanks);
        std::vector<bool> paired(nRanks);
        for (int r = 0; r < nRanks; r++) {
          PlaceRank(r);
          paired[r] = FindPair(phase, cross, &servers[r], &clients[r]);
        }
        int pairs = 0;
        for (int r = 0; r < nRanks; r++) {
          if (!paired[r]) {
            continue;
          }
          ASSERT_TRUE(servers[r] == r || clients[r] == r) << r;
          const int peer = servers[r] == r ? clients[r] : servers[r];
          ASSERT_TRUE(paired[peer]) << r << " names unpaired " << peer;
          EXPECT_EQ(servers[peer], servers[r]) << r;
          EXPECT_EQ(clients[peer], clients[r]) << r;
          pairs++;
        }
        EXPECT_GT(pairs, 0);
      }
    }
  }
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_OddRingSkipsOnlyTheWrappingPair) {
  int server, client;
  BuildComm(2, {{0}, {1}, {2}});
  SetDevices({"A", "A", "A"});
  EXPECT_FALSE(FindPair(0, false, &server, &client));
  ASSERT_TRUE(FindPair(1, false, &server, &client));
  EXPECT_EQ(server, 1);
  EXPECT_EQ(client, 2);
  comm_->rank = 0;
  comm_->node = 0;
  ASSERT_TRUE(FindPair(0, false, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 1);
  EXPECT_FALSE(FindPair(1, false, &server, &client));
}

TEST_F(DiagIbWriteBwMicrotest, FindPair_FalseWhenNoPeerExists) {
  int server, client;
  BuildComm(0, {{0, 1}, {2, 3}});
  SetDevices({"A", "A", "A", "A"});
  EXPECT_FALSE(FindPair(0, true, &server, &client));  // one client device leaves nothing to rotate onto
  SetDevices({"A", "A", "A", "B"});
  comm_->rank = 1;
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // second A on the server, one A on the client
  comm_->rank = 3;
  comm_->node = 1;
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // B is not on the server node
  SetDevices({"C", "C", "A", "A"});
  comm_->rank = 0;
  comm_->node = 0;
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // C is not on the client node
  SetDevices({"C", "A", "A", "A"});
  comm_->rank = 3;
  comm_->node = 1;
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // the second A on the client outnumbers the server's
  EXPECT_FALSE(FindPair(1, false, &server, &client));
  comm_->rank = 2;
  ASSERT_TRUE(FindPair(1, false, &server, &client));
  EXPECT_EQ(server, 2);
  EXPECT_EQ(client, 1);
}

TEST_F(DiagIbWriteBwMicrotest, BenchmarkPort_Truncates32BitSumThenWrapsIntoBand) {
  BuildComm(0, {{0}, {1}});
  comm_->commHash = 0x100000000ULL + 19995;
  EXPECT_EQ(benchmarkPort(comm_.get(), 7), 10002);
  EXPECT_EQ(benchmarkPort(comm_.get(), 4), 29999);
}

TEST_F(DiagIbWriteBwMicrotest, ToolFailure_NamesTimeoutAndMissingToolElseGeneric) {
  EXPECT_EQ(toolFailure(0), nullptr);
  EXPECT_STREQ(toolFailure(124), "tool run timed out");
  EXPECT_STREQ(toolFailure(127), "tool missing");
  EXPECT_STREQ(toolFailure(1), "tool run failed");
  EXPECT_STREQ(toolFailure(-1), "tool run failed");
}

TEST_F(DiagIbWriteBwMicrotest, PluginSupportsIb_ExactOrDelimitedKnownNames) {
  for (const char* name : {"IB", "IBext", "IBext_v8", "IB SHARP", "SPCX", "NCCL RDMA Plugin", "NCCL RDMA Plugin_v2"}) {
    EXPECT_TRUE(pluginSupportsIb(name)) << name;
  }
  for (const char* name : {"", "I", "IBX", "IBext-v8", "Socket", "SPCX2", "NCCL RDMA", "ib"}) {
    EXPECT_FALSE(pluginSupportsIb(name)) << name;
  }
  EXPECT_FALSE(pluginSupportsIb(nullptr));
}

TEST_F(DiagIbWriteBwMicrotest, ResolveDeviceName_CopiesSegmentAndCutsDmaSuffixSysfsDoesNotKnow) {
  char out[IB_BW_NAME_SIZE];
  const char merged[] = "mlx5_1+mlx5_0";
  EXPECT_TRUE(resolveDeviceName(merged, 6, out));
  EXPECT_STREQ(out, "mlx5_1");
  EXPECT_EQ(accessed_, (std::vector<std::string>{SysPath("mlx5_1")}));
  accessed_.clear();
  EXPECT_TRUE(resolveDeviceName("mlx5_0_dma", 10, out));
  EXPECT_STREQ(out, "mlx5_0");
  EXPECT_EQ(accessed_, (std::vector<std::string>{SysPath("mlx5_0_dma"), SysPath("mlx5_0")}));
  accessed_.clear();
  EXPECT_FALSE(resolveDeviceName("mlx5_9_dma", 10, out));
  EXPECT_EQ(accessed_, (std::vector<std::string>{SysPath("mlx5_9_dma"), SysPath("mlx5_9")}));
  accessed_.clear();
  EXPECT_FALSE(resolveDeviceName("mlx5_9", 6, out));
  EXPECT_FALSE(resolveDeviceName("_dma", 4, out));
  EXPECT_EQ(accessed_, (std::vector<std::string>{SysPath("mlx5_9"), SysPath("_dma")}));
  accessed_.clear();
  sysfs_.push_back(SysPath("x_dma"));
  EXPECT_TRUE(resolveDeviceName("x_dma", 5, out));
  EXPECT_STREQ(out, "x_dma");
  EXPECT_FALSE(resolveDeviceName("mlx5_0", 0, out));
  const std::string longest(IB_BW_NAME_SIZE - 1, 'n');
  sysfs_.push_back(SysPath(longest.c_str()));
  EXPECT_TRUE(resolveDeviceName(longest.c_str(), longest.size(), out));
  EXPECT_EQ(std::string(out), longest);
  const std::string tooLong(IB_BW_NAME_SIZE, 'n');
  EXPECT_FALSE(resolveDeviceName(tooLong.c_str(), tooLong.size(), out));
}

TEST_F(DiagIbWriteBwMicrotest, SegmentCount_CountsPlusSeparatedMembers) {
  EXPECT_EQ(segmentCount(nullptr), 0);
  EXPECT_EQ(segmentCount(""), 1);
  EXPECT_EQ(segmentCount("mlx5_0"), 1);
  EXPECT_EQ(segmentCount("mlx5_0+mlx5_1+mlx5_2"), 3);
}

TEST_F(DiagIbWriteBwMicrotest, ProbeCapabilities_ReadsCudaAndDmabufFlagsFromHelp) {
  LocalInfo local{};
  probeExit_ = 3;
  EXPECT_EQ(probeCapabilities(&local), 3);
  EXPECT_TRUE(local.cuda);
  EXPECT_TRUE(local.dmabuf);
  help_ = "  --use_cuda=<id>\n";
  EXPECT_EQ(probeCapabilities(&local), 3);
  EXPECT_TRUE(local.cuda);
  EXPECT_FALSE(local.dmabuf);
  help_ = "  --use_cuda_dmabuf\n";
  EXPECT_EQ(probeCapabilities(&local), 3);
  EXPECT_TRUE(local.cuda);
  EXPECT_TRUE(local.dmabuf);
  help_ = "  --use_dmabuf --use_cud\n";
  EXPECT_EQ(probeCapabilities(&local), 3);
  EXPECT_FALSE(local.cuda);
  EXPECT_FALSE(local.dmabuf);
}

TEST_F(DiagIbWriteBwMicrotest, ProbeCapabilities_RocmBuildHelpFallsBackToHostMemory) {
  LocalInfo local{};
  help_ = "  --use_rocm=<rocm device id>\n  --use_rocm_dmabuf\n";
  EXPECT_EQ(probeCapabilities(&local), 0);
  // ROCm perftest spells it --use_rocm, so GPU memory is never measured on AMD; a fix flips this pin.
  EXPECT_FALSE(local.cuda);
  EXPECT_FALSE(local.dmabuf);
}

TEST_F(DiagIbWriteBwMicrotest, DiscoverLocal_RecordsHostChannelZeroDeviceAndCapabilities) {
  BuildComm(1, {{0, 1}, {2, 3}});
  comm_->nChannels = 4;
  propName_ = "mlx5_1_dma+mlx5_0+mlx5_3";
  sysfs_ = {SysPath("mlx5_1")};
  const LocalInfo local = Discover();
  EXPECT_FALSE(local.setupFailed);
  EXPECT_STREQ(local.hostname, kSelfHost);
  EXPECT_EQ(hostnameLen_, IB_BW_HOSTNAME_SIZE - 1u);
  EXPECT_STREQ(local.device, "mlx5_1");
  EXPECT_EQ(local.deviceCount, 3);
  EXPECT_EQ(local.port, kPhysPort);
  EXPECT_TRUE(local.cuda);
  EXPECT_TRUE(local.dmabuf);
  EXPECT_EQ(localNetChannels_, (std::vector<int>{0}));
  EXPECT_EQ(propsDev_, kNetDev);
  EXPECT_EQ(local.hostname[sizeof(local.hostname) - 1], '\0');
  EXPECT_EQ(local.device[sizeof(local.device) - 1], '\0');
}

TEST_F(DiagIbWriteBwMicrotest, DiscoverLocal_CudaNeedsDeviceAndDmabufNeedsCommSupport) {
  BuildComm(0, {{0}, {1}});
  comm_->dmaBufSupport = false;
  LocalInfo local = Discover();
  EXPECT_TRUE(local.cuda);
  EXPECT_FALSE(local.dmabuf);
  comm_->dmaBufSupport = true;
  comm_->cudaDev = -1;
  local = Discover();
  EXPECT_FALSE(local.cuda);
  EXPECT_FALSE(local.dmabuf);
  comm_->cudaDev = 0;
  probeExit_ = 1;
  local = Discover();
  EXPECT_FALSE(local.setupFailed);
  EXPECT_TRUE(local.cuda);
  EXPECT_TRUE(local.dmabuf);
}

TEST_F(DiagIbWriteBwMicrotest, DiscoverLocal_EachSetupFailureReportsAndMarksRank) {
  struct Case {
    std::function<void()> arm;
    std::string line;
    bool reached;
  };
  const std::vector<Case> cases = {
      {[] { g_gethostname = [](char*, size_t) { return -1; }; }, "cannot determine local hostname", false},
      {[this] { probeExit_ = 127; }, "required external tool missing", false},
      {[this] { probeExit_ = 124; }, "capability probe timed out", false},
      {[this] { comm_->ncclNet = nullptr; }, "failed: selected network plugin does not support IB", false},
      {[this] { net_.name = "Socket"; }, "failed: selected network plugin does not support IB", false},
      {[this] { devicesResult_ = ncclSystemError; }, "cannot enumerate network devices", false},
      {[] {
         g_ibNetDevices = [](int* count) {
           *count = 0;
           return ncclSuccess;
         };
       },
       "failed: no usable IB device", false},
      {[this] { comm_->topo = nullptr; }, "failed: no usable IB device", false},
      {[this] { comm_->nChannels = 0; }, "invalid channel count=0", false},
      {[this] { localNetResult_ = ncclInternalError; }, "cannot select a local network device", true},
      {[this] { localNetDev_ = -1; }, "failed: no usable IB device", true},
      {[this] { localNetDev_ = kNetDevices; }, "failed: no usable IB device", true},
      {[this] { propsResult_ = ncclSystemError; }, "cannot query the selected network device", true},
      {[this] { propName_ = nullptr; }, "failed: cannot resolve the selected IB device", true},
      {[this] { propName_ = "+mlx5_0"; }, "failed: cannot resolve the selected IB device", true},
      {[this] { propName_ = "mlx5_7"; }, "failed: cannot resolve the selected IB device", true},
  };
  for (std::size_t i = 0; i < cases.size(); i++) {
    SCOPED_TRACE(i);
    ResetDiscoverScene();
    cases[i].arm();
    LocalInfo local{};
    const std::string out = CaptureStdout([&] { local = Discover(); });
    EXPECT_EQ(out, NetInfo(cases[i].line));
    EXPECT_TRUE(local.setupFailed);
    EXPECT_EQ(local.deviceCount, 0);
    EXPECT_EQ(localNetChannels_.size(), cases[i].reached ? 1u : 0u);
  }
}

TEST_F(DiagIbWriteBwMicrotest, BuildCommand_ServerOmitsHostClientAppendsItAndTruncationFails) {
  const LocalInfo local = Info("me", "mlx5_1");
  const LocalInfo server = Info("srv", "mlx5_0");
  char out[IB_BW_COMMAND_BYTES];
  ASSERT_TRUE(buildCommand(out, sizeof(out), true, local, server, false, true, 5, 12345, 4));
  EXPECT_STREQ(out, "ib_write_bw -d mlx5_1 -i 1 -s 65536 --report_gbits -q 4 -p 12345 -n 1000");
  ASSERT_TRUE(buildCommand(out, sizeof(out), false, local, server, true, false, 5, 12345, 4));
  EXPECT_STREQ(out, "ib_write_bw -d mlx5_1 -i 1 -s 65536 --report_gbits -q 4 -p 12345 -n 1000 --use_cuda=5 srv");
  ASSERT_TRUE(buildCommand(out, sizeof(out), true, local, server, true, true, 5, 12345, 4));
  EXPECT_STREQ(out,
               "ib_write_bw -d mlx5_1 -i 1 -s 65536 --report_gbits -q 4 -p 12345 -n 1000 --use_cuda=5 "
               "--use_cuda_dmabuf");
  const std::string fits = "ib_write_bw -d mlx5_1 -i 1 -s 65536 --report_gbits -q 4 -p 12345 -n 1000";
  EXPECT_TRUE(buildCommand(out, static_cast<int>(fits.size()) + 1, true, local, server, false, false, 5, 12345, 4));
  EXPECT_FALSE(buildCommand(out, static_cast<int>(fits.size()), true, local, server, false, false, 5, 12345, 4));
  EXPECT_FALSE(buildCommand(out, static_cast<int>(fits.size()) + 4, false, local, server, false, false, 5, 12345, 4));
  EXPECT_TRUE(buildCommand(out, static_cast<int>(fits.size()) + 5, false, local, server, false, false, 5, 12345, 4));
  EXPECT_FALSE(buildCommand(out, static_cast<int>(fits.size()), false, local, server, false, false, 5, 12345, 4));
}

TEST_F(DiagIbWriteBwMicrotest, ParseBandwidth_TakesAverageOfFirstFiniteResultRow) {
  double average = -5;
  EXPECT_TRUE(parseBandwidth(kClientOutput, average));
  EXPECT_DOUBLE_EQ(average, kAverage);
  EXPECT_TRUE(parseBandwidth("header\n 1 2 nan 4\n 1 2 3 inf\n 7 8 9.5 6.25\n 7 8 1 2\n", average));
  EXPECT_DOUBLE_EQ(average, 6.25);
  EXPECT_TRUE(parseBandwidth("1 2 3.5 4.5", average));
  EXPECT_DOUBLE_EQ(average, 4.5);
  EXPECT_FALSE(parseBandwidth("1 2 3\n#bytes 1 2 3\n\n", average));
  EXPECT_FALSE(parseBandwidth("", average));
  EXPECT_FALSE(parseBandwidth("1 2 3", average));
  EXPECT_FALSE(parseBandwidth(nullptr, average));
}

}  // namespace
