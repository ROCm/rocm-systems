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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
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
#include "fakes/hip_fakes.h"
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
constexpr char kServerOutput[] = "****\n* Waiting for client to connect... *\n****\n";

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

template <typename T>
std::string Bytes(const T& value) {
  return std::string(reinterpret_cast<const char*>(&value), sizeof(value));
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

int Port(int serverRank) {
  return IB_BW_PORT_BASE + static_cast<unsigned int>(kCommHash + kNvmlBase + serverRank) % IB_BW_PORT_SPAN;
}

std::string Command(const char* device, int port, const char* cuda, const char* serverHost, int ibPort = 1,
                    int qps = 1) {
  std::string command = "ib_write_bw -d " + std::string(device) + " -i " + std::to_string(ibPort) +
                        " -s 65536 --report_gbits -q " + std::to_string(qps) + " -p " + std::to_string(port) +
                        " -n 1000" + cuda;
  if (serverHost != nullptr) {
    command += std::string(" ") + serverHost;
  }
  return command;
}

std::function<ncclResult_t(int*)> g_ibNetDevices;
std::function<ncclResult_t(int, ncclNetProperties_t*)> g_ibNetGetProperties;
ncclResult_t IbNetDevices(int* count) {
  return g_ibNetDevices(count);
}
ncclResult_t IbNetGetProperties(int dev, ncclNetProperties_t* props) {
  return g_ibNetGetProperties(dev, props);
}

// One bootstrap point-to-point call, in issue order: 'S' send or 'R' recv, with the bytes that crossed.
struct IbMsg {
  char kind;
  int peer;
  int tag;
  std::string bytes;
};

constexpr int kSync = IB_BW_PAIR_SYNC_TAG;
constexpr int kReady = IB_BW_SERVER_READY_TAG;
constexpr char kCudaFlags[] = " --use_cuda=3 --use_cuda_dmabuf";

bool operator==(const IbMsg& a, const IbMsg& b) {
  return a.kind == b.kind && a.peer == b.peer && a.tag == b.tag && a.bytes == b.bytes;
}

std::ostream& operator<<(std::ostream& os, const IbMsg& m) {
  os << m.kind << " peer=" << m.peer << " tag=" << std::hex << m.tag << " bytes=";
  for (unsigned char byte : m.bytes) {
    os << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  }
  return os << std::setfill(' ') << std::dec;
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
      return scene_.devicesResult;
    };
    g_ibNetGetProperties = [this](int dev, ncclNetProperties_t* props) {
      propsDev_ = dev;
      props->name = const_cast<char*>(scene_.propName);
      props->port = kPhysPort;
      return scene_.propsResult;
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
      scene_.localNetChannels.push_back(channelId);
      *dev = scene_.localNetDev;
      return scene_.localNetResult;
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
      return probe ? scene_.probeExit : clientExit_;
    };
    g_ncclDiagChildRunStream = [this](const char* command, int timeoutSec, char* output, int outputSize,
                                      ncclDiagChildLineFn onLine, void* ctx, bool* truncated) {
      EXPECT_EQ(timeoutSec, IB_BW_TIMEOUT_SEC);
      EXPECT_EQ(output, nullptr);
      EXPECT_EQ(outputSize, 0);
      serverCommands_.push_back(command);
      DeliverChildOutput(serverOutput_, output, outputSize, onLine, ctx, truncated);
      return serverExit_;
    };
    g_devrBootstrapSend = [this](void* bs, int peer, int tag, void* data, int size) {
      EXPECT_EQ(bs, comm_->bootstrap);
      log_.push_back({'S', peer, tag, std::string(static_cast<const char*>(data), size)});
      return sendFailTag_ == tag ? ncclRemoteError : ncclSuccess;
    };
    g_devrBootstrapRecv = [this](void* bs, int peer, int tag, void* data, int size) {
      EXPECT_EQ(bs, comm_->bootstrap);
      std::deque<std::string>& queue = inbox_[tag];
      if (queue.empty()) {
        log_.push_back({'R', peer, tag, ""});
        return ncclRemoteError;
      }
      EXPECT_EQ(queue.front().size(), static_cast<std::size_t>(size));
      std::memcpy(data, queue.front().data(), std::min<std::size_t>(size, queue.front().size()));
      log_.push_back({'R', peer, tag, queue.front()});
      queue.pop_front();
      return ncclSuccess;
    };
    g_devrBootstrapAllGather = [this](void* bs, void* buf, int size) { return AllGather(bs, buf, size); };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
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
    ResetHipFakes();
    ResetNcclFakes();
    ResetDiagnosticsFakes();
    ResetLibcFakes();
    ResetTopoStubs();
  }

  // Gathers the scripted peers around this rank's slot, which is recorded rather than overwritten.
  ncclResult_t AllGather(void* bs, void* buf, int size) {
    EXPECT_EQ(bs, comm_->bootstrap);
    if (++gathers_ == gatherFailAt_) {
      return ncclRemoteError;
    }
    const int self = comm_->rank;
    if (size == static_cast<int>(sizeof(LocalInfo))) {
      LocalInfo* slots = static_cast<LocalInfo*>(buf);
      selfInfo_ = slots[self];
      for (int r = 0; r < comm_->nRanks; r++) {
        if (r != self) {
          slots[r] = info_[r];
        }
      }
    } else if (size == static_cast<int>(sizeof(bool))) {
      bool* slots = static_cast<bool*>(buf);
      myVotes_.push_back(slots[self]);
      for (int r = 0; r < comm_->nRanks; r++) {
        if (r != self) {
          slots[r] = r != peerVetoRank_;
        }
      }
    } else if (size == static_cast<int>(sizeof(RankBandwidth))) {
      RankBandwidth* slots = static_cast<RankBandwidth*>(buf);
      selfBw_ = slots[self];
      for (int r = 0; r < comm_->nRanks; r++) {
        if (r != self) {
          slots[r] = bw_[r];
        }
      }
    } else {
      ADD_FAILURE() << "unexpected allgather size " << size;
    }
    return ncclSuccess;
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
    bw_.assign(nRanks, RankBandwidth{-1, -1});
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
    ASSERT_LE(devices.size(), info_.size());
    for (std::size_t r = 0; r < devices.size(); r++) {
      info_[r] = Info(("host" + std::to_string(r)).c_str(), devices[r]);
    }
  }

  // Rank 0 on node A pairs with rank 2 on node B over mlx5_0 in both phases.
  void BuildTwoNodePair(int rank) {
    BuildComm(rank, {{0, 1}, {2, 3}});
    info_[0] = Info(kSelfHost, "mlx5_0");
    info_[1] = Info("nodeA1", "mlx5_1");
    info_[2] = Info("nodeB0", "mlx5_0");
    info_[3] = Info("nodeB1", "mlx5_1");
  }

  void ResetPairScene(int rank) {
    InstallHooks();
    BuildTwoNodePair(rank);
    inbox_.clear();
    log_.clear();
    clientCommands_.clear();
    serverCommands_.clear();
    clientOutput_ = kClientOutput;
    clientExit_ = 0;
    myVotes_.clear();
    gathers_ = 0;
    gatherFailAt_ = 0;
    params_.clear();
    g_ibCallocCalls = 0;
    serverOutput_ = kServerOutput;
    serverExit_ = 0;
    sendFailTag_ = 0;
    peerVetoRank_ = -1;
    g_ibCallocFailAt = 0;
  }

  bool RunSchedule(bool useCrossNic, bool allPairsRanPoison) {
    std::unique_ptr<bool[]> votes = std::make_unique<bool[]>(comm_->nRanks);
    result_ = {7, 7};
    allPairsRan_ = allPairsRanPoison;
    return runSchedule(comm_.get(), info_.data(), votes.get(), useCrossNic, result_, allPairsRan_);
  }

  std::string Report(const RankBandwidth* bandwidth, bool cross, bool complete) {
    std::vector<double> sorted(comm_->nRanks);
    return CaptureStdout(
        [&] { reportBandwidthStats(comm_.get(), info_.data(), bandwidth, sorted.data(), cross, complete); });
  }

  double RunPair(int serverRank, int clientRank, bool cross, std::string* out) {
    double sample = -9;
    *out = CaptureStdout([&] { sample = runPair(comm_.get(), info_.data(), serverRank, clientRank, cross); });
    return sample;
  }

  std::string PairLine(const char* mode, int serverRank, int clientRank, const std::string& tail) {
    const LocalInfo& server = info_[serverRank];
    const LocalInfo& client = info_[clientRank];
    return NetInfo(std::string("mode=") + mode + " server_rank=" + std::to_string(serverRank) +
                   " server_host=" + server.hostname + " server_device=" + server.device +
                   " client_rank=" + std::to_string(clientRank) + " client_host=" + client.hostname +
                   " client_device=" + client.device + " " + tail);
  }

  bool FindPair(int phase, bool cross, int* serverRank, int* clientRank) {
    *serverRank = -7;
    *clientRank = -7;
    return findPair(comm_.get(), info_.data(), phase, cross, *serverRank, *clientRank);
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

  void Run() {
    ncclDiagRunIbWriteBw(comm_.get());
  }

  std::unique_ptr<ncclComm> comm_;
  std::vector<std::vector<int>> nodes_;
  std::vector<ncclPeerInfo> peers_;
  std::vector<ncclNodeRanks> nodeRanks_;
  std::vector<LocalInfo> info_;
  std::vector<RankBandwidth> bw_;
  int topoStorage_ = 0;
  ncclNet_t net_{};
  std::size_t hostnameLen_ = 0;
  int propsDev_ = -1;
  struct DiscoverScene {  // Discovery knobs and observations; reset only by `scene_ = DiscoverScene()`.
    const char* propName = "mlx5_0";
    ncclResult_t devicesResult = ncclSuccess;
    ncclResult_t propsResult = ncclSuccess;
    int localNetDev = kNetDev;
    ncclResult_t localNetResult = ncclSuccess;
    int probeExit = 0;
    std::vector<int> localNetChannels;
  } scene_;
  std::vector<std::string> sysfs_ = {SysPath(""), SysPath("mlx5_0"), SysPath("mlx5_1")};
  std::vector<std::string> accessed_;
  std::string help_ = kHelp;
  std::string clientOutput_ = kClientOutput;
  int clientExit_ = 0;
  std::string serverOutput_ = kServerOutput;
  int serverExit_ = 0;
  std::vector<std::string> clientCommands_;
  std::vector<std::string> serverCommands_;
  std::vector<IbMsg> log_;
  std::map<int, std::deque<std::string>> inbox_;
  int sendFailTag_ = 0;
  // Allgathers per Run(), in order: 1 rankInfo, 2-3 the per-phase votes, 4 the closing vote, 5 the summary.
  int gathers_ = 0;
  int gatherFailAt_ = 0;
  int peerVetoRank_ = -1;
  LocalInfo selfInfo_{};
  std::vector<bool> myVotes_;
  RankBandwidth selfBw_{-9, -9};
  std::map<std::string, int64_t> params_;
  RankBandwidth result_{7, 7};
  bool allPairsRan_ = false;
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
  PlaceRank(6);
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
  PlaceRank(3);
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 3);
  EXPECT_EQ(client, 5);
  PlaceRank(4);
  ASSERT_TRUE(FindPair(0, true, &server, &client));
  EXPECT_EQ(server, 0);
  EXPECT_EQ(client, 4);
  PlaceRank(5);
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
  PlaceRank(3);
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
    int pairedRanks[2][2];  // [phase][cross]
  };
  const std::vector<Topology> topologies = {
      {{{0, 1, 2}, {3, 4, 5}}, {"A", "B", "C", "C", "A", "B"}, {{6, 6}, {6, 6}}},
      {{{0, 1, 2, 3}, {4, 5, 6, 7}}, {"A", "A", "B", "C", "C", "B", "A", "A"}, {{8, 6}, {8, 6}}},
      {{{0, 1}, {2, 3}, {4, 5}}, {"A", "B", "B", "A", "A", "C"}, {{4, 4}, {2, 2}}},
      {{{3, 0}, {1, 4}, {2, 5}, {6, 7}}, {"A", "B", "A", "B", "A", "B", "B", "A"}, {{8, 8}, {8, 8}}},
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
        EXPECT_EQ(pairs, topologies[t].pairedRanks[phase][cross]);
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
  PlaceRank(0);
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
  PlaceRank(1);
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // second A on the server, one A on the client
  PlaceRank(3);
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // B is not on the server node
  SetDevices({"C", "C", "A", "A"});
  PlaceRank(0);
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // C is not on the client node
  SetDevices({"C", "A", "A", "A"});
  PlaceRank(3);
  EXPECT_FALSE(FindPair(0, false, &server, &client));  // the second A on the client outnumbers the server's
  EXPECT_FALSE(FindPair(1, false, &server, &client));
  PlaceRank(2);
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
  EXPECT_TRUE(resolveDeviceName("mlx5_1+mlx5_0", 6, out));
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
  scene_.probeExit = 3;
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
  scene_.propName = "mlx5_1_dma+mlx5_0+mlx5_3";
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
  EXPECT_EQ(scene_.localNetChannels, (std::vector<int>{0}));
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
  scene_.probeExit = 1;
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
      {[this] { scene_.probeExit = 127; }, "required external tool missing", false},
      {[this] { scene_.probeExit = 124; }, "capability probe timed out", false},
      {[this] { comm_->ncclNet = nullptr; }, "failed: selected network plugin does not support IB", false},
      {[this] { net_.name = "Socket"; }, "failed: selected network plugin does not support IB", false},
      {[this] { scene_.devicesResult = ncclSystemError; }, "cannot enumerate network devices", false},
      {[] {
         g_ibNetDevices = [](int* count) {
           *count = 0;
           return ncclSuccess;
         };
       },
       "failed: no usable IB device", false},
      {[this] { comm_->topo = nullptr; }, "failed: no usable IB device", false},
      {[this] { comm_->nChannels = 0; }, "invalid channel count=0", false},
      {[this] { scene_.localNetResult = ncclInternalError; }, "cannot select a local network device", true},
      {[this] { scene_.localNetDev = -1; }, "failed: no usable IB device", true},
      {[this] { scene_.localNetDev = kNetDevices; }, "failed: no usable IB device", true},
      {[this] { scene_.propsResult = ncclSystemError; }, "cannot query the selected network device", true},
      {[this] { scene_.propName = nullptr; }, "failed: cannot resolve the selected IB device", true},
      {[this] { scene_.propName = "+mlx5_0"; }, "failed: cannot resolve the selected IB device", true},
      {[this] { scene_.propName = "mlx5_7"; }, "failed: cannot resolve the selected IB device", true},
  };
  for (std::size_t i = 0; i < cases.size(); i++) {
    SCOPED_TRACE(i);
    InstallHooks();
    BuildComm(0, {{0}, {1}});
    scene_ = DiscoverScene();
    cases[i].arm();
    LocalInfo local{};
    const std::string out = CaptureStdout([&] { local = Discover(); });
    EXPECT_EQ(out, NetInfo(cases[i].line));
    EXPECT_TRUE(local.setupFailed);
    EXPECT_EQ(local.deviceCount, 0);
    EXPECT_EQ(scene_.localNetChannels.size(), cases[i].reached ? 1u : 0u);
  }
}

TEST_F(DiagIbWriteBwMicrotest, BuildCommand_ServerOmitsHostClientAppendsItAndTruncationFails) {
  const LocalInfo local = Info("me", "mlx5_1");
  const LocalInfo server = Info("srv", "mlx5_0");
  char out[IB_BW_COMMAND_BYTES];
  const std::string fits = "ib_write_bw -d mlx5_1 -i 1 -s 65536 --report_gbits -q 4 -p 12345 -n 1000";
  ASSERT_TRUE(buildCommand(out, sizeof(out), true, local, server, false, true, 5, 12345, 4));
  EXPECT_EQ(out, fits);
  ASSERT_TRUE(buildCommand(out, sizeof(out), false, local, server, true, false, 5, 12345, 4));
  EXPECT_EQ(out, fits + " --use_cuda=5 srv");
  ASSERT_TRUE(buildCommand(out, sizeof(out), true, local, server, true, true, 5, 12345, 4));
  EXPECT_EQ(out, fits + " --use_cuda=5 --use_cuda_dmabuf");
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

TEST_F(DiagIbWriteBwMicrotest, ComputeBandwidthStats_SkipsUnmeasuredAndTakesMedianOfSorted) {
  const RankBandwidth bandwidth[5] = {{40, 7}, {-1, 1}, {10, -1}, {30, 5}, {20, 3}};
  double sorted[5] = {};
  double minimum = -1, median = -1, maximum = -1;
  EXPECT_EQ(computeBandwidthStats(bandwidth, 5, false, sorted, minimum, median, maximum), 4);
  EXPECT_DOUBLE_EQ(minimum, 10);
  EXPECT_DOUBLE_EQ(median, 25);
  EXPECT_DOUBLE_EQ(maximum, 40);
  EXPECT_EQ(computeBandwidthStats(bandwidth, 5, true, sorted, minimum, median, maximum), 4);
  EXPECT_DOUBLE_EQ(minimum, 1);
  EXPECT_DOUBLE_EQ(median, 4);
  EXPECT_DOUBLE_EQ(maximum, 7);
  EXPECT_EQ(computeBandwidthStats(bandwidth, 4, true, sorted, minimum, median, maximum), 3);
  EXPECT_DOUBLE_EQ(minimum, 1);
  EXPECT_DOUBLE_EQ(median, 5);
  EXPECT_DOUBLE_EQ(maximum, 7);
  minimum = median = maximum = -1;
  EXPECT_EQ(computeBandwidthStats(bandwidth + 1, 2, false, sorted, minimum, median, maximum), 1);
  EXPECT_DOUBLE_EQ(minimum, 10);
  EXPECT_DOUBLE_EQ(median, 10);
  EXPECT_DOUBLE_EQ(maximum, 10);
  minimum = median = maximum = -1;
  EXPECT_EQ(computeBandwidthStats(bandwidth + 1, 1, false, sorted, minimum, median, maximum), 0);
  EXPECT_DOUBLE_EQ(minimum, -1);
  const RankBandwidth zero = {0, -1};
  EXPECT_EQ(computeBandwidthStats(&zero, 1, false, sorted, minimum, median, maximum), 1);
  EXPECT_DOUBLE_EQ(maximum, 0);
}

TEST_F(DiagIbWriteBwMicrotest, BandwidthOutlier_StrictlyBeyondThirtyPercent) {
  EXPECT_FALSE(bandwidthOutlier(70, 100));
  EXPECT_FALSE(bandwidthOutlier(130, 100));
  EXPECT_TRUE(bandwidthOutlier(69.9, 100));
  EXPECT_TRUE(bandwidthOutlier(130.1, 100));
}

TEST_F(DiagIbWriteBwMicrotest, ServerListenLine_SendsReadyOnceOnListenBanner) {
  BuildComm(0, {{0}, {1}});
  std::vector<std::string> sends;
  g_devrBootstrapSend = [&](void* bs, int peer, int tag, void* data, int size) {
    EXPECT_EQ(std::make_pair(bs, tag), std::make_pair(comm_->bootstrap, IB_BW_SERVER_READY_TAG));
    sends.push_back(std::to_string(peer) + ":" + std::string(static_cast<char*>(data), size));
    return ncclRemoteError;
  };
  ServerReadyCtx ctx = {comm_.get(), 1, false};
  serverListenLine("Waiting for server\n", &ctx);
  EXPECT_TRUE(sends.empty());
  serverListenLine("* Waiting for client to connect... *\n", &ctx);
  serverListenLine("* Waiting for client to connect... *\n", &ctx);
  EXPECT_EQ(sends, std::vector<std::string>{"1:\1"});
}

TEST_F(DiagIbWriteBwMicrotest, CommUsesCrossNic_NeedsCrossNicGraphWithChannels) {
  BuildComm(0, {{0}, {1}});
  EXPECT_FALSE(commUsesCrossNic(comm_.get()));
  comm_->graphs[NCCL_NUM_ALGORITHMS - 1].crossNic = 1;
  EXPECT_FALSE(commUsesCrossNic(comm_.get()));
  comm_->graphs[NCCL_NUM_ALGORITHMS - 1].nChannels = 1;
  EXPECT_TRUE(commUsesCrossNic(comm_.get()));
  comm_->graphs[NCCL_NUM_ALGORITHMS - 1].crossNic = 0;
  comm_->graphs[0].nChannels = 1;
  EXPECT_FALSE(commUsesCrossNic(comm_.get()));
  comm_->graphs[0].crossNic = 1;
  EXPECT_TRUE(commUsesCrossNic(comm_.get()));
}

TEST_F(DiagIbWriteBwMicrotest, ReportBandwidthStats_SummaryOnRankZeroWithCappedOutlierLines) {
  BuildComm(0, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}});
  SetDevices(std::vector<const char*>(20, "A"));
  std::vector<RankBandwidth> bandwidth(20, RankBandwidth{100, 100});
  const double outliers[] = {10, 20, 30, 40, 50, 60, 69, 131, 200, 300};
  for (int i = 0; i < 10; i++) {
    bandwidth[i + 4].direct = outliers[i];
  }
  bandwidth[3].direct = -1;
  std::string shownLines;
  const char* const shown[] = {"10.0", "20.0", "30.0", "40.0", "50.0", "60.0", "69.0", "131.0"};
  for (int i = 0; i < 8; i++) {
    shownLines += NetInfo("same-nic rank " + std::to_string(i + 4) + " (host" + std::to_string(i + 4) + "): " +
                          shown[i] + " Gbit/s, >30% off median 100.0 Gbit/s");
  }
  EXPECT_EQ(Report(bandwidth.data(), false, true),
            NetInfo("10.0/100.0/300.0 Gbit/s min/median/max same-nic bw (across 19 ranks)") + shownLines +
                NetInfo("same-nic: 2 more ranks >30% off median"));
  bandwidth[12].direct = 100;
  bandwidth[13].direct = 100;
  EXPECT_EQ(Report(bandwidth.data(), false, true),
            NetInfo("10.0/100.0/131.0 Gbit/s min/median/max same-nic bw (across 19 ranks)") + shownLines);
  PlaceRank(1);
  EXPECT_EQ(Report(bandwidth.data(), false, true), "");
}

TEST_F(DiagIbWriteBwMicrotest, ReportBandwidthStats_OkOnlyWhenCompleteAndInBand) {
  BuildComm(0, {{0, 1}, {2, 3}});
  SetDevices({"A", "A", "A", "A"});
  const RankBandwidth bandwidth[4] = {{90, 75}, {100, 100}, {110, 120}, {-1, 100}};
  EXPECT_EQ(Report(bandwidth, false, true),
            DiagLine("NCCL DIAG [OK] net bw: 90.0/100.0/110.0 Gbit/s min/median/max same-nic bw (across 3 ranks) in "
                     "comm 0x1234abcd"));
  EXPECT_EQ(Report(bandwidth, false, false),
            NetInfo("90.0/100.0/110.0 Gbit/s min/median/max same-nic bw (across 3 ranks)"));
  EXPECT_EQ(Report(bandwidth, true, true),
            DiagLine("NCCL DIAG [OK] net bw: 75.0/100.0/120.0 Gbit/s min/median/max cross-nic bw (across 4 ranks) in "
                     "comm 0x1234abcd"));
  const RankBandwidth high[3] = {{100, -1}, {100, -1}, {150, -1}};
  const RankBandwidth low[3] = {{100, -1}, {60, -1}, {100, -1}};
  const RankBandwidth crossLow[3] = {{100, 100}, {40, 100}, {100, 40}};
  comm_->nRanks = 3;
  EXPECT_EQ(Report(high, false, true),
            NetInfo("100.0/100.0/150.0 Gbit/s min/median/max same-nic bw (across 3 ranks)") +
                NetInfo("same-nic rank 2 (host2): 150.0 Gbit/s, >30% off median 100.0 Gbit/s"));
  EXPECT_EQ(Report(low, false, true),
            NetInfo("60.0/100.0/100.0 Gbit/s min/median/max same-nic bw (across 3 ranks)") +
                NetInfo("same-nic rank 1 (host1): 60.0 Gbit/s, >30% off median 100.0 Gbit/s"));
  EXPECT_EQ(Report(high, true, true), "");
  EXPECT_EQ(Report(crossLow, true, true),
            NetInfo("40.0/100.0/100.0 Gbit/s min/median/max cross-nic bw (across 3 ranks)") +
                NetInfo("cross-nic rank 2 (host2): 40.0 Gbit/s, >30% off median 100.0 Gbit/s"));
}

TEST_F(DiagIbWriteBwMicrotest, AllRanksOk_TrueOnlyWhenEveryRankVotesTrue) {
  BuildComm(2, {{0, 1}, {2, 3}});
  bool votes[4] = {};
  bool transportOk = false;
  EXPECT_TRUE(allRanksOk(comm_.get(), votes, true, transportOk));
  EXPECT_TRUE(transportOk);
  for (int veto : {0, 3}) {
    peerVetoRank_ = veto;
    transportOk = false;
    EXPECT_FALSE(allRanksOk(comm_.get(), votes, true, transportOk)) << veto;
    EXPECT_TRUE(transportOk);
  }
  peerVetoRank_ = -1;
  EXPECT_FALSE(allRanksOk(comm_.get(), votes, false, transportOk));
  EXPECT_TRUE(transportOk);
  EXPECT_EQ(myVotes_, (std::vector<bool>{true, true, true, false}));
  gatherFailAt_ = gathers_ + 1;
  EXPECT_FALSE(allRanksOk(comm_.get(), votes, true, transportOk));
  EXPECT_FALSE(transportOk);
}

TEST_F(DiagIbWriteBwMicrotest, SyncPair_ClientSendsThenAdoptsEchoServerReceivesThenEchoes) {
  BuildComm(2, {{0, 1}, {2, 3}});
  double bandwidth = 7.25;
  inbox_[kSync] = {Bytes(42.5)};
  EXPECT_TRUE(syncPair(comm_.get(), false, 1, bandwidth));
  EXPECT_DOUBLE_EQ(bandwidth, 42.5);
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'S', 1, kSync, Bytes(7.25)}, {'R', 1, kSync, Bytes(42.5)}}));
  log_.clear();
  bandwidth = 7.25;
  inbox_[kSync] = {Bytes(42.5)};
  EXPECT_TRUE(syncPair(comm_.get(), true, 3, bandwidth));
  EXPECT_DOUBLE_EQ(bandwidth, 42.5);
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'R', 3, kSync, Bytes(42.5)}, {'S', 3, kSync, Bytes(42.5)}}));
  log_.clear();
  EXPECT_FALSE(syncPair(comm_.get(), true, 3, bandwidth));
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'R', 3, kSync, ""}}));
  log_.clear();
  inbox_[kSync] = {Bytes(1.0)};
  sendFailTag_ = kSync;
  EXPECT_FALSE(syncPair(comm_.get(), false, 1, bandwidth));
  ASSERT_EQ(log_.size(), 1u);
  EXPECT_FALSE(syncPair(comm_.get(), true, 1, bandwidth));
  sendFailTag_ = 0;
  EXPECT_FALSE(syncPair(comm_.get(), false, 1, bandwidth));
  EXPECT_EQ(log_.back(), (IbMsg{'R', 1, kSync, ""}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_ServerAnnouncesThenReturnsClientMeasurement) {
  BuildTwoNodePair(0);
  params_["IB_QPS_PER_CONNECTION"] = 3;
  info_[2].dmabuf = false;
  inbox_[kSync] = {Bytes(88.5)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, false, &out), 88.5);
  EXPECT_EQ(out, "");
  EXPECT_EQ(serverCommands_, (std::vector<std::string>{Command("mlx5_0", Port(0), " --use_cuda=3", nullptr, 1, 3)}));
  EXPECT_TRUE(clientCommands_.empty());
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'S', 2, kReady, Bytes(true)}, {'R', 2, kSync, Bytes(88.5)},
                                      {'S', 2, kSync, Bytes(88.5)}}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_ServerWithoutBannerSendsNotReady) {
  BuildTwoNodePair(0);
  serverOutput_ = "listening on port\n";
  inbox_[kSync] = {Bytes(-1.0)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, true, &out), -1);
  EXPECT_EQ(out, "");
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'S', 2, kReady, Bytes(false)}, {'R', 2, kSync, Bytes(-1.0)},
                                      {'S', 2, kSync, Bytes(-1.0)}}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_ServerToolFailureReportsAfterReady) {
  BuildTwoNodePair(0);
  serverExit_ = 124;
  inbox_[kSync] = {Bytes(12.0)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, true, &out), 12.0);
  EXPECT_EQ(out, PairLine("cross", 0, 2, "tool run timed out"));
  ASSERT_FALSE(log_.empty());
  EXPECT_EQ(log_.front(), (IbMsg{'S', 2, kReady, Bytes(true)}));
  EXPECT_EQ(log_.size(), 3u);
  ResetPairScene(1);
  serverExit_ = 1;
  serverOutput_ = "";
  inbox_[kSync] = {Bytes(-1.0)};
  EXPECT_DOUBLE_EQ(RunPair(1, 2, false, &out), -1);
  EXPECT_EQ(out, PairLine("same", 1, 2, "tool run failed"));
  ASSERT_FALSE(log_.empty());
  EXPECT_EQ(log_.front(), (IbMsg{'S', 2, kReady, Bytes(false)}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_ClientMeasuresAfterReadyAndAdoptsEcho) {
  BuildTwoNodePair(2);
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(55.5)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, false, &out), 55.5);
  EXPECT_EQ(out, "");
  EXPECT_EQ(clientCommands_, (std::vector<std::string>{Command("mlx5_0", Port(0), kCudaFlags, kSelfHost)}));
  EXPECT_TRUE(serverCommands_.empty());
  EXPECT_EQ(log_, (std::vector<IbMsg>{{'R', 0, kReady, Bytes(true)}, {'S', 0, kSync, Bytes(kAverage)},
                                      {'R', 0, kSync, Bytes(55.5)}}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_CrossClientRunsItsOwnDeviceAndPortAgainstServerHost) {
  BuildTwoNodePair(3);
  info_[3].port = 2;
  inbox_[kReady] = {Bytes(true), Bytes(true)};
  inbox_[kSync] = {Bytes(61.0), Bytes(-1.0)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 3, true, &out), 61.0);
  EXPECT_EQ(out, "");
  EXPECT_EQ(clientCommands_, (std::vector<std::string>{Command("mlx5_1", Port(0), kCudaFlags, kSelfHost, 2)}));
  clientExit_ = 1;
  EXPECT_DOUBLE_EQ(RunPair(0, 3, true, &out), -1);
  EXPECT_EQ(out, PairLine("cross", 0, 3, "memory=cuda+dmabuf tool run failed"));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_ClientFailuresReportWithMemoryKind) {
  struct Case {
    std::function<void()> arm;
    std::string line;
    bool reached;
  };
  const std::vector<Case> cases = {
      {[] {}, "memory=cuda+dmabuf cannot receive the server status", false},
      {[this] { inbox_[kReady] = {Bytes(false)}; }, "memory=cuda+dmabuf server did not start", false},
      {[this] {
         inbox_[kReady] = {Bytes(true)};
         clientExit_ = 127;
       },
       "memory=cuda+dmabuf tool missing", true},
      {[this] {
         inbox_[kReady] = {Bytes(true)};
         clientExit_ = 127;
         clientOutput_ = "no rows\n";
       },
       "memory=cuda+dmabuf tool missing", true},
      {[this] {
         inbox_[kReady] = {Bytes(true)};
         clientOutput_ = "no rows\n";
         info_[0].dmabuf = false;
       },
       "memory=cuda no bandwidth data", true},
      {[this] {
         inbox_[kReady] = {Bytes(false)};
         info_[2].cuda = false;
         std::memset(info_[0].hostname, 'h', IB_BW_HOSTNAME_SIZE - 1);
       },
       "memory=host cannot build command", false},
      {[this] {
         info_[0].cuda = false;
         std::memset(info_[0].hostname, 'h', IB_BW_HOSTNAME_SIZE - 1);
       },
       "memory=host cannot build command", false},
  };
  for (std::size_t i = 0; i < cases.size(); i++) {
    SCOPED_TRACE(i);
    ResetPairScene(2);
    cases[i].arm();
    inbox_[kSync] = {Bytes(-1.0)};
    std::string out;
    EXPECT_DOUBLE_EQ(RunPair(0, 2, false, &out), -1);
    EXPECT_EQ(clientCommands_.size(), cases[i].reached ? 1u : 0u);
    ASSERT_EQ(log_.size(), 3u);
    EXPECT_EQ(log_.front().kind, 'R');
    EXPECT_EQ(log_.front().tag, kReady);
    EXPECT_EQ(log_[1], (IbMsg{'S', 0, kSync, Bytes(-1.0)}));
    EXPECT_EQ(log_.back(), (IbMsg{'R', 0, kSync, Bytes(-1.0)}));
    EXPECT_EQ(out, PairLine("same", 0, 2, cases[i].line));
  }
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_FailedExchangeReportsPeer) {
  BuildTwoNodePair(2);
  inbox_[kReady] = {Bytes(true)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, false, &out), -1);
  EXPECT_EQ(out, NetInfo("cannot exchange the measurement with rank 0"));
  ASSERT_FALSE(log_.empty());
  EXPECT_EQ(log_.back(), (IbMsg{'R', 0, kSync, ""}));
}

TEST_F(DiagIbWriteBwMicrotest, RunPair_InvalidQpsReturnsBeforeAnyPeerExchange) {
  BuildTwoNodePair(2);
  params_["IB_QPS_PER_CONNECTION"] = 0;
  inbox_[kReady] = {Bytes(true)};
  std::string out;
  EXPECT_DOUBLE_EQ(RunPair(0, 2, false, &out), -1);
  EXPECT_EQ(out, NetInfo("invalid qps parameter=0"));
  // A peer whose own qps is valid then blocks on the skipped status byte and pair sync; a fix flips this pin.
  EXPECT_TRUE(log_.empty());
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_ServesThenMeasuresAndAveragesBothPhases) {
  BuildTwoNodePair(0);
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(90.0), Bytes(97.25)};
  EXPECT_TRUE(RunSchedule(false, false));
  EXPECT_DOUBLE_EQ(result_.direct, (90.0 + 97.25) / 2);
  EXPECT_DOUBLE_EQ(result_.cross, -1);
  EXPECT_TRUE(allPairsRan_);
  EXPECT_EQ(myVotes_, (std::vector<bool>{true, true, true}));
  EXPECT_EQ(serverCommands_.size(), 1u);
  EXPECT_EQ(clientCommands_, (std::vector<std::string>{Command("mlx5_0", Port(2), kCudaFlags, "nodeB0")}));
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_CrossPhaseFeedsCrossAndFailedPairClearsComplete) {
  BuildTwoNodePair(0);
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(0.0), Bytes(-1.0)};
  EXPECT_TRUE(RunSchedule(true, true));
  EXPECT_DOUBLE_EQ(result_.cross, 0.0);
  EXPECT_DOUBLE_EQ(result_.direct, -1);
  EXPECT_FALSE(allPairsRan_);
  EXPECT_EQ(myVotes_, (std::vector<bool>{true, true, false}));
  ASSERT_FALSE(log_.empty());
  EXPECT_EQ(log_.front(), (IbMsg{'S', 3, kReady, Bytes(true)}));
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_DeadlineStopsBeforeNextPhase) {
  for (const auto& [param, seconds] : {std::pair<int64_t, uint64_t>{0, 5}, {1, 1}, {2, 2}}) {
    SCOPED_TRACE(param);
    ResetPairScene(0);
    params_["DIAGNOSTICS_IB_BW_TIMEOUT"] = param;
    inbox_[kSync] = {Bytes(90.0)};
    const uint64_t start = 1000;
    const uint64_t deadline = start + seconds * 1000000000ULL;
    std::deque<uint64_t> clock = {start, deadline - 1, deadline};
    ScopedHook ticks(g_ibClockNano, [&] {
      if (clock.empty()) {
        ADD_FAILURE() << "clock read past the script";
        return deadline;
      }
      const uint64_t now = clock.front();
      clock.pop_front();
      return now;
    });
    const std::string out = CaptureStdout([&] { EXPECT_TRUE(RunSchedule(false, true)); });
    EXPECT_EQ(out, NetInfo("test unable to complete in allocated time; ran 1 of 2 test phases"));
    EXPECT_EQ(ticks.calls, 3);
    EXPECT_DOUBLE_EQ(result_.direct, 90.0);
    EXPECT_FALSE(allPairsRan_);
    EXPECT_EQ(myVotes_, (std::vector<bool>{true, false, false}));
  }
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_PeerVetoStopsSilentlyOffRankZero) {
  BuildTwoNodePair(2);
  peerVetoRank_ = 0;
  EXPECT_EQ(CaptureStdout([&] { EXPECT_TRUE(RunSchedule(false, true)); }), "");
  EXPECT_DOUBLE_EQ(result_.direct, -1);
  EXPECT_FALSE(allPairsRan_);
  EXPECT_TRUE(log_.empty());
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_UnpairedRankStillVotesEachPhase) {
  BuildTwoNodePair(0);
  info_[0] = Info(kSelfHost, "mlx5_9");
  EXPECT_TRUE(RunSchedule(false, false));
  EXPECT_DOUBLE_EQ(result_.direct, -1);
  EXPECT_DOUBLE_EQ(result_.cross, -1);
  EXPECT_TRUE(allPairsRan_);
  EXPECT_EQ(gathers_, 3);
  EXPECT_TRUE(log_.empty());
}

TEST_F(DiagIbWriteBwMicrotest, RunSchedule_TransportFailureOnAnyVoteFails) {
  for (int failAt : {1, 3}) {
    SCOPED_TRACE(failAt);
    ResetPairScene(0);
    gatherFailAt_ = failAt;
    inbox_[kReady] = {Bytes(true)};
    inbox_[kSync] = {Bytes(90.0), Bytes(97.25)};
    EXPECT_FALSE(RunSchedule(false, true));
    EXPECT_EQ(gathers_, failAt);
  }
}

TEST_F(DiagIbWriteBwMicrotest, Run_ReportsCoverageGapsThenSummaryWithOutlier) {
  BuildTwoNodePair(0);
  info_[3] = Info("nodeB1", "mlx5_0");
  info_[1].deviceCount = 2;
  info_[3].deviceCount = 2;
  bw_[1].direct = 93.0;
  bw_[2].direct = 94.0;
  bw_[3].direct = 50.0;
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(90.0), Bytes(97.44)};
  const std::string out = CaptureStdout([&] { Run(); });
  EXPECT_EQ(out, NetInfo("multiple net devices per rank detected, only the first device is tested") +
                     NetInfo("net devices are shared across ranks, concurrent use can lower the bandwidth") +
                     NetInfo("50.0/93.4/94.0 Gbit/s min/median/max same-nic bw (across 4 ranks)") +
                     NetInfo("same-nic rank 3 (nodeB1): 50.0 Gbit/s, >30% off median 93.4 Gbit/s"));
  EXPECT_STREQ(selfInfo_.hostname, kSelfHost);
  EXPECT_STREQ(selfInfo_.device, "mlx5_0");
  EXPECT_FALSE(selfInfo_.setupFailed);
  EXPECT_DOUBLE_EQ(selfBw_.direct, (90.0 + 97.44) / 2);
  EXPECT_DOUBLE_EQ(selfBw_.cross, -1);
  EXPECT_EQ(serverCommands_, (std::vector<std::string>{Command("mlx5_0", Port(0), kCudaFlags, nullptr)}));
  EXPECT_EQ(clientCommands_, (std::vector<std::string>{Command("mlx5_0", Port(2), kCudaFlags, "nodeB0")}));
  EXPECT_EQ(gathers_, 5);
  EXPECT_EQ(myVotes_, (std::vector<bool>{true, true, true}));
}

TEST_F(DiagIbWriteBwMicrotest, Run_NonZeroRankMeasuresWithoutReporting) {
  BuildTwoNodePair(2);
  info_[3].deviceCount = 2;
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(97.44), Bytes(90.0)};
  EXPECT_EQ(CaptureStdout([&] { Run(); }), "");
  EXPECT_STREQ(selfInfo_.hostname, kSelfHost);
  EXPECT_DOUBLE_EQ(selfBw_.direct, (97.44 + 90.0) / 2);
  EXPECT_EQ(clientCommands_, (std::vector<std::string>{Command("mlx5_0", Port(0), kCudaFlags, kSelfHost)}));
  EXPECT_EQ(serverCommands_, (std::vector<std::string>{Command("mlx5_0", Port(2), kCudaFlags, nullptr)}));
  EXPECT_EQ(gathers_, 5);
}

TEST_F(DiagIbWriteBwMicrotest, Run_CrossNicAddsCrossSummaryAndIncompleteIsInfo) {
  BuildTwoNodePair(0);
  comm_->graphs[1].nChannels = 2;
  comm_->graphs[1].crossNic = 1;
  for (int r = 1; r < 4; r++) {
    bw_[r] = {100.0, 80.0};
  }
  inbox_[kReady] = {Bytes(true)};
  inbox_[kSync] = {Bytes(-1.0), Bytes(100.0)};
  const std::string out = CaptureStdout([&] { Run(); });
  EXPECT_EQ(out, NetInfo("100.0/100.0/100.0 Gbit/s min/median/max same-nic bw (across 4 ranks)") +
                     NetInfo("80.0/80.0/80.0 Gbit/s min/median/max cross-nic bw (across 3 ranks)"));
  ASSERT_FALSE(log_.empty());
  EXPECT_EQ(log_.front(), (IbMsg{'S', 3, kReady, Bytes(true)}));
  EXPECT_EQ(myVotes_, (std::vector<bool>{true, true, false}));
}

TEST_F(DiagIbWriteBwMicrotest, Run_SetupFailureOnAnyRankEndsBeforeSchedule) {
  BuildTwoNodePair(0);
  info_[1].setupFailed = true;
  info_[3].setupFailed = true;
  EXPECT_EQ(CaptureStdout([&] { Run(); }), NetInfo("setup failed on rank 1"));
  scene_.probeExit = 127;
  gathers_ = 0;  // drop the first Run()'s gather; the count below covers only the next two
  EXPECT_EQ(CaptureStdout([&] { Run(); }),
            NetInfo("required external tool missing") + NetInfo("setup failed on rank 0"));
  EXPECT_TRUE(selfInfo_.setupFailed);
  scene_.probeExit = 0;
  PlaceRank(2);
  EXPECT_EQ(CaptureStdout([&] { Run(); }), "");
  EXPECT_EQ(gathers_, 2);
  EXPECT_TRUE(serverCommands_.empty());
  EXPECT_TRUE(clientCommands_.empty());
}

TEST_F(DiagIbWriteBwMicrotest, Run_BootstrapFailuresAbortWithoutSummary) {
  for (int failAt : {1, 2, 5}) {
    SCOPED_TRACE(failAt);
    ResetPairScene(0);
    gatherFailAt_ = failAt;
    inbox_[kReady] = {Bytes(true)};
    inbox_[kSync] = {Bytes(90.0), Bytes(97.44)};
    const std::string out = CaptureStdout([&] { Run(); });
    EXPECT_EQ(out, NetInfo(failAt == 1 ? "cannot share local configuration" : "bootstrap failure, aborting check"));
    EXPECT_EQ(gathers_, failAt);
  }
}

TEST_F(DiagIbWriteBwMicrotest, Run_AllocationFailureJoinsNoCollective) {
  for (int failAt = 1; failAt <= 4; failAt++) {
    SCOPED_TRACE(failAt);
    ResetPairScene(0);
    g_ibCallocFailAt = failAt;
    EXPECT_EQ(CaptureStdout([&] { Run(); }), NetInfo("cannot allocate diagnostic state"));
    EXPECT_EQ(gathers_, 0);
  }
}

TEST_F(DiagIbWriteBwMicrotest, Run_IncompleteCommIsANoOp) {
  BuildTwoNodePair(0);
  ncclDiagRunIbWriteBw(nullptr);
  comm_->nRanks = 0;
  Run();
  comm_->nRanks = 4;
  comm_->nodeRanks = nullptr;
  Run();
  comm_->nodeRanks = nodeRanks_.data();
  comm_->peerInfo = nullptr;
  EXPECT_EQ(CaptureStdout([&] { Run(); }), "");
  EXPECT_EQ(g_ibCallocCalls, 0);
}

}  // namespace
