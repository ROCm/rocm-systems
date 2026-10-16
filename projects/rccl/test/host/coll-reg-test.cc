/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/register/coll_reg.cc, which picks the buffer
// registration (NVLS, IPC, CollNet, NET) for a collective task. This TU
// #includes the hipified coll_reg.cc (COLL_REG_CC_PATH) so its file-static
// helpers are reachable. No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <set>
#include <vector>

#include "RegisterScene.h"
#include "fakes/nccl_fakes.h"
#include "fakes/transport_p2p_reg_fakes.h"
#include "fakes/transport_reg_stubs.h"
#include "fakes/transport_stubs.h"

// Routes NCCL_PARAM(MloPartRdmaEnable) through g_loadParam on each call.
#include "fakes/param_redirect.h"

#include "fakes/transport_p2p_reg_seam.h"
#include COLL_REG_CC_PATH
#include "fakes/transport_p2p_reg_seam_undef.h"

namespace {

constexpr int kRanks = 8;
constexpr size_t kCount = 1024;  // ncclFloat32 elements
constexpr size_t kBytes = kCount * 4;
const void* const kSend = Ptr(0x100000);
void* const kRecv = Ptr(0x200000);

// One recorded call into a registration seam.
struct RegCall {
  const void* buff;
  size_t size;
  int type;                            // collNetSend / collNetRecv, or the ncclIpcRegType
  std::vector<int> peers;              // IPC
  std::vector<ncclConnector*> conns;   // NET
  bool operator==(const RegCall& o) const {
    return buff == o.buff && size == o.size && type == o.type && peers == o.peers && conns == o.conns;
  }
};

// ncclTransports[0]: canConnect answers from `connectable` and records the peer rank asked about.
struct FakeP2pTransport {
  std::set<int> connectable;
  ncclResult_t result = ncclSuccess;
  std::vector<int> asked;
  ncclTopoGraph* lastGraph = nullptr;
  ncclPeerInfo* lastMine = nullptr;

  ncclResult_t canConnect(int* ret, ncclComm* comm, ncclTopoGraph* graph, ncclPeerInfo* mine, ncclPeerInfo* theirs) {
    int peer = static_cast<int>(theirs - comm->peerInfo);
    asked.push_back(peer);
    lastGraph = graph;
    lastMine = mine;
    *ret = connectable.count(peer) ? 1 : 0;
    return result;
  }
};
FakeP2pTransport* g_activeP2p = nullptr;
ncclResult_t TrampCanConnect(int* ret, ncclComm* comm, ncclTopoGraph* graph, ncclPeerInfo* mine,
                             ncclPeerInfo* theirs) {
  return g_activeP2p->canConnect(ret, comm, graph, mine, theirs);
}
ncclTransport g_fakeP2pTransport = {"FakeP2P", TrampCanConnect, {}, {}};

class RegisterCollMicrotestBase : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  ncclTaskColl task_{};
  ncclRegCleanupQueue cleanupQueue_{};
  void* outSend_[NCCL_MAX_LOCAL_RANKS] = {};
  void* outRecv_[NCCL_MAX_LOCAL_RANKS] = {};
  bool needConnect_ = false;
  FakeP2pTransport p2p_;

  std::vector<ncclPeerInfo> peerInfo_ = std::vector<ncclPeerInfo>(kRanks);
  std::deque<ncclChannelPeer> peerStore_;
  std::vector<std::vector<ncclChannelPeer*>> peerTables_;
  std::deque<ncclReg> regs_;
  std::vector<ncclReg*> regSlots_;
  uintptr_t ipcAddrs_[kRanks] = {};

  std::vector<RegCall> ipcLocal_, ipcGraph_, netLocal_, netGraph_, collnetLocal_, collnetGraph_;

  void SetUp() override {
    comm_->rank = 0;
    comm_->nRanks = kRanks;
    comm_->nNodes = 1;
    comm_->config.minCTAs = 1;
    comm_->config.maxCTAs = 64;
    comm_->peerInfo = peerInfo_.data();
    task_.func = ncclFuncAllReduce;
    task_.sendbuff = kSend;
    task_.recvbuff = kRecv;
    task_.count = kCount;
    task_.datatype = ncclFloat32;
    task_.opDev.op = ncclDevSum;
    task_.protocol = NCCL_PROTO_SIMPLE;
    task_.algorithm = NCCL_ALGO_RING;
    task_.nMaxChannels = 32;
    task_.regBufType = -1;
    // Graph registration on, as on ROCm >= 7.12, but no capture: a graph seam reached outside one fails loudly.
    g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"GRAPH_REGISTER", 1}});
    g_activeP2p = &p2p_;
    ncclTransports[0] = &g_fakeP2pTransport;
  }

  void TearDown() override {
    free(task_.sendNetHandles);
    free(task_.recvNetHandles);
    free(task_.srecvNetHandles);
    g_activeP2p = nullptr;
    ResetNcclFakes();
    ResetTransportP2pRegFakes();
    ResetTransportStubs();
  }

  ncclResult_t Run() {
    return ncclRegisterCollBuffers(comm_.get(), &task_, outSend_, outRecv_, &cleanupQueue_, &needConnect_);
  }

  // HIP graph capture with graph registration enabled.
  void UseGraphCapture(int localRegister = 1) {
    comm_->planner.persistent = true;
    g_loadParam = ParamOverride({{"LOCAL_REGISTER", localRegister}, {"GRAPH_REGISTER", 1}});
  }

  // nChannels channels, each with a peer table indexed by rank (one past the end for the tree's nRanks sentinel).
  void SetChannels(int nChannels) {
    comm_->nChannels = nChannels;
    peerTables_.assign(nChannels, {});
    for (int c = 0; c < nChannels; ++c) {
      for (int r = 0; r <= kRanks; ++r) peerTables_[c].push_back(&peerStore_.emplace_back());
      comm_->channels[c].peers = peerTables_[c].data();
      comm_->channels[c].ring = {};
      comm_->channels[c].tree = {-1, -1, {-1, -1, -1}};
      comm_->channels[c].collnetChain = {-1, -1, {-1, -1, -1}};
      for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; ++i) {
        comm_->channels[c].collnetDirect.up[i] = -1;
        comm_->channels[c].collnetDirect.down[i] = -1;
      }
    }
  }
  ncclConnector* RecvConn(int c, int peer) { return &comm_->channels[c].peers[peer]->recv[0]; }
  ncclConnector* SendConn(int c, int peer) { return &comm_->channels[c].peers[peer]->send[0]; }
  void ConnectP2p(ncclConnector* conn) {
    conn->connected = 1;
    conn->conn.flags |= NCCL_P2P_READ;
  }

  // Puts a cache record covering [buff, buff + size) where ncclRegFind looks.
  void AddRegRecord(const void* buff, size_t size) {
    ncclReg& reg = regs_.emplace_back();
    reg.begAddr = reinterpret_cast<uintptr_t>(buff);
    reg.endAddr = reg.begAddr + size;
    regSlots_.push_back(&reg);
    std::sort(regSlots_.begin(), regSlots_.end(), [](ncclReg* a, ncclReg* b) { return a->begAddr < b->begAddr; });
    comm_->regCache.slots = regSlots_.data();
    comm_->regCache.population = static_cast<int>(regSlots_.size());
  }

  // Registration seams: each records its call and reports success for the buffers (IPC, NET) or
  // collnet types listed.
  void IpcLocalRegisters(std::set<const void*> ok) {
    g_ncclIpcLocalRegisterBuffer = [this, ok](ncclComm*, const void* buff, size_t size, int* peers, int n,
                                              ncclIpcRegType type, int* flag, uintptr_t* offset, uintptr_t** addrs) {
      ipcLocal_.push_back({buff, size, type, std::vector<int>(peers, peers + n), {}});
      *flag = ok.count(buff) ? 1 : 0;
      *offset = 0x40;
      *addrs = ipcAddrs_;
      return ncclSuccess;
    };
  }
  void IpcGraphRegisters(std::set<const void*> ok) {
    g_ncclIpcGraphRegisterBuffer = [this, ok](ncclComm*, const void* buff, size_t size, int* peers, int n,
                                              ncclIpcRegType type, int* flag, uintptr_t* offset, uintptr_t** addrs,
                                              void* queue, int* nElts) {
      EXPECT_EQ(queue, &cleanupQueue_);
      EXPECT_EQ(nElts, &task_.nCleanupQueueElts);
      ipcGraph_.push_back({buff, size, type, std::vector<int>(peers, peers + n), {}});
      *flag = ok.count(buff) ? 1 : 0;
      *offset = 0x40;
      *addrs = ipcAddrs_;
      return ncclSuccess;
    };
  }
  // NET registration answers by call order, since the ring registers recvbuff twice.
  void NetLocalRegisters(std::vector<int> results) {
    g_ncclNetLocalRegisterBuffer = [this, results](ncclComm*, const void* buff, size_t size, ncclConnector** conns,
                                                   int n, int* flag, void** handle) {
      size_t i = netLocal_.size();
      netLocal_.push_back({buff, size, 0, {}, std::vector<ncclConnector*>(conns, conns + n)});
      *flag = i < results.size() ? results[i] : 0;
      *handle = Ptr(0xe0 + i);
      return ncclSuccess;
    };
  }
  void NetGraphRegisters(std::vector<int> results) {
    g_ncclNetGraphRegisterBuffer = [this, results](ncclComm*, const void* buff, size_t size, ncclConnector** conns,
                                                   int n, int* flag, void** handle, ncclRegCleanupQueue* queue,
                                                   int* nElts) {
      EXPECT_EQ(queue, &cleanupQueue_);
      EXPECT_EQ(nElts, &task_.nCleanupQueueElts);
      size_t i = netGraph_.size();
      netGraph_.push_back({buff, size, 0, {}, std::vector<ncclConnector*>(conns, conns + n)});
      *flag = i < results.size() ? results[i] : 0;
      *handle = Ptr(0xf0 + i);
      return ncclSuccess;
    };
  }
  void CollnetLocalRegisters(std::set<int> okTypes) {
    g_ncclCollnetLocalRegisterBuffer = [this, okTypes](ncclComm*, const void* buff, size_t size, int type, int* flag,
                                                       void** handle) {
      collnetLocal_.push_back({buff, size, type, {}, {}});
      *flag = okTypes.count(type) ? 1 : 0;
      *handle = Ptr(0xc0 + type);
      return ncclSuccess;
    };
  }
  void CollnetGraphRegisters(std::set<int> okTypes) {
    g_ncclCollnetGraphRegisterBuffer = [this, okTypes](ncclComm*, const void* buff, size_t size, int type, int* flag,
                                                       void** handle, ncclRegCleanupQueue* queue, int* nElts) {
      EXPECT_EQ(queue, &cleanupQueue_);
      EXPECT_EQ(nElts, &task_.nCleanupQueueElts);
      collnetGraph_.push_back({buff, size, type, {}, {}});
      *flag = okTypes.count(type) ? 1 : 0;
      *handle = Ptr(0xd0 + type);
      return ncclSuccess;
    };
  }

  void ExpectUnregistered() {
    EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
    EXPECT_TRUE(needConnect_);
  }
};

// ---------------------------------------------------------------------------
// isMloPartBufRdmaCapable
// ---------------------------------------------------------------------------

using RegisterMloPartMicrotest = RegisterCollMicrotestBase;

TEST_F(RegisterMloPartMicrotest, NoMloPart_IsRdmaCapableEvenWithoutBuffer) {
  bool capable = false;
  EXPECT_EQ(isMloPartBufRdmaCapable(comm_.get(), nullptr, &capable), ncclSuccess);
  EXPECT_TRUE(capable);
}

TEST_F(RegisterMloPartMicrotest, MloPartWithoutBuffer_IsNotRdmaCapable) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"MLOPART_RDMA_ENABLE", 1}});
  bool capable = true;
  EXPECT_EQ(isMloPartBufRdmaCapable(comm_.get(), nullptr, &capable), ncclSuccess);
  EXPECT_FALSE(capable);
}

TEST_F(RegisterMloPartMicrotest, MloPartBuffer_FollowsRdmaEnableParam) {
  comm_->hasMloPart = true;
  bool capable = true;
  g_loadParam = ParamOverride({{"MLOPART_RDMA_ENABLE", 0}});
  EXPECT_EQ(isMloPartBufRdmaCapable(comm_.get(), kRecv, &capable), ncclSuccess);
  EXPECT_FALSE(capable);
  g_loadParam = ParamOverride({{"MLOPART_RDMA_ENABLE", 1}});
  EXPECT_EQ(isMloPartBufRdmaCapable(comm_.get(), kRecv, &capable), ncclSuccess);
  EXPECT_TRUE(capable);
}

// ---------------------------------------------------------------------------
// ncclRegisterCollNvlsBuffers (its NVLS body is CUDA-only)
// ---------------------------------------------------------------------------

using RegisterCollNvlsBuffersMicrotest = RegisterCollMicrotestBase;

TEST_F(RegisterCollNvlsBuffersMicrotest, NvlsTask_LeavesBufferUnregistered) {
  task_.algorithm = NCCL_ALGO_NVLS;
  comm_->nvlsSupport = 1;
  comm_->nvlsRegSupport = 1;
  EXPECT_EQ(ncclRegisterCollNvlsBuffers(comm_.get(), &task_, outSend_, outRecv_, &cleanupQueue_, &needConnect_),
            ncclSuccess);
  ExpectUnregistered();
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: registration gates
// ---------------------------------------------------------------------------

// A CollNet Direct AllGather whose one unconnected peer makes registration probe a failing transport.
class RegisterCollGateMicrotest : public RegisterCollMicrotestBase {
 protected:
  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    SetChannels(1);
    task_.algorithm = NCCL_ALGO_COLLNET_DIRECT;
    task_.func = ncclFuncAllGather;
    comm_->isAllDirectP2p = true;
    comm_->channels[0].collnetDirect.up[0] = 1;
    p2p_.result = ncclInternalError;
  }
};

TEST_F(RegisterCollGateMicrotest, RegistrationEnabled_ProbesPeers) {
  EXPECT_EQ(Run(), ncclInternalError);
  EXPECT_EQ(p2p_.asked, (std::vector<int>{1}));
}

TEST_F(RegisterCollGateMicrotest, RegistrationDisabled_LeavesBufferUnregistered) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}, {"GRAPH_REGISTER", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
  EXPECT_TRUE(p2p_.asked.empty());
}

TEST_F(RegisterCollGateMicrotest, GraphRegisterOutsideCapture_LeavesBufferUnregistered) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}, {"GRAPH_REGISTER", 1}});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollGateMicrotest, CaptureWithoutGraphRegister_LeavesBufferUnregistered) {
  comm_->planner.persistent = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}, {"GRAPH_REGISTER", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollGateMicrotest, GraphRegisterInCapture_ProbesPeers) {
  UseGraphCapture(/*localRegister=*/0);
  EXPECT_EQ(Run(), ncclInternalError);
  EXPECT_EQ(p2p_.asked, (std::vector<int>{1}));
}

}  // namespace
