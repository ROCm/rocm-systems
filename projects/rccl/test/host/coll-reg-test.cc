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

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: NVLS
// ---------------------------------------------------------------------------

class RegisterCollNvlsMicrotest : public RegisterCollMicrotestBase {
 protected:
  int nvlsLocalCalls_ = 0;

  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    task_.algorithm = NCCL_ALGO_NVLS;
    comm_->nvlsSupport = 1;
    comm_->nvlsRegSupport = 1;
  }

  void NvlsLocalRegisters(int registered) {
    g_ncclNvlsLocalRegisterBuffer = [this, registered](ncclComm*, const void*, void*, size_t, size_t, int* reged,
                                                       void** outSend, void** outRecv) {
      ++nvlsLocalCalls_;
      EXPECT_EQ(outSend, outSend_);
      EXPECT_EQ(outRecv, outRecv_);
      *reged = registered;
      return ncclSuccess;
    };
  }
};

TEST_F(RegisterCollNvlsMicrotest, TransportDisabled_LeavesBufferUnregistered) {
  comm_->nvlsSupport = 0;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNvlsMicrotest, HostModeDisablesTransport_LeavesBufferUnregistered) {
  comm_->config.nvlsHostMode = ncclNvlsHostModeDisableTransport;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNvlsMicrotest, NoNvlsRegSupport_LeavesBufferUnregistered) {
  comm_->nvlsRegSupport = 0;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNvlsMicrotest, PreMulSum_LeavesBufferUnregistered) {
  task_.opDev.op = ncclDevPreMulSum;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNvlsMicrotest, LocalRegistered_MarksNvlsAndSkipsConnect) {
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
  EXPECT_FALSE(needConnect_);
}

TEST_F(RegisterCollNvlsMicrotest, NvlsTreeLocalRegistered_MarksNvls) {
  task_.algorithm = NCCL_ALGO_NVLS_TREE;
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsMicrotest, NotRegistered_LeavesBufferUnregisteredAndChannelsAlone) {
  NvlsLocalRegisters(0);
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollNvlsMicrotest, AllReduce_RegistersBothBuffersAtCountSize) {
  g_ncclNvlsLocalRegisterBuffer = [this](ncclComm*, const void* send, void* recv, size_t sendSize, size_t recvSize,
                                         int* reged, void**, void**) {
    EXPECT_EQ(send, kSend);
    EXPECT_EQ(recv, kRecv);
    EXPECT_EQ(sendSize, kBytes);
    EXPECT_EQ(recvSize, kBytes);
    ++nvlsLocalCalls_;
    *reged = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(nvlsLocalCalls_, 1);
}

TEST_F(RegisterCollNvlsMicrotest, AllGather_RegistersOnlyTheGatheredRecvBuffer) {
  task_.func = ncclFuncAllGather;
  g_ncclNvlsLocalRegisterBuffer = [this](ncclComm*, const void* send, void* recv, size_t sendSize, size_t recvSize,
                                         int* reged, void**, void**) {
    EXPECT_EQ(send, nullptr);
    EXPECT_EQ(recv, kRecv);
    EXPECT_EQ(sendSize, kBytes);
    EXPECT_EQ(recvSize, kRanks * kBytes);
    ++nvlsLocalCalls_;
    *reged = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(nvlsLocalCalls_, 1);
}

TEST_F(RegisterCollNvlsMicrotest, ReduceScatter_RegistersOnlyTheScatteredSendBuffer) {
  task_.func = ncclFuncReduceScatter;
  g_ncclNvlsLocalRegisterBuffer = [this](ncclComm*, const void* send, void* recv, size_t sendSize, size_t recvSize,
                                         int* reged, void**, void**) {
    EXPECT_EQ(send, kSend);
    EXPECT_EQ(recv, nullptr);
    EXPECT_EQ(sendSize, kRanks * kBytes);
    EXPECT_EQ(recvSize, kBytes);
    ++nvlsLocalCalls_;
    *reged = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(nvlsLocalCalls_, 1);
}

TEST_F(RegisterCollNvlsMicrotest, RegisteredSingleNode_CapsChannelsAtFour) {
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 4);
}

TEST_F(RegisterCollNvlsMicrotest, RegisteredSingleNodeReduceScatter_CapsChannelsAtFive) {
  task_.func = ncclFuncReduceScatter;
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 5);
}

TEST_F(RegisterCollNvlsMicrotest, RegisteredMultiNode_CapsChannelsAtSix) {
  task_.algorithm = NCCL_ALGO_NVLS_TREE;
  comm_->nNodes = 2;
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 6);
}

TEST_F(RegisterCollNvlsMicrotest, Registered_KeepsChannelsWithinCtaLimits) {
  NvlsLocalRegisters(1);
  comm_->config.maxCTAs = 2;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 2);
  comm_->config.maxCTAs = 64;
  comm_->config.minCTAs = 8;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 8);
}

TEST_F(RegisterCollNvlsMicrotest, LocalRegistered_SkipsGraphRegistration) {
  UseGraphCapture();
  NvlsLocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsMicrotest, LocalNotRegisteredInCapture_FallsBackToGraphRegistration) {
  UseGraphCapture();
  NvlsLocalRegisters(0);
  g_ncclNvlsGraphRegisterBuffer = [this](ncclComm*, const void*, void*, size_t, size_t, int* reged, void**, void**,
                                         ncclRegCleanupQueue* queue, int* nElts) {
    EXPECT_EQ(queue, &cleanupQueue_);
    EXPECT_EQ(nElts, &task_.nCleanupQueueElts);
    *reged = 1;
    return ncclSuccess;
  };
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(nvlsLocalCalls_, 1);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsMicrotest, CaptureWithoutLocalRegister_RegistersPerGraphOnly) {
  UseGraphCapture(/*localRegister=*/0);
  g_ncclNvlsGraphRegisterBuffer = [](ncclComm*, const void*, void*, size_t, size_t, int* reged, void**, void**,
                                     ncclRegCleanupQueue*, int*) {
    *reged = 1;
    return ncclSuccess;
  };
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

class RegisterCollNvlsCollnetMicrotest : public RegisterCollNvlsMicrotest {
 protected:
  void SetUp() override {
    RegisterCollNvlsMicrotest::SetUp();
    comm_->nNodes = 2;
    NvlsLocalRegisters(1);
  }
};

TEST_F(RegisterCollNvlsCollnetMicrotest, BothDirectionsRegistered_MarksNetWithHandles) {
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, collNetSend, {}, {}}, {kRecv, kBytes, collNetRecv, {}, {}}};
  EXPECT_EQ(collnetLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER | NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xc0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xc0 + collNetRecv));
}

TEST_F(RegisterCollNvlsCollnetMicrotest, SendSideNotRegistered_SkipsRecvSide) {
  CollnetLocalRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetLocal_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, RecvSideNotRegistered_LeavesNetUnmarked) {
  CollnetLocalRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetLocal_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, NvlsNotRegistered_StillMarksCollnet) {
  NvlsLocalRegisters(0);
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  EXPECT_TRUE(needConnect_);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, SingleNode_SkipsCollnet) {
  comm_->nNodes = 1;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, NvlsTree_SkipsCollnet) {
  task_.algorithm = NCCL_ALGO_NVLS_TREE;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, MloPartRecvBufferNotRdmaCapable_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, LocalNotRegisteredInCapture_FallsBackToGraphRegistration) {
  UseGraphCapture();
  CollnetLocalRegisters({});
  CollnetGraphRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, collNetSend, {}, {}}, {kRecv, kBytes, collNetRecv, {}, {}}};
  EXPECT_EQ(collnetGraph_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER | NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xd0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xd0 + collNetRecv));
}

TEST_F(RegisterCollNvlsCollnetMicrotest, LocalRegisteredInCapture_SkipsGraphRegistration) {
  UseGraphCapture();
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER | NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollNvlsCollnetMicrotest, GraphSendSideNotRegistered_SkipsGraphRecvSide) {
  UseGraphCapture(/*localRegister=*/0);
  g_ncclNvlsGraphRegisterBuffer = [](ncclComm*, const void*, void*, size_t, size_t, int* reged, void**, void**,
                                     ncclRegCleanupQueue*, int*) {
    *reged = 1;
    return ncclSuccess;
  };
  CollnetGraphRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetGraph_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_NVLS_REG_BUFFER);
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: IPC eligibility
// ---------------------------------------------------------------------------

// Ring, tree and chain scenes that register over IPC with P2P peers 1 and 2 unless a rule skips them.
class RegisterCollIpcGateMicrotest : public RegisterCollMicrotestBase {
 protected:
  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    SetChannels(1);
    comm_->isAllDirectP2p = true;
    ncclChannel& ch = comm_->channels[0];
    ch.ring.prev = 1;
    ch.ring.next = 2;
    ch.tree.up = 1;
    ch.collnetChain.up = 1;
    ConnectP2p(RecvConn(0, 1));
    ConnectP2p(SendConn(0, 2));
    AddRegRecord(kSend, kRanks * kBytes);
    AddRegRecord(kRecv, kRanks * kBytes);
    IpcLocalRegisters({kSend, kRecv});
  }
};

TEST_F(RegisterCollIpcGateMicrotest, RingOutOfPlaceAllReduce_RegistersOverIpc) {
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollIpcGateMicrotest, NonSimpleProtocol_LeavesBufferUnregistered) {
  task_.protocol = NCCL_PROTO_LL;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, RingReduceScatter_LeavesBufferUnregistered) {
  task_.func = ncclFuncReduceScatter;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, AllReduceWithBias_LeavesBufferUnregistered) {
  task_.acc = Ptr(0x300000);
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, BroadcastWithAccumulator_RegistersOverIpc) {
  task_.func = ncclFuncBroadcast;
  task_.acc = Ptr(0x300000);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollIpcGateMicrotest, RingInPlaceAllReduce_LeavesBufferUnregistered) {
  task_.sendbuff = kRecv;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, RingInPlaceBroadcast_RegistersOverIpc) {
  task_.func = ncclFuncBroadcast;
  task_.sendbuff = kRecv;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollIpcGateMicrotest, RingReduce_LeavesBufferUnregistered) {
  task_.func = ncclFuncReduce;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, TreeOutOfPlace_RegistersOverIpc) {
  task_.algorithm = NCCL_ALGO_TREE;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollIpcGateMicrotest, TreeInPlace_LeavesBufferUnregistered) {
  task_.algorithm = NCCL_ALGO_TREE;
  task_.sendbuff = kRecv;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, TreeCrossClique_LeavesBufferUnregistered) {
  task_.algorithm = NCCL_ALGO_TREE;
  comm_->p2pCrossClique = true;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, CollnetChainInPlaceWithLocalPeers_LeavesBufferUnregistered) {
  task_.algorithm = NCCL_ALGO_COLLNET_CHAIN;
  task_.sendbuff = kRecv;
  comm_->maxLocalRanks = 2;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollIpcGateMicrotest, CollnetChainInPlaceSingleLocalRank_RegistersOverIpc) {
  task_.algorithm = NCCL_ALGO_COLLNET_CHAIN;
  task_.sendbuff = kRecv;
  comm_->maxLocalRanks = 1;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollIpcGateMicrotest, CollnetChainOutOfPlaceWithLocalPeers_RegistersOverIpc) {
  task_.algorithm = NCCL_ALGO_COLLNET_CHAIN;
  comm_->maxLocalRanks = 2;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: CollNet Direct
// ---------------------------------------------------------------------------

class RegisterCollNetDirectMicrotest : public RegisterCollMicrotestBase {
 protected:
  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    SetChannels(1);
    task_.algorithm = NCCL_ALGO_COLLNET_DIRECT;
    task_.func = ncclFuncAllGather;
    comm_->isAllDirectP2p = true;
  }
  ncclDirect& Direct() { return comm_->channels[0].collnetDirect; }
  // Peers 1 (up) and 2 (down), both reached over P2P.
  void TwoP2pPeers() {
    Direct().up[0] = 1;
    Direct().down[0] = 2;
    ConnectP2p(RecvConn(0, 1));
    ConnectP2p(RecvConn(0, 2));
  }
};

TEST_F(RegisterCollNetDirectMicrotest, P2pPeers_RegistersBothBuffersForEachPeerOnce) {
  TwoP2pPeers();
  Direct().down[3] = 1;
  Direct().up[NCCL_MAX_DIRECT_ARITY - 1] = 3;
  ConnectP2p(RecvConn(0, 3));
  IpcLocalRegisters({kSend, kRecv});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kSend, kBytes, NCCL_IPC_COLLECTIVE, {1, 2, 3}, {}},
                                {kRecv, kRanks * kBytes, NCCL_IPC_COLLECTIVE, {1, 2, 3}, {}}};
  EXPECT_EQ(ipcLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
  EXPECT_EQ(task_.recvbuffRmtAddrs, ipcAddrs_);
  EXPECT_EQ(task_.recvbuffOffset, 0x40u);
}

TEST_F(RegisterCollNetDirectMicrotest, NetConnectedPeer_IsNotIpcRegistered) {
  Direct().up[0] = 1;
  RecvConn(0, 1)->connected = 1;
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNetDirectMicrotest, UnconnectedPeer_RegisteredWhenTransportCanConnect) {
  Direct().up[0] = 1;
  Direct().down[0] = 2;
  p2p_.connectable = {2};
  IpcLocalRegisters({kSend, kRecv});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(p2p_.asked, (std::vector<int>{1, 2}));
  EXPECT_EQ(p2p_.lastGraph, &comm_->graphs[NCCL_ALGO_COLLNET_DIRECT]);
  EXPECT_EQ(p2p_.lastMine, &comm_->peerInfo[comm_->rank]);
  ASSERT_EQ(ipcLocal_.size(), 2u);
  EXPECT_EQ(ipcLocal_[0].peers, (std::vector<int>{2}));
}

TEST_F(RegisterCollNetDirectMicrotest, CanConnectFails_ReturnsError) {
  Direct().up[0] = 1;
  p2p_.result = ncclInternalError;
  EXPECT_EQ(Run(), ncclInternalError);
}

TEST_F(RegisterCollNetDirectMicrotest, NoP2pPeer_SkipsIpc) {
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNetDirectMicrotest, NotAllDirectP2p_SkipsIpc) {
  TwoP2pPeers();
  comm_->isAllDirectP2p = false;
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNetDirectMicrotest, ReduceScatter_SkipsIpc) {
  TwoP2pPeers();
  task_.func = ncclFuncReduceScatter;
  IpcLocalRegisters({kSend, kRecv});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(ipcLocal_.empty());
}

TEST_F(RegisterCollNetDirectMicrotest, AllReduce_SkipsIpc) {
  TwoP2pPeers();
  task_.func = ncclFuncAllReduce;
  IpcLocalRegisters({kSend, kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(ipcLocal_.empty());
}

TEST_F(RegisterCollNetDirectMicrotest, BroadcastRegistered_LeavesIpcUnmarked) {
  TwoP2pPeers();
  task_.func = ncclFuncBroadcast;
  IpcLocalRegisters({kSend, kRecv});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcLocal_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, AllGatherSendOnlyRegistered_LeavesIpcUnmarked) {
  TwoP2pPeers();
  IpcLocalRegisters({kSend});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, GraphCapture_RegistersLocallyOnlyWhatGraphMissed) {
  TwoP2pPeers();
  UseGraphCapture();
  IpcGraphRegisters({kSend});
  IpcLocalRegisters({kRecv});
  CollnetGraphRegisters({});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(ipcGraph_.size(), 2u);
  ASSERT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(ipcLocal_[0].buff, kRecv);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, GraphRegisteredRecv_RegistersOnlySendLocally) {
  TwoP2pPeers();
  UseGraphCapture();
  IpcGraphRegisters({kRecv});
  IpcLocalRegisters({kSend});
  CollnetGraphRegisters({});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(ipcLocal_[0].buff, kSend);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, CaptureWithoutLocalRegister_RegistersPerGraphOnly) {
  TwoP2pPeers();
  UseGraphCapture(/*localRegister=*/0);
  IpcGraphRegisters({});
  CollnetGraphRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcGraph_.size(), 2u);
}

TEST_F(RegisterCollNetDirectMicrotest, AllGatherOneRpnBothRegistered_MarksNetOnOneChannel) {
  comm_->isOneRPN = true;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kSend, kBytes, collNetSend, {}, {}},
                                {kRecv, kRanks * kBytes, collNetRecv, {}, {}}};
  EXPECT_EQ(collnetLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xc0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xc0 + collNetRecv));
  EXPECT_EQ(task_.nMaxChannels, 1);
}

TEST_F(RegisterCollNetDirectMicrotest, AllGatherOneRpnRecvMissing_LeavesNetUnmarked) {
  comm_->isOneRPN = true;
  CollnetLocalRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollNetDirectMicrotest, AllGatherMultiRpnSendRegistered_MarksNetKeepingChannels) {
  CollnetLocalRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollNetDirectMicrotest, AllGatherMultiRpnSendMissing_LeavesNetUnmarked) {
  CollnetLocalRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, ReduceOneRpnBothRegistered_MarksNet) {
  comm_->isOneRPN = true;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncReduceScatter}) {
    task_.func = func;
    task_.regBufType = -1;
    EXPECT_EQ(Run(), ncclSuccess);
    EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER) << func;
  }
}

TEST_F(RegisterCollNetDirectMicrotest, ReduceOneRpnOneSideMissing_LeavesNetUnmarked) {
  comm_->isOneRPN = true;
  task_.func = ncclFuncReduceScatter;
  for (int missing : {collNetSend, collNetRecv}) {
    CollnetLocalRegisters({missing == collNetSend ? collNetRecv : collNetSend});
    EXPECT_EQ(Run(), ncclSuccess);
    EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER) << missing;
  }
}

TEST_F(RegisterCollNetDirectMicrotest, ReduceScatterMultiRpn_LeavesNetUnmarked) {
  task_.func = ncclFuncReduceScatter;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, AllReduceMultiRpn_SkipsCollnet) {
  task_.func = ncclFuncAllReduce;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNetDirectMicrotest, OtherCollectiveRegistered_LeavesNetUnmarked) {
  task_.func = ncclFuncBroadcast;
  comm_->isOneRPN = true;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetLocal_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollNetDirectMicrotest, ScalingReduction_SkipsCollnet) {
  for (ncclDevRedOp_t op : {ncclDevPreMulSum, ncclDevSumPostDiv}) {
    task_.opDev.op = op;
    EXPECT_EQ(Run(), ncclSuccess);
    ExpectUnregistered();
  }
}

TEST_F(RegisterCollNetDirectMicrotest, MloPartBufferNotRdmaCapable_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollNetDirectMicrotest, MloPartWithoutSendBuffer_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 1}});
  task_.sendbuff = nullptr;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(collnetLocal_.empty());
}

TEST_F(RegisterCollNetDirectMicrotest, MloPartWithoutRecvBuffer_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 1}});
  task_.recvbuff = nullptr;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(collnetLocal_.empty());
}

TEST_F(RegisterCollNetDirectMicrotest, GraphCapture_RegistersCollnetLocallyOnlyWhatGraphMissed) {
  UseGraphCapture();
  comm_->isOneRPN = true;
  CollnetGraphRegisters({collNetSend});
  CollnetLocalRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(collnetGraph_.size(), 2u);
  ASSERT_EQ(collnetLocal_.size(), 1u);
  EXPECT_EQ(collnetLocal_[0].type, collNetRecv);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xd0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xc0 + collNetRecv));
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, GraphRegisteredCollnetRecv_RegistersOnlySendLocally) {
  UseGraphCapture();
  comm_->isOneRPN = true;
  CollnetGraphRegisters({collNetRecv});
  CollnetLocalRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(collnetLocal_.size(), 1u);
  EXPECT_EQ(collnetLocal_[0].type, collNetSend);
  EXPECT_EQ(task_.recvMhandle, Ptr(0xd0 + collNetRecv));
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollNetDirectMicrotest, CaptureWithoutLocalRegister_RegistersCollnetPerGraphOnly) {
  UseGraphCapture(/*localRegister=*/0);
  CollnetGraphRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetGraph_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: Ring
// ---------------------------------------------------------------------------

// Channel 0: recv from P2P peer 1, send to NIC peer 2. Channel 1: recv from NIC peer 3, send to P2P peer 1.
class RegisterCollRingMicrotest : public RegisterCollMicrotestBase {
 protected:
  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    SetChannels(2);
    comm_->channels[0].ring.prev = 1;
    comm_->channels[0].ring.next = 2;
    comm_->channels[1].ring.prev = 3;
    comm_->channels[1].ring.next = 1;
    RecvConn(0, 1)->conn.flags = NCCL_P2P_READ;
    SendConn(0, 2)->conn.flags = NCCL_DIRECT_NIC;
    RecvConn(1, 3)->conn.flags = NCCL_DIRECT_NIC;
    SendConn(1, 1)->conn.flags = NCCL_P2P_WRITE;
    comm_->isAllDirectP2p = true;
    comm_->useGdr = true;
    AddRegRecord(kSend, kBytes);
    AddRegRecord(kRecv, kRanks * kBytes);
  }
  // Leaves only P2P links, so the NET path has no peer.
  void P2pOnly() {
    SendConn(0, 2)->conn.flags = NCCL_P2P_WRITE;
    RecvConn(1, 3)->conn.flags = NCCL_P2P_READ;
  }
  std::vector<ncclConnector*> SendNetConns() { return {SendConn(0, 2), nullptr}; }
  std::vector<ncclConnector*> RecvNetConns() { return {nullptr, RecvConn(1, 3)}; }
};

TEST_F(RegisterCollRingMicrotest, NoRecvRecordOutsideCapture_LeavesBufferUnregistered) {
  regSlots_.erase(regSlots_.begin() + 1);
  comm_->regCache.population = 1;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollRingMicrotest, MultiNodeWithoutSendRecord_LeavesBufferUnregistered) {
  regSlots_.erase(regSlots_.begin());
  comm_->regCache.population = 1;
  comm_->nNodes = 2;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollRingMicrotest, SingleNodeWithoutSendRecord_RegistersRecvBuffer) {
  P2pOnly();
  regSlots_.erase(regSlots_.begin());
  comm_->regCache.population = 1;
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, P2pNeighbours_RegistersRecvBufferForEachOnce) {
  P2pOnly();
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, NCCL_IPC_COLLECTIVE, {1, 2, 3}, {}}};
  EXPECT_EQ(ipcLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
  EXPECT_EQ(task_.recvbuffRmtAddrs, ipcAddrs_);
  EXPECT_EQ(task_.recvbuffOffset, 0x40u);
}

TEST_F(RegisterCollRingMicrotest, IpcNotRegistered_LeavesBufferUnregistered) {
  P2pOnly();
  IpcLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, NotAllDirectP2p_SkipsIpc) {
  P2pOnly();
  comm_->isAllDirectP2p = false;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, NoRecordsInCapture_RegistersPerGraph) {
  P2pOnly();
  regSlots_.clear();
  comm_->regCache.population = 0;
  comm_->nNodes = 2;
  UseGraphCapture();
  IpcGraphRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcGraph_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, GraphIpcNotRegistered_FallsBackToLocal) {
  P2pOnly();
  UseGraphCapture();
  IpcGraphRegisters({});
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, GraphIpcMissWithoutLocalRegister_LeavesBufferUnregistered) {
  P2pOnly();
  UseGraphCapture(/*localRegister=*/0);
  IpcGraphRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcGraph_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, NicNeighbours_RegistersSendThenRecvOverEachDirection) {
  comm_->isAllDirectP2p = false;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kSend, kBytes, 0, {}, SendNetConns()},
                                {kRecv, kBytes, 0, {}, SendNetConns()},
                                {kRecv, kBytes, 0, {}, RecvNetConns()}};
  EXPECT_EQ(netLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  ASSERT_NE(task_.sendNetHandles, nullptr);
  ASSERT_NE(task_.recvNetHandles, nullptr);
  ASSERT_NE(task_.srecvNetHandles, nullptr);
  EXPECT_EQ(task_.sendNetHandles[0], Ptr(0xe0));
  EXPECT_EQ(task_.srecvNetHandles[0], Ptr(0xe1));
  EXPECT_EQ(task_.recvNetHandles[0], Ptr(0xe2));
}

TEST_F(RegisterCollRingMicrotest, SendBufferNotRegistered_StopsNetRegistration) {
  comm_->isAllDirectP2p = false;
  NetLocalRegisters({0});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(netLocal_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
  EXPECT_EQ(task_.sendNetHandles, nullptr);
}

TEST_F(RegisterCollRingMicrotest, RecvSideNotRegistered_LeavesNetUnmarked) {
  comm_->isAllDirectP2p = false;
  NetLocalRegisters({1, 1, 0});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
  EXPECT_EQ(task_.recvNetHandles, nullptr);
}

TEST_F(RegisterCollRingMicrotest, OnlyRecvNicNeighbour_RegistersRecvBuffer) {
  comm_->isAllDirectP2p = false;
  SendConn(0, 2)->conn.flags = 0;
  NetLocalRegisters({1});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, 0, {}, RecvNetConns()}};
  EXPECT_EQ(netLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, OnlySendNicNeighbour_RegistersSendSide) {
  comm_->isAllDirectP2p = false;
  RecvConn(1, 3)->conn.flags = 0;
  NetLocalRegisters({1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(netLocal_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, IpcAndNet_MarksBoth) {
  IpcLocalRegisters({kRecv});
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(netLocal_.size(), 3u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER | NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, IpcRegisteredNetNot_MarksOnlyIpc) {
  IpcLocalRegisters({kRecv});
  NetLocalRegisters({0});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(netLocal_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, IpcAndNetSingleNode_KeepsChannels) {
  IpcLocalRegisters({kRecv});
  NetLocalRegisters({1, 1, 1});
  task_.nMaxChannels = 20;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 20);
}

TEST_F(RegisterCollRingMicrotest, IpcRegisteredWithoutNicNeighbours_MarksOnlyIpc) {
  P2pOnly();
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
  EXPECT_EQ(task_.sendNetHandles, nullptr);
}

TEST_F(RegisterCollRingMicrotest, NetPxn_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  comm_->useNetPXN = true;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(netLocal_.empty());
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, NoGdr_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  comm_->useGdr = false;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(netLocal_.empty());
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, UnpackNetDevice_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  comm_->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(netLocal_.empty());
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, ScalingAllReduce_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  for (ncclDevRedOp_t op : {ncclDevPreMulSum, ncclDevSumPostDiv}) {
    task_.opDev.op = op;
    EXPECT_EQ(Run(), ncclSuccess);
    EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
  }
}

TEST_F(RegisterCollRingMicrotest, ScalingBroadcast_RegistersNet) {
  comm_->isAllDirectP2p = false;
  task_.func = ncclFuncBroadcast;
  task_.opDev.op = ncclDevPreMulSum;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, MloPartBufferNotRdmaCapable_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, MloPartWithoutSendBuffer_SkipsNetRegistration) {
  comm_->isAllDirectP2p = false;
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 1}});
  task_.sendbuff = nullptr;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, GraphCapture_RegistersNetPerGraph) {
  comm_->isAllDirectP2p = false;
  UseGraphCapture();
  NetGraphRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kSend, kBytes, 0, {}, SendNetConns()},
                                {kRecv, kBytes, 0, {}, SendNetConns()},
                                {kRecv, kBytes, 0, {}, RecvNetConns()}};
  EXPECT_EQ(netGraph_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.sendNetHandles[0], Ptr(0xf0));
}

TEST_F(RegisterCollRingMicrotest, GraphSendNotRegistered_FallsBackToLocal) {
  comm_->isAllDirectP2p = false;
  UseGraphCapture();
  NetGraphRegisters({0});
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(netGraph_.size(), 1u);
  EXPECT_EQ(netLocal_.size(), 3u);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, GraphNetMissWithoutLocalRegister_LeavesBufferUnregistered) {
  comm_->isAllDirectP2p = false;
  UseGraphCapture(/*localRegister=*/0);
  NetGraphRegisters({0});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.regBufType, NCCL_REGULAR_BUFFER);
}

TEST_F(RegisterCollRingMicrotest, OneRpnGatherRegistered_CapsChannelsAtNetCount) {
  comm_->isAllDirectP2p = false;
  comm_->isOneRPN = true;
  comm_->minNetCount = 3;
  NetLocalRegisters({1, 1, 1});
  for (ncclFunc_t func : {ncclFuncAllGather, ncclFuncBroadcast}) {
    task_.func = func;
    task_.nMaxChannels = 32;
    EXPECT_EQ(Run(), ncclSuccess);
    EXPECT_EQ(task_.nMaxChannels, 3) << func;
    free(task_.sendNetHandles);
    free(task_.recvNetHandles);
    free(task_.srecvNetHandles);
    task_.sendNetHandles = task_.recvNetHandles = task_.srecvNetHandles = nullptr;
    netLocal_.clear();
  }
}

TEST_F(RegisterCollRingMicrotest, OneRpnGatherRegistered_KeepsChannelsWithinCtaLimits) {
  comm_->isAllDirectP2p = false;
  comm_->isOneRPN = true;
  comm_->minNetCount = 3;
  comm_->config.minCTAs = 5;
  task_.func = ncclFuncBroadcast;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 5);
}

TEST_F(RegisterCollRingMicrotest, OneRpnAllReduceRegistered_KeepsChannels) {
  comm_->isAllDirectP2p = false;
  comm_->isOneRPN = true;
  comm_->minNetCount = 3;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollRingMicrotest, MultiRpnGatherRegistered_KeepsChannels) {
  comm_->isAllDirectP2p = false;
  comm_->minNetCount = 3;
  task_.func = ncclFuncBroadcast;
  NetLocalRegisters({1, 1, 1});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(task_.nMaxChannels, 32);
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: Tree and CollNet chain
// ---------------------------------------------------------------------------

// Channel 0's tree and chain: P2P parent 1, P2P child 2, network child 3, and an nRanks "no peer" slot.
class RegisterCollTreeMicrotest : public RegisterCollMicrotestBase {
 protected:
  void SetUp() override {
    RegisterCollMicrotestBase::SetUp();
    SetChannels(1);
    task_.algorithm = NCCL_ALGO_TREE;
    comm_->isAllDirectP2p = true;
    for (ncclTree* tree : {&comm_->channels[0].tree, &comm_->channels[0].collnetChain}) *tree = {0, 1, {2, kRanks, 3}};
    ConnectP2p(RecvConn(0, 1));
    ConnectP2p(RecvConn(0, 2));
    RecvConn(0, 3)->connected = 1;
    AddRegRecord(kRecv, kBytes);
  }
};

TEST_F(RegisterCollTreeMicrotest, P2pTreePeers_RegistersRecvBufferForEach) {
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, NCCL_IPC_COLLECTIVE, {2, 1}, {}}};
  EXPECT_EQ(ipcLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
  EXPECT_EQ(task_.recvbuffRmtAddrs, ipcAddrs_);
}

TEST_F(RegisterCollTreeMicrotest, PeersAcrossChannels_RegisteredOnce) {
  SetChannels(2);
  for (int c = 0; c < 2; ++c) {
    comm_->channels[c].tree = {0, 1, {-1, -1, -1}};
    ConnectP2p(RecvConn(c, 1));
  }
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(ipcLocal_[0].peers, (std::vector<int>{1}));
}

TEST_F(RegisterCollTreeMicrotest, UnconnectedPeer_AsksTransportWithAlgorithmGraph) {
  RecvConn(0, 2)->connected = 0;
  p2p_.connectable = {};
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(p2p_.asked, (std::vector<int>{2}));
  EXPECT_EQ(p2p_.lastGraph, &comm_->graphs[NCCL_ALGO_TREE]);
  ASSERT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(ipcLocal_[0].peers, (std::vector<int>{1}));
}

TEST_F(RegisterCollTreeMicrotest, CanConnectFails_ReturnsError) {
  RecvConn(0, 2)->connected = 0;
  p2p_.result = ncclInternalError;
  EXPECT_EQ(Run(), ncclInternalError);
}

TEST_F(RegisterCollTreeMicrotest, NoRecvRecordOutsideCapture_LeavesBufferUnregistered) {
  regSlots_.clear();
  comm_->regCache.population = 0;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollTreeMicrotest, NotAllDirectP2p_SkipsIpc) {
  comm_->isAllDirectP2p = false;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollTreeMicrotest, NoP2pPeer_SkipsIpc) {
  comm_->channels[0].tree = {0, 3, {-1, -1, -1}};
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollTreeMicrotest, IpcNotRegistered_LeavesBufferUnregistered) {
  IpcLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollTreeMicrotest, NoRecordsInCapture_RegistersPerGraph) {
  regSlots_.clear();
  comm_->regCache.population = 0;
  UseGraphCapture();
  IpcGraphRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcGraph_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollTreeMicrotest, GraphIpcNotRegistered_FallsBackToLocal) {
  UseGraphCapture();
  IpcGraphRegisters({});
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER);
}

TEST_F(RegisterCollTreeMicrotest, GraphIpcMissWithoutLocalRegister_LeavesBufferUnregistered) {
  UseGraphCapture(/*localRegister=*/0);
  IpcGraphRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(ipcGraph_.size(), 1u);
  ExpectUnregistered();
}

TEST_F(RegisterCollTreeMicrotest, TreeOneRpn_SkipsCollnet) {
  comm_->isOneRPN = true;
  IpcLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

class RegisterCollChainMicrotest : public RegisterCollTreeMicrotest {
 protected:
  void SetUp() override {
    RegisterCollTreeMicrotest::SetUp();
    task_.algorithm = NCCL_ALGO_COLLNET_CHAIN;
    comm_->isAllDirectP2p = false;
    comm_->isOneRPN = true;
  }
};

TEST_F(RegisterCollChainMicrotest, OneRpnBothRegistered_MarksNetOnOneChannel) {
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kSend, kBytes, collNetSend, {}, {}}, {kRecv, kBytes, collNetRecv, {}, {}}};
  EXPECT_EQ(collnetLocal_, expected);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xc0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xc0 + collNetRecv));
  EXPECT_EQ(task_.nMaxChannels, 1);
}

TEST_F(RegisterCollChainMicrotest, IpcAndCollnet_MarksBoth) {
  comm_->isAllDirectP2p = true;
  IpcLocalRegisters({kRecv});
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  ASSERT_EQ(ipcLocal_.size(), 1u);
  EXPECT_EQ(ipcLocal_[0].peers, (std::vector<int>{2, 1}));
  EXPECT_TRUE(p2p_.asked.empty());
  EXPECT_EQ(task_.regBufType, NCCL_IPC_REG_BUFFER | NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollChainMicrotest, UnconnectedPeer_AsksTransportWithChainGraph) {
  comm_->isAllDirectP2p = true;
  RecvConn(0, 2)->connected = 0;
  IpcLocalRegisters({});
  CollnetLocalRegisters({});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(p2p_.lastGraph, &comm_->graphs[NCCL_ALGO_COLLNET_CHAIN]);
}

TEST_F(RegisterCollChainMicrotest, SendSideNotRegistered_SkipsRecvSide) {
  CollnetLocalRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetLocal_.size(), 1u);
  ExpectUnregistered();
  EXPECT_EQ(task_.nMaxChannels, 32);
}

TEST_F(RegisterCollChainMicrotest, RecvSideNotRegistered_LeavesNetUnmarked) {
  CollnetLocalRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetLocal_.size(), 2u);
  ExpectUnregistered();
}

TEST_F(RegisterCollChainMicrotest, MultiRpn_SkipsCollnet) {
  comm_->isOneRPN = false;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollChainMicrotest, ScalingReduction_SkipsCollnet) {
  for (ncclDevRedOp_t op : {ncclDevPreMulSum, ncclDevSumPostDiv}) {
    task_.opDev.op = op;
    EXPECT_EQ(Run(), ncclSuccess);
    ExpectUnregistered();
  }
}

TEST_F(RegisterCollChainMicrotest, MloPartWithoutSendBuffer_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 1}});
  task_.sendbuff = nullptr;
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(collnetLocal_.empty());
}

TEST_F(RegisterCollChainMicrotest, MloPartRdmaDisabled_SkipsCollnet) {
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 0}});
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_TRUE(collnetLocal_.empty());
}

TEST_F(RegisterCollChainMicrotest, MloPartWithoutRecvBufferInCapture_SkipsCollnet) {
  UseGraphCapture();
  comm_->hasMloPart = true;
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"GRAPH_REGISTER", 1}, {"MLOPART_RDMA_ENABLE", 1}});
  task_.recvbuff = nullptr;
  EXPECT_EQ(Run(), ncclSuccess);
  ExpectUnregistered();
}

TEST_F(RegisterCollChainMicrotest, GraphCapture_RegistersBothPerGraph) {
  UseGraphCapture();
  CollnetGraphRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetGraph_.size(), 2u);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xd0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xd0 + collNetRecv));
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollChainMicrotest, GraphSendNotRegistered_RegistersBothLocally) {
  UseGraphCapture();
  CollnetGraphRegisters({});
  CollnetLocalRegisters({collNetSend, collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetGraph_.size(), 1u);
  EXPECT_EQ(collnetLocal_.size(), 2u);
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollChainMicrotest, GraphRecvNotRegistered_RegistersOnlyRecvLocally) {
  UseGraphCapture();
  CollnetGraphRegisters({collNetSend});
  CollnetLocalRegisters({collNetRecv});
  EXPECT_EQ(Run(), ncclSuccess);
  std::vector<RegCall> expected{{kRecv, kBytes, collNetRecv, {}, {}}};
  EXPECT_EQ(collnetLocal_, expected);
  EXPECT_EQ(task_.sendMhandle, Ptr(0xd0 + collNetSend));
  EXPECT_EQ(task_.recvMhandle, Ptr(0xc0 + collNetRecv));
  EXPECT_EQ(task_.regBufType, NCCL_NET_REG_BUFFER);
}

TEST_F(RegisterCollChainMicrotest, GraphRecvMissWithoutLocalRegister_LeavesBufferUnregistered) {
  UseGraphCapture(/*localRegister=*/0);
  CollnetGraphRegisters({collNetSend});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(collnetGraph_.size(), 2u);
  ExpectUnregistered();
}

// ---------------------------------------------------------------------------
// ncclRegisterCollBuffers: channel cap for single-node IPC buffers
// ---------------------------------------------------------------------------

class RegisterCollIpcChannelsMicrotest : public RegisterCollTreeMicrotest {
 protected:
  int ChannelsFor(int requested) {
    task_.nMaxChannels = requested;
    EXPECT_EQ(Run(), ncclSuccess);
    return task_.nMaxChannels;
  }
};

TEST_F(RegisterCollIpcChannelsMicrotest, SingleNodeIpc_CapsSeventeenToTwentyFourAtSixteen) {
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(ChannelsFor(16), 16);
  EXPECT_EQ(ChannelsFor(17), 16);
  EXPECT_EQ(ChannelsFor(24), 16);
  EXPECT_EQ(ChannelsFor(25), 25);
}

TEST_F(RegisterCollIpcChannelsMicrotest, MultiNodeIpc_KeepsChannels) {
  comm_->nNodes = 2;
  IpcLocalRegisters({kRecv});
  EXPECT_EQ(ChannelsFor(20), 20);
}

TEST_F(RegisterCollIpcChannelsMicrotest, Unregistered_KeepsChannels) {
  IpcLocalRegisters({});
  EXPECT_EQ(ChannelsFor(20), 20);
}

}  // namespace
