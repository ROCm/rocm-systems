/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/rma_socket_setup.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/socket_fakes.h"
#include "transport/RmaSocketTest.h"

#include RMA_SOCKET_SETUP_CC_PATH

namespace {

using rma_socket_test::Scene;

class ScopedConnectHandshake {
 private:
  std::vector<int> acceptedRanks_;
  size_t recvIndex_ = 0;
  uint64_t nextMagic_;
  int readyPollsPerPeer_;
  int readyPolls_ = 0;

 public:
  ScopedConnectHandshake(std::initializer_list<int> acceptedRanks, uint64_t firstMagic = 0,
                         int readyPollsPerPeer = 1)
      : acceptedRanks_(acceptedRanks),
        nextMagic_(firstMagic),
        readyPollsPerPeer_(readyPollsPerPeer > 0 ? readyPollsPerPeer : 1),
        init(g_ncclSocketInit,
             [this](ncclSocket* sock, const ncclSocketAddress* addr, uint64_t magic, ncclSocketType,
                    volatile uint32_t*, int, int) {
               if (addr != nullptr) outgoingConnects.push_back({sock, addr, magic});
               return ncclSuccess;
             }),
        connect(g_ncclSocketConnect, [](ncclSocket*) { return ncclSuccess; }),
        send(g_ncclSocketSend, [this](ncclSocket*, void* data, int size) {
          if (data == nullptr || size != static_cast<int>(sizeof(int))) {
            ADD_FAILURE() << "connect handshake sent an invalid rank buffer";
            return ncclInternalError;
          }
          sentRanks.push_back(*static_cast<int*>(data));
          return ncclSuccess;
        }),
        accept(g_ncclSocketAccept, [this](ncclSocket* sock, ncclSocket*, bool) {
          sock->magic = nextMagic_++;
          return ncclSuccess;
        }),
        ready(g_ncclSocketReady, [this](ncclSocket*, int* running) {
          *running = (++readyPolls_ % readyPollsPerPeer_) == 0;
          return ncclSuccess;
        }),
        recv(g_ncclSocketRecv, [this](ncclSocket*, void* data, int size) {
          if (data == nullptr || size != static_cast<int>(sizeof(int))) {
            ADD_FAILURE() << "connect handshake received an invalid rank buffer";
            return ncclInternalError;
          }
          if (recvIndex_ >= acceptedRanks_.size()) {
            ADD_FAILURE() << "connect handshake exhausted its accepted-rank script";
            return ncclInternalError;
          }
          *static_cast<int*>(data) = acceptedRanks_[recvIndex_++];
          return ncclSuccess;
        }),
        move(g_ncclSocketMove, [](ncclSocket* dst, ncclSocket* src) {
          dst->magic = src->magic;
          src->magic = 0;
        }),
        close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; }) {}

  struct OutgoingConnect {
    ncclSocket* sock;
    const ncclSocketAddress* addr;
    uint64_t magic;
  };

  std::vector<int> sentRanks;
  std::vector<OutgoingConnect> outgoingConnects;
  ScopedHook<ncclResult_t(ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int,
                          int)>
      init;
  ScopedHook<ncclResult_t(ncclSocket*)> connect;
  ScopedHook<ncclResult_t(ncclSocket*, void*, int)> send;
  ScopedHook<ncclResult_t(ncclSocket*, ncclSocket*, bool)> accept;
  ScopedHook<ncclResult_t(ncclSocket*, int*)> ready;
  ScopedHook<ncclResult_t(ncclSocket*, void*, int)> recv;
  ScopedHook<void(ncclSocket*, ncclSocket*)> move;
  ScopedHook<ncclResult_t(ncclSocket*, bool)> close;
};

class RmaSocketSetupMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { g_hipAsyncOpsResult = hipSuccess; }

  void TearDown() override {
    ResetSocketFakes();
    ResetOsFakes();
    ResetHipFakes();
    ResetNcclFakes();
  }
};

ncclRmaSocketProxyCollComm* AllocateOwnedComm(int nranks) {
  auto* comm = static_cast<ncclRmaSocketProxyCollComm*>(std::calloc(1, sizeof(ncclRmaSocketProxyCollComm)));
  if (comm == nullptr) return nullptr;
  comm->nranks = nranks;
  comm->peerSender =
      static_cast<ncclRmaSocketProxyPeerSender*>(std::calloc(nranks, sizeof(*comm->peerSender)));
  comm->peerReceiver =
      static_cast<ncclRmaSocketProxyPeerReceiver*>(std::calloc(nranks, sizeof(*comm->peerReceiver)));
  if (comm->peerSender == nullptr || comm->peerReceiver == nullptr) {
    std::free(comm->peerSender);
    std::free(comm->peerReceiver);
    std::free(comm);
    return nullptr;
  }
  return comm;
}

void FreeRejectedComm(ncclRmaSocketProxyCollComm* comm) {
  comm->nContexts = 0;
  comm->mrHead = nullptr;
  for (int peer = 0; peer < comm->nranks; ++peer) {
    comm->peerSender[peer].count = 0;
    comm->peerReceiver[peer].recvState = ncclRmaSocketProxyRecvStateControlMsg;
    comm->peerReceiver[peer].controlMsgOffset = 0;
  }
  EXPECT_EQ(ncclRmaSocketProxyFreeCollComm(comm), ncclSuccess);
}

TEST_F(RmaSocketSetupMicrotest, SetSendQueueCapacity_InvalidContextCountIsRejected) {
  ncclRmaSocketProxyCollComm comm{};

  EXPECT_EQ(ncclRmaSocketProxySetSendQueueCapacity(nullptr, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxySetSendQueueCapacity(&comm, -1), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxySetSendQueueCapacity(
                &comm, INT_MAX / (2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER) + 1),
            ncclInvalidArgument);
}

TEST_F(RmaSocketSetupMicrotest, SetSendQueueCapacity_GrowthKeepsQueuedTasksInFifoOrder) {
  ncclRmaSocketProxyCollComm comm{};
  comm.nranks = 1;
  comm.peerSender = static_cast<ncclRmaSocketProxyPeerSender*>(
      std::calloc(1, sizeof(ncclRmaSocketProxyPeerSender)));
  ASSERT_NE(comm.peerSender, nullptr);
  auto* sender = comm.peerSender;
  sender->capacity = 3;
  sender->tasks =
      static_cast<ncclRmaSocketProxySendTask*>(std::calloc(sender->capacity, sizeof(*sender->tasks)));
  ASSERT_NE(sender->tasks, nullptr);
  sender->head = 2;
  sender->tail = 1;
  sender->count = 2;
  sender->tasks[2].controlMsg.requestId = 41;
  sender->tasks[0].controlMsg.requestId = 42;

  ASSERT_EQ(ncclRmaSocketProxySetSendQueueCapacity(&comm, 1), ncclSuccess);

  EXPECT_EQ(sender->capacity, 2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER);
  EXPECT_EQ(sender->head, 0);
  EXPECT_EQ(sender->tail, 2);
  EXPECT_EQ(sender->count, 2);
  EXPECT_EQ(sender->tasks[0].controlMsg.requestId, 41u);
  EXPECT_EQ(sender->tasks[1].controlMsg.requestId, 42u);
  std::free(sender->tasks);
  std::free(comm.peerSender);
}

TEST_F(RmaSocketSetupMicrotest, SetSendQueueCapacity_InvalidExistingQueueStateIsRejected) {
  ncclRmaSocketProxyCollComm comm{};
  ncclRmaSocketProxyPeerSender sender{};
  comm.nranks = 1;
  comm.peerSender = &sender;
  sender.capacity = 1;
  sender.count = 2;

  EXPECT_EQ(ncclRmaSocketProxySetSendQueueCapacity(&comm, 1), ncclInternalError);
}

TEST_F(RmaSocketSetupMicrotest, Connect_InvalidArgumentsAreRejectedWithoutPublishingComm) {
  ncclSocket listen{};
  ncclRmaSocketProxyHandle handle{};
  void* handles[] = {&handle};
  void* collComm = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, nullptr, 1, 0, &listen, &collComm), ncclInvalidArgument);
  EXPECT_EQ(collComm, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, 0, nullptr, &collComm), ncclInvalidArgument);
  EXPECT_EQ(collComm, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, 0, &listen, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, -1, &listen, &collComm), ncclInvalidArgument);
  EXPECT_EQ(collComm, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, 1, &listen, &collComm), ncclInvalidArgument);
  EXPECT_EQ(collComm, reinterpret_cast<void*>(0x1));
}

TEST_F(RmaSocketSetupMicrotest, Connect_AllPeersConnectedBuildsBidirectionalPeerState) {
  constexpr int nranks = 2;
  ncclSocket listen{};
  std::array<ncclRmaSocketProxyHandle, nranks> handleStorage{};
  handleStorage[0].magic = 0x1111;
  handleStorage[1].magic = 0x2222;
  std::array<void*, nranks> handles{{&handleStorage[0], &handleStorage[1]}};
  ScopedConnectHandshake handshake({1, 0}, /*firstMagic=*/100, /*readyPollsPerPeer=*/2);
  void* opaque = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyConnect(nullptr, handles.data(), nranks, 0, &listen, &opaque), ncclSuccess);
  auto* comm = static_cast<ncclRmaSocketProxyCollComm*>(opaque);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->rank, 0);
  EXPECT_EQ(comm->nranks, nranks);
  EXPECT_EQ(comm->nextGlobalRequestId, 0u);
  EXPECT_EQ(comm->lastError, ncclSuccess);
  EXPECT_EQ(handshake.sentRanks, (std::vector<int>{0, 0}));
  EXPECT_EQ(handshake.init.calls, nranks * 2);
  EXPECT_EQ(handshake.connect.calls, nranks);
  EXPECT_EQ(handshake.send.calls, nranks);
  EXPECT_EQ(handshake.accept.calls, nranks);
  EXPECT_EQ(handshake.recv.calls, nranks);
  EXPECT_EQ(handshake.ready.calls, nranks * 2);
  EXPECT_EQ(handshake.move.calls, nranks);
  EXPECT_EQ(comm->peerReceiver[0].sock.magic, 101u);
  EXPECT_EQ(comm->peerReceiver[1].sock.magic, 100u);
  ASSERT_EQ(handshake.outgoingConnects.size(), static_cast<size_t>(nranks));
  for (int peer = 0; peer < nranks; ++peer) {
    EXPECT_EQ(handshake.outgoingConnects[peer].sock, &comm->peerSender[peer].sock);
    EXPECT_EQ(handshake.outgoingConnects[peer].addr, &handleStorage[peer].connectAddr);
    EXPECT_EQ(handshake.outgoingConnects[peer].magic, handleStorage[peer].magic);
  }
  for (int peer = 0; peer < nranks; ++peer) {
    EXPECT_NE(comm->peerSender[peer].buffer, nullptr);
    EXPECT_NE(comm->peerReceiver[peer].buffer, nullptr);
    EXPECT_EQ(comm->peerSender[peer].lastCompletedRequestGlobalId, UINT64_MAX);
    EXPECT_EQ(comm->peerReceiver[peer].lastCompletedRequestGlobalId, UINT64_MAX);
  }

  EXPECT_EQ(ncclRmaSocketProxyCloseColl(comm), ncclSuccess);
  EXPECT_EQ(handshake.close.calls, nranks * 2);
}

TEST_F(RmaSocketSetupMicrotest, Connect_MissingPeerHandleReturnsErrorAndLeavesOutputNull) {
  ncclSocket listen{};
  void* handles[] = {nullptr};
  void* opaque = reinterpret_cast<void*>(0x1);
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });

  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, 0, &listen, &opaque), ncclInvalidArgument);
  EXPECT_EQ(opaque, nullptr);
  EXPECT_EQ(close.calls, 2);
}

TEST_F(RmaSocketSetupMicrotest, Connect_InvalidAcceptedRankReturnsErrorAndCleansPartialState) {
  ncclSocket listen{};
  ncclRmaSocketProxyHandle handle{};
  void* handles[] = {&handle};
  void* opaque = reinterpret_cast<void*>(0x1);
  ScopedConnectHandshake handshake({1});

  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles, 1, 0, &listen, &opaque), ncclInternalError);
  EXPECT_EQ(opaque, nullptr);
  EXPECT_EQ(handshake.move.calls, 0);
  EXPECT_GE(handshake.close.calls, 3);
}

TEST_F(RmaSocketSetupMicrotest, Connect_DuplicateAcceptedRankReturnsErrorAndCleansPartialState) {
  constexpr int nranks = 2;
  ncclSocket listen{};
  std::array<ncclRmaSocketProxyHandle, nranks> handleStorage{};
  std::array<void*, nranks> handles{{&handleStorage[0], &handleStorage[1]}};
  void* opaque = reinterpret_cast<void*>(0x1);
  ScopedConnectHandshake handshake({0, 0}, /*firstMagic=*/99);

  EXPECT_EQ(ncclRmaSocketProxyConnect(nullptr, handles.data(), nranks, 0, &listen, &opaque), ncclInternalError);
  EXPECT_EQ(opaque, nullptr);
  EXPECT_EQ(handshake.move.calls, 1);
  EXPECT_EQ(handshake.close.calls, nranks * 2 + 1);
}

TEST_F(RmaSocketSetupMicrotest, CreateContext_InvalidConfigurationIsRejected) {
  ncclRmaSocketProxyCollComm comm{};
  comm.nranks = 4;
  ncclRmaConfig_t config{};
  void* ctx = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyCreateContext(nullptr, &config, &ctx), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, nullptr, &ctx), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, nullptr), ncclInvalidArgument);
  config.nContexts = 0;
  config.rankStride = 1;
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, &ctx), ncclInvalidArgument);
  config.nContexts = 1;
  config.rankStride = 0;
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, &ctx), ncclInvalidArgument);
  config.rankStride = 3;
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, &ctx), ncclInvalidArgument);
  config.rankStride = 1;
  comm.nContexts = INT_MAX;
  EXPECT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, &ctx), ncclInvalidArgument);
}

TEST_F(RmaSocketSetupMicrotest, CreateAndDestroyContext_PublishesRequestsAndUpdatesLiveCount) {
  ncclRmaSocketProxyCollComm comm{};
  comm.nranks = 2;
  comm.peerSender = static_cast<ncclRmaSocketProxyPeerSender*>(
      std::calloc(comm.nranks, sizeof(ncclRmaSocketProxyPeerSender)));
  ASSERT_NE(comm.peerSender, nullptr);
  ncclRmaConfig_t config{};
  config.nContexts = 2;
  config.rankStride = 1;
  void* opaque = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyCreateContext(&comm, &config, &opaque), ncclSuccess);
  auto* contexts = static_cast<ncclRmaSocketProxyCtx*>(opaque);
  ASSERT_NE(contexts, nullptr);
  EXPECT_EQ(comm.nContexts, 2);
  for (int c = 0; c < config.nContexts; ++c) {
    EXPECT_EQ(contexts[c].collComm, &comm);
    EXPECT_EQ(contexts[c].nContexts, config.nContexts);
    EXPECT_EQ(contexts[c].requestCount, NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER * comm.nranks);
    EXPECT_NE(contexts[c].requests, nullptr);
  }
  for (int peer = 0; peer < comm.nranks; ++peer) {
    EXPECT_EQ(comm.peerSender[peer].capacity, 2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER * config.nContexts);
  }

  EXPECT_EQ(ncclRmaSocketProxyDestroyContext(contexts), ncclSuccess);
  EXPECT_EQ(comm.nContexts, 0);
  for (int peer = 0; peer < comm.nranks; ++peer) std::free(comm.peerSender[peer].tasks);
  std::free(comm.peerSender);
}

TEST_F(RmaSocketSetupMicrotest, CreateContext_SuccessiveBatchesGrowCapacityFromCumulativeContextCount) {
  ncclRmaSocketProxyCollComm comm{};
  comm.nranks = 1;
  comm.peerSender =
      static_cast<ncclRmaSocketProxyPeerSender*>(std::calloc(1, sizeof(ncclRmaSocketProxyPeerSender)));
  ASSERT_NE(comm.peerSender, nullptr);
  ncclRmaConfig_t firstConfig{};
  firstConfig.nContexts = 1;
  firstConfig.rankStride = 1;
  ncclRmaConfig_t secondConfig{};
  secondConfig.nContexts = 2;
  secondConfig.rankStride = 1;
  void* first = nullptr;
  void* second = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyCreateContext(&comm, &firstConfig, &first), ncclSuccess);
  EXPECT_EQ(comm.nContexts, 1);
  EXPECT_EQ(comm.peerSender[0].capacity, 2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER);
  ASSERT_EQ(ncclRmaSocketProxyCreateContext(&comm, &secondConfig, &second), ncclSuccess);
  EXPECT_EQ(comm.nContexts, 3);
  EXPECT_EQ(comm.peerSender[0].capacity, 2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER * 3);

  EXPECT_EQ(ncclRmaSocketProxyDestroyContext(second), ncclSuccess);
  EXPECT_EQ(comm.nContexts, 1);
  EXPECT_EQ(ncclRmaSocketProxyDestroyContext(first), ncclSuccess);
  EXPECT_EQ(comm.nContexts, 0);
  std::free(comm.peerSender[0].tasks);
  std::free(comm.peerSender);
}

TEST_F(RmaSocketSetupMicrotest, DestroyContext_ActiveOperationKeepsContextLive) {
  ncclRmaSocketProxyCollComm comm{};
  comm.nContexts = 1;
  auto* ctx = static_cast<ncclRmaSocketProxyCtx*>(std::calloc(1, sizeof(ncclRmaSocketProxyCtx)));
  ASSERT_NE(ctx, nullptr);
  ctx->collComm = &comm;
  ctx->nContexts = 1;
  ctx->activeOps = 1;
  ctx->requests = static_cast<ncclRmaSocketProxyRequest*>(std::calloc(1, sizeof(ncclRmaSocketProxyRequest)));
  ASSERT_NE(ctx->requests, nullptr);

  const ncclResult_t result = ncclRmaSocketProxyDestroyContext(ctx);

  EXPECT_EQ(result, ncclInvalidUsage);
  EXPECT_EQ(comm.nContexts, 1);
  if (result != ncclSuccess) {
    std::free(ctx->requests);
    std::free(ctx);
  }
}

TEST_F(RmaSocketSetupMicrotest, DestroyContext_InvalidOwnershipIsRejected) {
  EXPECT_EQ(ncclRmaSocketProxyDestroyContext(nullptr), ncclSuccess);

  auto* orphan = static_cast<ncclRmaSocketProxyCtx*>(std::calloc(1, sizeof(ncclRmaSocketProxyCtx)));
  ASSERT_NE(orphan, nullptr);
  orphan->nContexts = 1;
  const ncclResult_t orphanResult = ncclRmaSocketProxyDestroyContext(orphan);
  EXPECT_EQ(orphanResult, ncclInvalidUsage);
  if (orphanResult != ncclSuccess) std::free(orphan);

  ncclRmaSocketProxyCollComm comm{};
  comm.nContexts = 1;
  auto* empty = static_cast<ncclRmaSocketProxyCtx*>(std::calloc(1, sizeof(ncclRmaSocketProxyCtx)));
  ASSERT_NE(empty, nullptr);
  empty->collComm = &comm;
  const ncclResult_t emptyResult = ncclRmaSocketProxyDestroyContext(empty);
  EXPECT_EQ(emptyResult, ncclInvalidUsage);
  EXPECT_EQ(comm.nContexts, 1);
  if (emptyResult != ncclSuccess) std::free(empty);

  auto* contexts = static_cast<ncclRmaSocketProxyCtx*>(std::calloc(2, sizeof(ncclRmaSocketProxyCtx)));
  ASSERT_NE(contexts, nullptr);
  for (int c = 0; c < 2; ++c) {
    contexts[c].collComm = &comm;
    contexts[c].nContexts = 2;
  }
  comm.nContexts = 1;
  const ncclResult_t countResult = ncclRmaSocketProxyDestroyContext(contexts);
  EXPECT_EQ(countResult, ncclInvalidUsage);
  if (countResult != ncclSuccess) std::free(contexts);
}

TEST_F(RmaSocketSetupMicrotest, CloseColl_LiveResourcesOrOperationsAreRejected) {
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  auto expectRejected = [](auto configure) {
    auto* comm = AllocateOwnedComm(2);
    ASSERT_NE(comm, nullptr);
    ncclRmaSocketProxyMrHandle mr{};
    configure(comm, &mr);

    const ncclResult_t result = ncclRmaSocketProxyCloseColl(comm);

    EXPECT_EQ(result, ncclInvalidUsage);
    if (result != ncclSuccess) FreeRejectedComm(comm);
  };

  expectRejected([](auto* comm, auto*) { comm->nContexts = 1; });
  expectRejected([](auto* comm, auto* mr) { comm->mrHead = mr; });
  expectRejected([](auto* comm, auto*) { comm->peerSender[0].count = 1; });
  expectRejected([](auto* comm, auto*) { comm->peerReceiver[0].recvState = ncclRmaSocketProxyRecvStatePayload; });
  expectRejected([](auto* comm, auto*) { comm->peerReceiver[0].controlMsgOffset = 1; });
}

TEST_F(RmaSocketSetupMicrotest, CloseColl_QuiescentCommClosesPeerSocketsAndFreesComm) {
  auto* comm = AllocateOwnedComm(2);
  ASSERT_NE(comm, nullptr);
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool wait) {
    EXPECT_FALSE(wait);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyCloseColl(comm), ncclSuccess);
  EXPECT_EQ(close.calls, 4);
  EXPECT_EQ(ncclRmaSocketProxyCloseColl(nullptr), ncclSuccess);
}

TEST_F(RmaSocketSetupMicrotest, CloseColl_SocketCloseFailureStillClosesEveryPeerAndReturnsError) {
  auto* comm = AllocateOwnedComm(2);
  ASSERT_NE(comm, nullptr);
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaSocketProxyCloseColl(comm), ncclSystemError);
  EXPECT_EQ(close.calls, 4);
}

}  // namespace
