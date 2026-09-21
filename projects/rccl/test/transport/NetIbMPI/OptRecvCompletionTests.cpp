/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Optional recv completion is process-lifetime (NCCL_PARAM cache). Each case
// lives in its own net_ib_transport.json config; skip if that env is absent.

#define OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP()                                                       \
  do {                                                                                             \
    const char* _opt = getenv("RCCL_IB_OPT_RECV_COMPLETION");                                      \
    if (!_opt || strcmp(_opt, "1") != 0) {                                                         \
      GTEST_SKIP() << "Requires RCCL_IB_OPT_RECV_COMPLETION=1 "                                     \
                      "(use cast_opt_recv in net_ib_transport.json)";                              \
    }                                                                                              \
    const char* _sched = getenv("RCCL_IB_QP_SCHED_ENABLE");                                        \
    if (!_sched || strcmp(_sched, "0") != 0) {                                                     \
      GTEST_SKIP() << "Requires RCCL_IB_QP_SCHED_ENABLE=0 "                                         \
                      "(QP scheduling forces optRecvCompletion off)";                              \
    }                                                                                              \
  } while (0)

// =============================================================================
// OptRecvCompletionFlagEnabled
// Control on, QP sched off: commBase.optRecvCompletion must be 1 after connect.
// =============================================================================
TEST_F(NetIbMPITest, OptRecvCompletionFlagEnabled) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  SetupConnectionWithGuard(0, pair, connGuard);

  const int rank = MPIEnvironment::world_rank;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  int flag = 0;
  ASSERT_EQ(ncclIbCastGetOptRecvCompletion(comm, &flag), ncclSuccess);
  EXPECT_EQ(flag, 1) << "optRecvCompletion should be enabled (control=1, qpSched=0)";
}

// =============================================================================
// OptRecvCompletionFlagDisabledByQpSched
// cast_base enables the WRR scheduler, which must force the flag off.
// =============================================================================
TEST_F(NetIbMPITest, OptRecvCompletionFlagDisabledByQpSched) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  CAST_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  void* listenComm = nullptr;
  void* sendComm = nullptr;
  void* recvComm = nullptr;
  SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

  const int rank = MPIEnvironment::world_rank;
  void* comm = (rank == 0) ? recvComm : sendComm;
  int flag = 1;
  EXPECT_EQ(ncclIbCastGetOptRecvCompletion(comm, &flag), ncclSuccess);
  EXPECT_EQ(flag, 0) << "QP scheduling must disable optRecvCompletion";

  // EXPECT_ so a close failure cannot skip the barrier and hang the peer.
  if (rank == 0) {
    EXPECT_EQ(CloseRecvComm(recvComm), ncclSuccess);
    EXPECT_EQ(CloseListenComm(listenComm), ncclSuccess);
  } else {
    EXPECT_EQ(CloseSendComm(sendComm), ncclSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// OptRecvCompletionSkipWritePath
// 1-req send/recv with the NCCL_NET_OPTIONAL_RECV_COMPLETION hint. Recv may
// complete as soon as test() runs (no RQ WQE); wait for the send CQE first so
// host-buffer data is visible before verify.
// =============================================================================
TEST_F(NetIbMPITest, OptRecvCompletionSkipWritePath) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  const int rank = MPIEnvironment::world_rank;
  const int senderRank = 1;

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  SetupConnectionWithGuard(0, pair, connGuard);

  const size_t bufferSize = kSmallBufferSize;
  const int tag = 77;

  void* buffer = malloc(bufferSize);
  ASSERT_NE(buffer, nullptr);
  auto bufferGuard = makeHostBufferAutoGuard(buffer);

  void* mhandle = nullptr;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

  void* request = nullptr;
  if (rank == 0) {
    PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  } else {
    fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank));
    PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  }

  // Barrier first (same as SimpleSendRecv). ASSERT_ before this would abort one
  // rank while the peer waits here.
  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_NE(request, nullptr) << "Request must be non-NULL before waiting";
  EXPECT_NE(request, (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION);

  int sizes[1] = {0};
  // Send CQE is the data-visibility fence on the WRITE path. EXPECT_ so a
  // failed wait still reaches the second barrier.
  if (rank == 1 && request && request != (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0 && request && request != (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
    EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(senderRank)))
        << "Data validation failed on optional-recv WRITE path";
  }
}

// =============================================================================
// OptRecvCompletionMultiRecvNoHint
// n>1 must not use the optional-recv hint (IbCastCanSkipRecvCompletion returns ncclInternalError).
// One 4-wide irecv, matched by four isends. The sends can only be posted as a
// group: IbCastMultiSend fires once every slot of the multi-recv is filled, so
// waiting on any one send before the rest are posted would hang.
// =============================================================================
TEST_F(NetIbMPITest, OptRecvCompletionMultiRecvNoHint) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  const int rank = MPIEnvironment::world_rank;
  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  SetupConnectionWithGuard(0, pair, connGuard);

  constexpr int kN = 4;
  size_t sizes[kN] = {64, 1024, 4096, 65536};
  int tags[kN] = {201, 202, 203, 204};
  void* bufs[kN] = {};
  void* mhandles[kN] = {};

  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  auto cleanup = makeScopeGuard([&]() {
    for (int i = 0; i < kN; i++) {
      if (mhandles[i]) { DeregisterMemory(comm, mhandles[i]); mhandles[i] = nullptr; }
      if (bufs[i]) { free(bufs[i]); bufs[i] = nullptr; }
    }
  });

  for (int i = 0; i < kN; i++) {
    bufs[i] = malloc(sizes[i]);
    ASSERT_NE(bufs[i], nullptr) << "malloc failed for slot " << i;
    if (rank == 0) memset(bufs[i], 0xCC, sizes[i]);
    else fillHostBufferWithPattern<uint8_t>(bufs[i], sizes[i], makeBytePattern(tags[i]));
    ASSERT_EQ(RegisterMemory(comm, bufs[i], sizes[i], NCCL_PTR_HOST, &mhandles[i]), ncclSuccess)
        << "RegisterMemory failed for slot " << i;
  }

  if (rank == 0) {
    void* request = nullptr;
    // No hint: n>1 must never take the skip path, even with the flag enabled.
    EXPECT_EQ(PostRecv(pair.recvComm, kN, bufs, sizes, tags, mhandles, &request), ncclSuccess);
    EXPECT_NE(request, nullptr);

    MPI_Barrier(MPI_COMM_WORLD);

    int recvSizes[kN] = {};
    if (request) {
      EXPECT_EQ(WaitForCompletion(request, recvSizes), ncclSuccess);
      for (int i = 0; i < kN; i++) {
        EXPECT_EQ(recvSizes[i], static_cast<int>(sizes[i])) << "recv size mismatch at slot " << i;
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(bufs[i], sizes[i], makeBytePattern(tags[i])))
            << "data mismatch at slot " << i;
      }
    }
  } else {
    MPI_Barrier(MPI_COMM_WORLD);

    void* reqs[kN] = {};
    for (int i = 0; i < kN; i++) {
      PostSendWithRetry(pair.sendComm, bufs[i], sizes[i], tags[i], mhandles[i], &reqs[i]);
      EXPECT_NE(reqs[i], nullptr) << "PostSend returned null request for slot " << i;
    }
    for (int i = 0; i < kN; i++) {
      if (!reqs[i]) continue;
      int sentSize[1] = {0};
      EXPECT_EQ(WaitForCompletion(reqs[i], sentSize), ncclSuccess)
          << "send completion failed for slot " << i;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
