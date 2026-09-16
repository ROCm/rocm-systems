/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

// optRecvCompletion is 1 when the control is on and QP sched is off.
TEST_F(NetIbMPITest, OptRecvCompletionFlagEnabled) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  ASSERT_SETUP_CONNECTION(0, pair, connGuard);

  const int rank = MPIEnvironment::world_rank;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  int flag = 0;
  ASSERT_EQ(ncclIbCastGetOptRecvCompletion(comm, &flag), ncclSuccess);
  EXPECT_EQ(flag, 1) << "optRecvCompletion should be enabled (control=1, qpSched=0)";
}

// QP scheduling forces optRecvCompletion off.
TEST_F(NetIbMPITest, OptRecvCompletionFlagDisabledByQpSched) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  CAST_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  ASSERT_SETUP_CONNECTION(0, pair, connGuard);

  const int rank = MPIEnvironment::world_rank;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  int flag = 1;
  ASSERT_EQ(ncclIbCastGetOptRecvCompletion(comm, &flag), ncclSuccess);
  EXPECT_EQ(flag, 0) << "QP scheduling must disable optRecvCompletion";
}

// With the gate off, a hinted transfer reports the posted size.
TEST_F(NetIbMPITest, OptRecvCompletionDisabledKeepsRecvSize) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  CAST_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  const int rank = MPIEnvironment::world_rank;
  const int senderRank = 1;

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  ASSERT_SETUP_CONNECTION(0, pair, connGuard);

  const size_t bufferSize = kSmallBufferSize;
  const int tag = 78;

  void* buffer = malloc(bufferSize);
  EXPECT_NE(buffer, nullptr) << "malloc failed";
  auto bufferGuard = makeHostBufferAutoGuard(buffer);

  void* mhandle = nullptr;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  ncclResult_t regRc = ncclSystemError;
  if (buffer) regRc = RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle);
  EXPECT_EQ(regRc, ncclSuccess) << "RegisterMemory failed";
  NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

  const bool ready = buffer != nullptr && regRc == ncclSuccess;
  int notReady = ready ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &notReady, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  void* request = nullptr;
  if (notReady) {
    ADD_FAILURE() << "buffer setup failed on at least one rank";
  } else if (rank == 0) {
    PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  } else {
    fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank));
    PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int sizes[1] = {0};
  if (rank == 1 && IsRealRequest(request)) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0 && IsRealRequest(request)) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
    EXPECT_EQ(sizes[0], static_cast<int>(bufferSize))
        << "disabled optRecvCompletion must report the posted size, not the skip path's 0";
    EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(senderRank)));
  }
}

// Hinted 1-req transfer skips the receiver completion.
TEST_F(NetIbMPITest, OptRecvCompletionSkipWritePath) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();

  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  const int rank = MPIEnvironment::world_rank;
  const int senderRank = 1;

  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  ASSERT_SETUP_CONNECTION(0, pair, connGuard);

  const size_t bufferSize = kSmallBufferSize;
  const int tag = 77;

  void* buffer = malloc(bufferSize);
  EXPECT_NE(buffer, nullptr) << "malloc failed";
  auto bufferGuard = makeHostBufferAutoGuard(buffer);

  void* mhandle = nullptr;
  void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
  ncclResult_t regRc = ncclSystemError;
  if (buffer) regRc = RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle);
  EXPECT_EQ(regRc, ncclSuccess) << "RegisterMemory failed";
  NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

  // Both ranks must agree before posting, so one failure cannot leave the other in a barrier.
  const bool ready = buffer != nullptr && regRc == ncclSuccess;
  int notReady = ready ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &notReady, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  void* request = nullptr;
  if (notReady) {
    ADD_FAILURE() << "buffer setup failed on at least one rank";
  } else if (rank == 0) {
    PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  } else {
    fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank));
    PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request, /*optRecvHint=*/true);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (!notReady) {
    EXPECT_TRUE(IsRealRequest(request)) << "Request must be a real handle before waiting";
  }

  int sizes[1] = {0};
  if (rank == 1 && IsRealRequest(request)) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0 && IsRealRequest(request)) {
    EXPECT_EQ(WaitForCompletion(request, sizes), ncclSuccess);
    EXPECT_EQ(sizes[0], 0) << "optional-recv skip path must report recv size 0";
    EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(senderRank)))
        << "Data validation failed on optional-recv WRITE path";
  }
}

void NetIbMPITest::OptRecvCompletionRunMultiRecv(bool optRecvHint) {
  net_ = &netIbCast;
  AssertInitAndGetDevices(nullptr);

  const int rank = MPIEnvironment::world_rank;
  ConnectionPair pair;
  NetConnectionGuard connGuard(net_);
  ASSERT_SETUP_CONNECTION(0, pair, connGuard);

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

  int setupFailed = 0;
  for (int i = 0; i < kN && !setupFailed; i++) {
    bufs[i] = malloc(sizes[i]);
    if (!bufs[i]) {
      ADD_FAILURE() << "malloc failed for slot " << i;
      setupFailed = 1;
      break;
    }
    if (rank == 0) memset(bufs[i], 0xCC, sizes[i]);
    else fillHostBufferWithPattern<uint8_t>(bufs[i], sizes[i], makeBytePattern(tags[i]));
    if (RegisterMemory(comm, bufs[i], sizes[i], NCCL_PTR_HOST, &mhandles[i]) != ncclSuccess) {
      ADD_FAILURE() << "RegisterMemory failed for slot " << i;
      setupFailed = 1;
    }
  }
  // Both ranks return together if setup fails.
  MPI_Allreduce(MPI_IN_PLACE, &setupFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (setupFailed) {
    ADD_FAILURE() << "buffer setup failed on at least one rank";
    return;
  }

  if (rank == 0) {
    void* request = nullptr;
    EXPECT_EQ(PostRecv(pair.recvComm, kN, bufs, sizes, tags, mhandles, &request, optRecvHint), ncclSuccess);
    EXPECT_TRUE(IsRealRequest(request));

    MPI_Barrier(MPI_COMM_WORLD);

    int recvSizes[kN] = {};
    if (IsRealRequest(request)) {
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
      PostSendWithRetry(pair.sendComm, bufs[i], sizes[i], tags[i], mhandles[i], &reqs[i], optRecvHint);
      EXPECT_TRUE(IsRealRequest(reqs[i])) << "PostSend returned null request for slot " << i;
    }
    for (int i = 0; i < kN; i++) {
      if (!IsRealRequest(reqs[i])) continue;
      int sentSize[1] = {0};
      EXPECT_EQ(WaitForCompletion(reqs[i], sentSize), ncclSuccess)
          << "send completion failed for slot " << i;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// n>1 without the hint still completes.
TEST_F(NetIbMPITest, OptRecvCompletionMultiRecvNoHint) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();
  OptRecvCompletionRunMultiRecv(/*optRecvHint=*/false);
}

// n>1 with the hint still completes.
TEST_F(NetIbMPITest, OptRecvCompletionMultiRecvWithHint) {
  SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
  OPT_RECV_ENABLED_ENV_CHECK_OR_SKIP();
  OptRecvCompletionRunMultiRecv(/*optRecvHint=*/true);
}

#endif // MPI_TESTS_ENABLED
