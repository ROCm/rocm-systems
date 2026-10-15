/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// End-to-end communicator lifecycle with TLS on RCCL's bootstrap and proxy sockets
// (ncclSetEncryption). Communicator init runs the bootstrap ring connect and the
// (bidirectional, by default) ring AllGather; ncclCommSplit runs the blocking ring
// connect of bootstrapSplit; P2P connection setup runs asynchronous bootstrapSend.
// Run these under both NCCL_BOOTSTRAP_BIDIR_ALLGATHER=0 and =1 (see the bootstrap_tls
// categories in test_categories_mpi.yaml).

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "TestChecks.hpp"

#include <hip/hip_runtime.h>
#include <vector>

namespace {

constexpr const char* kPsk = "rccl-bootstrap-tls-mpi-test-key-0123456789";

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
constexpr bool kTlsBuild = true;
#else
constexpr bool kTlsBuild = false;
#endif

ncclResult_t SetPsk(const char* psk) {
  ncclEncryptionConfig_t cfg = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  cfg.mode = NCCL_ENCRYPTION_MODE_PSK;
  cfg.psk = psk;
  return ncclSetEncryption(&cfg);
}

// Sum-AllReduce of (rank + 1) on `comm`, checked on the host.
::testing::AssertionResult AllReduceRankSum(ncclComm_t comm, hipStream_t stream) {
  int rank = 0, nranks = 0;
  if (ncclCommUserRank(comm, &rank) != ncclSuccess || ncclCommCount(comm, &nranks) != ncclSuccess)
    return ::testing::AssertionFailure() << "comm query failed";
  constexpr size_t kCount = 1 << 16;
  std::vector<float> host(kCount, (float)(rank + 1));
  float* dev = nullptr;
  if (hipMalloc(&dev, kCount * sizeof(float)) != hipSuccess) return ::testing::AssertionFailure() << "hipMalloc";
  ncclResult_t res = ncclSuccess;
  bool ok = hipMemcpy(dev, host.data(), kCount * sizeof(float), hipMemcpyHostToDevice) == hipSuccess &&
            (res = ncclAllReduce(dev, dev, kCount, ncclFloat, ncclSum, comm, stream)) == ncclSuccess &&
            hipStreamSynchronize(stream) == hipSuccess &&
            hipMemcpy(host.data(), dev, kCount * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
  (void)hipFree(dev);
  if (!ok) return ::testing::AssertionFailure() << "AllReduce failed: " << ncclGetErrorString(res);
  const float expected = nranks * (nranks + 1) / 2.0f;
  for (size_t i = 0; i < kCount; i++)
    if (host[i] != expected)
      return ::testing::AssertionFailure() << "element " << i << " = " << host[i] << ", expected " << expected;
  return ::testing::AssertionSuccess();
}

// Each rank sends its rank id to the next rank and receives from the previous one.
::testing::AssertionResult RingSendRecv(ncclComm_t comm, hipStream_t stream) {
  int rank = 0, nranks = 0;
  if (ncclCommUserRank(comm, &rank) != ncclSuccess || ncclCommCount(comm, &nranks) != ncclSuccess)
    return ::testing::AssertionFailure() << "comm query failed";
  constexpr size_t kCount = 1 << 14;
  std::vector<int> host(kCount, rank);
  int *sendBuf = nullptr, *recvBuf = nullptr;
  if (hipMalloc(&sendBuf, kCount * sizeof(int)) != hipSuccess || hipMalloc(&recvBuf, kCount * sizeof(int)) != hipSuccess)
    return ::testing::AssertionFailure() << "hipMalloc";
  bool ok = hipMemcpy(sendBuf, host.data(), kCount * sizeof(int), hipMemcpyHostToDevice) == hipSuccess &&
            ncclGroupStart() == ncclSuccess;
  if (ok) {
    // ncclGroupEnd must run even if an enqueue fails, or the group stays open for later calls.
    bool enqueued = ncclSend(sendBuf, kCount, ncclInt32, (rank + 1) % nranks, comm, stream) == ncclSuccess &&
                    ncclRecv(recvBuf, kCount, ncclInt32, (rank + nranks - 1) % nranks, comm, stream) == ncclSuccess;
    ok = ncclGroupEnd() == ncclSuccess && enqueued && hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(host.data(), recvBuf, kCount * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess;
  }
  (void)hipFree(sendBuf);
  (void)hipFree(recvBuf);
  if (!ok) return ::testing::AssertionFailure() << "send/recv failed";
  const int expected = (rank + nranks - 1) % nranks;
  for (size_t i = 0; i < kCount; i++)
    if (host[i] != expected)
      return ::testing::AssertionFailure() << "element " << i << " = " << host[i] << ", expected " << expected;
  return ::testing::AssertionSuccess();
}

bool AllRanksTrue(bool local) {
  int v = local ? 1 : 0, all = 0;
  MPI_Allreduce(&v, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return all != 0;
}

} // namespace

class BootstrapTlsMPITest : public MPITestBase {
protected:
  void TearDown() override {
    MPITestBase::TearDown();
    (void)ncclSetEncryption(nullptr);
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

TEST_F(BootstrapTlsMPITest, EncryptedInitCollectivesSplitSendRecv) {
  SKIP_UNLESS_MPI_PREREQS(2);
  if (!kTlsBuild) GTEST_SKIP() << "RCCL was built without TLS_BACKEND=OPENSSL3";
  ASSERT_EQ(ncclSuccess, SetPsk(kPsk));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  EXPECT_TRUE(AllReduceRankSum(comm, stream));
  EXPECT_TRUE(RingSendRecv(comm, stream));

  ncclComm_t sub = nullptr;
  int rank = MPIEnvironment::world_rank;
  ASSERT_EQ(ncclSuccess, ncclCommSplit(comm, rank % 2, rank, &sub, nullptr));
  int subCount = 0;
  ASSERT_EQ(ncclSuccess, ncclCommCount(sub, &subCount));
  EXPECT_EQ((MPIEnvironment::world_size + 1 - rank % 2) / 2, subCount);
  if (subCount > 1) {
    EXPECT_TRUE(AllReduceRankSum(sub, stream));
    EXPECT_TRUE(RingSendRecv(sub, stream));
  }
  EXPECT_EQ(ncclSuccess, ncclCommDestroy(sub));
  EXPECT_TRUE(AllRanksTrue(!HasFailure())) << "failure on another rank";
}

// Encryption can be enabled between communicators: a plaintext communicator created
// before ncclSetEncryption keeps working next to an encrypted one created after it.
TEST_F(BootstrapTlsMPITest, EnablingEncryptionBetweenCommunicators) {
  SKIP_UNLESS_MPI_PREREQS(2);
  if (!kTlsBuild) GTEST_SKIP() << "RCCL was built without TLS_BACKEND=OPENSSL3";
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t plain = getActiveCommunicator();
  hipStream_t stream = getActiveStream();
  EXPECT_TRUE(AllReduceRankSum(plain, stream));

  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(ncclSuccess, SetPsk(kPsk));
  ncclUniqueId id;
  int rank = MPIEnvironment::world_rank, nranks = MPIEnvironment::world_size;
  int idRes = ncclSuccess;
  if (rank == 0) idRes = ncclGetUniqueId(&id);
  MPI_Bcast(&idRes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  ASSERT_EQ(ncclSuccess, idRes) << "ncclGetUniqueId failed on rank 0";
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t encrypted = nullptr;
  ASSERT_EQ(ncclSuccess, ncclCommInitRank(&encrypted, nranks, id, rank));
  EXPECT_TRUE(AllReduceRankSum(encrypted, stream));
  EXPECT_TRUE(AllReduceRankSum(plain, stream));
  EXPECT_TRUE(RingSendRecv(encrypted, stream));
  EXPECT_EQ(ncclSuccess, ncclCommDestroy(encrypted));
  EXPECT_TRUE(AllRanksTrue(!HasFailure())) << "failure on another rank";
}

// Without TLS support, a configured key makes communicator creation fail on every rank
// rather than fall back to plaintext or hang.
TEST_F(BootstrapTlsMPITest, KeyWithoutTlsSupportFailsInit) {
  SKIP_UNLESS_MPI_PREREQS(2);
  if (kTlsBuild) GTEST_SKIP() << "covers builds without TLS_BACKEND=OPENSSL3";
  ASSERT_EQ(ncclSuccess, SetPsk(kPsk));
  int rank = MPIEnvironment::world_rank, nranks = MPIEnvironment::world_size;
  ncclUniqueId id = {};
  int idRes = ncclSuccess;
  if (rank == 0) idRes = ncclGetUniqueId(&id);
  MPI_Bcast(&idRes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (idRes != ncclSuccess) {
    if (rank == 0) EXPECT_EQ(ncclInvalidUsage, idRes);
    return;
  }
  ncclComm_t comm = nullptr;
  ncclResult_t res = ncclCommInitRank(&comm, nranks, id, rank);
  EXPECT_NE(ncclSuccess, res) << "communicator came up in plaintext although a key was configured";
  if (res == ncclSuccess) (void)ncclCommDestroy(comm);
}

#endif // MPI_TESTS_ENABLED
