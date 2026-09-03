/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// MPI tests for the new NCCL device compute APIs ported into RCCL:
//   Copy       (ncclLocalCopy / ncclLsaCopy)
//   ReduceSum  (ncclLocalReduceSum / ncclLsaReduceSum)
//   ReduceCopy (ncclLocalReduceSumCopy / ncclLsaReduceSumCopy)

#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __CUDACC_EXTENDED_LAMBDA__ 1
#endif

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "SymmetricMemPrereq.hpp"
#include "TestChecks.hpp"

#include "nccl_device.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <string>
#include <type_traits>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLTestGuards;

namespace {

// Symmetric memory (cuMem) is required for window registration used by the LSA
// paths; the Local paths only need plain device memory but we keep a single
// gate for simplicity of the suite. Match RCCL NCCL_PARAM parsing (strtoll).
std::string cuMemReason() { return symmetricMemEnvAndRuntimeSkipReason(); }

// Number of MPI ranks co-located on this rank's node.
int nodeLocalRanks() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int n = 0;
  MPI_Comm_size(nodeComm, &n);
  MPI_Comm_free(&nodeComm);
  return n;
}

// CTA size used for all kernels. Wave64 friendly.
constexpr int kThreads = 256;

// Count sweep exercises the 16B/4B/scalar vectorization tiers plus the aligned
// bulk vs scalar-remainder split in reduce_copy__impl.h.
const std::vector<size_t>& countSweep() {
  static const std::vector<size_t> v = {1, 3, 15, 16, 17, 1024, 1025};
  return v;
}

int lsaBaseRank(int rank, int lsaSize) {
  return (rank / lsaSize) * lsaSize;
}

}  // namespace

// ===========================================================================
// Fixture
// ===========================================================================

class DeviceApiMPITests : public MPITestBase {};

// ===========================================================================
// Local Copy: ncclLocalCopy (1 source -> N strided local destinations)
// ===========================================================================

__global__ void localCopyKernel(float* src, float* dstBase, int nDst, size_t displ, size_t count) {
  ncclLocalCopy<float>(ncclCoopCta(), src, nDst, dstBase, displ, count);
}

TEST_F(DeviceApiMPITests, Local_Copy) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  constexpr int nDst = 4;
  for (size_t count : countSweep()) {
    const size_t displElts = count + 8;  // pad so strided dsts do not overlap
    std::vector<float> hSrc(count);
    for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>(i % 97) + 0.5f;

    float* dSrc = nullptr;
    float* dDst = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, count * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, nDst * displElts * sizeof(float)));
    auto guard = makeScopeGuard([&]() {
      if (dSrc) (void)hipFree(dSrc);
      if (dDst) (void)hipFree(dDst);
    });
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), count * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, nDst * displElts * sizeof(float)));

    localCopyKernel<<<1, kThreads, 0, stream>>>(dSrc, dDst, nDst, displElts, count);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<float> hDst(nDst * displElts);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(hDst.data(), dDst, nDst * displElts * sizeof(float), hipMemcpyDeviceToHost));
    for (int d = 0; d < nDst; ++d)
      for (size_t i = 0; i < count; ++i)
        ASSERT_EQ(hSrc[i], hDst[d * displElts + i])
            << "count=" << count << " dst=" << d << " elt=" << i;
  }
}

// ===========================================================================
// Local ReduceSum: ncclLocalReduceSum (N strided sources -> 1 destination)
// ===========================================================================

__global__ void localReduceSumKernel(int nSrc, float* srcBase, size_t displ, float* dst, size_t count) {
  ncclLocalReduceSum<float>(ncclCoopCta(), nSrc, srcBase, displ, dst, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSum) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  // nSrc sweep hits the nSrc%4 unrolled path (4) plus remainder paths (1,2,5).
  for (int nSrc : {1, 2, 4, 5}) {
    for (size_t count : countSweep()) {
      const size_t displElts = count + 8;
      std::vector<float> hSrc(nSrc * displElts, 0.0f);
      for (int s = 0; s < nSrc; ++s)
        for (size_t i = 0; i < count; ++i)
          hSrc[s * displElts + i] = static_cast<float>((s + 1) * (i % 13));

      float* dSrc = nullptr;
      float* dDst = nullptr;
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, nSrc * displElts * sizeof(float)));
      ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, count * sizeof(float)));
      auto guard = makeScopeGuard([&]() {
        if (dSrc) (void)hipFree(dSrc);
        if (dDst) (void)hipFree(dDst);
      });
      ASSERT_MPI_EQ(hipSuccess,
                    hipMemcpy(dSrc, hSrc.data(), nSrc * displElts * sizeof(float), hipMemcpyHostToDevice));
      ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, count * sizeof(float)));

      localReduceSumKernel<<<1, kThreads, 0, stream>>>(nSrc, dSrc, displElts, dDst, count);
      ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

      std::vector<float> hDst(count);
      ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, count * sizeof(float), hipMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) {
        float expected = 0.0f;
        for (int s = 0; s < nSrc; ++s) expected += hSrc[s * displElts + i];
        ASSERT_FLOAT_EQ(expected, hDst[i]) << "nSrc=" << nSrc << " count=" << count << " elt=" << i;
      }
    }
  }
}

// ===========================================================================
// Local ReduceSumCopy: ncclLocalReduceSumCopy (N sources -> M destinations,
// each destination receives the full reduction)
// ===========================================================================

__global__ void localReduceSumCopyKernel(int nSrc, float* srcBase, size_t srcDispl,
                                          int nDst, float* dstBase, size_t dstDispl, size_t count) {
  ncclLocalReduceSumCopy<float>(ncclCoopCta(), nSrc, srcBase, srcDispl, nDst, dstBase, dstDispl, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSumCopy) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  constexpr int nSrc = 4;
  constexpr int nDst = 3;
  for (size_t count : countSweep()) {
    const size_t srcDispl = count + 8;
    const size_t dstDispl = count + 8;
    std::vector<float> hSrc(nSrc * srcDispl, 0.0f);
    for (int s = 0; s < nSrc; ++s)
      for (size_t i = 0; i < count; ++i)
        hSrc[s * srcDispl + i] = static_cast<float>((s + 1) + (i % 11));

    float* dSrc = nullptr;
    float* dDst = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, nSrc * srcDispl * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, nDst * dstDispl * sizeof(float)));
    auto guard = makeScopeGuard([&]() {
      if (dSrc) (void)hipFree(dSrc);
      if (dDst) (void)hipFree(dDst);
    });
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(dSrc, hSrc.data(), nSrc * srcDispl * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, nDst * dstDispl * sizeof(float)));

    localReduceSumCopyKernel<<<1, kThreads, 0, stream>>>(nSrc, dSrc, srcDispl, nDst, dDst, dstDispl, count);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<float> hDst(nDst * dstDispl);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(hDst.data(), dDst, nDst * dstDispl * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      float expected = 0.0f;
      for (int s = 0; s < nSrc; ++s) expected += hSrc[s * srcDispl + i];
      for (int d = 0; d < nDst; ++d)
        ASSERT_FLOAT_EQ(expected, hDst[d * dstDispl + i]) << "count=" << count << " dst=" << d << " elt=" << i;
    }
  }
}

// ===========================================================================
// LSA helpers
// ===========================================================================

namespace {

std::string lsaLocalSkipReason() {
  if (auto r = cuMemReason(); !r.empty()) return r;
  if (nodeLocalRanks() < 2) return "Requires >=2 ranks co-located on a node for a non-trivial LSA team";
  return "";
}
std::string lsaSkipReason() {
  const std::string local = lsaLocalSkipReason();
  return mpiCoordinatedSkipReason(!local.empty(), local.empty() ? nullptr : local.c_str());
}

// [5.1e] LSA sub-team split needs >=4 even co-located ranks (innerFactor pair groups).
std::string differentTeamsSkipReason() {
  std::string local = lsaLocalSkipReason();
  if (local.empty()) {
    const int lsaSize = nodeLocalRanks();
    if (lsaSize < 4) local = "Requires >=4 co-located ranks for LSA inner sub-team (5.1e)";
    else if (lsaSize % 2 != 0) local = "Requires even co-located rank count for inner sub-team split";
  }
  return mpiCoordinatedSkipReason(!local.empty(), local.empty() ? nullptr : local.c_str());
}

}  // namespace

// ---------------------------------------------------------------------------
// LSA Copy: origin's local source broadcast to every LSA peer's symmetric dst.
// Each rank runs the copy from its own source into the shared symmetric window
// addressed via the LSA team, then validates its own destination slot.
// ---------------------------------------------------------------------------

__global__ void lsaCopyKernel(float* src, ncclWindow_t dstWin, size_t dstOff, size_t count,
                              struct ncclDevComm devComm, int doCopy) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (doCopy)
    ncclLsaCopy<float>(coop, src, dstWin, dstOff, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_Copy) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);

  const size_t count = 1024;
  const size_t bytes = count * sizeof(float);

  // Symmetric destination window (written by all LSA peers).
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto dstMemCleanup = makeScopeGuard([&]() { if (dDst) (void)ncclMemFree(dDst); });
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin); });

  // Local (non-symmetric) source seeded with this rank's identity so we can
  // verify that peer p's slot received peer p's data.
  float* dSrc = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, bytes));
  auto srcCleanup = makeScopeGuard([&]() { if (dSrc) (void)hipFree(dSrc); });
  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>(rank * 1000 + (i % 251));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  MPI_Barrier(MPI_COMM_WORLD);
  const int lsaBase = lsaBaseRank(rank, nodeLocalRanks());
  const int doCopy = (rank == lsaBase) ? 1 : 0;
  lsaCopyKernel<<<1, kThreads, 0, stream>>>(dSrc, dstWin, /*dstOff=*/0, count, devComm, doCopy);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Only lsaBase issues ncclLsaCopy (NCCL: single rank per destination region). All ranks
  // still enter the kernel so the LSA barrier team is complete. Every rank verifies it
  // received lsaBase's payload.
  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  std::vector<float> hExpect(count);
  for (size_t i = 0; i < count; ++i) hExpect[i] = static_cast<float>(lsaBase * 1000 + (i % 251));
  for (size_t i = 0; i < count; ++i)
    ASSERT_FLOAT_EQ(hExpect[i], hDst[i]) << "rank=" << rank << " elt=" << i;
}

// ---------------------------------------------------------------------------
// LSA ReduceSum: reduce a symmetric buffer across the LSA team into a local dst.
// Each rank seeds its symmetric buffer with a rank-dependent value; the reduced
// result on every rank must equal the sum over the LSA team.
// ---------------------------------------------------------------------------

__global__ void lsaReduceSumKernel(ncclWindow_t srcWin, size_t srcOff, float* dst, size_t count,
                                   struct ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  ncclLsaReduceSum<float>(coop, srcWin, srcOff, dst, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSum) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = (rank / lsaSize) * lsaSize;  // world rank of LSA-local rank 0

  const size_t count = 1024;
  const size_t bytes = count * sizeof(float);

  void* dSrc = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  auto srcMemCleanup = makeScopeGuard([&]() { if (dSrc) (void)ncclMemFree(dSrc); });
  ncclWindow_t srcWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin); });

  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, bytes));
  auto dstCleanup = makeScopeGuard([&]() { if (dDst) (void)hipFree(dDst); });

  // Seed with a per-rank value; keep values small so the fp32 sum is exact.
  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>((rank + 1) + (i % 7));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceSumKernel<<<1, kThreads, 0, stream>>>(srcWin, /*srcOff=*/0, dDst, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>((lsaBase + p + 1) + (i % 7));
    ASSERT_FLOAT_EQ(expected, hDst[i]) << "rank=" << rank << " elt=" << i;
  }
}

// ---------------------------------------------------------------------------
// LSA ReduceSumCopy: N sources -> N destinations across the LSA team, each
// destination receiving the full team reduction (same-team variant).
// ---------------------------------------------------------------------------

__global__ void lsaReduceSumCopyKernel(ncclWindow_t srcWin, size_t srcOff,
                                       ncclWindow_t dstWin, size_t dstOff, size_t count,
                                       struct ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  ncclLsaReduceSumCopy<float>(coop, srcWin, srcOff, dstWin, dstOff, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSumCopy) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = (rank / lsaSize) * lsaSize;

  const size_t count = 1024;
  const size_t bytes = count * sizeof(float);

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>((rank + 1) + (i % 5));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceSumCopyKernel<<<1, kThreads, 0, stream>>>(srcWin, /*srcOff=*/0, dstWin, /*dstOff=*/0, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>((lsaBase + p + 1) + (i % 5));
    ASSERT_FLOAT_EQ(expected, hDst[i]) << "rank=" << rank << " elt=" << i;
  }
}

// ---------------------------------------------------------------------------
// LSA ReduceSumCopy [5.1e]: full LSA-team sources → inner sub-team destinations.
// Single-node only: sub-teams must stay within the mapped lsaFlatBase domain.
// Requires >=4 even co-located ranks (ncclTeamInnerFactor(lsa, 2)).
// ---------------------------------------------------------------------------

__global__ void lsaReduceSumCopyDifferentTeamsKernel(ncclWindow_t srcWin, size_t srcOff, ncclWindow_t dstWin,
                                                   size_t dstOff, size_t count, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  constexpr int kInnerTeamSize = 2;
  ncclTeam dstInner = ncclTeamInnerFactor(lsa, kInnerTeamSize);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  ncclSymPtr<float> src{srcWin, srcOff};
  ncclSymPtr<float> dst{dstWin, dstOff};
  ncclLsaReduceSumCopy<float>(coop, src, lsa, dst, dstInner, count);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSumCopy_DifferentTeams) {
  if (!validateTestPrerequisites(/*min_processes=*/4, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 4-8 ranks";
  if (auto reason = differentTeamsSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = (rank / lsaSize) * lsaSize;

  const size_t count = 256;
  const size_t bytes = count * sizeof(float);

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ncclWindow_t srcWin = nullptr;
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i)
    hSrc[i] = static_cast<float>((rank + 1) + (i % 7));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  MPI_Barrier(MPI_COMM_WORLD);
  lsaReduceSumCopyDifferentTeamsKernel<<<1, kThreads, 0, stream>>>(srcWin, 0, dstWin, 0, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p)
      expected += static_cast<float>((lsaBase + p + 1) + (i % 7));
    ASSERT_FLOAT_EQ(expected, hDst[i]) << "rank=" << rank << " elt=" << i;
  }
}

// ===========================================================================
// Group A — Memory & LSA pointers + barriers
// ===========================================================================

__global__ void localPointerWriteKernel(ncclWindow_t win, size_t byteOff, int value, int* outOk) {
  if (threadIdx.x == 0) {
    int* p = static_cast<int*>(ncclGetLocalPointer(win, byteOff));
    p[0] = value;
    outOk[0] = 1;
  }
}

TEST_F(DeviceApiMPITests, Pointer_Local_GetLocalPointer) {
  if (auto reason = cuMemReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  constexpr size_t kByteOff = 16;
  constexpr size_t kWinBytes = 32;
  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, kWinBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, kWinBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  int* dOk = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOk, sizeof(int)));
  auto okCleanup = makeScopeGuard([&]() { if (dOk) (void)hipFree(dOk); });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dWin, 0, kWinBytes));

  constexpr int kValue = 0x12345678;
  localPointerWriteKernel<<<1, 1, 0, stream>>>(win, kByteOff, kValue, dOk);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hOk = 0;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hOk, dOk, sizeof(int), hipMemcpyDeviceToHost));
  ASSERT_EQ(1, hOk);

  std::vector<int> hBuf(kWinBytes / sizeof(int), 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hBuf.data(), dWin, kWinBytes, hipMemcpyDeviceToHost));
  ASSERT_EQ(kValue, hBuf[kByteOff / sizeof(int)]);
}

__global__ void lsaPointerWriteKernel(ncclWindow_t win, size_t byteOff, int dstLsaPeer, int value, bool doWrite,
                                      ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (doWrite && threadIdx.x == 0) {
    int* peer = static_cast<int*>(ncclGetLsaPointer(win, byteOff, dstLsaPeer));
    peer[0] = value;
  }
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Pointer_Lsa_GetLsaPointer) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;

  constexpr size_t kBytes = sizeof(int);
  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, kBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, kBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dWin, 0, kBytes));

  constexpr int kMarker = 0xABCDEF01;
  const int dstLsaPeer = 1;  // rank 0 writes into LSA peer 1's slot (lsaSize >= 2)
  const bool doWrite = (myLsa == 0);
  lsaPointerWriteKernel<<<1, 1, 0, stream>>>(win, 0, dstLsaPeer, kMarker, doWrite, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hostVal = 0;
  const hipError_t hostMemcpyErr =
      (rank == lsaBase + dstLsaPeer)
          ? hipMemcpy(&hostVal, dWin, kBytes, hipMemcpyDeviceToHost)
          : hipSuccess;
  ASSERT_MPI_HIP_OK_ON_RANK(rank, lsaBase + dstLsaPeer, hostMemcpyErr);
  ASSERT_MPI_EQ_ON_RANK(rank, lsaBase + dstLsaPeer, kMarker, hostVal);
}

__global__ void peerPointerMatchKernel(ncclWindow_t win, size_t byteOff, int worldPeer, int lsaPeer,
                                       int* outMatch) {
  void* pPeer = ncclGetPeerPointer(win, byteOff, worldPeer);
  void* pLsa = ncclGetLsaPointer(win, byteOff, lsaPeer);
  if (threadIdx.x == 0)
    outMatch[0] = (pPeer == pLsa) ? 1 : 0;
}

TEST_F(DeviceApiMPITests, Pointer_Peer_GetPeerPointer) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int peerWorld = lsaBase + 1;
  const int peerLsa = 1;

  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, sizeof(int)));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, sizeof(int), &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  int* dMatch = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dMatch, sizeof(int)));
  auto matchCleanup = makeScopeGuard([&]() { if (dMatch) (void)hipFree(dMatch); });

  peerPointerMatchKernel<<<1, 1, 0, stream>>>(win, 0, peerWorld, peerLsa, dMatch);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hMatch = 0;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hMatch, dMatch, sizeof(int), hipMemcpyDeviceToHost));
  ASSERT_EQ(1, hMatch);
}

__global__ void peerTeamPointerMatchKernel(ncclWindow_t win, size_t byteOff, int lsaPeer, int* outMatch,
                                           ncclDevComm devComm) {
  ncclTeam team = ncclTeamLsa(devComm);
  const int worldPeer = devComm.rank - devComm.lsaRank + lsaPeer;
  void* pWorld = ncclGetPeerPointer(win, byteOff, worldPeer);
  void* pTeam = ncclGetPeerPointer(win, byteOff, team, lsaPeer);
  if (threadIdx.x == 0)
    outMatch[0] = (pWorld == pTeam) ? 1 : 0;
}

TEST_F(DeviceApiMPITests, Pointer_Peer_TeamOverload) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, sizeof(int)));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, sizeof(int), &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  int* dMatch = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dMatch, sizeof(int)));
  auto matchCleanup = makeScopeGuard([&]() { if (dMatch) (void)hipFree(dMatch); });

  peerTeamPointerMatchKernel<<<1, 1, 0, stream>>>(win, 0, /*lsaPeer=*/1, dMatch, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hMatch = 0;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hMatch, dMatch, sizeof(int), hipMemcpyDeviceToHost));
  ASSERT_EQ(1, hMatch);
}

__global__ void hostPtrMatchKernel(ncclWindow_t win, size_t byteOff, int lsaPeer, void* hostResolved, int* outMatch) {
  void* dev = ncclGetLsaPointer(win, byteOff, lsaPeer);
  if (threadIdx.x == 0)
    outMatch[0] = (dev == hostResolved) ? 1 : 0;
}

TEST_F(DeviceApiMPITests, Pointer_Host_GetLsaDevicePointer) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;

  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, sizeof(int)));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, sizeof(int), &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  void* hostPeerPtr = nullptr;
  const int peerLsa = (myLsa + 1) % lsaSize;
  ASSERT_MPI_EQ(ncclSuccess, ncclGetLsaDevicePointer(win, 0, peerLsa, &hostPeerPtr));
  ASSERT_NE(nullptr, hostPeerPtr);

  int* dMatch = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dMatch, sizeof(int)));
  auto matchCleanup = makeScopeGuard([&]() { if (dMatch) (void)hipFree(dMatch); });

  hostPtrMatchKernel<<<1, 1, 0, stream>>>(win, 0, peerLsa, hostPeerPtr, dMatch);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hMatch = 0;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hMatch, dMatch, sizeof(int), hipMemcpyDeviceToHost));
  ASSERT_EQ(1, hMatch);

  void* hostPeerViaWorld = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclGetPeerDevicePointer(win, 0, lsaBase + peerLsa, &hostPeerViaWorld));
  ASSERT_EQ(hostPeerPtr, hostPeerViaWorld);
}

__global__ void barrierSyncHandoffKernel(ncclWindow_t win, int dstLsaPeer, int writeVal, int* readBack,
                                         ncclDevComm devComm, int mode) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (mode == 0 && threadIdx.x == 0) {
    int* peer = static_cast<int*>(ncclGetLsaPointer(win, 0, dstLsaPeer));
    peer[0] = writeVal;
  }
  bar.sync(coop, cuda::memory_order_release);
  if (mode == 1 && threadIdx.x == 0) {
    int* local = static_cast<int*>(ncclGetLocalPointer(win, 0));
    readBack[0] = local[0];
  }
}

TEST_F(DeviceApiMPITests, Barrier_Lsa_SyncOrdering) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;

  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, sizeof(int)));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, sizeof(int), &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dWin, 0, sizeof(int)));

  int* dRead = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dRead, sizeof(int)));
  auto readCleanup = makeScopeGuard([&]() { if (dRead) (void)hipFree(dRead); });

  constexpr int kValue = 4242;
  const int dstLsaPeer = 1;  // rank 0 writes into LSA peer 1's slot (lsaSize >= 2)
  const int mode = (myLsa == 0) ? 0 : (myLsa == dstLsaPeer) ? 1 : 2;
  MPI_Barrier(MPI_COMM_WORLD);
  barrierSyncHandoffKernel<<<1, 1, 0, stream>>>(win, dstLsaPeer, kValue, dRead, devComm, mode);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hostRead = 0;
  const hipError_t hostMemcpyErr =
      (rank == lsaBase + dstLsaPeer)
          ? hipMemcpy(&hostRead, dRead, sizeof(int), hipMemcpyDeviceToHost)
          : hipSuccess;
  ASSERT_MPI_HIP_OK_ON_RANK(rank, lsaBase + dstLsaPeer, hostMemcpyErr);
  ASSERT_MPI_EQ_ON_RANK(rank, lsaBase + dstLsaPeer, kValue, hostRead);
}

__global__ void barrierArriveWaitKernel(ncclWindow_t win, int dstLsaPeer, int writeVal, int* readBack,
                                        ncclDevComm devComm, int mode) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x);
  if (mode == 0 && threadIdx.x == 0) {
    int* peer = static_cast<int*>(ncclGetLsaPointer(win, 0, dstLsaPeer));
    peer[0] = writeVal;
  }
  bar.arrive(coop, cuda::memory_order_release);
  bar.wait(coop, cuda::memory_order_acquire);
  if (mode == 1 && threadIdx.x == 0) {
    int* local = static_cast<int*>(ncclGetLocalPointer(win, 0));
    readBack[0] = local[0];
  }
}

TEST_F(DeviceApiMPITests, Barrier_Lsa_ArriveWaitSplit) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;

  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, sizeof(int)));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, sizeof(int), &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dWin, 0, sizeof(int)));

  int* dRead = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dRead, sizeof(int)));
  auto readCleanup = makeScopeGuard([&]() { if (dRead) (void)hipFree(dRead); });

  constexpr int kValue = 8080;
  const int dstLsaPeer = 1;  // rank 0 writes into LSA peer 1's slot (lsaSize >= 2)
  const int mode = (myLsa == 0) ? 0 : (myLsa == dstLsaPeer) ? 1 : 2;
  MPI_Barrier(MPI_COMM_WORLD);
  barrierArriveWaitKernel<<<1, 1, 0, stream>>>(win, dstLsaPeer, kValue, dRead, devComm, mode);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hostRead = 0;
  const hipError_t hostMemcpyErr =
      (rank == lsaBase + dstLsaPeer)
          ? hipMemcpy(&hostRead, dRead, sizeof(int), hipMemcpyDeviceToHost)
          : hipSuccess;
  ASSERT_MPI_HIP_OK_ON_RANK(rank, lsaBase + dstLsaPeer, hostMemcpyErr);
  ASSERT_MPI_EQ_ON_RANK(rank, lsaBase + dstLsaPeer, kValue, hostRead);
}

__global__ void multiCtaBarrierKernel(ncclWindow_t win, int* outBlockId, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (threadIdx.x == 0) {
    int* p = static_cast<int*>(ncclGetLocalPointer(win, blockIdx.x * sizeof(int)));
    p[0] = static_cast<int>(blockIdx.x + 1);
    outBlockId[blockIdx.x] = static_cast<int>(blockIdx.x);
  }
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Barrier_Lsa_MultiCtaIndex) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  constexpr int kBlocks = 2;
  constexpr size_t kBytes = kBlocks * sizeof(int);
  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, kBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, kBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  int* dBlockIds = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dBlockIds, kBlocks * sizeof(int)));
  auto idsCleanup = makeScopeGuard([&]() { if (dBlockIds) (void)hipFree(dBlockIds); });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dWin, 0, kBytes));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dBlockIds, -1, kBlocks * sizeof(int)));

  multiCtaBarrierKernel<<<kBlocks, kThreads, 0, stream>>>(win, dBlockIds, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<int> hIds(kBlocks);
  std::vector<int> hWin(kBlocks);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hIds.data(), dBlockIds, kBytes, hipMemcpyDeviceToHost));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hWin.data(), dWin, kBytes, hipMemcpyDeviceToHost));
  for (int b = 0; b < kBlocks; ++b) {
    ASSERT_EQ(b, hIds[b]);
    ASSERT_EQ(b + 1, hWin[b]);
  }
}

// ===========================================================================
// Group B — Reduce/Copy fixed overload extensions
// ===========================================================================

__global__ void lsaReduceSumSymPtrKernel(ncclSymPtr<float> src, float* dst, size_t count, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  ncclLsaReduceSum<float>(coop, src, dst, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSum_SymPtr) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  const size_t count = 256;
  const size_t bytes = count * sizeof(float);
  void* dSrc = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  auto memCleanup = makeScopeGuard([&]() { if (dSrc) (void)ncclMemFree(dSrc); });
  ncclWindow_t srcWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin); });

  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, bytes));
  auto dstCleanup = makeScopeGuard([&]() { if (dDst) (void)hipFree(dDst); });

  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>((rank + 1) + (i % 7));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  ncclSymPtr<float> srcSym(srcWin, 0);
  lsaReduceSumSymPtrKernel<<<1, kThreads, 0, stream>>>(srcSym, dDst, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>((lsaBase + p + 1) + (i % 7));
    ASSERT_FLOAT_EQ(expected, hDst[i]) << "elt=" << i;
  }
}

__global__ void lsaCopyOffsetKernel(float* src, ncclWindow_t dstWin, size_t dstOff, size_t count,
                                    struct ncclDevComm devComm, int doCopy) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (doCopy)
    ncclLsaCopy<float>(coop, src, dstWin, dstOff, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_Copy_NonZeroOffset) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);

  constexpr size_t kHeaderBytes = 64;
  const size_t count = 128;
  const size_t payloadBytes = count * sizeof(float);
  const size_t winBytes = kHeaderBytes + payloadBytes;

  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, winBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dDst) (void)ncclMemFree(dDst); });
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, winBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin); });

  float* dSrc = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, payloadBytes));
  auto srcCleanup = makeScopeGuard([&]() { if (dSrc) (void)hipFree(dSrc); });

  std::vector<float> hSrc(count);
  const int lsaBase = lsaBaseRank(rank, nodeLocalRanks());
  for (size_t i = 0; i < count; ++i) hSrc[i] = static_cast<float>(lsaBase * 100 + i);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), payloadBytes, hipMemcpyHostToDevice));

  std::vector<char> hHeader(kHeaderBytes);
  for (size_t i = 0; i < kHeaderBytes; ++i) hHeader[i] = static_cast<char>(0xA5);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hHeader.data(), kHeaderBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(static_cast<char*>(dDst) + kHeaderBytes, 0, payloadBytes));

  MPI_Barrier(MPI_COMM_WORLD);
  const int doCopy = (rank == lsaBase) ? 1 : 0;
  lsaCopyOffsetKernel<<<1, kThreads, 0, stream>>>(dSrc, dstWin, kHeaderBytes, count, devComm, doCopy);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<char> hWin(winBytes);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hWin.data(), dDst, winBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kHeaderBytes; ++i)
    ASSERT_EQ(static_cast<char>(0xA5), hWin[i]) << "header byte " << i;
  std::vector<float> hExpect(count);
  for (size_t i = 0; i < count; ++i) hExpect[i] = static_cast<float>(lsaBase * 100 + i);
  for (size_t i = 0; i < count; ++i) {
    float got = 0.0f;
    std::memcpy(&got, hWin.data() + kHeaderBytes + i * sizeof(float), sizeof(float));
    ASSERT_FLOAT_EQ(hExpect[i], got) << "payload elt " << i;
  }
}

__global__ void lsaReduceSumDevCommKernel(ncclWindow_t srcWin, size_t srcOff, float* dst, size_t count,
                                          ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  ncclLsaReduceSum<float>(coop, srcWin, srcOff, dst, count, devComm);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSum_DevCommOverload) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  const size_t count = 64;
  const size_t bytes = count * sizeof(float);
  void* dSrc = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  auto memCleanup = makeScopeGuard([&]() { if (dSrc) (void)ncclMemFree(dSrc); });
  ncclWindow_t srcWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin); });

  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, bytes));
  auto dstCleanup = makeScopeGuard([&]() { if (dDst) (void)hipFree(dDst); });

  std::vector<float> hSrc(count, static_cast<float>(rank + 1));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceSumDevCommKernel<<<1, kThreads, 0, stream>>>(srcWin, 0, dDst, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hDst(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>(lsaBase + p + 1);
    ASSERT_FLOAT_EQ(expected, hDst[i]);
  }
}

template<typename T>
__global__ void localReduceSumTypedKernel(int nSrc, T* srcBase, size_t displElts, T* dst, size_t count) {
  ncclLocalReduceSum<T>(ncclCoopCta(), nSrc, srcBase, displElts, dst, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSum_DtypeSweep) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  auto runCase = [&](auto tag) {
    using T = decltype(tag);
    constexpr int kNSrc = 3;
    constexpr size_t kCount = 17;
    constexpr size_t kDisplPadElts = 4;
    constexpr size_t kDisplElts = kCount + kDisplPadElts;
    constexpr int kInt8SeedBase = 100;
    std::vector<T> hSrc(kNSrc * kDisplElts, T{0});
    for (int s = 0; s < kNSrc; ++s) {
      for (size_t i = 0; i < kCount; ++i) {
        if constexpr (std::is_same_v<T, int8_t>) {
          // Per-source constants so the sum wraps (exercises OpSum<int8> __vadd4 path).
          hSrc[s * kDisplElts + i] = static_cast<int8_t>(kInt8SeedBase + s);
        } else {
          hSrc[s * kDisplElts + i] = static_cast<T>(s + 1 + i);
        }
      }
    }

    T* dSrc = nullptr;
    T* dDst = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, kNSrc * kDisplElts * sizeof(T)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, kCount * sizeof(T)));
    auto guard = makeScopeGuard([&]() {
      if (dSrc) (void)hipFree(dSrc);
      if (dDst) (void)hipFree(dDst);
    });
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), kNSrc * kDisplElts * sizeof(T), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, kCount * sizeof(T)));

    localReduceSumTypedKernel<T><<<1, kThreads, 0, stream>>>(kNSrc, dSrc, kDisplElts, dDst, kCount);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<T> hDst(kCount);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, kCount * sizeof(T), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kCount; ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        double expected = 0.0;
        for (int s = 0; s < kNSrc; ++s)
          expected += static_cast<double>(hSrc[s * kDisplElts + i]);
        ASSERT_NEAR(expected, static_cast<double>(hDst[i]), 1e-5) << "i=" << i;
      } else {
        T expected = T{0};
        for (int s = 0; s < kNSrc; ++s)
          expected = static_cast<T>(expected + hSrc[s * kDisplElts + i]);
        ASSERT_EQ(expected, hDst[i]) << "i=" << i;
      }
    }
  };

  runCase(float{});
  runCase(double{});
  runCase(int8_t{});
}

__global__ void localReduceSumCoopThreadKernel(int nSrc, float* srcBase, size_t displ, float* dst, size_t count) {
  ncclCoopThread coop;
  if (coop.thread_rank() == 0)
    ncclLocalReduceSum<float>(coop, nSrc, srcBase, displ, dst, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSum_CoopThread) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  constexpr int nSrc = 2;
  constexpr size_t kCount = 8;
  constexpr size_t kDisplPadElts = 4;
  constexpr size_t kDisplElts = kCount + kDisplPadElts;
  constexpr float kSrc0Value = 3.0f;
  constexpr float kSrc1Value = 5.0f;
  constexpr float kExpectedSum = kSrc0Value + kSrc1Value;
  std::vector<float> hSrc(nSrc * kDisplElts, 0.0f);
  hSrc[0] = kSrc0Value;
  hSrc[kDisplElts] = kSrc1Value;

  float* dSrc = nullptr;
  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, nSrc * kDisplElts * sizeof(float)));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, kCount * sizeof(float)));
  auto guard = makeScopeGuard([&]() {
    if (dSrc) (void)hipFree(dSrc);
    if (dDst) (void)hipFree(dDst);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), nSrc * kDisplElts * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, kCount * sizeof(float)));

  localReduceSumCoopThreadKernel<<<1, kThreads, 0, stream>>>(nSrc, dSrc, kDisplElts, dDst, kCount);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  float h0 = 0.0f;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&h0, dDst, sizeof(float), hipMemcpyDeviceToHost));
  ASSERT_FLOAT_EQ(kExpectedSum, h0);
}

__global__ void lsaReduceSumCopyInPlaceKernel(ncclWindow_t win, size_t byteOff, size_t count, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, lsa, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  // Only LSA rank 0 performs the N->N reduce-copy; peers barrier in/out so concurrent
  // invocations do not store into each other's source slots mid-reduce.
  if (devComm.lsaRank == 0)
    ncclLsaReduceSumCopy<float>(coop, win, byteOff, win, byteOff, count, lsa);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, ReduceCopy_InPlace) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  constexpr size_t kCount = 64;
  constexpr size_t kBytes = kCount * sizeof(float);
  constexpr size_t kByteOff = 0;
  void* dWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dWin, kBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dWin) (void)ncclMemFree(dWin); });
  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dWin, kBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (win) (void)ncclCommWindowDeregister(comm, win); });

  std::vector<float> hSeed(kCount, static_cast<float>(rank + 1));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dWin, hSeed.data(), kBytes, hipMemcpyHostToDevice));

  lsaReduceSumCopyInPlaceKernel<<<1, kThreads, 0, stream>>>(win, kByteOff, kCount, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hOut(kCount);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), dWin, kBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kCount; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>(lsaBase + p + 1);
    ASSERT_FLOAT_EQ(expected, hOut[i]);
  }
}

__global__ void localReduceSumCountUint32Kernel(int nSrc, float* srcBase, size_t displ, float* dst, uint32_t count) {
  ncclLocalReduceSum<float, ncclCoopCta, uint32_t>(ncclCoopCta(), nSrc, srcBase, displ, dst, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSum_CountUint32) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  constexpr int nSrc = 2;
  constexpr uint32_t kCount = 1024;
  constexpr size_t kDisplPadElts = 8;
  constexpr size_t kDisplElts = kCount + kDisplPadElts;
  constexpr float kSrcValue = 1.0f;
  constexpr float kExpectedSum = static_cast<float>(nSrc) * kSrcValue;
  float* dSrc = nullptr;
  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, nSrc * kDisplElts * sizeof(float)));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, kCount * sizeof(float)));
  auto guard = makeScopeGuard([&]() {
    if (dSrc) (void)hipFree(dSrc);
    if (dDst) (void)hipFree(dDst);
  });
  std::vector<float> hSrc(nSrc * kDisplElts, kSrcValue);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), nSrc * kDisplElts * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, kCount * sizeof(float)));

  localReduceSumCountUint32Kernel<<<1, kThreads, 0, stream>>>(nSrc, dSrc, kDisplElts, dDst, kCount);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  float h0 = 0.0f;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&h0, dDst, sizeof(float), hipMemcpyDeviceToHost));
  ASSERT_FLOAT_EQ(kExpectedSum, h0);
}

// ===========================================================================
// Group C — Lambda-based layouts
// ===========================================================================

__global__ void lsaReduceSumExcludeSelfKernel(ncclWindow_t srcWin, size_t srcOff, float* dst, size_t count,
                                              ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, team, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  const int myLsa = devComm.lsaRank;
  const int nSrc = team.nRanks - 1;
  auto srcLambda = [=] __device__(int i) -> float* {
    int peer = (i < myLsa) ? i : i + 1;
    return static_cast<float*>(ncclGetLsaPointer(srcWin, srcOff, peer));
  };
  ncclLsaReduceSum<float>(coop, srcLambda, nSrc, dst, count);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSum_LambdaExcludeSelf) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;

  constexpr size_t kCount = 32;
  constexpr size_t kBytes = kCount * sizeof(float);
  constexpr size_t kSrcOff = 0;
  constexpr int kSeedScale = 10;
  void* dSrc = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dSrc) (void)ncclMemFree(dSrc); });
  ncclWindow_t srcWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, kBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin); });

  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, kBytes));
  auto dstCleanup = makeScopeGuard([&]() { if (dDst) (void)hipFree(dDst); });

  const float seed = static_cast<float>((rank + 1) * kSeedScale);
  std::vector<float> hSrc(kCount, seed);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), kBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, kBytes));

  lsaReduceSumExcludeSelfKernel<<<1, kThreads, 0, stream>>>(srcWin, kSrcOff, dDst, kCount, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hDst(kCount);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hDst.data(), dDst, kBytes, hipMemcpyDeviceToHost));
  float expected = 0.0f;
  for (int p = 0; p < lsaSize; ++p) {
    if (p == myLsa) continue;
    expected += static_cast<float>((lsaBase + p + 1) * kSeedScale);
  }
  for (size_t i = 0; i < kCount; ++i)
    ASSERT_FLOAT_EQ(expected, hDst[i]) << "rank=" << rank << " i=" << i;
}

__global__ void lsaCopyLambdaScatterKernel(float* src, ncclWindow_t dstWin, size_t baseOff, size_t peerSkipBytes,
                                           size_t count, ncclDevComm devComm, int doCopy) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, team, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  if (doCopy) {
    auto dstLambda = [=] __device__(int i) -> float* {
      return static_cast<float*>(ncclGetLsaPointer(dstWin, baseOff + static_cast<size_t>(i) * peerSkipBytes, i));
    };
    ncclLsaCopy<float>(coop, src, dstLambda, team.nRanks, count);
  }
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_Copy_LambdaScatter) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);

  constexpr size_t kCount = 16;
  constexpr size_t kPeerSkipBytes = 32;
  constexpr int kBaseScale = 1000;
  constexpr size_t kBaseOff = 0;
  const size_t winBytes = kPeerSkipBytes * static_cast<size_t>(nodeLocalRanks()) + kCount * sizeof(float);
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, winBytes));
  auto memCleanup = makeScopeGuard([&]() { if (dDst) (void)ncclMemFree(dDst); });
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, winBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() { if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin); });

  float* dSrc = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSrc, kCount * sizeof(float)));
  auto srcCleanup = makeScopeGuard([&]() { if (dSrc) (void)hipFree(dSrc); });

  std::vector<float> hSrc(kCount);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);
  const int myLsa = rank - lsaBase;
  for (size_t i = 0; i < kCount; ++i) hSrc[i] = static_cast<float>(lsaBase * kBaseScale + i);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, winBytes));

  MPI_Barrier(MPI_COMM_WORLD);
  const int doCopy = (rank == lsaBase) ? 1 : 0;
  lsaCopyLambdaScatterKernel<<<1, kThreads, 0, stream>>>(dSrc, dstWin, kBaseOff, kPeerSkipBytes, kCount, devComm, doCopy);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Lambda scatter writes LSA peer i at baseOff + i*peerSkipBytes in that peer's window.
  const size_t localPayloadOff = static_cast<size_t>(myLsa) * kPeerSkipBytes;
  std::vector<float> hLocal(kCount);
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(hLocal.data(), static_cast<char*>(dDst) + localPayloadOff, kCount * sizeof(float),
                          hipMemcpyDeviceToHost));
  std::vector<float> hExpect(kCount);
  for (size_t i = 0; i < kCount; ++i) hExpect[i] = static_cast<float>(lsaBase * kBaseScale + i);
  for (size_t i = 0; i < kCount; ++i)
    ASSERT_FLOAT_EQ(hExpect[i], hLocal[i]) << "rank=" << rank << " i=" << i;
}

__global__ void lsaReduceSumCopyLambdaKernel(ncclWindow_t srcWin, size_t srcOff, ncclWindow_t dstWin, size_t dstOff,
                                             size_t count, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, team, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  auto srcLambda = [=] __device__(int i) -> float* {
    return static_cast<float*>(ncclGetLsaPointer(srcWin, srcOff, i));
  };
  auto dstLambda = [=] __device__(int i) -> float* {
    return static_cast<float*>(ncclGetLsaPointer(dstWin, dstOff, i));
  };
  ncclLsaReduceSumLsaCopy<float>(coop, srcLambda, team.nRanks, dstLambda, team.nRanks, count);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceSumCopy_LambdaLsaLsa) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  const size_t count = 32;
  const size_t bytes = count * sizeof(float);
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ncclWindow_t srcWin = nullptr;
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  std::vector<float> hSrc(count, static_cast<float>(rank + 1));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceSumCopyLambdaKernel<<<1, kThreads, 0, stream>>>(srcWin, 0, dstWin, 0, count, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hOut(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (int p = 0; p < lsaSize; ++p) expected += static_cast<float>(lsaBase + p + 1);
    ASSERT_FLOAT_EQ(expected, hOut[i]);
  }
}

// ===========================================================================
// Group D — Custom RedOp (ncclLsaReduceLsaCopy with user-defined functors)
// NCCL ships only OpSum in reduce_copy__types.h; min/max are test-local RedOps.
// ===========================================================================

namespace {

template<typename T>
struct TestOpMin {
  using EltType = T;
  __device__ T operator()(const T& a, const T& b) const {
    return a < b ? a : b;
  }
};

template<typename T>
struct TestOpMax {
  using EltType = T;
  __device__ T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

float peerSeedValue(int worldRank, size_t i) {
  return static_cast<float>((worldRank + 1) * 10 + (i % 5));
}

float reducePeerSeed(int lsaSize, int lsaBase, size_t i, float (*combine)(float, float)) {
  float acc = peerSeedValue(lsaBase, i);
  for (int p = 1; p < lsaSize; ++p)
    acc = combine(acc, peerSeedValue(lsaBase + p, i));
  return acc;
}

float combineMax(float a, float b) {
  return a > b ? a : b;
}

float combineMin(float a, float b) {
  return a < b ? a : b;
}

}  // namespace

template<typename RedOp>
__global__ void lsaReduceLsaCopyCustomKernel(ncclWindow_t srcWin, size_t srcOff, ncclWindow_t dstWin, size_t dstOff,
                                             size_t count, ncclDevComm devComm, RedOp redOp) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar(coop, devComm, team, devComm.lsaBarrier, blockIdx.x);
  bar.sync(coop, cuda::memory_order_acquire);
  auto srcLambda = [=] __device__(int i) -> float* {
    return static_cast<float*>(ncclGetLsaPointer(srcWin, srcOff, i));
  };
  auto dstLambda = [=] __device__(int i) -> float* {
    return static_cast<float*>(ncclGetLsaPointer(dstWin, dstOff, i));
  };
  ncclLsaReduceLsaCopy<float>(coop, srcLambda, team.nRanks, dstLambda, team.nRanks, redOp, count);
  bar.sync(coop, cuda::memory_order_release);
}

TEST_F(DeviceApiMPITests, Lsa_ReduceLsaCopy_OpMax) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  const size_t count = 64;
  const size_t bytes = count * sizeof(float);
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ncclWindow_t srcWin = nullptr;
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i)
    hSrc[i] = peerSeedValue(rank, i);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceLsaCopyCustomKernel<TestOpMax<float>><<<1, kThreads, 0, stream>>>(
    srcWin, 0, dstWin, 0, count, devComm, TestOpMax<float>{});
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hOut(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    const float expected = reducePeerSeed(lsaSize, lsaBase, i, combineMax);
    ASSERT_FLOAT_EQ(expected, hOut[i]) << "rank=" << rank << " i=" << i;
  }
}

TEST_F(DeviceApiMPITests, Lsa_ReduceLsaCopy_OpMin) {
  if (auto reason = lsaSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() { (void)ncclDevCommDestroy(comm, &devComm); });

  int rank = -1;
  ncclCommUserRank(comm, &rank);
  const int lsaSize = nodeLocalRanks();
  const int lsaBase = lsaBaseRank(rank, lsaSize);

  const size_t count = 64;
  const size_t bytes = count * sizeof(float);
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, bytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ncclWindow_t srcWin = nullptr;
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dSrc, bytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, dDst, bytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  std::vector<float> hSrc(count);
  for (size_t i = 0; i < count; ++i)
    hSrc[i] = peerSeedValue(rank, i);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hSrc.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, bytes));

  lsaReduceLsaCopyCustomKernel<TestOpMin<float>><<<1, kThreads, 0, stream>>>(
    srcWin, 0, dstWin, 0, count, devComm, TestOpMin<float>{});
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  std::vector<float> hOut(count);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), dDst, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    const float expected = reducePeerSeed(lsaSize, lsaBase, i, combineMin);
    ASSERT_FLOAT_EQ(expected, hOut[i]) << "rank=" << rank << " i=" << i;
  }
}

__global__ void localReduceSumLambdaKernel(float* s0, float* s1, float* s2, float* dst, size_t count) {
  ncclCoopCta coop = ncclCoopCta();
  auto srcLambda = [=] __device__(int i) -> float* {
    return (i == 0) ? s0 : (i == 1) ? s1 : s2;
  };
  ncclLocalReduceSum<float>(coop, srcLambda, 3, dst, count);
}

TEST_F(DeviceApiMPITests, Local_ReduceSum_Lambda) {
  if (!validateTestPrerequisites(/*min_processes=*/1))
    GTEST_SKIP() << "Requires >=1 process";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  hipStream_t stream = getActiveStream();

  constexpr size_t kCount = 8;
  constexpr float kSrc0Value = 1.0f;
  constexpr float kSrc1Value = 2.0f;
  constexpr float kSrc2Value = 4.0f;
  constexpr float kExpectedSum = kSrc0Value + kSrc1Value + kSrc2Value;
  float* d0 = nullptr;
  float* d1 = nullptr;
  float* d2 = nullptr;
  float* dDst = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&d0, kCount * sizeof(float)));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&d1, kCount * sizeof(float)));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&d2, kCount * sizeof(float)));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dDst, kCount * sizeof(float)));
  auto guard = makeScopeGuard([&]() {
    if (d0) (void)hipFree(d0);
    if (d1) (void)hipFree(d1);
    if (d2) (void)hipFree(d2);
    if (dDst) (void)hipFree(dDst);
  });

  std::vector<float> h0(kCount, kSrc0Value);
  std::vector<float> h1(kCount, kSrc1Value);
  std::vector<float> h2(kCount, kSrc2Value);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d0, h0.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d1, h1.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d2, h2.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dDst, 0, kCount * sizeof(float)));

  localReduceSumLambdaKernel<<<1, kThreads, 0, stream>>>(d0, d1, d2, dDst, kCount);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  float hSum = 0.0f;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hSum, dDst, sizeof(float), hipMemcpyDeviceToHost));
  ASSERT_FLOAT_EQ(kExpectedSum, hSum);
}

#endif  // MPI_TESTS_ENABLED
