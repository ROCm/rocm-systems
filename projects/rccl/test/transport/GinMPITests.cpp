/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// MPI tests for the device-side GIN proxy backend. Each test launches a real
// gin.{put|signal|waitSignal|...} kernel against a real proxy thread + IB.

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl_device.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <ios>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLTestGuards;

namespace {

// Honor explicit user disable.
std::string ginEnvDisabledReason() {
  if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN explicitly disabled by environment (NCCL_GIN_ENABLE=0)";
  return "";
}

// Plugin selector must be 2 (proxy)
std::string ginTypeReason() {
  const char* ginType = std::getenv("NCCL_GIN_TYPE");
  if (!ginType)
    return "GIN type not set (required NCCL_GIN_TYPE=2)";
  if (std::atoi(ginType) != 2)
    return std::string("Invalid GIN type: ") + ginType + " (required NCCL_GIN_TYPE=2)";
  return "";
}

// ncclDevCommCreate gates on comm->symmetricSupport which needs cuMem.
std::string cuMemReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || std::strcmp(cumem, "1") != 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE=1)";
  return "";
}

// On single-node, ncclTopoTrimSystem removes NET nodes unless intranet is forced,
// which leaves GIN no path to bind to.
std::string intranetReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0, worldSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize != worldSize) return "";  // multi-node, intranet is irrelevant
  const char* intra = std::getenv("RCCL_ENABLE_INTRANET");
  if (!intra || std::strcmp(intra, "1") != 0)
    return "Intranet mode required for single-node run (RCCL_ENABLE_INTRANET=1)";
  return "";
}

// Cross-node-only skip helper. Uses MPI_COMM_TYPE_SHARED (the same primitive
// intranetReason uses) to confirm at least two physical nodes are involved;
// if all ranks share a node, the IB plugin would silently fall back to
// loopback and the test wouldn't actually exercise the fabric.
std::string crossNodeReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0, worldSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize == worldSize)
    return "Cross-node test requires ranks on >=2 physical nodes";
  return "";
}

// Returns "" if every prerequisite for a GIN proxy-backend MPI test is met,
// else the first failing helper's reason (suitable for GTEST_SKIP()).
std::string ginProxyTestSkipReason() {
  for (auto check : {ginEnvDisabledReason, ginTypeReason, cuMemReason, intranetReason}) {
    if (auto reason = check(); !reason.empty()) return reason;
  }
  return "";
}

}  // namespace

// ---------------------------------------------------------------------------
// Put_BasicAndOffsets
//   Smallest end-to-end exercise of the device put -> ring -> proxy thread
//   -> IB -> peer signal cell chain. Rank 0 issues a single gin.put with a
//   SignalInc to rank 1 using non-zero src/dst/signal offsets.
// ---------------------------------------------------------------------------

__global__ void putBasicProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, dstOff,
            srcWin, srcOff,
            bytes,
            ncclGin_SignalInc{sigIdx});
  }
  // Collective flush: drain posted GFDs before the kernel exits.
  gin.flush(ncclCoopCta());
}

__global__ void putBasicConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

class GinMPIDeviceTests : public MPITestBase {
 protected:
  // Tiny put + waitSignal round-trip used by the Invalid_*Pool tests
  // to confirm comm bring-up + data path still work after pool clamp.
  void runBasicPutSelfCheck() {
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    ASSERT_EQ(2, nRanks);

    constexpr size_t          kBufBytes      = 64;
    constexpr size_t          kTransferBytes = 64;
    constexpr ncclGinSignal_t kSigIdx        = 0;
    constexpr int             kPeer          = 1;

    void* dSrc = nullptr;
    void* dDst = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSrc) (void)ncclMemFree(dSrc);
      if (dDst) (void)ncclMemFree(dDst);
    });

    ncclWindow_t srcWin = nullptr, dstWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
      if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    });

    ncclDevCommRequirements reqs{};
    reqs.railGinBarrierCount = 1;
    reqs.ginSignalCount      = 1;
    ncclDevComm devComm{};
    ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
    auto devCommCleanup = makeScopeGuard([&]() {
      (void)ncclDevCommDestroy(comm, &devComm);
    });

    std::vector<uint8_t> hostSrc(kBufBytes, 0);
    std::vector<uint8_t> hostDst(kBufBytes, 0);
    for (size_t i = 0; i < kTransferBytes; i++) {
      hostSrc[i] = static_cast<uint8_t>(0xA0 + (i & 0x3F));
    }
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      putBasicProducerKernel<<<1, 32, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          kTransferBytes, kSigIdx, kPeer,
          devComm);
    } else {
      putBasicConsumerKernel<<<1, 32, 0, stream>>>(
          kSigIdx, /*expectedSignalValue=*/1, devComm);
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
      std::vector<uint8_t> hostResult(kBufBytes, 0);
      ASSERT_EQ(hipSuccess,
                hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < kTransferBytes; i++) {
        ASSERT_EQ(hostSrc[i], hostResult[i]) << "byte " << i << " mismatched";
      }
    }
  }
};

TEST_F(GinMPIDeviceTests, Put_BasicAndOffsets) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // 8 KiB symmetric buffers; transfer 4 KiB starting at byte 4 KiB on src
  // and at byte 2 KiB on dst. Signal at index 1 (non-zero offset within the
  // per-context signal pool of size 2).
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;  // rank 0 -> rank 1

  // Symmetric src + dst (every rank allocates).
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Window registration is collective for SYMMETRIC-mode windows.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // ginSignalCount=2 so signal index 1 is valid; railGinBarrierCount>0
  // because the runtime allocates per-CTA barrier state and we launch with
  // 1 CTA. We don't actually invoke the barrier in this test.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // - Rank 0's src has a deterministic byte pattern in [srcOff, srcOff+xfer).
  // - Both ranks' dst is zeroed; any spurious write outside [dstOff, dstOff+xfer)
  //   on rank 1 surfaces as a verification mismatch.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (size_t i = 0; i < kTransferBytes; i++) {
    hostSrc[kSrcOff + i] = static_cast<uint8_t>(0x40 + (i & 0xFF));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // All ranks finished setup before any kernel launches.
  MPI_Barrier(MPI_COMM_WORLD);

  // Producer (rank 0) and consumer (rank 1) launch their own kernels.
  // Consumer's waitSignal returns once rank 0's put -> proxy -> IB -> signal
  // bump completes; that's the GIN-defined release point for the payload.
  if (rank == 0) {
    putBasicProducerKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff,
        dstWin, kDstOff,
        kTransferBytes, kSigIdx, kPeer,
        devComm);
  } else {
    putBasicConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Both kernels are done; rank 1's payload + signal cell are settled.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies the dst payload landed at dstOff and nowhere else.
  // Use plain ASSERT_EQ here -- this branch is rank-divergent, so
  // ASSERT_MPI_EQ would deadlock at its internal MPI_Allreduce.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < kTransferBytes; i++) {
      const uint8_t expected = static_cast<uint8_t>(0x40 + (i & 0xFF));
      ASSERT_EQ(expected, hostResult[kDstOff + i])
          << "byte " << i << " in [dstOff, dstOff+xfer) differs";
    }
    for (size_t i = 0; i < kDstOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " before dstOff was unexpectedly written";
    }
    for (size_t i = kDstOff + kTransferBytes; i < kBufBytes; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " after dstOff+xfer was unexpectedly written";
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// Put_CrossNode
//   Same wire-level put as Put.BasicAndOffsets, but enforces that the two
//   ranks live on different physical nodes so the IB plugin actually goes
//   through the fabric instead of single-node loopback. Single-node uses
//   IB loopback (or a faster intra-node path); multi-node hits the real
//   fabric and exercises a different IB plugin path. Skips cleanly if both
//   ranks landed on the same host (e.g. running with --map-by slot and
//   slots>=2 on a single host).
//
//   Run with: NPROC=2 bash run-mpitest.sh --gtest_filter='GinMPI_Put.CrossNode'
//   from the multi-node-docker harness (which sets --map-by node).
// ---------------------------------------------------------------------------

TEST_F(GinMPIDeviceTests, Put_CrossNode) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // The only thing that distinguishes CrossNode from BasicAndOffsets: the
  // two ranks must be on different physical nodes.
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // Same buffer geometry as BasicAndOffsets: 8 KiB symmetric src/dst,
  // 4 KiB transfer, non-zero offsets on both sides, signal at index 1.
  // Replicating these constants instead of factoring them out keeps each
  // test's geometry self-evident in-place; the duplication is intentional.
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;  // rank 0 -> rank 1

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (size_t i = 0; i < kTransferBytes; i++) {
    hostSrc[kSrcOff + i] = static_cast<uint8_t>(0x40 + (i & 0xFF));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    putBasicProducerKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff,
        dstWin, kDstOff,
        kTransferBytes, kSigIdx, kPeer,
        devComm);
  } else {
    putBasicConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies the dst payload landed at dstOff and nowhere else.
  // Plain ASSERT_EQ here -- this branch is rank-divergent.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < kTransferBytes; i++) {
      const uint8_t expected = static_cast<uint8_t>(0x40 + (i & 0xFF));
      ASSERT_EQ(expected, hostResult[kDstOff + i])
          << "byte " << i << " in [dstOff, dstOff+xfer) differs";
    }
    for (size_t i = 0; i < kDstOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " before dstOff was unexpectedly written";
    }
    for (size_t i = kDstOff + kTransferBytes; i < kBufBytes; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " after dstOff+xfer was unexpectedly written";
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// PutValue_Inline
//   Rank 0 sends an 8-byte uint64_t value to rank 1 via gin.putValue. The
//   value rides inside the GFD itself (no source MR is dereferenced); the
//   host proxy reconstructs it from inlineValLow + inlineValLow2 + inlineValHigh
//   at gin_host_proxy.cc:248-261. Picks a value that exercises all three pieces
//   of the 4+2+2 byte split so a regression in any field surfaces.
// ---------------------------------------------------------------------------

__global__ void putValueInlineProducerKernel(
    ncclWindow_t dstWin, size_t dstOff,
    uint64_t value, ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.putValue<uint64_t>(ncclTeamWorld(devComm), peer,
                           dstWin, dstOff, value,
                           ncclGin_SignalInc{sigIdx});
  }
  // Collective flush: drain posted GFDs before the kernel exits.
  gin.flush(ncclCoopCta());
}

__global__ void putValueInlineConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

TEST_F(GinMPIDeviceTests, PutValue_Inline) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // 4 KiB symmetric dst buffer; inline value lands at byte 1 KiB on rank 1.
  // kValue exercises all three pieces of the inline 4+2+2 byte split that
  // the host proxy reconstructs at gin_host_proxy.cc:248-261, so a regression
  // in any of {inlineValLow, inlineValLow2, inlineValHigh} surfaces.
  constexpr size_t   kBufBytes = 4 * 1024;
  constexpr size_t   kDstOff   = 1 * 1024;
  constexpr uint64_t kValue    = 0x123456789ABCDEF0ULL;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;  // rank 0 -> rank 1

  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dDst) (void)ncclMemFree(dDst);
  });

  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // ginSignalCount=2 so signal index 1 is valid; railGinBarrierCount>0
  // because the runtime allocates per-CTA barrier state and we launch with
  // 1 CTA. We don't actually invoke the barrier in this test.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Both ranks zero their dst; any spurious write outside the 8-byte landing
  // window on rank 1 surfaces as a verification mismatch.
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // All ranks finished setup before any kernel launches.
  MPI_Barrier(MPI_COMM_WORLD);

  // Producer (rank 0) and consumer (rank 1) launch their own kernels.
  // Consumer's waitSignal returns once rank 0's putValue -> proxy -> IB -> signal
  // bump completes; that's the GIN-defined release point for the inline value.
  if (rank == 0) {
    putValueInlineProducerKernel<<<1, 32, 0, stream>>>(
        dstWin, kDstOff, kValue, kSigIdx, kPeer, devComm);
  } else {
    putValueInlineConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Both kernels are done; rank 1's payload + signal cell are settled.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies the inline value landed at dstOff and nowhere else.
  // Use plain ASSERT_EQ here -- this branch is rank-divergent, so
  // ASSERT_MPI_EQ would deadlock at its internal MPI_Allreduce.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    uint64_t got = 0;
    std::memcpy(&got, hostResult.data() + kDstOff, sizeof(got));
    ASSERT_EQ(kValue, got)
        << "inline value mismatch (4+2+2 split likely corrupted)";

    for (size_t i = 0; i < kDstOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " before dstOff was unexpectedly written";
    }
    for (size_t i = kDstOff + sizeof(uint64_t); i < kBufBytes; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " after dstOff+8 was unexpectedly written";
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// MultiContext_AllFourRoute
//   Verifies all 4 GIN contexts deliver end-to-end. Producer launches 4 blocks
//   on rank 0; block b constructs ncclGin{devComm, b} so the ctor's
//   power-of-2 mask path (gin__funcs.h:24-26) maps it to contextId=b. Each
//   block puts into its own per-context dst slot with a per-context byte
//   pattern and signals SignalInc{b}. Consumer mirrors with 4 blocks each
//   waiting on its own context's signal. Confirms each contextId has a
//   working ginComms[contextId] + proxy ring + IB QP all the way through;
//   ginWins[contextId] selection at gin__funcs.h:139-142 is exercised
//   transitively. Slot tails are asserted zero to catch cross-context
//   contamination.
// ---------------------------------------------------------------------------

__global__ void multiContextProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin,
    size_t slotStride, size_t bytes, int peer,
    struct ncclDevComm devComm) {
  // contextIndex = blockIdx.x; for ginContextCount=4 the ctor takes the
  // power-of-2 mask arm so contextId = blockIdx.x & 3 = blockIdx.x.
  ncclGin gin{devComm, (int)blockIdx.x};
  const size_t off = (size_t)blockIdx.x * slotStride;
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, off,
            srcWin, off,
            bytes,
            ncclGin_SignalInc{(ncclGinSignal_t)blockIdx.x});
  }
  gin.flush(ncclCoopCta());
}

__global__ void multiContextConsumerKernel(
    uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, (int)blockIdx.x};
  gin.waitSignal(ncclCoopCta(), (ncclGinSignal_t)blockIdx.x, expectedSignalValue);
}

TEST_F(GinMPIDeviceTests, MultiContext_AllFourRoute) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // 4 per-context slots; transfer 1 KiB into each 4 KiB slot so the tail
  // [kTransferBytes, kSlotStride) stays zero and can pin down any
  // cross-context address-arithmetic regression in ginWins[contextId] /
  // proxy-side ring routing.
  constexpr int    kNumContexts    = NCCL_GIN_MAX_CONTEXTS;  // 4
  constexpr size_t kSlotStride     = 4 * 1024;
  constexpr size_t kTransferBytes  = 1 * 1024;
  constexpr size_t kBufBytes       = kNumContexts * kSlotStride;  // 16 KiB
  constexpr int    kPeer           = 1;  // rank 0 -> rank 1

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // ginContextCount on the requirements struct is a hint
  // (core.h:75); the authoritative value comes from the env-controlled
  // ginState.ginCommCount (gin_host.cc:95) and is read back from
  // devComm.ginContextCount post-create. We still set the hint to express
  // intent and skip below if the runtime gave us fewer.
  // ginSignalCount=kNumContexts because each block uses signal id = blockIdx.x
  // within its own context (signal pools are per-context).
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = kNumContexts;
  reqs.ginSignalCount      = kNumContexts;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount != kNumContexts) {
    GTEST_SKIP() << "Test requires " << kNumContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << kNumContexts << ")";
  }

  // Per-block byte pattern: 0x10 + b. Distinct patterns let any cross-context
  // contamination (block b's payload landing in block b'!=b's slot) surface
  // as a value mismatch rather than just a missing write.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (int b = 0; b < kNumContexts; b++) {
    const uint8_t pattern = static_cast<uint8_t>(0x10 + b);
    std::fill_n(hostSrc.begin() + b * kSlotStride, kTransferBytes, pattern);
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    multiContextProducerKernel<<<kNumContexts, 32, 0, stream>>>(
        srcWin, dstWin, kSlotStride, kTransferBytes, kPeer, devComm);
  } else {
    multiContextConsumerKernel<<<kNumContexts, 32, 0, stream>>>(
        /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies each context's slot. Use plain ASSERT_EQ here -- this
  // branch is rank-divergent, so ASSERT_MPI_EQ would deadlock at its
  // internal MPI_Allreduce.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (int b = 0; b < kNumContexts; b++) {
      const uint8_t pattern = static_cast<uint8_t>(0x10 + b);
      const size_t base = (size_t)b * kSlotStride;
      for (size_t i = 0; i < kTransferBytes; i++) {
        ASSERT_EQ(pattern, hostResult[base + i])
            << "ctx " << b << ": byte " << i
            << " in transferred range mismatched (expected 0x" << std::hex
            << (int)pattern << ")";
      }
      for (size_t i = kTransferBytes; i < kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << b << ": byte " << i
            << " in slot tail was unexpectedly written";
      }
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// MultiContext_NonPowerOf2
//   Exercises the only non-power-of-2 supported context count (3). The
//   ncclGin ctor at gin__funcs.h:24-26 picks between two arms:
//     - power-of-2 mask: contextId = contextIndex & (n-1)   for n in {1,2,4}
//     - true modulo   : contextId = contextIndex % 3        for n == 3
//   AllFourRoute exercises the mask arm; this is the only test that hits
//   the modulo arm. Producer launches 6 blocks; pairs (0,3), (1,4), (2,5)
//   collide on contextId 0/1/2 respectively. Each pair writes disjoint
//   sub-slots within the same ctx-id slot (so we can verify both arrived
//   without a race) and signals SignalInc on signal 0 of its context;
//   the consumer waitSignal(0, expected=2) per context confirms both
//   producers landed.
//
//   Requires NCCL_GIN_NCONTEXTS=3 (otherwise gin_host.cc:95 clamps to 4
//   and we skip).
// ---------------------------------------------------------------------------

__global__ void multiContextNpo2ProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin,
    int numContexts, size_t slotStride, size_t subSlotBytes, int peer,
    struct ncclDevComm devComm) {
  // For ginContextCount=3 the ctor takes the modulo arm so
  // contextId = blockIdx.x % 3. Pairs of blocks collide on the same ctx.
  ncclGin gin{devComm, (int)blockIdx.x};
  const int    ctx       = (int)blockIdx.x % numContexts;
  const int    subSlotIx = (int)blockIdx.x / numContexts;  // 0 or 1
  const size_t off       = (size_t)ctx * slotStride + (size_t)subSlotIx * subSlotBytes;
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, off,
            srcWin, off,
            subSlotBytes,
            // One signal per context (signal id 0 in each); two producer blocks
            // per ctx → consumer waits for the cell to reach 2.
            ncclGin_SignalInc{(ncclGinSignal_t)0});
  }
  gin.flush(ncclCoopCta());
}

__global__ void multiContextNpo2ConsumerKernel(
    uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  // contextIndex = blockIdx.x in [0, numContexts); modulo arm trivially
  // gives contextId == blockIdx.x for that range.
  ncclGin gin{devComm, (int)blockIdx.x};
  gin.waitSignal(ncclCoopCta(), (ncclGinSignal_t)0, expectedSignalValue);
}

TEST_F(GinMPIDeviceTests, MultiContext_NonPowerOf2) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // 3 ctx slots * 4 KiB stride; within each slot, 2 sub-slots of 1 KiB
  // (one per producer block hitting that ctx) leave a 2 KiB tail asserted
  // zero to catch cross-ctx contamination.
  constexpr int    kNumContexts    = 3;
  constexpr int    kBlocksPerCtx   = 2;
  constexpr int    kProducerBlocks = kNumContexts * kBlocksPerCtx;  // 6
  constexpr size_t kSubSlotBytes   = 1 * 1024;
  constexpr size_t kSlotStride     = 4 * 1024;
  constexpr size_t kBufBytes       = kNumContexts * kSlotStride;   // 12 KiB
  constexpr int    kPeer           = 1;  // rank 0 -> rank 1

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // ginSignalCount=1 because each context uses signal id 0; signal pools are
  // per-context so the three contexts' signal-0 cells are independent.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = kNumContexts;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount != kNumContexts) {
    GTEST_SKIP() << "Test requires " << kNumContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << kNumContexts << ")";
  }

  // Per-block byte pattern 0x10 + b for blocks 0..5; distinct patterns let a
  // cross-block landing surface as a value mismatch rather than just a
  // missing write.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (int b = 0; b < kProducerBlocks; b++) {
    const int    ctx       = b % kNumContexts;
    const int    subSlotIx = b / kNumContexts;
    const size_t off       = (size_t)ctx * kSlotStride + (size_t)subSlotIx * kSubSlotBytes;
    const uint8_t pattern  = static_cast<uint8_t>(0x10 + b);
    std::fill_n(hostSrc.begin() + off, kSubSlotBytes, pattern);
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    multiContextNpo2ProducerKernel<<<kProducerBlocks, 32, 0, stream>>>(
        srcWin, dstWin, kNumContexts, kSlotStride, kSubSlotBytes, kPeer, devComm);
  } else {
    multiContextNpo2ConsumerKernel<<<kNumContexts, 32, 0, stream>>>(
        /*expectedSignalValue=*/static_cast<uint64_t>(kBlocksPerCtx), devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies each ctx slot holds the bytes from BOTH producer blocks
  // that mapped to it (one per sub-slot), plus an untouched tail.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (int ctx = 0; ctx < kNumContexts; ctx++) {
      const size_t base = (size_t)ctx * kSlotStride;
      for (int subSlotIx = 0; subSlotIx < kBlocksPerCtx; subSlotIx++) {
        const int     b       = subSlotIx * kNumContexts + ctx;  // inverse of (b%3, b/3)
        const uint8_t pattern = static_cast<uint8_t>(0x10 + b);
        const size_t  off     = base + (size_t)subSlotIx * kSubSlotBytes;
        for (size_t i = 0; i < kSubSlotBytes; i++) {
          ASSERT_EQ(pattern, hostResult[off + i])
              << "ctx " << ctx << " sub-slot " << subSlotIx
              << " (block " << b << "): byte " << i
              << " mismatched (expected 0x" << std::hex << (int)pattern << ")";
        }
      }
      // Tail after both sub-slots must remain zero -- catches a stray put
      // landing in the wrong ctx's slot.
      const size_t tailStart = base + (size_t)kBlocksPerCtx * kSubSlotBytes;
      for (size_t i = tailStart; i < base + kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[i])
            << "ctx " << ctx << ": byte " << (i - base)
            << " in slot tail was unexpectedly written";
      }
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// LargeBufferSweep
//   Aligned + unaligned size matrix. Re-uses putBasicProducerKernel /
//   putBasicConsumerKernel and loops over the sweep with a single comm,
//   single window pair, single signal cell that is monotonically bumped
//   (SignalInc each iter -> consumer waits for i+1). dDst is re-zeroed
//   on rank 1 between iterations so post-payload bytes can be asserted
//   zero -- catches RDMA size overshoot / stale tail-byte handling.
//   Verification is std::memcmp on two ranges per size (payload + tail);
//   ASSERT_EQ aborts on first failure with the size in the message.
// ---------------------------------------------------------------------------

TEST_F(GinMPIDeviceTests, LargeBuffer_Sweep) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // Aligned + unaligned size matrix from the shortlist. Plus-one variants
  // catch tail-byte handling regressions (RDMA fragments aligned at page
  // boundaries; the +1 byte forces a separate trailing fragment).
  const std::vector<size_t> kSizes = {
      1,
      64,
      4 * 1024,
      4 * 1024 + 1,
      1 * 1024 * 1024,
      4 * 1024 * 1024,
      16 * 1024 * 1024,
      16 * 1024 * 1024 + 1,
  };
  const size_t kMaxBytes = kSizes.back();
  constexpr ncclGinSignal_t kSigIdx = 0;
  constexpr int kPeer = 1;  // rank 0 -> rank 1

  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kMaxBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kMaxBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kMaxBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kMaxBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Source pattern is size-independent so we only upload once: src[i]=i&0xFF.
  // Different bytes-modulo-256 values guarantee any byte-shift / off-by-one
  // in the RDMA path surfaces as a value mismatch.
  std::vector<uint8_t> hostSrc(kMaxBytes, 0);
  for (size_t i = 0; i < kMaxBytes; i++) hostSrc[i] = static_cast<uint8_t>(i & 0xFF);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kMaxBytes, hipMemcpyHostToDevice));

  const std::vector<uint8_t> hostZero(kMaxBytes, 0);
  std::vector<uint8_t> hostResult(kMaxBytes, 0);

  uint64_t signalExpected = 0;
  for (size_t iter = 0; iter < kSizes.size(); iter++) {
    const size_t sz = kSizes[iter];

    // Re-zero dDst on both ranks so post-payload bytes can be asserted zero
    // post-put. (The previous iter's larger size could have left non-zero
    // bytes here.)
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostZero.data(), kMaxBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    // Each iter bumps signal[0] by 1 via SignalInc; consumer expects iter+1.
    signalExpected++;
    if (rank == 0) {
      putBasicProducerKernel<<<1, 32, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          sz, kSigIdx, kPeer, devComm);
    } else {
      putBasicConsumerKernel<<<1, 32, 0, stream>>>(
          kSigIdx, signalExpected, devComm);
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 1 verifies the dst payload landed bit-for-bit at offset 0 and
    // that nothing landed past the put size. Use plain ASSERT_EQ here --
    // this branch is rank-divergent.
    if (rank == 1) {
      ASSERT_EQ(hipSuccess,
                hipMemcpy(hostResult.data(), dDst, kMaxBytes, hipMemcpyDeviceToHost));

      // Bulk payload check (memcmp is vectorized; ASSERT aborts at first
      // failure with the size, so 16M-byte iterations don't blow up the log).
      const int payloadCmp = std::memcmp(hostResult.data(), hostSrc.data(), sz);
      ASSERT_EQ(0, payloadCmp)
          << "size=" << sz << ": payload range [0," << sz
          << ") differs from source";

      // Bulk tail check: anything past sz must remain zero.
      if (sz < kMaxBytes) {
        const int tailCmp =
          std::memcmp(hostResult.data() + sz, hostZero.data(), kMaxBytes - sz);
        ASSERT_EQ(0, tailCmp)
            << "size=" << sz << ": bytes in [" << sz << "," << kMaxBytes
            << ") were unexpectedly written";
      }
    }
  }

  // ScopeGuards run in reverse order on return.
}

// ---------------------------------------------------------------------------
// Disable_GIN_Error
//   Negative test: with NCCL_GIN_ENABLE=0, a GIN-requiring path must fail
//   with ncclInternalError -- no hang, no crash. Inverse env gate: opts in
//   only when NCCL_GIN_ENABLE=0 (positive tests opt out via the same env).
// ---------------------------------------------------------------------------

TEST_F(GinMPIDeviceTests, Disable_Error) {
  // Inverse of ginEnvDisabledReason: opt INTO running when GIN is off.
  const char* e = std::getenv("NCCL_GIN_ENABLE");
  if (!e || std::strcmp(e, "0") != 0)
    GTEST_SKIP() << "Negative-path test; opt in by setting NCCL_GIN_ENABLE=0";

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Deliberately skips ginProxyTestSkipReason: GIN_TYPE / CUMEM / INTRANET
  // gate the data path, none of which is exercised here. Plain comm
  // bring-up does not call into GIN, so it succeeds even with the plugin
  // disabled.
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  // ginSignalCount > 0 forces ncclDevCommCreate to bring up GIN; with
  // GIN disabled it must fail before the plugin attaches.
  ncclDevCommRequirements reqs{};
  reqs.ginSignalCount = 1;
  ncclDevComm devComm{};
  ncclResult_t r = ncclDevCommCreate(comm, &reqs, &devComm);

  ASSERT_EQ(ncclInternalError, r)
      << "ncclDevCommCreate should fail with ncclInternalError when "
         "NCCL_GIN_ENABLE=0; got result " << (int)r;

  // No devCommDestroy: create failed, nothing to tear down. The test
  // fixture cleans up the underlying comm.
}

TEST_F(GinMPIDeviceTests, Invalid_SignalPool) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Opt in: only run when the user has explicitly requested an oversized
  // signal pool (>= 1 GiB) so the runtime's clamp path is exercised.
  const char* env = std::getenv("NCCL_GIN_SIGNAL_POOL_SIZE");
  if (!env || std::strtoull(env, nullptr, 0) < (1ULL << 30))
    GTEST_SKIP() << "Set NCCL_GIN_SIGNAL_POOL_SIZE>=0x40000000 to opt in";

  runBasicPutSelfCheck();
}

TEST_F(GinMPIDeviceTests, Invalid_CounterPool) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  const char* env = std::getenv("NCCL_GIN_COUNTER_POOL_SIZE");
  if (!env || std::strtoull(env, nullptr, 0) < (1ULL << 30))
    GTEST_SKIP() << "Set NCCL_GIN_COUNTER_POOL_SIZE>=0x40000000 to opt in";

  runBasicPutSelfCheck();
}

TEST_F(GinMPIDeviceTests, Teardown_NoLeaks) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Inner scope so window/devComm/mem guards fire before we call
  // cleanupTestCommunicator() below (they hold refs to the comm).
  {
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    constexpr size_t          kBufBytes      = 64;
    constexpr ncclGinSignal_t kSigIdx        = 0;
    constexpr int             kPeer          = 1;

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    ASSERT_EQ(2, nRanks);

    void* dSrc = nullptr;
    void* dDst = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSrc) (void)ncclMemFree(dSrc);
      if (dDst) (void)ncclMemFree(dDst);
    });

    ncclWindow_t srcWin = nullptr, dstWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
      if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    });

    ncclDevCommRequirements reqs{};
    reqs.railGinBarrierCount = 1;
    reqs.ginSignalCount      = 1;
    ncclDevComm devComm{};
    ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
    auto devCommCleanup = makeScopeGuard([&]() {
      (void)ncclDevCommDestroy(comm, &devComm);
    });

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      putBasicProducerKernel<<<1, 32, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          kBufBytes, kSigIdx, kPeer,
          devComm);
    } else {
      putBasicConsumerKernel<<<1, 32, 0, stream>>>(
          kSigIdx, /*expectedSignalValue=*/1, devComm);
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
  }  // <- mem/window/devComm guards fire here while comm is still live.

  // Explicit destroy + verify it succeeded (test goes beyond the
  // fixture's silent TearDown). cleanupTestCommunicator() returning
  // ncclSuccess implies ncclGinFinalize ran to completion (proxy
  // thread joined, MR/QP released).
  ASSERT_EQ(ncclSuccess, cleanupTestCommunicator());
  ASSERT_EQ(nullptr, getActiveCommunicator());
  ASSERT_EQ(nullptr, getActiveStream());
}

TEST_F(GinMPIDeviceTests, Init_Destroy_Stress) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Each iter brings up the comm, triggers GIN connect via a symmetric
  // window register, then tears down. Catches refcount/mutex regressions
  // in ncclGinFinalize that only surface across repeated lifecycles.
  constexpr int kIterations = 10;
  for (int i = 0; i < kIterations; ++i) {
    ASSERT_EQ(ncclSuccess, createTestCommunicator()) << "iter " << i;
    ncclComm_t comm = getActiveCommunicator();

    void* d = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&d, 64));

    ncclWindow_t win = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, d, 64, &win, NCCL_WIN_COLL_SYMMETRIC));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemFree(d));

    ASSERT_EQ(ncclSuccess, cleanupTestCommunicator()) << "iter " << i;
  }
}

#endif  // MPI_TESTS_ENABLED
