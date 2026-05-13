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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
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

class GinMPI_Put : public MPITestBase {};

TEST_F(GinMPI_Put, BasicAndOffsets) {
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

#endif  // MPI_TESTS_ENABLED
