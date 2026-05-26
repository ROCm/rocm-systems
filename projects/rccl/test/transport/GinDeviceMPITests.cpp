/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// MPI tests for the device-side GIN proxy backend. Each test launches a real
// gin.{put|putValue|waitSignal|...} kernel against a real proxy thread + IB,
// and validates the wire-level result on the receiving rank.

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

std::string ginEnvDisabledReason() {
  if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN explicitly disabled by environment (NCCL_GIN_ENABLE=0)";
  return "";
}

std::string ginTypeReason() {
  const char* ginType = std::getenv("NCCL_GIN_TYPE");
  if (!ginType)
    return "GIN type not set (required NCCL_GIN_TYPE=2)";
  if (std::atoi(ginType) != 2)
    return std::string("Invalid GIN type: ") + ginType + " (required NCCL_GIN_TYPE=2)";
  return "";
}

std::string cuMemReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || std::strcmp(cumem, "1") != 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE=1)";
  return "";
}

// Single-node runs need intranet mode -- otherwise the topology pruner
// removes the NET node and GIN has no path to bind.
std::string intranetReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0, worldSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize != worldSize) return "";
  const char* intra = std::getenv("RCCL_ENABLE_INTRANET");
  if (!intra || std::strcmp(intra, "1") != 0)
    return "Intranet mode required for single-node run (RCCL_ENABLE_INTRANET=1)";
  return "";
}

// Skip if all ranks share a node -- IB would silently loopback.
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

// First failing prerequisite, or "" if all met.
std::string ginProxyTestSkipReason() {
  for (auto check : {ginEnvDisabledReason, ginTypeReason, cuMemReason, intranetReason}) {
    if (auto reason = check(); !reason.empty()) return reason;
  }
  return "";
}

}  // namespace

// Producer: thread 0 of block 0 issues one put with a SignalInc; CTA flushes.
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
  // Drain the posted GFD before the kernel exits.
  gin.flush(ncclCoopCta());
}

// Consumer: whole CTA cooperatively waits for the signal to reach the target.
__global__ void putBasicConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Combined producer + consumer for alltoall: thread 0 puts to every non-self
// peer (one slot each), the CTA flushes, then waits for the same number of
// signal increments to arrive from peers.
__global__ void alltoallKernel(
    ncclWindow_t sendWin,
    ncclWindow_t recvWin,
    size_t bytesPerSlot,
    int nRanks, int myRank,
    size_t slotStrideBytes,
    ncclGinSignal_t sigIdx,
    uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto team = ncclTeamWorld(devComm);
    // Send slot p of our send buffer into peer p's recv slot for our rank.
    for (int p = 0; p < nRanks; ++p) {
      if (p == myRank) continue;
      gin.put(team, p,
              recvWin, /*dstOff=*/(size_t)myRank * slotStrideBytes,
              sendWin, /*srcOff=*/(size_t)p     * slotStrideBytes,
              bytesPerSlot,
              ncclGin_SignalInc{sigIdx});
    }
  }
  gin.flush(ncclCoopCta());
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

class GinMPIDeviceTests : public MPITestBase {
 protected:
  // Minimal 64-byte put + waitSignal round-trip from rank 0 to rank 1.
  // Used by the Invalid_*Pool tests to confirm comm bring-up + the GIN
  // data path still work after the runtime clamps an oversized pool.
  void runBasicPutSelfCheck() {
    // Bring up the comm + stream from the fixture.
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    ASSERT_EQ(2, nRanks);

    // Tiny geometry: full-buffer transfer, signal 0, peer is rank 1.
    constexpr size_t          kBufBytes      = 64;
    constexpr size_t          kTransferBytes = 64;
    constexpr ncclGinSignal_t kSigIdx        = 0;
    constexpr int             kPeer          = 1;

    // Allocate symmetric src/dst on every rank; freed on scope exit.
    void* dSrc = nullptr;
    void* dDst = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSrc) (void)ncclMemFree(dSrc);
      if (dDst) (void)ncclMemFree(dDst);
    });

    // Register collective symmetric windows so the device side can address
    // peer memory through srcWin/dstWin.
    ncclWindow_t srcWin = nullptr, dstWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
      if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    });

    // Bring up the GIN device comm (1 barrier slot, 1 signal cell).
    ncclDevCommRequirements reqs{};
    reqs.railGinBarrierCount = 1;
    reqs.ginSignalCount      = 1;
    ncclDevComm devComm{};
    ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
    auto devCommCleanup = makeScopeGuard([&]() {
      (void)ncclDevCommDestroy(comm, &devComm);
    });

    // Stage a deterministic byte pattern in the source buffer; dst stays zero.
    std::vector<uint8_t> hostSrc(kBufBytes, 0);
    std::vector<uint8_t> hostDst(kBufBytes, 0);
    for (size_t i = 0; i < kTransferBytes; i++) {
      hostSrc[i] = static_cast<uint8_t>(0xA0 + (i & 0x3F));
    }
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

    // Sync so neither rank launches before the other has finished setup.
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 puts the payload + bumps the signal; rank 1 waits on it.
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

    // Sync so rank 1's verify isn't racing rank 0's kernel completion.
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 1 reads dst back and checks every byte landed.
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

// Smallest end-to-end exercise of the device put -> proxy -> IB -> peer
// signal chain, with non-zero src/dst/signal offsets so address-arithmetic
// regressions surface as a verification mismatch.
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

  // 8 KiB symmetric buffers; 4 KiB transfer at non-zero src/dst offsets;
  // signal at non-zero index. Forces every offset to be exercised.
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;

  // Allocate symmetric src/dst on every rank.
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register collective windows over the symmetric buffers.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN with 2 signals so kSigIdx=1 is a valid in-pool index.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Stage source pattern in [kSrcOff, kSrcOff+kTransferBytes); rest is zero.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (size_t i = 0; i < kTransferBytes; i++) {
    hostSrc[kSrcOff + i] = static_cast<uint8_t>(0x40 + (i & 0xFF));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches its kernel before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts payload + bumps signal; rank 1 waits on the same signal.
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

  // Sync before verify; both ranks have finished their kernel.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies: payload landed exactly in [kDstOff, +kTransferBytes),
  // and bytes before/after that range are still zero.
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
}

// Same wire-level put as Put_BasicAndOffsets, but requires the two ranks to
// live on different physical nodes so IB actually traverses the fabric
// instead of falling back to a single-node loopback path.
TEST_F(GinMPIDeviceTests, Put_CrossNode) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Skip on single-node runs; otherwise we'd just be retesting loopback.
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(2, nRanks);

  // Same geometry as Put_BasicAndOffsets so the comparison is apples-to-apples.
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;

  // Allocate symmetric src/dst on every rank.
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register collective windows over the symmetric buffers.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN with 2 signals so kSigIdx=1 is valid.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Stage source pattern in [kSrcOff, kSrcOff+kTransferBytes); rest zero.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (size_t i = 0; i < kTransferBytes; i++) {
    hostSrc[kSrcOff + i] = static_cast<uint8_t>(0x40 + (i & 0xFF));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts + signals; rank 1 waits.
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

  // Verify payload + zero-tails on rank 1.
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
}

// Producer: putValue carries the 8-byte payload inside the GFD itself
// (no source MR is dereferenced); also bumps a signal so the receiver
// can wait synchronously.
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
  gin.flush(ncclCoopCta());
}

// Consumer: just waits for the inline-value signal.
__global__ void putValueInlineConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Sends a single 8-byte uint64_t inline (no source buffer / MR involved)
// and verifies it lands intact at the requested offset on the peer.
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

  // kValue exercises all three pieces of the inline 4+2+2 byte split,
  // so any field-reconstruction regression in the host proxy surfaces.
  constexpr size_t   kBufBytes = 4 * 1024;
  constexpr size_t   kDstOff   = 1 * 1024;
  constexpr uint64_t kValue    = 0x123456789ABCDEF0ULL;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;

  // Allocate symmetric dst on every rank (no src needed: value is inline).
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register dst window collectively.
  ncclWindow_t dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN with 2 signals so kSigIdx=1 is valid.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Zero dst so any spurious write outside the 8-byte landing surfaces.
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 sends the inline value + signal; rank 1 waits.
  if (rank == 0) {
    putValueInlineProducerKernel<<<1, 32, 0, stream>>>(
        dstWin, kDstOff, kValue, kSigIdx, kPeer, devComm);
  } else {
    putValueInlineConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies the 8 bytes at dstOff match kValue and nothing else moved.
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
}

// Producer: thread 0 issues a zero-byte gin.signal (no src/dst windows).
// The GFD becomes a non-inline put with size=0, hasInline=false; the proxy
// dispatches it through the signal-only iputSignal path
// (gin_host_proxy.cc:271-273).
__global__ void signalNoPayloadProducerKernel(
    ncclGinSignal_t sigIdx, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.signal(ncclTeamWorld(devComm), peer, ncclGin_SignalInc{sigIdx});
  }
  // Drain the posted GFD before the kernel exits.
  gin.flush(ncclCoopCta());
}

// Consumer: whole CTA cooperatively waits for the signal to reach the target.
__global__ void signalNoPayloadConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Bare-minimum signal RTT: rank 0 issues a zero-byte gin.signal that only
// bumps rank 1's signal cell -- no data, no src/dst windows. It's the
// latency-floor primitive the barrier is built on and exercises the proxy's
// signal-only path. Signal index is non-zero so a regression that drops the
// index multiplier would land at cell 0 and the consumer's waitSignal(1)
// would never observe the bump.
TEST_F(GinMPIDeviceTests, Signal_NoPayload) {
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

  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int             kPeer   = 1;  // rank 0 -> rank 1

  // No buffers, no windows: gin.signal is a zero-byte put. The runtime's
  // own signal pool is the only memory touched by the GFD; it's allocated
  // and registered by ncclDevCommCreate based on ginSignalCount.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Sync so neither rank launches its kernel before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Producer (rank 0) emits a single signal; consumer (rank 1) waits for
  // signal cell kSigIdx to reach 1. The successful return of waitSignal on
  // the consumer IS the assertion -- there is no payload to verify.
  if (rank == 0) {
    signalNoPayloadProducerKernel<<<1, 32, 0, stream>>>(kSigIdx, kPeer, devComm);
  } else {
    signalNoPayloadConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Both kernels are done; rank 1's signal cell is settled.
  MPI_Barrier(MPI_COMM_WORLD);
}

// Producer-only kernel: rank 0 issues a put with no remote action and a
// CounterInc local action, then waits on its OWN counter. The counter is
// bumped by the local proxy thread when the IB CQE for the put lands
// (gin_host_proxy.cc:147-152, inside proxyGinPollCompletions). That's a
// distinct path from the remote-signal write dispatched via iputSignal
// (gin_host_proxy.cc:271-273) which the prior put/signal tests exercise.
__global__ void waitCounterLocalProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, ncclGinCounter_t cntIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, dstOff, srcWin, srcOff, bytes,
            ncclGin_None{},                  // no remote action (no signal)
            ncclGin_CounterInc{cntIdx});     // local: bump cntIdx on IB CQE
  }
  // CTA-collective wait; only thread 0 spins on the cell. Once it returns,
  // the proxy thread has observed the IB CQE for the put and bumped the
  // counter.
  gin.waitCounter(ncclCoopCta(), cntIdx, /*least=*/1);
}

// Exercises the local IB-completion -> counter-bump path. Rank 1 is a
// passive RDMA target (no kernel) - the payload lands in its dst window
// via the IB plugin. The successful return of waitCounter(1) on rank 0
// (gated by hipStreamSynchronize) IS the assertion.
TEST_F(GinMPIDeviceTests, WaitCounter_Local) {
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

  // 8 KiB symmetric buffers; 4 KiB payload at non-zero src/dst offsets so
  // the put is a realistic data move. The counter (not the payload) is
  // what we verify -- this test is about the LOCAL completion path.
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinCounter_t kCntIdx = 1;  // non-zero counter index
  constexpr int kPeer = 1;

  // Symmetric src + dst (every rank allocates because window registration
  // is collective for SYMMETRIC-mode windows).
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

  // ginCounterCount=2 so kCntIdx=1 is in range; no ginSignalCount needed
  // (the put has no remote-signal action). railGinBarrierCount=1 because
  // the runtime allocates per-CTA barrier state for any 1-CTA launch.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginCounterCount     = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Known fill on src so the put has real bytes to move; both ranks zero
  // their dst as a baseline. We don't verify the destination -- this test
  // is about the local counter, not data delivery.
  std::vector<uint8_t> hostSrc(kBufBytes, 0xCC);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches its kernel before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Only rank 0 launches a kernel; the local counter lives on rank 0. Rank
  // 1 is purely the RDMA target -- the IB plugin lands the payload in its
  // dst window passively.
  if (rank == 0) {
    waitCounterLocalProducerKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff,
        kTransferBytes, kCntIdx, kPeer, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Sync so ScopeGuards (window/comm teardown is collective) see both
  // ranks past the kernel phase.
  MPI_Barrier(MPI_COMM_WORLD);
}

// Producer kernel: rank 0 issues a single put carrying BOTH a remote
// SignalInc action and a local CounterInc action, then waits on its OWN
// counter. The remote SignalInc bumps the peer's signal cell when the IB
// write lands at the target (gin_host_proxy.cc:271-273); the local
// CounterInc bumps rank 0's counter when the IB CQE for the put is
// observed by the local proxy (gin_host_proxy.cc:147-152). One put,
// two completion sites.
__global__ void waitCounterAndSignalProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes,
    ncclGinSignal_t sigIdx, ncclGinCounter_t cntIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, dstOff, srcWin, srcOff, bytes,
            ncclGin_SignalInc{sigIdx},        // remote: bump peer's signal cell
            ncclGin_CounterInc{cntIdx});      // local: bump cntIdx on IB CQE
  }
  // CTA-collective wait gating kernel exit on local CQ completion. On the
  // proxy backend, the local CQE strictly follows the remote SignalInc
  // landing, so once this returns the consumer's waitSignal is unblocked
  // (or already past).
  gin.waitCounter(ncclCoopCta(), cntIdx, /*least=*/1);
}

// Consumer kernel: rank 1 waits on its signal cell to reach 1. No put is
// issued from this side -- the IB plugin lands rank 0's payload + signal
// passively.
__global__ void waitCounterAndSignalConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Exercises both completion sites of a single gin.put with combined
// (SignalInc, CounterInc) actions:
//   - Remote signal notification: rank 1's waitSignal returns when the
//     IB write of the SignalInc lands on its signal cell.
//   - Local IB-CQE -> counter bump: rank 0's waitCounter returns when the
//     local proxy thread has observed the CQE for the put.
// Together with the post-sync host-side payload check on rank 1, this
// proves the same put delivered (a) bytes, (b) remote signal, and
// (c) local counter -- all driven by one gin.put.
TEST_F(GinMPIDeviceTests, WaitCounterAndSignal) {
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

  // 8 KiB symmetric buffers; 4 KiB payload at non-zero src/dst offsets so
  // the put is a realistic data move. Both signal and counter indices are
  // non-zero so a regression that drops the index multiplier would land
  // at cell 0 and the waits would never observe the bump.
  constexpr size_t kBufBytes      = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t  kSigIdx = 1;
  constexpr ncclGinCounter_t kCntIdx = 1;
  constexpr int kPeer = 1;

  // Symmetric src + dst on every rank (window registration is collective
  // for SYMMETRIC-mode windows).
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

  // ginSignalCount=2 and ginCounterCount=2 so kSigIdx=1 and kCntIdx=1 are
  // both in range. railGinBarrierCount=1 covers the 1-CTA launch.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  reqs.ginCounterCount     = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Deterministic byte pattern in src so the post-sync data check on
  // rank 1 catches wrong-bytes / wrong-offset regressions. Both ranks
  // zero their dst as a baseline.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (size_t i = 0; i < kTransferBytes; i++) {
    hostSrc[kSrcOff + i] = static_cast<uint8_t>(0x40 + (i & 0xFF));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts + bumps remote signal + bumps local counter; rank 1 waits
  // for the signal. Each side's gin wait IS its assertion (counter on
  // rank 0, signal on rank 1); the host-side memcmp on rank 1 below
  // verifies the payload.
  if (rank == 0) {
    waitCounterAndSignalProducerKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff,
        kTransferBytes, kSigIdx, kCntIdx, kPeer, devComm);
  } else {
    waitCounterAndSignalConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Sync so rank 1's verify isn't racing rank 0's kernel completion.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies the payload landed at [kDstOff, kDstOff+kTransferBytes)
  // and nothing outside that range was touched.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kDstOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " before dstOff was unexpectedly written";
    }
    for (size_t i = 0; i < kTransferBytes; i++) {
      ASSERT_EQ(hostSrc[kSrcOff + i], hostResult[kDstOff + i])
          << "payload byte " << i << " mismatched";
    }
    for (size_t i = kDstOff + kTransferBytes; i < kBufBytes; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " after dstOff+kTransferBytes was unexpectedly written";
    }
  }

  // Final sync so collective teardown (ScopeGuards) see both ranks past
  // the verification phase.
  MPI_Barrier(MPI_COMM_WORLD);
}

// Collective kernel: every rank runs the same code. The barrier composes
// signal + waitSignal + an epoch counter (gin_barrier__funcs.h:42-60) -- each
// sync sends SignalInc to every other rank and waits for the local signal
// cell to reach a higher epoch, so consecutive syncs don't collide on the
// same expected value. The ncclTeamTagRail{} overload routes to
// comm.railGinBarrier and the rail team automatically.
__global__ void barrier2RanksKernel(int iters, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  // barrierIndex must be < railGinBarrierCount (we set =1, so 0).
  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagRail{}, /*barrierIndex=*/0};
  for (int i = 0; i < iters; i++) {
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  }
}

// First test that actually USES the railGinBarrier resource (prior tests
// only provisioned it). Kernel completion after kIters rounds proves both
// ranks moved through the barrier in lockstep -- if the epoch math broke,
// iter 1 would deadlock waiting for an epoch the peer never reaches.
TEST_F(GinMPIDeviceTests, Barrier_TwoRanks) {
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

  // Large enough that a stuck-at-zero epoch would deadlock at iter 1
  // (not silently pass on iter 0); small enough to stay fast. Each round
  // sends 1 SignalInc per peer (1 in the 2-rank case) and waits for the
  // corresponding signal cell to reach the new epoch.
  constexpr int kIters = 16;

  // railGinBarrierCount=1 covers our 1-CTA launch (barrierIndex=0). The
  // barrier session uses the rail-team GIN barrier pool that the runtime
  // allocates via ncclGinBarrierCreateRequirement; its signal cells live
  // at (handle.signal0 + barrierIndex), separate from the user-facing
  // ginSignalCount pool.
  //
  // ginForceEnable is REQUIRED. ncclDevCommCreateInternal only activates
  // GIN (populating ginHandles[]/ginSignalBase) when nNodes>1 OR
  // ginForceEnable OR ginSignalCount!=0 OR ginCounterCount!=0 -- the gate
  // intentionally ignores railGinBarrierCount. Without this flag a
  // single-node test that asks only for railGinBarrierCount gets NULL
  // ginHandles[0], and the barrier's internal waitSignal -> getSignalPtr
  // dispatch dereferences a NULL ncclGinProxyGpuCtx and faults at (nil)
  // on the GPU. Other tests dodge the gate accidentally by setting
  // ginSignalCount/ginCounterCount non-zero; this one has neither so we
  // ask for GIN explicitly.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginForceEnable      = true;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // All ranks finished setup before any kernel launches.
  MPI_Barrier(MPI_COMM_WORLD);

  // Both ranks launch the same kernel -- barrier is collective, no
  // producer/consumer split. Each rank's iter i sends SignalInc to the
  // other rank and waits for its own signal cell to reach epoch+1; if the
  // barrier raced past the peer (epoch broken), iter i+1 would either spin
  // forever or read a stale epoch. Kernel completion means all kIters
  // rounds stayed in lockstep.
  barrier2RanksKernel<<<1, 32, 0, stream>>>(kIters, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Successful return of the kernel IS the assertion: signal + waitSignal
  // + epoch all worked together for kIters rounds.
  MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// SignalAdd_AndShadow
//   Walks the signal-shadow API family end-to-end. SignalAdd (variable add)
//   takes a different proxy host path than SignalInc (+1):
//   gin_proxy.h -> ncclGinProxyOpWithSignalAdd. The shadow-side calls
//   (readSignal, increaseSignalShadow, waitSignalMeetShadow,
//   waitSignalFollowShadow) are not exercised by any other test. The shadow
//   lives in GPU memory at comm.ginSignalShadows + ... so it persists across
//   kernel launches on the same rank; only increaseSignalShadow and
//   resetSignal mutate it.
//
//   Phases (rank 0 = producer, rank 1 = consumer):
//     P1: producer 1x put(SignalAdd+5, CounterInc); waitCounter(1).
//         consumer waitSignal(5); readSignal -> must be 5.
//     P2: producer 3x put(SignalAdd+7, CounterInc); waitCounter(4).
//         consumer waitSignalFollowShadow(leastDelta=1, &b, &d) -> b=0,
//         d=26 (signal moved 0->26 since the last shadow snapshot). Then
//         increaseSignalShadow(+100): shadow := 126.
//     P3: producer 1x put(SignalAdd+100); no counter; drained via flush.
//         consumer waitSignalMeetShadow returns when signal >= 126.
//     P4: producer readCounter -> must be 4 (4 counter-bearing puts; P3
//         carried no counter).
//
//   Why waitCounter at the end of P1/P2 producer kernels: local CQ
//   completion (counter bump) implies the remote SignalAdd write for that
//   put has landed on the consumer's signal cell. Without it, the
//   inter-phase MPI_Barrier would not guarantee in-flight signal writes
//   have reached the peer, and the consumer could observe an intermediate
//   signal value (e.g. 5 instead of 26 in P2).
//
//   P2 ordering quirk: waitSignalFollowShadow(leastDelta=1) returns as soon
//   as signal > shadow. If P2's puts and the consumer's wait launched
//   concurrently, the consumer could observe signal=5 (still just P1) and
//   return with delta=5 -- the d==26 assertion would fail spuriously. So
//   P2 producer runs to completion first, then an MPI_Barrier, then the P2
//   consumer launches. P1 and P3 don't have this issue and run concurrently.
// ---------------------------------------------------------------------------

// Producer with counter. Used by P1 (nPuts=1, signalDelta=5) and P2
// (nPuts=3, signalDelta=7). waitCounter at the end gates exit on local CQ
// completion for all `nPuts` puts, which on the proxy backend implies the
// remote SignalAdd writes have landed at the peer.
__global__ void signalAddProducerWithCounterKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes,
    ncclGinSignal_t sigIdx, uint64_t signalDelta,
    ncclGinCounter_t cntIdx, uint64_t counterExpected,
    int nPuts, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < nPuts; i++) {
      gin.put(ncclTeamWorld(devComm), peer,
              dstWin, dstOff, srcWin, srcOff, bytes,
              ncclGin_SignalAdd{sigIdx, signalDelta},   // remote: bump peer's signal by delta
              ncclGin_CounterInc{cntIdx});              // local: bump cntIdx on IB CQE
    }
  }
  gin.waitCounter(ncclCoopCta(), cntIdx, counterExpected);
}

// Producer without counter. Used by P3: a single put carrying only
// ncclGin_SignalAdd{sig, 100}. flush() drains posted GFDs before exit so
// the kernel doesn't return while the SignalAdd is still in flight on the
// proxy.
__global__ void signalAddProducerNoCounterKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes,
    ncclGinSignal_t sigIdx, uint64_t signalDelta,
    int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, dstOff, srcWin, srcOff, bytes,
            ncclGin_SignalAdd{sigIdx, signalDelta});
  }
  gin.flush(ncclCoopCta());
}

// Producer P4: read the counter back. Must be 4 = 1 (P1) + 3 (P2); P3 had
// no counter so it cannot have bumped it.
__global__ void signalAddProducerReadCounterKernel(
    ncclGinCounter_t cntIdx, uint64_t* outCounter,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outCounter = gin.readCounter(cntIdx);
  }
}

// Consumer P1: waitSignal(>=5) then readSignal. After P1's single
// SignalAdd+5 the signal cell is exactly 5, so readSignal must return 5.
__global__ void signalAddConsumerPhase1Kernel(
    ncclGinSignal_t sigIdx, uint64_t least,
    uint64_t* outReadSignal,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, least);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outReadSignal = gin.readSignal(sigIdx);
  }
}

// Consumer P2: waitSignalFollowShadow(leastDelta=1) returns once
// signal > shadow by >= 1; P2 producer's waitCounter(4) guarantees signal
// has reached 5+3*7=26 here, so before==0 (initial shadow) and delta==26.
// Then increaseSignalShadow(+100) raises shadow from 26 to 126 for P3.
__global__ void signalAddConsumerPhase2Kernel(
    ncclGinSignal_t sigIdx,
    uint64_t leastDelta,
    uint64_t shadowBump,
    uint64_t* outBefore, uint64_t* outDelta,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  uint64_t before = 0, delta = 0;
  gin.waitSignalFollowShadow(ncclCoopCta(), sigIdx, leastDelta, &before, &delta);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outBefore = before;
    *outDelta  = delta;
    gin.increaseSignalShadow(sigIdx, shadowBump);
  }
}

// Consumer P3: shadow was raised to 126 in P2; this waits for the
// producer's SignalAdd+100 to bring signal from 26 -> 126.
__global__ void signalAddConsumerPhase3Kernel(
    ncclGinSignal_t sigIdx, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignalMeetShadow(ncclCoopCta(), sigIdx);
}

TEST_F(GinMPIDeviceTests, SignalAdd_AndShadow) {
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

  // The put is just a vehicle for the SignalAdd + CounterInc actions; 8
  // bytes is the smallest reasonable payload. Bytes correctness is covered
  // by Put_BasicAndOffsets and is not asserted here.
  constexpr size_t kBufBytes      = 64;
  constexpr size_t kTransferBytes = 8;
  constexpr size_t kSrcOff        = 0;
  constexpr size_t kDstOff        = 0;
  constexpr ncclGinSignal_t  kSigIdx = 1;  // non-zero so signal-pool indexing is exercised
  constexpr ncclGinCounter_t kCntIdx = 1;
  constexpr int kPeer = 1;  // rank 0 -> rank 1

  // Symmetric src + dst (window registration is collective for
  // SYMMETRIC-mode windows, so every rank allocates).
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

  // sigIdx=1 and cntIdx=1 -> ginSignalCount/ginCounterCount must be >= 2.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  reqs.ginCounterCount     = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Producer needs a known fill (bytes aren't asserted but the put must
  // have something to move). Consumer's dst gets zeroed.
  if (rank == 0) {
    std::vector<uint8_t> hostSrc(kBufBytes, 0xA5);
    ASSERT_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  }
  std::vector<uint8_t> hostZero(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostZero.data(), kBufBytes, hipMemcpyHostToDevice));

  // Per-rank result buffer. Layout (uint64_t slots):
  //   producer (rank 0): [0] readCounter (P4)
  //   consumer (rank 1): [0] readSignal (P1), [1] before (P2), [2] delta (P2)
  // Initialized to a sentinel so an unwritten slot is obvious in failure logs.
  constexpr size_t kResultSlots = 4;
  constexpr uint64_t kSentinel  = 0xFFFFFFFFFFFFFFFFull;
  uint64_t* dResults = nullptr;
  ASSERT_EQ(hipSuccess, hipMalloc(&dResults, kResultSlots * sizeof(uint64_t)));
  auto resultsCleanup = makeScopeGuard([&]() { if (dResults) (void)hipFree(dResults); });
  std::vector<uint64_t> hostInit(kResultSlots, kSentinel);
  ASSERT_EQ(hipSuccess,
            hipMemcpy(dResults, hostInit.data(), kResultSlots * sizeof(uint64_t),
                      hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  // --- P1: producer 1x SignalAdd+5/CounterInc, consumer waitSignal(5) +
  //         readSignal. Concurrent: the consumer's waitSignal blocks until
  //         the put lands, and no later producer activity is in flight that
  //         could push signal>5.
  if (rank == 0) {
    signalAddProducerWithCounterKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/5,
        kCntIdx, /*counterExpected=*/1,
        /*nPuts=*/1, kPeer, devComm);
  } else {
    signalAddConsumerPhase1Kernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*least=*/5, &dResults[0], devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // --- P2: producer 3x SignalAdd+7/CounterInc, ending with waitCounter(4).
  //         MUST complete BEFORE the consumer's waitSignalFollowShadow runs
  //         (see banner: leastDelta=1 could otherwise observe signal=5 from
  //         P1 and return with delta=5 instead of 26).
  if (rank == 0) {
    signalAddProducerWithCounterKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/7,
        kCntIdx, /*counterExpected=*/4,
        /*nPuts=*/3, kPeer, devComm);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
  }
  MPI_Barrier(MPI_COMM_WORLD);  // gate consumer P2 on producer P2 done

  if (rank == 1) {
    signalAddConsumerPhase2Kernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*leastDelta=*/1, /*shadowBump=*/100,
        &dResults[1], &dResults[2], devComm);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
  }
  MPI_Barrier(MPI_COMM_WORLD);  // shadow on rank 1 is now 26+100=126

  // --- P3: producer 1x SignalAdd+100 (no counter, drained via flush);
  //         consumer waitSignalMeetShadow returns when signal >= 126.
  //         Concurrent: consumer is gated on the SignalAdd landing.
  if (rank == 0) {
    signalAddProducerNoCounterKernel<<<1, 32, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/100,
        kPeer, devComm);
  } else {
    signalAddConsumerPhase3Kernel<<<1, 32, 0, stream>>>(kSigIdx, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // --- P4: producer reads counter back. The 4 counter-bearing puts (P1: 1,
  //         P2: 3) were all waited for via waitCounter in their kernels;
  //         P3 had no counter. Counter must be exactly 4.
  if (rank == 0) {
    signalAddProducerReadCounterKernel<<<1, 1, 0, stream>>>(
        kCntIdx, &dResults[0], devComm);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Pull results back per-rank and assert. These branches are
  // rank-divergent, so use plain ASSERT_EQ (ASSERT_MPI_EQ would deadlock on
  // its internal MPI_Allreduce).
  std::vector<uint64_t> hostResults(kResultSlots, 0);
  ASSERT_EQ(hipSuccess,
            hipMemcpy(hostResults.data(), dResults, kResultSlots * sizeof(uint64_t),
                      hipMemcpyDeviceToHost));
  if (rank == 0) {
    ASSERT_EQ(4ull, hostResults[0]) << "P4 readCounter must equal 4 (1 from P1 + 3 from P2)";
  } else {
    ASSERT_EQ(5ull,  hostResults[0]) << "P1 readSignal must equal 5";
    ASSERT_EQ(0ull,  hostResults[1]) << "P2 waitSignalFollowShadow before must be 0";
    ASSERT_EQ(26ull, hostResults[2]) << "P2 waitSignalFollowShadow delta must be 26";
  }

  // ScopeGuards run in reverse order on return.
}

// Producer: drives BOTH symPtr-form shims in a single kernel:
//   1) gin.put<float>(team, peer, dstSym, srcSym, nElts, SignalInc) -- the
//      typed-pointer put shim (gin__funcs.h:168) unpacks
//      ncclSymPtr<T>::{window,offset} and dispatches to the window-form put
//      with byte-size = nElts*sizeof(T).
//   2) gin.putValue<uint32_t>(team, peer, valDstSym, value, SignalInc) --
//      the typed-pointer putValue shim (gin__funcs.h:231) unpacks valDstSym
//      and dispatches to the window-form putValue with the value carried
//      inline in the GFD itself. This drives the 4-byte inline branch of
//      buildGfd (only inlineLow.inlineValLow set, inlineValLow2/inlineValHigh
//      stay zero because sizeof(T) is not > 4 and not > 6 -- gin_proxy.h:84-95),
//      distinct from the uint64_t 4+2+2 split exercised by PutValue_Inline.
// Both posts bump kSigIdx by 1 via SignalInc; the consumer waits on 2.
__global__ void symPtrProducerKernel(
    ncclSymPtr<float> dstSym, ncclSymPtr<float> srcSym, size_t nElts,
    ncclSymPtr<uint32_t> valDstSym, uint32_t inlineValue,
    ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstSym, srcSym, nElts,
            ncclGin_SignalInc{sigIdx});
    gin.putValue<uint32_t>(ncclTeamWorld(devComm), peer,
                           valDstSym, inlineValue,
                           ncclGin_SignalInc{sigIdx});
  }
  // Drain both posted GFDs before exit.
  gin.flush(ncclCoopCta());
}

// Consumer: whole CTA waits for both signal increments from the two posts.
__global__ void symPtrConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Exercises the typed-pointer shims (ncclSymPtr<T>) for both gin.put and
// gin.putValue. Rank 0 sends a 512-float array via symPtr put and a 4-byte
// uint32 sentinel via symPtr putValue<uint32_t>; rank 1 verifies both
// land at their respective offsets and nothing else moved.
TEST_F(GinMPIDeviceTests, SymPtr_PutAndPutValue) {
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

  // Layout: src window holds a 512-float array starting at srcOff. dst
  // window receives the float array at dstOff (deliberately != srcOff to
  // exercise non-zero offsets), and the 4-byte uint32 inline value lands
  // at valOff (well past the float region so a misrouted put surfaces as
  // a verification mismatch). No src memory is needed for the inline
  // putValue -- the 4 bytes ride inside the GFD itself.
  constexpr size_t   kBufBytes    = 8 * 1024;
  constexpr size_t   kNElts       = 512;
  constexpr size_t   kSrcOff      = 1 * 1024;             // bytes
  constexpr size_t   kDstOff      = 2 * 1024;             // bytes
  constexpr size_t   kValOff      = 6 * 1024;             // bytes (non-overlapping)
  constexpr uint32_t kInlineValue = 0xCAFEBABEu;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;

  static_assert(kSrcOff + kNElts * sizeof(float) <= kBufBytes, "src overflow");
  static_assert(kDstOff + kNElts * sizeof(float) <= kBufBytes, "dst overflow");
  static_assert(kValOff + sizeof(uint32_t)       <= kBufBytes, "val overflow");
  static_assert(kValOff >= kDstOff + kNElts * sizeof(float),
                "val region must not overlap dst float region");

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

  // ginSignalCount=2 so kSigIdx=1 is in range.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Rank 0 fills its src float array with a deterministic pattern. Both
  // ranks zero their dst so any spurious write outside [dstOff, dstOff+xfer)
  // or near kValOff surfaces as a verification mismatch.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  if (rank == 0) {
    float* srcFloats = reinterpret_cast<float*>(hostSrc.data() + kSrcOff);
    for (size_t i = 0; i < kNElts; i++) {
      srcFloats[i] = static_cast<float>(i) + 0.5f;
    }
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches its kernel before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Producer issues both the float-array put and the inline uint32 putValue
  // from a single kernel; consumer waitSignal(2) drains both. Two posts
  // each bump kSigIdx by 1 via SignalInc, so the final value is 2.
  if (rank == 0) {
    ncclSymPtr<float>    srcSym(srcWin, kSrcOff);
    ncclSymPtr<float>    dstSym(dstWin, kDstOff);
    ncclSymPtr<uint32_t> valDstSym(dstWin, kValOff);
    symPtrProducerKernel<<<1, 32, 0, stream>>>(
        dstSym, srcSym, kNElts,
        valDstSym, kInlineValue,
        kSigIdx, kPeer, devComm);
  } else {
    symPtrConsumerKernel<<<1, 32, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/2, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Receiver-side verification.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    // 1) The float array landed at dstOff and matches rank 0's source.
    const float* gotFloats = reinterpret_cast<const float*>(hostResult.data() + kDstOff);
    for (size_t i = 0; i < kNElts; i++) {
      const float expected = static_cast<float>(i) + 0.5f;
      ASSERT_EQ(expected, gotFloats[i])
          << "float[" << i << "] mismatch at dstOff via symPtr put";
    }
    // 2) The 4-byte uint32 inline value landed at valOff. These bytes came
    //    from the inline GFD slot (putValue<uint32_t>), not from any source
    //    MR -- the symPtr shim only carried the destination offset.
    uint32_t gotInline = 0;
    std::memcpy(&gotInline, hostResult.data() + kValOff, sizeof(gotInline));
    ASSERT_EQ(kInlineValue, gotInline)
        << "inline uint32 mismatch at valOff (symPtr putValue path)";
    // 3) Nothing was written outside the two destination regions.
    for (size_t i = 0; i < kDstOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " before float-dst region was unexpectedly written";
    }
    for (size_t i = kDstOff + kNElts * sizeof(float); i < kValOff; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " between float-dst and val-dst was unexpectedly written";
    }
    for (size_t i = kValOff + sizeof(uint32_t); i < kBufBytes; i++) {
      ASSERT_EQ(0u, hostResult[i])
          << "byte " << i << " after val-dst region was unexpectedly written";
    }
  }
}

// Reference single-node alltoall: every rank puts its slice for every peer
// into that peer's recvbuf via gin.put + SignalInc, then waits on its own
// signal cell for nRanks increments. Ported one-for-one from
// examples/06_device_api/02_gin_alltoall_pure/main.cu so a regression in
// put + signal + barrier + flush composed under realistic load surfaces here.
__global__ void alltoallPureKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, struct ncclDevComm devComm) {
  constexpr int ginContext = 0;
  constexpr unsigned int signalIndex = 0;
  ncclGin gin{devComm, ginContext};
  // Capture current signal cell so a count-sweep that reuses the same
  // kernel/devComm doesn't conflate increments from past iterations.
  uint64_t signalValue = gin.readSignal(signalIndex);

  // Cross-rank sync ensures every rank's sendbuf is registered before any
  // peer reads from it via gin.put below.
  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamWorld(devComm),
      devComm.railGinBarrier, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Each rank writes its per-peer slice into the destination's recvbuf
  // slot (offset = my rank * size). All puts bump signalIndex by 1 on the
  // destination, so the destination's signal cell receives nRanks
  // increments (one per source rank including self).
  const size_t size = count * sizeof(float);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

// Single-node 2-8 ranks; count sweep {1, 1024, 1<<16}. Bit-for-bit verifies
// the deterministic sendbuf[i] = rank*1000 + dst*100 + i pattern landed on
// the right peer slot. The 1<<16 case (256 KiB / direction / peer) saturates
// ring credit to exercise the postGfd credit-wait path.
TEST_F(GinMPIDeviceTests, Alltoall_PureReference) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_GE(nRanks, 2);
  ASSERT_LE(nRanks, 8);

  // ginSignalCount=1 covers signalIndex=0; railGinBarrierCount=1 matches
  // our single-CTA launch. Both non-zero so GIN activates without
  // ginForceEnable.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // 1 (alignment/tail edges), 1024 (medium), 65536 (saturating).
  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};
  constexpr int kCTAs          = 1;   // == railGinBarrierCount
  constexpr int kThreadsPerCTA = 512; // matches the production example launch

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t totalElements = count * static_cast<size_t>(nRanks);
    const size_t sizeBytes     = totalElements * sizeof(float);

    void* dSend = nullptr;
    void* dRecv = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, sizeBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, sizeBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSend) (void)ncclMemFree(dSend);
      if (dRecv) (void)ncclMemFree(dRecv);
    });

    ncclWindow_t sendWin = nullptr, recvWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dSend, sizeBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dRecv, sizeBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
      if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
    });

    // Per-source/per-dest pattern: every byte of every slice has a unique
    // expected value, so a misrouted slice surfaces as a mismatch.
    std::vector<float> hostSend(totalElements, 0.0f);
    std::vector<float> hostRecv(totalElements, 0.0f);
    for (int dst = 0; dst < nRanks; dst++) {
      for (size_t i = 0; i < count; i++) {
        hostSend[static_cast<size_t>(dst) * count + i] =
            static_cast<float>(rank * 1000 + dst * 100 + static_cast<int>(i));
      }
    }
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dSend, hostSend.data(), sizeBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dRecv, hostRecv.data(), sizeBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    alltoallPureKernel<<<kCTAs, kThreadsPerCTA, 0, stream>>>(
        sendWin, /*sendoffset=*/0,
        recvWin, /*recvoffset=*/0,
        count, devComm);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    // recvbuf[src*count + i] on rank R should equal src*1000 + R*100 + i.
    ASSERT_EQ(hipSuccess,
        hipMemcpy(hostRecv.data(), dRecv, sizeBytes, hipMemcpyDeviceToHost));
    for (int src = 0; src < nRanks; src++) {
      for (size_t i = 0; i < count; i++) {
        const float expected =
            static_cast<float>(src * 1000 + rank * 100 + static_cast<int>(i));
        const float got = hostRecv[static_cast<size_t>(src) * count + i];
        ASSERT_EQ(expected, got)
            << "rank=" << rank << " src=" << src << " i=" << i;
      }
    }
  }
}

// Hybrid alltoall: LSA stores for intra-node peers + gin.put for cross-node
// peers, bracketed by entry/exit barriers. Ported from
// examples/06_device_api/03_gin_alltoall_hybrid/main.cu (HybridAlltoAllKernel).
// On single-node nNodes==1 so world.nRanks==lsa.nRanks; numRemotePeers==0
// and the gin.put loops are no-ops -- this run primarily exercises the
// LSA-store path + entry/exit barrier composition. The cross-node arm is
// covered by Alltoall_CrossNode per the plan.
__global__ void alltoallHybridKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, struct ncclDevComm devComm) {
  constexpr int ginContext = 0;
  constexpr unsigned int signalIndex = 0;
  ncclGin gin{devComm, ginContext};
  uint64_t signalValue = gin.readSignal(signalIndex);

  // ncclBarrierSession (not the gin-only variant) routes through both LSA
  // and rail-team GIN under ncclTeamTagWorld(). Single-node degenerates to
  // an LSA-team barrier; multi-node crosses rails.
  ncclBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa   = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;

  // Cross-node peers: covers everything outside this node's LSA team
  // (no-op on single-node).
  const size_t size = count * sizeof(float);
  for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }
  for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  // Intra-node peers: write each LSA peer's slot in their recv buffer
  // directly. ncclGetLocalPointer resolves our own sendbuf;
  // ncclGetLsaPointer resolves an LSA peer's recvbuf into our address space.
  float* sendLocal = (float*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      float* recvPtr = (float*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }

  // Only wait for remote peers' increments; LSA peers don't bump the
  // signal cell. On single-node this returns immediately (delta=0).
  int numRemotePeers = world.nRanks - lsa.nRanks;
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
  gin.flush(ncclCoopCta());

  // Final barrier: every rank's LSA stores into peer recvbufs are issued
  // and the local writes from peers are visible before any rank consumes
  // its recvbuf.
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

// Same shape and verification as Alltoall_PureReference but exercises the
// LSA-store path and the dual-pool barrier. On single-node the gin.put arm
// is a no-op; the test still validates LSA + rail-team-GIN barrier
// composition + readSignal/waitSignal plumbing with delta=0.
TEST_F(GinMPIDeviceTests, AlltoallHybrid_Reference) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_GE(nRanks, 2);
  ASSERT_LE(nRanks, 8);

  // Both barrier pools (LSA + rail-GIN) are used by ncclBarrierSession
  // under ncclTeamTagWorld(); both must cover our single-CTA launch
  // (barrierIndex=0). ginSignalCount=1 covers the cross-node signal cell
  // and triggers GIN activation.
  ncclDevCommRequirements reqs{};
  reqs.lsaBarrierCount     = 1;
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};
  constexpr int kCTAs          = 1;
  constexpr int kThreadsPerCTA = 512;

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t totalElements = count * static_cast<size_t>(nRanks);
    const size_t sizeBytes     = totalElements * sizeof(float);

    void* dSend = nullptr;
    void* dRecv = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, sizeBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, sizeBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSend) (void)ncclMemFree(dSend);
      if (dRecv) (void)ncclMemFree(dRecv);
    });

    ncclWindow_t sendWin = nullptr, recvWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dSend, sizeBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dRecv, sizeBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
      if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
    });

    // Same deterministic pattern as Alltoall_PureReference.
    std::vector<float> hostSend(totalElements, 0.0f);
    std::vector<float> hostRecv(totalElements, 0.0f);
    for (int dst = 0; dst < nRanks; dst++) {
      for (size_t i = 0; i < count; i++) {
        hostSend[static_cast<size_t>(dst) * count + i] =
            static_cast<float>(rank * 1000 + dst * 100 + static_cast<int>(i));
      }
    }
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dSend, hostSend.data(), sizeBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dRecv, hostRecv.data(), sizeBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    alltoallHybridKernel<<<kCTAs, kThreadsPerCTA, 0, stream>>>(
        sendWin, /*sendoffset=*/0,
        recvWin, /*recvoffset=*/0,
        count, devComm);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_EQ(hipSuccess,
        hipMemcpy(hostRecv.data(), dRecv, sizeBytes, hipMemcpyDeviceToHost));
    for (int src = 0; src < nRanks; src++) {
      for (size_t i = 0; i < count; i++) {
        const float expected =
            static_cast<float>(src * 1000 + rank * 100 + static_cast<int>(i));
        const float got = hostRecv[static_cast<size_t>(src) * count + i];
        ASSERT_EQ(expected, got)
            << "rank=" << rank << " src=" << src << " i=" << i;
      }
    }
  }
}

// Producer: one block per GIN context. Block b uses ginContext=b, puts into
// its own slot, and signals signal[b] in that context.
__global__ void multiContextProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin,
    size_t slotStride, size_t bytes, int peer,
    struct ncclDevComm devComm) {
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

// Consumer: block b waits on signal[b] in its own context, mirroring the
// producer-side mapping.
__global__ void multiContextConsumerKernel(
    uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, (int)blockIdx.x};
  gin.waitSignal(ncclCoopCta(), (ncclGinSignal_t)blockIdx.x, expectedSignalValue);
}

// Drives all NCCL_GIN_MAX_CONTEXTS contexts in parallel, each with its own
// slot + per-context signal. Confirms every contextId has a working
// proxy ring + IB QP and that there's no cross-context contamination.
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

  // 4 per-context slots; 1 KiB payload into each 4 KiB slot. The 3 KiB
  // tail per slot is asserted zero to catch cross-context contamination.
  constexpr int    kNumContexts    = NCCL_GIN_MAX_CONTEXTS;  // 4
  constexpr size_t kSlotStride     = 4 * 1024;
  constexpr size_t kTransferBytes  = 1 * 1024;
  constexpr size_t kBufBytes       = kNumContexts * kSlotStride;
  constexpr int    kPeer           = 1;

  // Allocate symmetric src/dst large enough for all contexts' slots.
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register collective windows over the symmetric buffers.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN. ginContextCount on reqs is a hint; the authoritative
  // value is env-driven and read back from devComm. Each block uses a
  // signal id == blockIdx.x, so we need kNumContexts signal cells per ctx.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = kNumContexts;
  reqs.ginSignalCount      = kNumContexts;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Skip if the runtime didn't actually give us the requested number of ctxs.
  if ((int)devComm.ginContextCount != kNumContexts) {
    GTEST_SKIP() << "Test requires " << kNumContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << kNumContexts << ")";
  }

  // Stage a distinct pattern (0x10 + b) per context slot so cross-context
  // landings show up as value mismatches rather than missing writes.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (int b = 0; b < kNumContexts; b++) {
    const uint8_t pattern = static_cast<uint8_t>(0x10 + b);
    std::fill_n(hostSrc.begin() + b * kSlotStride, kTransferBytes, pattern);
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  // Launch kNumContexts blocks on each side; one block per context.
  if (rank == 0) {
    multiContextProducerKernel<<<kNumContexts, 32, 0, stream>>>(
        srcWin, dstWin, kSlotStride, kTransferBytes, kPeer, devComm);
  } else {
    multiContextConsumerKernel<<<kNumContexts, 32, 0, stream>>>(
        /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify every context's slot independently: payload range matches
  // pattern, slot tail is still zero.
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
}

// Exercises the non-power-of-2 context count (3), which forces the modulo
// arm of the ncclGin ctor. Producer launches 6 blocks; pairs (0,3), (1,4),
// (2,5) collide on contextId 0/1/2 and each pair writes a disjoint sub-slot.
// Each producer signals signal 0 of its ctx; consumer waits for value 2.
// Requires NCCL_GIN_NCONTEXTS=3 (skipped otherwise).
__global__ void multiContextNpo2ProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin,
    int numContexts, size_t slotStride, size_t subSlotBytes, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, (int)blockIdx.x};
  const int    ctx       = (int)blockIdx.x % numContexts;
  const int    subSlotIx = (int)blockIdx.x / numContexts;
  const size_t off       = (size_t)ctx * slotStride + (size_t)subSlotIx * subSlotBytes;
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, off,
            srcWin, off,
            subSlotBytes,
            ncclGin_SignalInc{(ncclGinSignal_t)0});
  }
  gin.flush(ncclCoopCta());
}

__global__ void multiContextNpo2ConsumerKernel(
    uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
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

  // 3 ctx slots * 4 KiB stride; 2 sub-slots of 1 KiB per slot (one per
  // producer block hitting that ctx); 2 KiB tail per slot asserted zero.
  constexpr int    kNumContexts    = 3;
  constexpr int    kBlocksPerCtx   = 2;
  constexpr int    kProducerBlocks = kNumContexts * kBlocksPerCtx;
  constexpr size_t kSubSlotBytes   = 1 * 1024;
  constexpr size_t kSlotStride     = 4 * 1024;
  constexpr size_t kBufBytes       = kNumContexts * kSlotStride;
  constexpr int    kPeer           = 1;

  // Allocate symmetric src/dst.
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register collective windows over the symmetric buffers.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN with 3 contexts, 1 signal each (signal pools are per-ctx,
  // so signal 0 is independent across the three contexts).
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = kNumContexts;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Skip unless the runtime gave us exactly 3 contexts (this test is the
  // only thing exercising the modulo arm of the ctor).
  if ((int)devComm.ginContextCount != kNumContexts) {
    GTEST_SKIP() << "Test requires " << kNumContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << kNumContexts << ")";
  }

  // Stage a distinct pattern per producer block (0x10 + b) so a stray
  // landing surfaces as a value mismatch.
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

  // Producer launches 6 blocks (2 per ctx); consumer launches one block
  // per ctx and waits for value 2 on signal 0 of that ctx.
  if (rank == 0) {
    multiContextNpo2ProducerKernel<<<kProducerBlocks, 32, 0, stream>>>(
        srcWin, dstWin, kNumContexts, kSlotStride, kSubSlotBytes, kPeer, devComm);
  } else {
    multiContextNpo2ConsumerKernel<<<kNumContexts, 32, 0, stream>>>(
        /*expectedSignalValue=*/static_cast<uint64_t>(kBlocksPerCtx), devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify both sub-slots of every ctx hold the right pattern, and the
  // tail past the two sub-slots is still zero.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (int ctx = 0; ctx < kNumContexts; ctx++) {
      const size_t base = (size_t)ctx * kSlotStride;
      for (int subSlotIx = 0; subSlotIx < kBlocksPerCtx; subSlotIx++) {
        const int     b       = subSlotIx * kNumContexts + ctx;
        const uint8_t pattern = static_cast<uint8_t>(0x10 + b);
        const size_t  off     = base + (size_t)subSlotIx * kSubSlotBytes;
        for (size_t i = 0; i < kSubSlotBytes; i++) {
          ASSERT_EQ(pattern, hostResult[off + i])
              << "ctx " << ctx << " sub-slot " << subSlotIx
              << " (block " << b << "): byte " << i
              << " mismatched (expected 0x" << std::hex << (int)pattern << ")";
        }
      }
      const size_t tailStart = base + (size_t)kBlocksPerCtx * kSubSlotBytes;
      for (size_t i = tailStart; i < base + kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[i])
            << "ctx " << ctx << ": byte " << (i - base)
            << " in slot tail was unexpectedly written";
      }
    }
  }
}

// Sweep a matrix of aligned + unaligned transfer sizes through the same
// put kernel; reuses one comm / window pair / signal cell across iters
// (signal monotonically bumped, consumer waits for iter+1).
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

  // Aligned + unaligned matrix; the +1 variants force a trailing fragment
  // so RDMA tail-byte handling regressions surface.
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
  constexpr int kPeer = 1;

  // Allocate symmetric src/dst sized for the largest sweep entry.
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kMaxBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kMaxBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });

  // Register collective windows; reused across all iterations.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kMaxBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kMaxBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });

  // Bring up GIN with one signal cell; we'll bump it once per iter.
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Stage src once: src[i] = i & 0xFF. Distinct bytes mod 256 catch
  // byte-shift / off-by-one regressions in the RDMA path.
  std::vector<uint8_t> hostSrc(kMaxBytes, 0);
  for (size_t i = 0; i < kMaxBytes; i++) hostSrc[i] = static_cast<uint8_t>(i & 0xFF);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kMaxBytes, hipMemcpyHostToDevice));

  // Reusable buffers: zero pattern for clearing dst, scratch for verify.
  const std::vector<uint8_t> hostZero(kMaxBytes, 0);
  std::vector<uint8_t> hostResult(kMaxBytes, 0);

  uint64_t signalExpected = 0;
  for (size_t iter = 0; iter < kSizes.size(); iter++) {
    const size_t sz = kSizes[iter];

    // Clear dDst so the previous iter's larger payload doesn't leak into
    // this iter's tail-byte assertions.
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostZero.data(), kMaxBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    // Each iter bumps signal[0] by 1; consumer waits for the cumulative count.
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

    // Rank 1 verifies: payload [0,sz) matches src; tail [sz,kMaxBytes) zero.
    if (rank == 1) {
      ASSERT_EQ(hipSuccess,
                hipMemcpy(hostResult.data(), dDst, kMaxBytes, hipMemcpyDeviceToHost));

      const int payloadCmp = std::memcmp(hostResult.data(), hostSrc.data(), sz);
      ASSERT_EQ(0, payloadCmp)
          << "size=" << sz << ": payload range [0," << sz
          << ") differs from source";

      if (sz < kMaxBytes) {
        const int tailCmp =
          std::memcmp(hostResult.data() + sz, hostZero.data(), kMaxBytes - sz);
        ASSERT_EQ(0, tailCmp)
            << "size=" << sz << ": bytes in [" << sz << "," << kMaxBytes
            << ") were unexpectedly written";
      }
    }
  }
}

// Negative test: with NCCL_GIN_ENABLE=0, ncclDevCommCreate on a GIN-
// requiring config must fail with ncclInternalError -- no hang, no crash.
TEST_F(GinMPIDeviceTests, Disable_Error) {
  const char* e = std::getenv("NCCL_GIN_ENABLE");
  if (!e || std::strcmp(e, "0") != 0)
    GTEST_SKIP() << "Negative-path test; opt in by setting NCCL_GIN_ENABLE=0";

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Skip ginProxyTestSkipReason: bare comm bring-up does not call into GIN,
  // so the data-path gates don't apply here.
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  // ginSignalCount > 0 forces ncclDevCommCreate to bring up GIN.
  ncclDevCommRequirements reqs{};
  reqs.ginSignalCount = 1;
  ncclDevComm devComm{};
  ncclResult_t r = ncclDevCommCreate(comm, &reqs, &devComm);

  ASSERT_EQ(ncclInternalError, r)
      << "ncclDevCommCreate should fail with ncclInternalError when "
         "NCCL_GIN_ENABLE=0; got result " << (int)r;
}

// With an oversized signal pool the runtime should silently clamp to its
// internal max; confirm the data path still works after the clamp.
TEST_F(GinMPIDeviceTests, Invalid_SignalPool) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Opt in with an oversized signal pool (>= 1 GiB) to exercise the clamp.
  const char* env = std::getenv("NCCL_GIN_SIGNAL_POOL_SIZE");
  if (!env || std::strtoull(env, nullptr, 0) < (1ULL << 30))
    GTEST_SKIP() << "Set NCCL_GIN_SIGNAL_POOL_SIZE>=0x40000000 to opt in";

  // If clamp + data path are healthy, the basic put round-trip succeeds.
  runBasicPutSelfCheck();
}

// Counterpart of Invalid_SignalPool for the counter pool.
TEST_F(GinMPIDeviceTests, Invalid_CounterPool) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Opt in with an oversized counter pool (>= 1 GiB).
  const char* env = std::getenv("NCCL_GIN_COUNTER_POOL_SIZE");
  if (!env || std::strtoull(env, nullptr, 0) < (1ULL << 30))
    GTEST_SKIP() << "Set NCCL_GIN_COUNTER_POOL_SIZE>=0x40000000 to opt in";

  runBasicPutSelfCheck();
}

// Run a put round-trip, then explicitly tear the comm down and check
// cleanup returns success. Failure here indicates a leak / unjoined proxy
// thread / undeleased MR or QP somewhere in ncclGinFinalize.
TEST_F(GinMPIDeviceTests, Teardown_NoLeaks) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // Inner scope so window/devComm/mem guards fire before we call
  // cleanupTestCommunicator() below (they hold refs to the comm).
  {
    // Setup: comm, stream, geometry.
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

    // Allocate symmetric src/dst.
    void* dSrc = nullptr;
    void* dDst = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSrc) (void)ncclMemFree(dSrc);
      if (dDst) (void)ncclMemFree(dDst);
    });

    // Register collective windows.
    ncclWindow_t srcWin = nullptr, dstWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dSrc, kBufBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, dDst, kBufBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
      if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    });

    // Bring up GIN (1 signal cell suffices for a single round-trip).
    ncclDevCommRequirements reqs{};
    reqs.railGinBarrierCount = 1;
    reqs.ginSignalCount      = 1;
    ncclDevComm devComm{};
    ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
    auto devCommCleanup = makeScopeGuard([&]() {
      (void)ncclDevCommDestroy(comm, &devComm);
    });

    MPI_Barrier(MPI_COMM_WORLD);

    // Run a single basic put + waitSignal round-trip.
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
  }  // mem/window/devComm guards fire here while comm is still live.

  // Explicit destroy: success implies ncclGinFinalize ran to completion
  // (proxy thread joined, MR/QP released).
  ASSERT_EQ(ncclSuccess, cleanupTestCommunicator());
  ASSERT_EQ(nullptr, getActiveCommunicator());
  ASSERT_EQ(nullptr, getActiveStream());
}

// Hammer create/destroy in a loop. Catches refcount, mutex, and
// finalize-order regressions that only surface across multiple lifecycles.
TEST_F(GinMPIDeviceTests, Init_Destroy_Stress) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  constexpr int kIterations = 10;
  for (int i = 0; i < kIterations; ++i) {
    // Fresh comm each iter.
    ASSERT_EQ(ncclSuccess, createTestCommunicator()) << "iter " << i;
    ncclComm_t comm = getActiveCommunicator();

    // Allocate + register a tiny window to actually trigger the GIN
    // connect machinery (not just bare comm bring-up).
    void* d = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&d, 64));

    ncclWindow_t win = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, d, 64, &win, NCCL_WIN_COLL_SYMMETRIC));

    MPI_Barrier(MPI_COMM_WORLD);

    // Tear everything down in reverse order before the next iter.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemFree(d));

    ASSERT_EQ(ncclSuccess, cleanupTestCommunicator()) << "iter " << i;
  }
}

// Cross-node alltoall using device-side put. Each rank sends one slot to
// every other rank and waits for nRanks-1 signal increments per iter.
// Sweeps tiny/medium/saturating sizes through the same comm and buffers.
TEST_F(GinMPIDeviceTests, Alltoall_CrossNode) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  // Single-node would loopback IB; require real cross-node ranks.
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2))
    GTEST_SKIP() << "Requires >=2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);

  // Geometry: per-peer slot is sized for the largest sweep entry; smaller
  // counts just use the head of each slot.
  using T = uint32_t;
  static constexpr size_t kCounts[]   = {1, 1u << 10, 1u << 20};
  static constexpr size_t kMaxCount   = 1u << 20;
  const size_t slotStrideBytes        = kMaxCount * sizeof(T);
  const size_t bufBytes               = (size_t)nRanks * slotStrideBytes;
  constexpr ncclGinSignal_t kSigIdx   = 0;

  // Allocate symmetric send/recv buffers (one slot per peer).
  void* dSend = nullptr;
  void* dRecv = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, bufBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, bufBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSend) (void)ncclMemFree(dSend);
    if (dRecv) (void)ncclMemFree(dRecv);
  });

  // Register collective windows over send/recv.
  ncclWindow_t sendWin = nullptr, recvWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSend, bufBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dRecv, bufBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
    if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
  });

  // Bring up GIN with one signal cell (cumulative across iters).
  ncclDevCommRequirements reqs{};
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // Encoding: byte[3] = sender rank, byte[2] = dest rank, byte[1:0] =
  // element index. Lets the receiver decode both source and dest from a slot.
  auto pack = [](int sender, int dest, size_t i) -> T {
    return (T)((((uint32_t)sender & 0xFFu) << 24) |
               (((uint32_t)dest   & 0xFFu) << 16) |
               ((uint32_t)i       & 0xFFFFu));
  };

  // Stage send buffer: slot p contains values addressed to peer p.
  std::vector<T> hostSend((size_t)nRanks * kMaxCount);
  for (int p = 0; p < nRanks; ++p) {
    for (size_t i = 0; i < kMaxCount; ++i) {
      hostSend[(size_t)p * kMaxCount + i] = pack(rank, p, i);
    }
  }
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(dSend, hostSend.data(), bufBytes, hipMemcpyHostToDevice));

  // Cumulative across iters; the signal cell is never reset between iters.
  std::vector<T> hostRecv((size_t)nRanks * kMaxCount);
  uint64_t expectedSignal = 0;

  for (size_t count : kCounts) {
    // Kernel skips p == myRank; fill our self slot locally so verify
    // covers all nRanks rows uniformly.
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpyAsync((uint8_t*)dRecv + (size_t)rank * slotStrideBytes,
                                 (uint8_t*)dSend + (size_t)rank * slotStrideBytes,
                                 count * sizeof(T),
                                 hipMemcpyDeviceToDevice,
                                 stream));

    // We expect to receive nRanks-1 signal increments per iter.
    expectedSignal += (uint64_t)(nRanks - 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Single combined kernel: put to every peer + flush + waitSignal.
    alltoallKernel<<<1, 32, 0, stream>>>(
        sendWin, recvWin,
        count * sizeof(T),
        nRanks, rank,
        slotStrideBytes,
        kSigIdx, expectedSignal,
        devComm);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    // Verify each row of the recv buffer holds the bytes packed by sender r.
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostRecv.data(), dRecv, bufBytes, hipMemcpyDeviceToHost));
    for (int r = 0; r < nRanks; ++r) {
      for (size_t i = 0; i < count; ++i) {
        T expected = pack(r, rank, i);
        T actual   = hostRecv[(size_t)r * kMaxCount + i];
        ASSERT_EQ(expected, actual)
            << "count=" << count << " sender=" << r << " i=" << i;
      }
    }

    // Hold all ranks here so a fast peer can't start the next iter and
    // overwrite a slow rank's recvbuf before it has verified.
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

#endif  // MPI_TESTS_ENABLED