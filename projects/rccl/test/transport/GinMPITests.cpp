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
