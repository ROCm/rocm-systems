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
#include "comm.h"

#include "nccl_device.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <ios>
#include <string>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLTestGuards;

namespace {

std::string ginEnvDisabledReason() {
  if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN explicitly disabled by environment (NCCL_GIN_ENABLE=0)";
  return "";
}

// GIN type requested for this run (proxy / rocshmem-gda / anvil-sdma), 0 if unset.
int requestedGinType() {
  const char* t = std::getenv("NCCL_GIN_TYPE");
  return t ? std::atoi(t) : 0;
}

// Valid NCCL_GIN_TYPE values, derived from the enum so the numbers stay correct
// if the type numbering ever changes.
std::string ginTypeUsage() {
  return "required NCCL_GIN_TYPE=" +
         std::to_string(NCCL_NET_DEVICE_GIN_PROXY) + " [proxy], " +
         std::to_string(NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA) + " [rocshmem-gda] or " +
         std::to_string(NCCL_NET_DEVICE_GIN_ANVIL_SDMA) + " [anvil-sdma]";
}

std::string ginTypeReason() {
  const char* ginType = std::getenv("NCCL_GIN_TYPE");
  if (!ginType)
    return "GIN type not set (" + ginTypeUsage() + ")";
  int t = requestedGinType();
  if (t != NCCL_NET_DEVICE_GIN_PROXY && t != NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA &&
      t != NCCL_NET_DEVICE_GIN_ANVIL_SDMA)
    return std::string("Invalid GIN type: ") + ginType + " (" + ginTypeUsage() + ")";
  return "";
}

std::string cuMemReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || cumem[0] == '\0')
    return "Symmetric memory required (set NCCL_CUMEM_ENABLE to a non-zero value)";
  errno = 0;
  if (std::strtoll(cumem, nullptr, 0) == 0 && errno == 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE must be non-zero)";
  return "";
}

// Number of MPI ranks co-located on this rank's node.
int nodeLocalRanks() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int n = 0;
  MPI_Comm_size(nodeComm, &n);
  MPI_Comm_free(&nodeComm);
  return n;
}

// Collective topology check: every rank receives the same verdict, so an
// uneven placement cannot make one node skip while another enters NCCL setup.
bool uniformNodeLocalRanks(int* ranksPerNode) {
  int local = nodeLocalRanks();
  int minimum = local;
  int maximum = local;
  MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (ranksPerNode) *ranksPerNode = minimum;
  return minimum == maximum;
}

// Single-node runs need intranet mode -- otherwise the topology pruner
// removes the NET node and GIN has no path to bind.
std::string intranetReason() {
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (nodeLocalRanks() != worldSize) return "";
  // anvil-sdma works single-node without intranet mode
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return "";
  const char* intra = std::getenv("RCCL_ENABLE_INTRANET");
  if (!intra || std::strcmp(intra, "1") != 0)
    return "Intranet mode required for single-node run (RCCL_ENABLE_INTRANET=1)";
  return "";
}

// Skip if all ranks share a node -- IB would silently loopback.
std::string crossNodeReason() {
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (nodeLocalRanks() == worldSize)
    return "Cross-node test requires ranks on >=2 physical nodes";
  return "";
}

// anvil-SDMA is a single-node backend. Skip (rather than fail/hang) if it is
// accidentally launched with ranks spanning more than one physical node.
std::string anvilSingleNodeReason() {
  if (requestedGinType() != NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return "";
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (nodeLocalRanks() != worldSize)
    return "anvil-SDMA is a single-node backend; skipping multi-node run";
  return "";
}

// First failing prerequisite, or "" if all met.
std::string ginProxyTestSkipReason() {
  for (auto check : {ginEnvDisabledReason, ginTypeReason, cuMemReason, intranetReason,
                     anvilSingleNodeReason}) {
    if (auto reason = check(); !reason.empty()) return reason;
  }
  return "";
}

// rocSHMEM-GDA and SDMA implement only INDEXED signals; VA signals are
// unsupported (Put/PutValue address the signal via indexedSignal.signalId).
std::string vaSignalTestSkipReason() {
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA)
    return "VA signals not supported by rocSHMEM-GDA (NCCL_GIN_TYPE=" +
           std::to_string(NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA) + ")";
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ANVIL_SDMA)
    return "VA signals not supported by SDMA (NCCL_GIN_TYPE=" +
           std::to_string(NCCL_NET_DEVICE_GIN_ANVIL_SDMA) + ")";
  return "";
}

// The GIN-SDMA alltoall exists only on the SDMA backend.
std::string sdmaInternalA2AEnvSkipReason() {
  if (requestedGinType() != NCCL_NET_DEVICE_GIN_ANVIL_SDMA)
    return "GIN-SDMA alltoall requires NCCL_GIN_TYPE=" +
           std::to_string(NCCL_NET_DEVICE_GIN_ANVIL_SDMA);
  if (const char* e = std::getenv("NCCL_GIN_A2A_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN-SDMA alltoall disabled (NCCL_GIN_A2A_ENABLE=0)";
  // The count sweep needs both thresholds at their defaults: a raised floor drops
  // the smallest count, a raised SDMA cutoff keeps the largest on the LSA kernel.
  if (std::getenv("NCCL_GIN_A2A_MIN_BYTES"))
    return "count sweep assumes the default NCCL_GIN_A2A_MIN_BYTES";
  if (std::getenv("NCCL_GIN_A2A_SDMA_MIN_BYTES"))
    return "SDMA tier coverage assumes the default NCCL_GIN_A2A_SDMA_MIN_BYTES";
  return "";
}

// Comm-side subset of ncclAllToAllGinSdmaEligible in gin_alltoall_sdma.cu, which is
// the source of truth. Skipping keeps an ineligible setup from failing opaquely.
std::string sdmaInternalA2ACommSkipReason(ncclComm_t comm) {
  if (comm->nNodes != 1)
    return "GIN-SDMA alltoall is scaleup-only (requires nNodes=1)";
  if (!comm->symmetricSupport)
    return "GIN-SDMA alltoall requires symmetric memory support";
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL)
    return "GIN-SDMA alltoall requires full GIN connectivity";
  return "";
}

// Initialized reqs with ginConnectionType=FULL; default for all tests here.
ncclDevCommRequirements defaultGinReqs() {
  ncclDevCommRequirements r = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  r.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  return r;
}

// No-ops on backends other than anvil-sdma. acquire() asserts, so callers wrap
// it in ASSERT_NO_FATAL_FAILURE.
struct SdmaScratchWindow {
  ncclComm_t   comm = nullptr;
  void*        buf  = nullptr;
  ncclWindow_t win  = nullptr;

  ~SdmaScratchWindow() {
    if (win) (void)ncclCommWindowDeregister(comm, win);
    if (buf) (void)ncclMemFree(buf);
  }

  void acquire(ncclComm_t c) {
    if (requestedGinType() != NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return;
    constexpr size_t kBytes = 4 * 1024;
    comm = c;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&buf, kBytes));
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(c, buf, kBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  }
};

// Bounded stream drain: poll with a deadline so a device-side hang becomes a
// reported timeout instead of an infinite hipStreamSynchronize. Returns the
// final stream status so the caller can distinguish the three outcomes:
//   hipSuccess       - stream completed cleanly
//   hipErrorNotReady - deadline elapsed while the stream was still busy (hang)
//   any other error  - a genuine device/launch error, propagated as-is
hipError_t syncStreamWithinTimeout(hipStream_t stream, int seconds) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  for (;;) {
    hipError_t q = hipStreamQuery(stream);
    if (q != hipErrorNotReady) return q;  // hipSuccess or a real error
    if (std::chrono::steady_clock::now() >= deadline) return hipErrorNotReady;  // timed out
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
// Multi-context trigger for the Tier-1 *_MultiContext tests. Returns the
// requested context count N (>=2) parsed from NCCL_GIN_NCONTEXTS, clamped to
// NCCL_GIN_MAX_CONTEXTS; returns 0 when the var is unset or <=1 so the caller
// can skip and keep the default suite single-context. NCCL_GIN_NCONTEXTS is the
// same knob the runtime uses to provision contexts, so setting it both triggers
// the test and makes the runtime grant the contexts.
int ginEnvContextCount() {
  const char* e = std::getenv("NCCL_GIN_NCONTEXTS");
  if (!e || !*e) return 0;
  int n = std::atoi(e);
  if (n <= 1) return 0;
  return std::min(n, NCCL_GIN_MAX_CONTEXTS);
}

// Standard single-CTA GIN kernel launch (thread 0 issues, whole CTA waits/flushes).
constexpr int kGinKernelBlocks  = 1;
constexpr int kGinKernelThreads = 32;

// Single-thread GIN kernel launch, for kernels where one thread both issues and
// polls so there is no CTA-wide wait or flush to widen.
constexpr int kGinSingleThreadBlocks  = 1;
constexpr int kGinSingleThreadThreads = 1;

// An LSA team only becomes non-trivial once a node hosts more than one rank; at
// one rank per node it has size 1 and any LSA assertion is vacuous.
constexpr int kMinLsaRanksPerNode = 2;

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
    ncclDevCommRequirements reqs = defaultGinReqs();
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
      putBasicProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          kTransferBytes, kSigIdx, kPeer,
          devComm);
    } else {
      putBasicConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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

  // Tier-1 test bodies, parameterized by GIN context count. Each is defined
  // out-of-line just before its TEST_F (after that test's kernels are declared).
  // The single-context TEST_F calls run*(1); the *_MultiContext sibling passes
  // the count from NCCL_GIN_NCONTEXTS (see ginEnvContextCount()).
  void runPutBasicAndOffsets(int nContexts);
  void runPutValueInline(int nContexts);
  void runWaitCounterAndSignal(int nContexts);
  void runVASignalPut(int nContexts);
};

// Context-aware producer/consumer for Put_BasicAndOffsets: one block per GIN
// context (ctx = blockIdx.x), each writing into its own slot at base =
// ctx*slotStride and signaling sigIdx in its own per-context signal pool.
// Launched with 1 block this reproduces the original single-context behavior.
__global__ void putMultiCtxProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, size_t slotStride,
    ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  const int    ctx  = (int)blockIdx.x;
  const size_t base = (size_t)ctx * slotStride;
  ncclGin gin{devComm, ctx};
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, base + dstOff,
            srcWin, base + srcOff,
            bytes,
            ncclGin_SignalInc{sigIdx});
  }
  gin.flush(ncclCoopCta());
}

__global__ void putMultiCtxConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, (int)blockIdx.x};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Smallest end-to-end exercise of the device put -> proxy -> IB -> peer
// signal chain, with non-zero src/dst/signal offsets so address-arithmetic
// regressions surface as a verification mismatch. Runs nContexts contexts in
// parallel (one block each); nContexts=1 is the default single-context path.
void GinMPIDeviceTests::runPutBasicAndOffsets(int nContexts) {
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

  // 8 KiB per-context slot; 4 KiB transfer at non-zero src/dst offsets;
  // signal at non-zero index. Forces every offset to be exercised.
  constexpr size_t kSlotStride    = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;
  const size_t kBufBytes = (size_t)nContexts * kSlotStride;

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

  // Bring up GIN with nContexts contexts and 2 signals so kSigIdx=1 is a valid
  // in-pool index within each per-context signal pool.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = nContexts;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount < nContexts) {
    GTEST_SKIP() << "Test requires >= " << nContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << nContexts << ")";
  }

  // Stage per-context source patterns in [base+kSrcOff, +kTransferBytes); rest
  // is zero. The +ctx salt makes a stray cross-context landing a value mismatch.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (int ctx = 0; ctx < nContexts; ctx++) {
    const size_t base = (size_t)ctx * kSlotStride;
    for (size_t i = 0; i < kTransferBytes; i++) {
      hostSrc[base + kSrcOff + i] = static_cast<uint8_t>(0x40 + ((i + ctx) & 0xFF));
    }
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches its kernel before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts payload + bumps signal on each context; rank 1 waits per ctx.
  if (rank == 0) {
    putMultiCtxProducerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff,
        dstWin, kDstOff,
        kTransferBytes, kSlotStride, kSigIdx, kPeer,
        devComm);
  } else {
    putMultiCtxConsumerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Sync before verify; both ranks have finished their kernel.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies each context's slot independently: payload landed exactly
  // in [base+kDstOff, +kTransferBytes), and bytes before/after are still zero.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (int ctx = 0; ctx < nContexts; ctx++) {
      const size_t base = (size_t)ctx * kSlotStride;
      for (size_t i = 0; i < kTransferBytes; i++) {
        const uint8_t expected = static_cast<uint8_t>(0x40 + ((i + ctx) & 0xFF));
        ASSERT_EQ(expected, hostResult[base + kDstOff + i])
            << "ctx " << ctx << " byte " << i << " in [dstOff, dstOff+xfer) differs";
      }
      for (size_t i = 0; i < kDstOff; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " before dstOff was unexpectedly written";
      }
      for (size_t i = kDstOff + kTransferBytes; i < kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " after dstOff+xfer was unexpectedly written";
      }
    }
  }
}

TEST_F(GinMPIDeviceTests, Put_BasicAndOffsets) {
  runPutBasicAndOffsets(/*nContexts=*/1);
}

TEST_F(GinMPIDeviceTests, Put_BasicAndOffsets_MultiContext) {
  int n = ginEnvContextCount();
  if (n == 0) GTEST_SKIP() << "Set NCCL_GIN_NCONTEXTS>1 to run the multi-context variant";
  runPutBasicAndOffsets(n);
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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
    putBasicProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff,
        dstWin, kDstOff,
        kTransferBytes, kSigIdx, kPeer,
        devComm);
  } else {
    putBasicConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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

// Whole CTA cooperatively issues one gin.put via ncclCoopCta (every lane enters put()).
__global__ void putCoopCtaProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.put(ncclTeamWorld(devComm), peer,
          dstWin, dstOff,
          srcWin, srcOff,
          bytes,
          ncclGin_SignalInc{sigIdx},
          ncclGin_None{},
          ncclCoopCta());
  gin.flush(ncclCoopCta());
}

// A cooperative whole-CTA gin.put must complete rather than deadlock: rank 0 puts a
// payload + signal, rank 1 waits and verifies. A hang is caught by the bounded drain.
TEST_F(GinMPIDeviceTests, Put_CoopCta_Regression) {
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

  // Minimal geometry: single-slot full-buffer transfer, signal 0, peer rank 1.
  constexpr size_t          kBufBytes      = 4 * 1024;
  constexpr size_t          kTransferBytes = 4 * 1024;
  constexpr ncclGinSignal_t kSigIdx        = 0;
  constexpr int             kPeer          = 1;
  // A full wavefront (>=64 on gfx9) so the coop group spans many lanes; this is
  // what exercises the coop/thread_rank mismatch. 256 threads = several warps.
  constexpr int             kThreadsPerCta = 256;

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

  ncclDevCommRequirements reqs = defaultGinReqs();
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
    hostSrc[i] = static_cast<uint8_t>(0xC0 + (i & 0x3F));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    putCoopCtaProducerKernel<<<1, kThreadsPerCta, 0, stream>>>(
        srcWin, /*srcOff=*/0,
        dstWin, /*dstOff=*/0,
        kTransferBytes, kSigIdx, kPeer,
        devComm);
  } else {
    putBasicConsumerKernel<<<1, kThreadsPerCta, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }

  // Bounded drain: a hang never completes, so cap the wait and fail instead of blocking.
  const hipError_t drainStatus = syncStreamWithinTimeout(stream, /*seconds=*/60);

  // Surface the timeout on both ranks so neither is left waiting at the barrier.
  int localTimedOut = (drainStatus == hipErrorNotReady) ? 1 : 0;
  int anyTimedOut   = 0;
  MPI_Allreduce(&localTimedOut, &anyTimedOut, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(0, anyTimedOut)
      << "cooperative gin.put(ncclCoopCta) did not complete within the timeout "
         "(missing coop->ncclCoopThread fix)";

  // A genuine device/launch error must fail loudly rather than be treated as success.
  ASSERT_EQ(hipSuccess, drainStatus)
      << "stream drain reported a device error: " << hipGetErrorString(drainStatus);

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

// Warp-span variant: same cooperative put through a multi-warp ncclCoopWarpSpan, whose
// sync() is the AMD software named barrier. Unlike ncclCoopCta's convergent __syncthreads
// it deadlocks under divergent participation without the coop->ncclCoopThread fix, giving
// AMD coverage the whole-CTA test cannot.
__global__ void putCoopWarpSpanProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  ncclCoopNamedBarrierInit();  // zero LDS barrier slots (all threads)
  ncclGin gin{devComm, /*ginContext=*/0};
  const int nWarps = blockDim.x / warpSize;  // >=2 warps -> software barrier active
  gin.put(ncclTeamWorld(devComm), peer,
          dstWin, dstOff,
          srcWin, srcOff,
          bytes,
          ncclGin_SignalInc{sigIdx},
          ncclGin_None{},
          ncclCoopWarpSpan(/*warp0=*/0, nWarps, /*id=*/0));
  gin.flush(ncclCoopCta());
}

// Warp-span flavor: with the fix this drains and verifies; without it lane 0 deadlocks
// in the software barrier and the bounded drain turns the hang into a failure.
TEST_F(GinMPIDeviceTests, Put_CoopWarpSpan_Regression_AMD) {
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

  constexpr size_t          kBufBytes      = 4 * 1024;
  constexpr size_t          kTransferBytes = 4 * 1024;
  constexpr ncclGinSignal_t kSigIdx        = 0;
  constexpr int             kPeer          = 1;
  constexpr int             kThreadsPerCta = 256;  // 4 warps -> warp-span software barrier active

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

  ncclDevCommRequirements reqs = defaultGinReqs();
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
    hostSrc[i] = static_cast<uint8_t>(0xC0 + (i & 0x3F));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    putCoopWarpSpanProducerKernel<<<1, kThreadsPerCta, 0, stream>>>(
        srcWin, /*srcOff=*/0,
        dstWin, /*dstOff=*/0,
        kTransferBytes, kSigIdx, kPeer,
        devComm);
  } else {
    putBasicConsumerKernel<<<1, kThreadsPerCta, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }

  const hipError_t drainStatus = syncStreamWithinTimeout(stream, /*seconds=*/60);

  int localTimedOut = (drainStatus == hipErrorNotReady) ? 1 : 0;
  int anyTimedOut   = 0;
  MPI_Allreduce(&localTimedOut, &anyTimedOut, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(0, anyTimedOut)
      << "cooperative gin.put(ncclCoopWarpSpan) did not complete within the timeout "
         "(missing coop->ncclCoopThread fix)";

  // A genuine device/launch error must fail loudly rather than be treated as success.
  ASSERT_EQ(hipSuccess, drainStatus)
      << "stream drain reported a device error: " << hipGetErrorString(drainStatus);

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

// Producer: putValue carries the 8-byte payload inside the GFD itself
// (no source MR is dereferenced); also bumps a signal so the receiver
// can wait synchronously. Context-aware: one block per GIN context
// (ctx = blockIdx.x), each writing value+ctx into its own slot at
// base = ctx*slotStride. Launched with 1 block this is the original path.
__global__ void putValueInlineProducerKernel(
    ncclWindow_t dstWin, size_t dstOff, size_t slotStride,
    uint64_t value, ncclGinSignal_t sigIdx, int peer,
    struct ncclDevComm devComm) {
  const int    ctx  = (int)blockIdx.x;
  const size_t base = (size_t)ctx * slotStride;
  ncclGin gin{devComm, ctx};
  if (threadIdx.x == 0) {
    gin.putValue<uint64_t>(ncclTeamWorld(devComm), peer,
                           dstWin, base + dstOff, value + (uint64_t)ctx,
                           ncclGin_SignalInc{sigIdx});
  }
  gin.flush(ncclCoopCta());
}

// Consumer: just waits for the inline-value signal on its own context.
__global__ void putValueInlineConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, (int)blockIdx.x};
  gin.waitSignal(ncclCoopCta(), sigIdx, expectedSignalValue);
}

// Sends a single 8-byte uint64_t inline (no source buffer / MR involved)
// and verifies it lands intact at the requested offset on the peer. Runs
// nContexts contexts in parallel, each reconstructing an independent inline
// value (hostGpuCtx->inlines is per-context); nContexts=1 is the default path.
void GinMPIDeviceTests::runPutValueInline(int nContexts) {
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
  constexpr size_t   kSlotStride = 4 * 1024;
  constexpr size_t   kDstOff     = 1 * 1024;
  constexpr uint64_t kValue      = 0x123456789ABCDEF0ULL;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;
  const size_t kBufBytes = (size_t)nContexts * kSlotStride;

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

  // Bring up GIN with nContexts contexts and 2 signals so kSigIdx=1 is valid
  // within each per-context signal pool.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = nContexts;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount < nContexts) {
    GTEST_SKIP() << "Test requires >= " << nContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << nContexts << ")";
  }

  // Zero dst so any spurious write outside the 8-byte landing surfaces.
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 sends the inline value + signal on each context; rank 1 waits.
  if (rank == 0) {
    putValueInlineProducerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        dstWin, kDstOff, kSlotStride, kValue, kSigIdx, kPeer, devComm);
  } else {
    putValueInlineConsumerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies, per context, the 8 bytes at base+dstOff match kValue+ctx
  // and nothing else in the slot moved.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));

    for (int ctx = 0; ctx < nContexts; ctx++) {
      const size_t base = (size_t)ctx * kSlotStride;
      uint64_t got = 0;
      std::memcpy(&got, hostResult.data() + base + kDstOff, sizeof(got));
      ASSERT_EQ(kValue + (uint64_t)ctx, got)
          << "ctx " << ctx << " inline value mismatch (4+2+2 split likely corrupted)";

      for (size_t i = 0; i < kDstOff; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " before dstOff was unexpectedly written";
      }
      for (size_t i = kDstOff + sizeof(uint64_t); i < kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " after dstOff+8 was unexpectedly written";
      }
    }
  }
}

TEST_F(GinMPIDeviceTests, PutValue_Inline) {
  runPutValueInline(/*nContexts=*/1);
}

TEST_F(GinMPIDeviceTests, PutValue_Inline_MultiContext) {
  int n = ginEnvContextCount();
  if (n == 0) GTEST_SKIP() << "Set NCCL_GIN_NCONTEXTS>1 to run the multi-context variant";
  runPutValueInline(n);
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

  // anvil-sdma binds signal routes only when >=1 window is registered
  // (dev_runtime.cc gates on winSortedCount>0). Proxy backends need no window.
  SdmaScratchWindow scratch;
  ASSERT_NO_FATAL_FAILURE(scratch.acquire(comm));

  ncclDevCommRequirements reqs = defaultGinReqs();
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
    signalNoPayloadProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kSigIdx, kPeer, devComm);
  } else {
    signalNoPayloadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Both kernels are done; rank 1's signal cell is settled.
  MPI_Barrier(MPI_COMM_WORLD);
}

// Validates the NCCL 2.29.7 defect (AICOMRCCL-1115): "Fix a 16-bit overflow of
// signal and counter ids with GIN proxy." Pre-fix, the proxy GPU-flush
// descriptor packed signalId/counterId into uint16_t fields
// (gin_proxy_device_host_common.h), so any id >= 65536 truncated to (id &
// 0xFFFF). The fix widens those fields to uint32_t bitfields (signalId : 24,
// counterId : 23) and grows the pools/validation to match. Here rank 0 bumps a
// signal whose index is exactly one past the old 16-bit range; rank 1 waits on
// that same index. With the fix the bump lands at cell 65536 and waitSignal
// returns; pre-fix the id truncates to cell 0, the waiter on 65536 never
// observes it, and the consumer hangs (surfaced as a timeout) -- exactly the
// discrimination we want. This reuses the Signal_NoPayload kernels unchanged;
// only the index and pool size differ from the in-range control above.
TEST_F(GinMPIDeviceTests, Signal_HighIdNoOverflow) {
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

  // Index one past the old uint16_t range. ginSignalCount must exceed it so the
  // cell is a valid in-pool index (post-fix default pool is 512Ki).
  constexpr ncclGinSignal_t kSigIdx          = 65536;  // > 0xFFFF
  constexpr uint32_t        kSignalPoolCount = 65537;
  constexpr int             kPeer            = 1;       // rank 0 -> rank 1

  // anvil-sdma needs >=1 window for signal routes to bind (see Signal_NoPayload).
  // Proxy backends keep the no-window path.
  SdmaScratchWindow scratch;
  ASSERT_NO_FATAL_FAILURE(scratch.acquire(comm));

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = kSignalPoolCount;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  MPI_Barrier(MPI_COMM_WORLD);

  // Producer (rank 0) bumps signal cell kSigIdx; consumer (rank 1) waits for it
  // to reach 1. The consumer's successful return IS the assertion -- pre-fix it
  // never returns because the bump truncates to cell 0.
  if (rank == 0) {
    signalNoPayloadProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kSigIdx, kPeer, devComm);
  } else {
    signalNoPayloadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
    waitCounterLocalProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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
// Context-aware: one block per GIN context (ctx = blockIdx.x), each putting
// into its own slot at base = ctx*slotStride and using sigIdx/cntIdx in its
// own per-context pools. Launched with 1 block this is the original path.
__global__ void waitCounterAndSignalProducerKernel(
    ncclWindow_t srcWin, size_t srcOff,
    ncclWindow_t dstWin, size_t dstOff,
    size_t bytes, size_t slotStride,
    ncclGinSignal_t sigIdx, ncclGinCounter_t cntIdx, int peer,
    struct ncclDevComm devComm) {
  const int    ctx  = (int)blockIdx.x;
  const size_t base = (size_t)ctx * slotStride;
  ncclGin gin{devComm, ctx};
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, base + dstOff, srcWin, base + srcOff, bytes,
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
  ncclGin gin{devComm, (int)blockIdx.x};
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
void GinMPIDeviceTests::runWaitCounterAndSignal(int nContexts) {
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

  // 8 KiB per-context slot; 4 KiB payload at non-zero src/dst offsets so
  // the put is a realistic data move. Both signal and counter indices are
  // non-zero so a regression that drops the index multiplier would land
  // at cell 0 and the waits would never observe the bump.
  constexpr size_t kSlotStride    = 8 * 1024;
  constexpr size_t kTransferBytes = 4 * 1024;
  constexpr size_t kSrcOff        = 4 * 1024;
  constexpr size_t kDstOff        = 2 * 1024;
  constexpr ncclGinSignal_t  kSigIdx = 1;
  constexpr ncclGinCounter_t kCntIdx = 1;
  constexpr int kPeer = 1;
  const size_t kBufBytes = (size_t)nContexts * kSlotStride;

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
  // both in range within each per-context pool. railGinBarrierCount=1 covers
  // the launch; ginContextCount=nContexts provisions the contexts.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = nContexts;
  reqs.ginSignalCount      = 2;
  reqs.ginCounterCount     = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount < nContexts) {
    GTEST_SKIP() << "Test requires >= " << nContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << nContexts << ")";
  }

  // Deterministic per-context byte pattern in src so the post-sync data check
  // on rank 1 catches wrong-bytes / wrong-offset / cross-context regressions.
  // Both ranks zero their dst as a baseline.
  std::vector<uint8_t> hostSrc(kBufBytes, 0);
  std::vector<uint8_t> hostDst(kBufBytes, 0);
  for (int ctx = 0; ctx < nContexts; ctx++) {
    const size_t base = (size_t)ctx * kSlotStride;
    for (size_t i = 0; i < kTransferBytes; i++) {
      hostSrc[base + kSrcOff + i] = static_cast<uint8_t>(0x40 + ((i + ctx) & 0xFF));
    }
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hostSrc.data(), kBufBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  // Sync so neither rank launches before setup is done globally.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts + bumps remote signal + bumps local counter on each context;
  // rank 1 waits for the signal. Each side's gin wait IS its assertion
  // (counter on rank 0, signal on rank 1); the host-side memcmp on rank 1
  // below verifies the payload.
  if (rank == 0) {
    waitCounterAndSignalProducerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff,
        kTransferBytes, kSlotStride, kSigIdx, kCntIdx, kPeer, devComm);
  } else {
    waitCounterAndSignalConsumerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*expectedSignalValue=*/1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Sync so rank 1's verify isn't racing rank 0's kernel completion.
  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies, per context, the payload landed at
  // [base+kDstOff, +kTransferBytes) and nothing outside that range was touched.
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    ASSERT_EQ(hipSuccess,
              hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));
    for (int ctx = 0; ctx < nContexts; ctx++) {
      const size_t base = (size_t)ctx * kSlotStride;
      for (size_t i = 0; i < kDstOff; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " before dstOff was unexpectedly written";
      }
      for (size_t i = 0; i < kTransferBytes; i++) {
        ASSERT_EQ(hostSrc[base + kSrcOff + i], hostResult[base + kDstOff + i])
            << "ctx " << ctx << " payload byte " << i << " mismatched";
      }
      for (size_t i = kDstOff + kTransferBytes; i < kSlotStride; i++) {
        ASSERT_EQ(0u, hostResult[base + i])
            << "ctx " << ctx << " byte " << i << " after dstOff+kTransferBytes was unexpectedly written";
      }
    }
  }

  // Final sync so collective teardown (ScopeGuards) see both ranks past
  // the verification phase.
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(GinMPIDeviceTests, WaitCounterAndSignal) {
  runWaitCounterAndSignal(/*nContexts=*/1);
}

TEST_F(GinMPIDeviceTests, WaitCounterAndSignal_MultiContext) {
  int n = ginEnvContextCount();
  if (n == 0) GTEST_SKIP() << "Set NCCL_GIN_NCONTEXTS>1 to run the multi-context variant";
  runWaitCounterAndSignal(n);
}

// Collective kernel: every rank runs the same code. The barrier composes
// signal + waitSignal over a per-peer signal window (gin_barrier__funcs.h):
// each sync sends a SignalInc to every other rank -- bumping cell
// (base + myRank) on that peer -- then waits on its own (base + peer) cell
// for every peer. The expected value comes from the local signal shadow,
// incremented once per sync, so consecutive syncs don't collide on the same
// value. The ncclTeamTagRail{} overload routes to comm.railGinBarrier and
// the rail team automatically.
__global__ void barrier2RanksKernel(int iters, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  // barrierIndex must be < railGinBarrierCount (we set =1, so 0).
  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagRail{}, /*barrierIndex=*/0};
  for (int i = 0; i < iters; i++) {
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  }
}

// Direct rail-team barrier coverage. Exactly one rank per node makes the
// two-rank rail team non-trivial and avoids conflating it with the world test.
TEST_F(GinMPIDeviceTests, Barrier_TwoRanks) {
  constexpr int kNodes = 2;
  constexpr int kRanksPerNode = 1;
  constexpr int kWorldRanks = kNodes * kRanksPerNode;

  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/kWorldRanks,
                                 /*max_processes=*/kWorldRanks,
                                 /*require_power_of_two=*/false,
                                 /*min_nodes=*/kNodes, /*max_nodes=*/kNodes))
    GTEST_SKIP() << "Requires exactly " << kWorldRanks << " ranks on " << kNodes
                 << " nodes";
  int ranksPerNode = 0;
  if (!uniformNodeLocalRanks(&ranksPerNode) || ranksPerNode != kRanksPerNode)
    GTEST_SKIP() << "Requires exactly " << kRanksPerNode << " rank per node";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(kWorldRanks, nRanks);

  // Large enough that a stuck-at-zero expected value would deadlock at
  // iter 1 (not silently pass on iter 0); small enough to stay fast. Each
  // round sends 1 SignalInc per peer (1 in the 2-rank case) and waits for
  // that peer's own cell to reach the incremented shadow value.
  constexpr int kIters = 16;

  // railGinBarrierCount=1 covers our 1-CTA launch (barrierIndex=0). The
  // barrier session uses the rail-team GIN barrier pool that the runtime
  // allocates via ncclGinBarrierCreateRequirement, which reserves
  // nBarriers*railTeam.nRanks cells -- one per source rank per barrier -- so
  // a cell lives at
  // (handle.signal0 + barrierIndex*railTeam.nRanks + srcRank). That pool is
  // appended after the user-facing ginSignalCount range, so it never shifts
  // user signal indices.
  //
  // GIN activation matters here: without it ginHandles[0] is NULL and the
  // barrier's internal waitSignal -> getSignalPtr dispatch faults on a NULL
  // ncclGinProxyGpuCtx. defaultGinReqs() asks for it explicitly by setting
  // ginConnectionType=FULL, which is what a single-node run needs since this
  // test requests no signal or counter pool of its own.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // All ranks finished setup before any kernel launches.
  MPI_Barrier(MPI_COMM_WORLD);

  // Both ranks launch the same kernel -- barrier is collective, no
  // producer/consumer split. Each rank's iter i sends SignalInc to the other
  // rank's cell and waits for its own cell for that peer to reach the newly
  // incremented shadow value; if the barrier raced past the peer, iter i+1
  // would either spin forever or match a stale count. Kernel completion
  // means all kIters rounds stayed in lockstep.
  barrier2RanksKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kIters, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Successful return of the kernel IS the assertion: signal + waitSignal
  // + shadow bookkeeping all worked together for kIters rounds.
  MPI_Barrier(MPI_COMM_WORLD);
}

// Same barrier as barrier2RanksKernel but over the world team, so every rank
// has 3 peers instead of 1.
__global__ void barrier4RanksKernel(int iters, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  // barrierIndex must be < worldGinBarrierCount (we set =1, so 0).
  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagWorld{}, /*barrierIndex=*/0};
  for (int i = 0; i < iters; i++) {
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  }
}

// Multi-peer counterpart to Barrier_TwoRanks. At 2 ranks the barrier has a
// single peer, so it never exercises the rotating peer order, the distribution
// of peers across the coop group, or several per-peer waits outstanding at
// once. With 4 ranks every rank has 3 peers and all three are in play.
//
// Deliberately asserts only that the barrier completes, same as
// Barrier_TwoRanks: how the barrier lays out or accounts for its signals is an
// internal detail that has already changed once (2.29.7) and may change again,
// so this test stays behavioural. A regression in the arrival accounting still
// surfaces -- as a deadlock at some round, not a silent pass.
//
// Uses the WORLD GIN barrier, not the rail one: ncclTeamRail has
// nRanks = comm.nRanks / lsaSize, so a 4-rank job only yields a 4-rank rail
// team at 4 nodes x 1 GPU, whereas the world team is 4 ranks on any layout
// (including the 2-node x 2-GPU config the other 4-rank GIN tests run on).
TEST_F(GinMPIDeviceTests, Barrier_FourRanks) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/4, /*max_processes=*/4))
    GTEST_SKIP() << "Requires exactly 4 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int nRanks = -1;
  ncclCommCount(comm, &nRanks);
  ASSERT_EQ(4, nRanks);

  // Same reasoning as Barrier_TwoRanks: large enough that stuck bookkeeping
  // deadlocks rather than passing on round 0, small enough to stay fast.
  constexpr int kIters = 16;

  // worldGinBarrierCount=1 covers the 1-CTA launch (barrierIndex=0) and sizes
  // the pool at 1*worldTeam.nRanks = 4 cells. GIN activation comes from
  // defaultGinReqs()'s ginConnectionType=FULL, as in Barrier_TwoRanks.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.worldGinBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // All ranks finished setup before any kernel launches.
  MPI_Barrier(MPI_COMM_WORLD);

  // All four ranks run the same collective loop. Each round every rank signals
  // its 3 peers and waits for all 3 to arrive; if the arrival accounting broke
  // for any peer, that round would spin forever rather than pass.
  barrier4RanksKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kIters, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Successful return of the kernel IS the assertion, as in Barrier_TwoRanks:
  // all kIters rounds kept 4 ranks in lockstep across 3 peers each.
  MPI_Barrier(MPI_COMM_WORLD);
}

struct WorldBarrierObservation {
  int teamNRanks;
  int teamRank;
  int teamStride;
  uint32_t handleSignal0;
  uint32_t activeSignal;
  int selectionMatched;
  int completedIters;
  int signalMismatches;
};

// Observe the baked-in team and handle before synchronizing. A wrong world
// constructor returns cleanly with selectionMatched=0 instead of hanging on an
// unallocated or differently sized rail pool.
__global__ void worldBarrierObservationKernel(
    int iters, WorldBarrierObservation* observation,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagWorld{}, /*barrierIndex=*/0};
  ncclTeam world = ncclTeamWorld(devComm);
  bool selectionMatched =
      bar.team.nRanks == world.nRanks &&
      bar.team.rank == world.rank &&
      bar.team.stride == world.stride &&
      bar.handle.signal0 == devComm.worldGinBarrier.signal0 &&
      bar.signal == devComm.worldGinBarrier.signal0;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    observation->teamNRanks = bar.team.nRanks;
    observation->teamRank = bar.team.rank;
    observation->teamStride = bar.team.stride;
    observation->handleSignal0 = bar.handle.signal0;
    observation->activeSignal = bar.signal;
    observation->selectionMatched = selectionMatched ? 1 : 0;
  }
  if (!selectionMatched) return;

  for (int i = 0; i < iters; ++i) {
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int mismatches = 0;
    for (int peer = 0; peer < world.nRanks; ++peer) {
      uint64_t expected = peer == world.rank ? 0 : static_cast<uint64_t>(iters);
      if (gin.readSignal(bar.signal + peer) != expected) ++mismatches;
    }
    observation->completedIters = iters;
    observation->signalMismatches = mismatches;
  }
}

// Two ranks per node make the world team (4 ranks, stride 1) observably
// different from each non-trivial rail team (2 ranks, stride 2).
TEST_F(GinMPIDeviceTests, Barrier_WorldTeamUsesWorldPool) {
  constexpr int kNodes = 2;
  constexpr int kRanksPerNode = 2;
  constexpr int kWorldRanks = kNodes * kRanksPerNode;
  constexpr int kRailRanks = kNodes;

  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/kWorldRanks,
                                 /*max_processes=*/kWorldRanks,
                                 /*require_power_of_two=*/false,
                                 /*min_nodes=*/kNodes, /*max_nodes=*/kNodes))
    GTEST_SKIP() << "Requires exactly " << kWorldRanks << " ranks on " << kNodes
                 << " nodes";
  int ranksPerNode = 0;
  if (!uniformNodeLocalRanks(&ranksPerNode) || ranksPerNode != kRanksPerNode)
    GTEST_SKIP() << "Requires exactly " << kRanksPerNode << " ranks per node";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();
  ncclTeam_t world = ncclTeamWorld(comm);
  ncclTeam_t rail = ncclTeamRail(comm);
  ASSERT_EQ(kWorldRanks, world.nRanks);
  ASSERT_EQ(kRailRanks, rail.nRanks);
  ASSERT_NE(world.stride, rail.stride);

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.worldGinBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  WorldBarrierObservation* dObservation = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dObservation, sizeof(WorldBarrierObservation)));
  auto observationCleanup = makeScopeGuard([&]() {
    if (dObservation) (void)hipFree(dObservation);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dObservation, 0, sizeof(WorldBarrierObservation)));

  constexpr int kIters = 16;
  MPI_Barrier(MPI_COMM_WORLD);
  worldBarrierObservationKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
      kIters, dObservation, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  WorldBarrierObservation observation{};
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(&observation, dObservation, sizeof(observation), hipMemcpyDeviceToHost));
  ASSERT_MPI_EQ(world.nRanks, observation.teamNRanks);
  ASSERT_MPI_EQ(world.rank, observation.teamRank);
  ASSERT_MPI_EQ(world.stride, observation.teamStride);
  ASSERT_MPI_EQ(devComm.worldGinBarrier.signal0, observation.handleSignal0);
  ASSERT_MPI_EQ(devComm.worldGinBarrier.signal0, observation.activeSignal);
  ASSERT_MPI_EQ(1, observation.selectionMatched);
  ASSERT_MPI_EQ(kIters, observation.completedIters);
  ASSERT_MPI_EQ(0, observation.signalMismatches);
  MPI_Barrier(MPI_COMM_WORLD);
}

// Direct tests for ncclBarrierSession (barrier.h), separate from the alltoall
// kernels that use it incidentally: BarrierSession_LsaOnly (LSA-only subset,
// ncclTeamTagLsa, no GIN) and BarrierSession_Hybrid (world-team: inner LSA +
// outer rail-GIN).

// LSA-team collective for the LSA-only subset. Each round every rank stamps the
// iteration into its slot of each peer's window, barriers, then checks all
// peers' stamps are visible. acq_rel publishes our writes and acquires theirs;
// a broken barrier leaves a stale slot in dErr (or deadlocks on a broken epoch).
__global__ void barrierSessionLsaKernel(
    ncclWindow_t win, size_t off, int iters, int* dErr,
    struct ncclDevComm devComm) {
  // Dedicated LSA barrier session reads comm.lsaBarrier (sized by reqs.lsaBarrierCount, no GIN); the composite
  // ncclBarrierSession(ncclTeamTagLsa) reads the 0-sized comm.hybridLsaBarrier and overruns for LSA teams >2 ranks.
  ncclLsaBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/blockIdx.x};
  ncclTeam lsa = ncclTeamLsa(devComm);
  int* myBuf = static_cast<int*>(ncclGetLocalPointer(win, off));
  for (int it = 1; it <= iters; ++it) {
    if (threadIdx.x == 0) {
      for (int p = 0; p < lsa.nRanks; ++p) {
        int* peer = static_cast<int*>(ncclGetLsaPointer(win, off, p));
        peer[lsa.rank] = it;
      }
    }
    // Publish our writes to LSA peers and acquire theirs (acq_rel so the loads
    // below are ordered after every peer's store, not relaxed).
    bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
    if (threadIdx.x == 0) {
      for (int p = 0; p < lsa.nRanks; ++p) {
        if (myBuf[p] != it) atomicExch(dErr, it);
      }
    }
    // Gate the next round's overwrites on all peers having read this round.
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  }
}

// Exercises the LSA-only subset in isolation. Needs symmetric memory and >=2
// ranks co-located on a node; it does NOT require the GIN proxy, so it gates on
// cuMem + a node-local size check rather than the full GIN prerequisites.
TEST_F(GinMPIDeviceTests, BarrierSession_LsaOnly) {
  if (auto reason = cuMemReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  int ranksPerNode = 0;
  if (!uniformNodeLocalRanks(&ranksPerNode) || ranksPerNode < kMinLsaRanksPerNode)
    GTEST_SKIP() << "Requires a uniform placement with >=" << kMinLsaRanksPerNode
                 << " ranks per node";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int nRanks = -1;
  ncclCommCount(comm, &nRanks);
  ASSERT_GE(nRanks, 2);
  ASSERT_LE(nRanks, 8);

  // LSA-only path: start from the plain initializer (GIN disabled) and ask only
  // for the inner LSA barrier pool, so ncclDevCommCreate does not activate GIN.
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // One int slot per possible LSA peer (<= nRanks). Symmetric window so peers
  // can store into our buffer via ncclGetLsaPointer.
  constexpr int kIters = 16;
  const size_t  bytes  = static_cast<size_t>(nRanks) * sizeof(int);
  void* dBuf = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dBuf, bytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dBuf) (void)ncclMemFree(dBuf);
  });

  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dBuf, bytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (win) (void)ncclCommWindowDeregister(comm, win);
  });

  // Device-side error flag: set to the failing iteration if a peer's write
  // was not visible after the barrier.
  int* dErr = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dErr, sizeof(int)));
  auto errCleanup = makeScopeGuard([&]() {
    if (dErr) (void)hipFree(dErr);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dErr, 0, sizeof(int)));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dBuf, 0, bytes));

  MPI_Barrier(MPI_COMM_WORLD);

  // Collective: every rank runs the same kernel on its node's LSA team.
  barrierSessionLsaKernel<<<1, 64, 0, stream>>>(win, /*off=*/0, kIters, dErr, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  int hErr = 0;
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(&hErr, dErr, sizeof(int), hipMemcpyDeviceToHost));
  ASSERT_EQ(0, hErr) << "LSA barrier failed to make a peer's write visible at iteration " << hErr;

  MPI_Barrier(MPI_COMM_WORLD);
}

// AICOMRCCL-2241: registering a symmetric window after ncclDevCommCreate is
// legal. Anvil SDMA resolves the user VA via ncclDevrGetLsaSelfAddr, which
// only walks memHead, so ginEnabled must already see this mem on the list.
// BarrierSession_LsaOnly also registers after create, but it never activates
// GIN (plain NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER), so it cannot catch this.
TEST_F(GinMPIDeviceTests, WindowRegister_AfterDevCommCreate) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginSignalCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  constexpr size_t kBytes = 64;
  void* dBuf = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dBuf, kBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dBuf) (void)ncclMemFree(dBuf);
  });

  ncclWindow_t win = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dBuf, kBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (win) (void)ncclCommWindowDeregister(comm, win);
  });

  MPI_Barrier(MPI_COMM_WORLD);
}

// Collective over the world team: composes the inner LSA barrier with the
// outer rail-GIN barrier (ncclTeamTagWorld ctor). Mirrors Barrier_TwoRanks'
// "completion == success" philosophy -- a broken inner LSA epoch or outer
// per-peer signal count deadlocks at iter 1 -- but routes through both
// substrates. On single node the outer rail arm degenerates; multi-node it
// crosses rails.
__global__ void barrierSessionHybridKernel(int iters, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  ncclBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), ncclTeamTagWorld(), gin, /*index=*/blockIdx.x};
  for (int i = 0; i < iters; i++) {
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  }
}

TEST_F(GinMPIDeviceTests, BarrierSession_Hybrid) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  // The hybrid session needs both a non-trivial LSA team and a cross-node rail,
  // so at least two nodes each hosting a non-trivial LSA team. The upper bound
  // is the harness limit shared with the other barrier tests.
  constexpr int kMinNodes = 2;
  constexpr int kMinRanks = kMinNodes * kMinLsaRanksPerNode;
  constexpr int kMaxRanks = 8;

  if (!validateTestPrerequisites(/*min_processes=*/kMinRanks,
                                 /*max_processes=*/kMaxRanks))
    GTEST_SKIP() << "Requires " << kMinRanks << "-" << kMaxRanks << " ranks";
  int ranksPerNode = 0;
  if (!uniformNodeLocalRanks(&ranksPerNode) || ranksPerNode < kMinLsaRanksPerNode)
    GTEST_SKIP() << "Requires a uniform placement with at least "
                 << kMinLsaRanksPerNode << " ranks per node";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  int nRanks = -1;
  ncclCommCount(comm, &nRanks);
  ASSERT_GE(nRanks, 2);
  ASSERT_LE(nRanks, 8);

  constexpr int kIters = 16;

  // Only barrierCount provisions the three pools consumed by the generic
  // world-team session. Keeping every specialized count at zero ensures an
  // accidental constructor or requirement-routing regression cannot be masked.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.barrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  ASSERT_MPI_EQ(1, devComm.hybridLsaBarrier.nBarriers);
  ASSERT_MPI_EQ(0, devComm.lsaBarrier.nBarriers);
  // barrierCount sizes both outer GIN arms: one cell per source rank of the rail team and of the world team.
  ASSERT_MPI_EQ(ncclTeamRail(comm).nRanks + ncclTeamWorld(comm).nRanks, devComm.ginSignalCount);

  MPI_Barrier(MPI_COMM_WORLD);

  // All ranks run the same collective barrier loop; completion means every
  // round's inner LSA epochs and outer rail-GIN per-peer signal counts
  // stayed in lockstep.
  barrierSessionHybridKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kIters, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);
}

struct BarrierSelectionObservation {
  uint64_t genericLsaBufHandle;
  int genericLsaNBarriers;
  uint32_t genericRailSignal0;
  uint64_t dedicatedLsaBufHandle;
  int dedicatedLsaNBarriers;
  uint32_t directRailSignal0;
  uint32_t directWorldSignal0;
  int genericLsaNRanks;
  int genericRailNRanks;
  int dedicatedLsaNRanks;
  int directRailNRanks;
  int directWorldNRanks;
};

// Selection-only kernel: constructing every session is enough to observe which
// pool it selected. Deliberately do not sync; valid specialized allocations
// must not hide a generic constructor routed to the wrong pool.
__global__ void barrierSelectionObservationKernel(
    BarrierSelectionObservation* observation,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  ncclBarrierSession<ncclCoopCta> generic{
      ncclCoopCta(), ncclTeamTagWorld{}, gin, /*index=*/0};
  ncclLsaBarrierSession<ncclCoopCta> dedicatedLsa{
      ncclCoopCta(), devComm, ncclTeamTagLsa{}, /*index=*/0};
  ncclGinBarrierSession<ncclCoopCta> directRail{
      ncclCoopCta(), gin, ncclTeamTagRail{}, /*barrierIndex=*/0};
  ncclGinBarrierSession<ncclCoopCta> directWorld{
      ncclCoopCta(), gin, ncclTeamTagWorld{}, /*barrierIndex=*/0};

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto& genericLsa = generic.lsaBarrier();
    auto& genericRail = generic.ginBarrier();
    observation->genericLsaBufHandle = genericLsa.handle.bufHandle;
    observation->genericLsaNBarriers = genericLsa.handle.nBarriers;
    observation->genericRailSignal0 = genericRail.handle.signal0;
    observation->dedicatedLsaBufHandle = dedicatedLsa.handle.bufHandle;
    observation->dedicatedLsaNBarriers = dedicatedLsa.handle.nBarriers;
    observation->directRailSignal0 = directRail.handle.signal0;
    observation->directWorldSignal0 = directWorld.handle.signal0;
    observation->genericLsaNRanks = genericLsa.team.nRanks;
    observation->genericRailNRanks = genericRail.team.nRanks;
    observation->dedicatedLsaNRanks = dedicatedLsa.team.nRanks;
    observation->directRailNRanks = directRail.team.nRanks;
    observation->directWorldNRanks = directWorld.team.nRanks;
  }
}

TEST_F(GinMPIDeviceTests, BarrierPools_AreIsolated) {
  // Two ranks per node keeps the LSA, rail, and world teams distinct, so each
  // barrier flavor has its own team to select.
  constexpr int kNodes = 2;
  constexpr int kRanksPerNode = 2;
  constexpr int kWorldRanks = kNodes * kRanksPerNode;

  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/kWorldRanks,
                                 /*max_processes=*/kWorldRanks,
                                 /*require_power_of_two=*/false,
                                 /*min_nodes=*/kNodes, /*max_nodes=*/kNodes))
    GTEST_SKIP() << "Requires exactly " << kWorldRanks << " ranks on " << kNodes
                 << " nodes";
  int ranksPerNode = 0;
  if (!uniformNodeLocalRanks(&ranksPerNode) || ranksPerNode != kRanksPerNode)
    GTEST_SKIP() << "Requires exactly " << kRanksPerNode << " ranks per node";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();
  ncclTeam_t lsa = ncclTeamLsa(comm);
  ncclTeam_t rail = ncclTeamRail(comm);
  ncclTeam_t world = ncclTeamWorld(comm);

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginSignalCount = 1;
  reqs.barrierCount = 1;
  reqs.lsaBarrierCount = 1;
  reqs.railGinBarrierCount = 1;
  reqs.worldGinBarrierCount = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  using Range = std::array<uint64_t, 2>;
  std::array<Range, 4> ginRanges{{
      {0, 1},
      {devComm.hybridRailGinBarrier.signal0,
       devComm.hybridRailGinBarrier.signal0 + static_cast<uint64_t>(rail.nRanks)},
      {devComm.railGinBarrier.signal0,
       devComm.railGinBarrier.signal0 + static_cast<uint64_t>(rail.nRanks)},
      {devComm.worldGinBarrier.signal0,
       devComm.worldGinBarrier.signal0 + static_cast<uint64_t>(world.nRanks)},
  }};
  bool ginRangesValid = true;
  for (size_t i = 0; i < ginRanges.size(); ++i) {
    ginRangesValid =
        ginRangesValid && ginRanges[i][0] < ginRanges[i][1] &&
        ginRanges[i][1] <= static_cast<uint64_t>(devComm.ginSignalCount);
    for (size_t j = i + 1; j < ginRanges.size(); ++j) {
      ginRangesValid =
          ginRangesValid &&
          (ginRanges[i][1] <= ginRanges[j][0] ||
           ginRanges[j][1] <= ginRanges[i][0]);
    }
  }
  ASSERT_MPI_TRUE(ginRangesValid);

  uint64_t lsaBytes = static_cast<uint64_t>(3 + lsa.nRanks) * sizeof(uint32_t);
  Range hybridLsaRange{
      ncclGetResourceBufferOffset(devComm.hybridLsaBarrier.bufHandle),
      ncclGetResourceBufferOffset(devComm.hybridLsaBarrier.bufHandle) + lsaBytes};
  Range dedicatedLsaRange{
      ncclGetResourceBufferOffset(devComm.lsaBarrier.bufHandle),
      ncclGetResourceBufferOffset(devComm.lsaBarrier.bufHandle) + lsaBytes};
  ASSERT_MPI_TRUE(hybridLsaRange[1] <= dedicatedLsaRange[0] ||
                  dedicatedLsaRange[1] <= hybridLsaRange[0]);

  BarrierSelectionObservation* dObservation = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dObservation, sizeof(BarrierSelectionObservation)));
  auto observationCleanup = makeScopeGuard([&]() {
    if (dObservation) (void)hipFree(dObservation);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dObservation, 0, sizeof(BarrierSelectionObservation)));
  barrierSelectionObservationKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
      dObservation, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  BarrierSelectionObservation observation{};
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(&observation, dObservation, sizeof(observation), hipMemcpyDeviceToHost));
  ASSERT_MPI_EQ(devComm.hybridLsaBarrier.bufHandle, observation.genericLsaBufHandle);
  ASSERT_MPI_EQ(devComm.hybridLsaBarrier.nBarriers, observation.genericLsaNBarriers);
  ASSERT_MPI_EQ(devComm.hybridRailGinBarrier.signal0, observation.genericRailSignal0);
  ASSERT_MPI_EQ(devComm.lsaBarrier.bufHandle, observation.dedicatedLsaBufHandle);
  ASSERT_MPI_EQ(devComm.lsaBarrier.nBarriers, observation.dedicatedLsaNBarriers);
  ASSERT_MPI_EQ(devComm.railGinBarrier.signal0, observation.directRailSignal0);
  ASSERT_MPI_EQ(devComm.worldGinBarrier.signal0, observation.directWorldSignal0);
  ASSERT_MPI_EQ(lsa.nRanks, observation.genericLsaNRanks);
  ASSERT_MPI_EQ(rail.nRanks, observation.genericRailNRanks);
  ASSERT_MPI_EQ(lsa.nRanks, observation.dedicatedLsaNRanks);
  ASSERT_MPI_EQ(rail.nRanks, observation.directRailNRanks);
  ASSERT_MPI_EQ(world.nRanks, observation.directWorldNRanks);
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
//   inter-phase sync would not guarantee in-flight signal writes have
//   reached the peer, and the consumer could observe an intermediate signal
//   value (e.g. 5 instead of 26 in P2).
//
//   P2 ordering quirk: waitSignalFollowShadow(leastDelta=1) returns as soon
//   as signal > shadow. If P2's puts and the consumer's wait launched
//   concurrently, the consumer could observe signal=5 (still just P1) and
//   return with delta=5 -- the d==26 assertion would fail spuriously. So
//   P2 producer runs to completion first, then ASSERT_MPI_EQ(stream sync)
//   (MPI_Allreduce) gates the P2 consumer launch. P1 and P3 run concurrently.
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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
    signalAddProducerWithCounterKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/5,
        kCntIdx, /*counterExpected=*/1,
        /*nPuts=*/1, kPeer, devComm);
  } else {
    signalAddConsumerPhase1Kernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*least=*/5, &dResults[0], devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // --- P2: producer 3x SignalAdd+7/CounterInc, ending with waitCounter(4).
  //         MUST complete BEFORE the consumer's waitSignalFollowShadow runs
  //         (see banner: leastDelta=1 could otherwise observe signal=5 from
  //         P1 and return with delta=5 instead of 26).
  if (rank == 0) {
    signalAddProducerWithCounterKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/7,
        kCntIdx, /*counterExpected=*/4,
        /*nPuts=*/3, kPeer, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  if (rank == 1) {
    signalAddConsumerPhase2Kernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, /*leastDelta=*/1, /*shadowBump=*/100,
        &dResults[1], &dResults[2], devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // --- P3: producer 1x SignalAdd+100 (no counter, drained via flush);
  //         consumer waitSignalMeetShadow returns when signal >= 126.
  //         Concurrent: consumer is gated on the SignalAdd landing.
  if (rank == 0) {
    signalAddProducerNoCounterKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, kSrcOff, dstWin, kDstOff, kTransferBytes,
        kSigIdx, /*signalDelta=*/100,
        kPeer, devComm);
  } else {
    signalAddConsumerPhase3Kernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kSigIdx, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // --- P4: producer reads counter back. The 4 counter-bearing puts (P1: 1,
  //         P2: 3) were all waited for via waitCounter in their kernels;
  //         P3 had no counter. Counter must be exactly 4.
  if (rank == 0) {
    signalAddProducerReadCounterKernel<<<1, 1, 0, stream>>>(
        kCntIdx, &dResults[0], devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  // Pull results back per-rank; each rank validates its own slots via MPI-aware
  // checks so a failure on one rank aborts all ranks together.
  std::vector<uint64_t> hostResults(kResultSlots, 0);
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(hostResults.data(), dResults, kResultSlots * sizeof(uint64_t),
                          hipMemcpyDeviceToHost));
  ASSERT_MPI_EQ_ON_RANK(rank, 0, uint64_t{4}, hostResults[0]);
  ASSERT_MPI_EQ_ON_RANK(rank, 1, uint64_t{5}, hostResults[0]);
  ASSERT_MPI_EQ_ON_RANK(rank, 1, uint64_t{0}, hostResults[1]);
  ASSERT_MPI_EQ_ON_RANK(rank, 1, uint64_t{26}, hostResults[2]);

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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
    symPtrProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        dstSym, srcSym, kNElts,
        valDstSym, kInlineValue,
        kSigIdx, kPeer, devComm);
  } else {
    symPtrConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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
// signal cell for nRanks increments. Mirrors ginAlltoAllBody (device impl 3) in
// rccl-tests/src/alltoall.cu, minus its trailing release sync, so a regression
// in put + signal + barrier + flush composed under load surfaces here.
// Body shared by the two reference kernels below. bar is whichever session the
// caller built, so the identical put + signal + flush composition runs under
// both the generic barrier and the dedicated world-team GIN barrier.
template <typename BarT>
__device__ void alltoallPureBody(
    ncclGin& gin, BarT& bar, unsigned int signalIndex, uint64_t signalValue,
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, struct ncclDevComm devComm) {
  // Cross-rank sync ensures every rank's sendbuf is registered before any
  // peer reads from it via gin.put below.
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

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

  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

// Generic session, provisioned from reqs.barrierCount. This is the rccl-tests
// composition.
__global__ void alltoallPureKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, struct ncclDevComm devComm) {
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin{devComm, /*ginContext=*/0};
  // Capture current signal cell so a count-sweep that reuses the same
  // kernel/devComm doesn't conflate increments from past iterations.
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  alltoallPureBody(gin, bar, signalIndex, signalValue,
      sendwin, sendoffset, recvwin, recvoffset, count, devComm);
}

// Same collective on the dedicated world GIN pool, provisioned from
// reqs.worldGinBarrierCount. This is the composition in
// examples/06_device_api/02_alltoall_gin/c/main.cu.
__global__ void alltoallWorldGinKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, struct ncclDevComm devComm) {
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin{devComm, /*ginContext=*/0};
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagWorld{}, blockIdx.x};
  alltoallPureBody(gin, bar, signalIndex, signalValue,
      sendwin, sendoffset, recvwin, recvoffset, count, devComm);
}

// One CTA keeps barrier index 0 the only index in play; 512 threads is the
// rccl-tests production launch shape.
constexpr int kAlltoallRefCTAs    = 1;
constexpr int kAlltoallRefThreads = 512;

// Count sweep shared by the two alltoall reference tests: 1 (alignment/tail
// edges), 1024 (medium), 65536 (saturating). Only reqs and the kernel differ
// between them, so each caller supplies those. launch receives the registered
// windows, the current count and the devComm. Asserts, so callers wrap this in
// ASSERT_NO_FATAL_FAILURE.
template <typename LaunchT>
void runAlltoallReferenceSweep(ncclComm_t comm, hipStream_t stream, int rank, int nRanks,
                               const ncclDevCommRequirements& reqs, LaunchT&& launch) {
  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};

  // Register windows ONCE (largest count) and reuse, BEFORE ncclDevCommCreate
  // activates GIN: some backends resolve a buffer's address at registration,
  // which only succeeds pre-activation.
  const size_t maxBytes = counts.back() * static_cast<size_t>(nRanks) * sizeof(float);

  void* dSend = nullptr;
  void* dRecv = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, maxBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, maxBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSend) (void)ncclMemFree(dSend);
    if (dRecv) (void)ncclMemFree(dRecv);
  });

  ncclWindow_t sendWin = nullptr, recvWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dSend, maxBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dRecv, maxBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
    if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
  });

  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t totalElements = count * static_cast<size_t>(nRanks);
    const size_t sizeBytes     = totalElements * sizeof(float);

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

    launch(sendWin, recvWin, count, devComm);
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

// Single-node 2-8 ranks. Bit-for-bit verifies the deterministic
// sendbuf[i] = rank*1000 + dst*100 + i pattern landed on the right peer slot.
// The 1<<16 case (256 KiB / direction / peer) saturates ring credit to
// exercise the postGfd credit-wait path.
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

  // ncclBarrierSession(ncclTeamTagWorld) draws on the three hybrid pools, all
  // sized by reqs.barrierCount rather than lsa/railGinBarrierCount, so leaving
  // it 0 hangs the GIN arm. ginSignalCount=1 covers the kernel's signalIndex=0.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.barrierCount   = 1;
  reqs.ginSignalCount = 1;

  ASSERT_NO_FATAL_FAILURE(runAlltoallReferenceSweep(comm, stream, rank, nRanks, reqs,
      [&](ncclWindow_t sendWin, ncclWindow_t recvWin, size_t count, ncclDevComm& devComm) {
        alltoallPureKernel<<<kAlltoallRefCTAs, kAlltoallRefThreads, 0, stream>>>(
            sendWin, /*sendoffset=*/0,
            recvWin, /*recvoffset=*/0,
            count, devComm);
      }));
}

// Same collective and same verification as Alltoall_PureReference, on the
// dedicated world GIN barrier pool instead of the generic one. Keeps the
// ncclTeamTagWorld constructor and worldGinBarrierCount covered, which the
// generic session never touches.
TEST_F(GinMPIDeviceTests, Alltoall_WorldGinBarrierReference) {
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

  // worldGinBarrierCount allocates the world pool that the ncclTeamTagWorld
  // constructor binds to, so no manual barrier requirement is needed.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.worldGinBarrierCount = 1;
  reqs.ginSignalCount       = 1;

  ASSERT_NO_FATAL_FAILURE(runAlltoallReferenceSweep(comm, stream, rank, nRanks, reqs,
      [&](ncclWindow_t sendWin, ncclWindow_t recvWin, size_t count, ncclDevComm& devComm) {
        alltoallWorldGinKernel<<<kAlltoallRefCTAs, kAlltoallRefThreads, 0, stream>>>(
            sendWin, /*sendoffset=*/0,
            recvWin, /*recvoffset=*/0,
            count, devComm);
      }));
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

  // The world-team ncclBarrierSession sizes its hybrid LSA+rail-GIN barriers from
  // reqs.barrierCount (not lsa/railGinBarrierCount); 0 hangs the rail arm.
  // ginSignalCount=1 covers the cross-node signal cell and triggers GIN activation.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.barrierCount        = 1;
  reqs.ginSignalCount      = 1;

  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};
  constexpr int kCTAs          = 1;
  constexpr int kThreadsPerCTA = 512;

  // Register windows ONCE (largest count) and reuse, BEFORE ncclDevCommCreate
  // activates GIN: some backends resolve a buffer's address at registration,
  // which only succeeds pre-activation.
  const size_t maxBytes = counts.back() * static_cast<size_t>(nRanks) * sizeof(float);

  void* dSend = nullptr;
  void* dRecv = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, maxBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, maxBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSend) (void)ncclMemFree(dSend);
    if (dRecv) (void)ncclMemFree(dRecv);
  });

  ncclWindow_t sendWin = nullptr, recvWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dSend, maxBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dRecv, maxBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
    if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
  });

  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t totalElements = count * static_cast<size_t>(nRanks);
    const size_t sizeBytes     = totalElements * sizeof(float);

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

// A 2.29.7-versioned request for indexed GIN resources is rejected because
// those device layouts are not compatible with the 2.30 GIN layout. This is
// intentionally narrower than claiming every possible legacy pure-put binary
// is detected by the compatibility filter.
TEST_F(GinMPIDeviceTests, DevComm_LegacyGinSignalRequestRejected) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  // GIN device code compiled against 2.29.7 is ABI-incompatible with 2.30.
  ncclDevCommRequirements legacyReqs = defaultGinReqs();
  legacyReqs.version = NCCL_VERSION(2, 29, 7);
  legacyReqs.ginSignalCount = 1;
  ncclDevComm legacyDevComm{};
  ncclResult_t result = ncclDevCommCreate(comm, &legacyReqs, &legacyDevComm);
  if (result == ncclSuccess) {
    (void)ncclDevCommDestroy(comm, &legacyDevComm);
  }
  ASSERT_MPI_EQ(ncclInvalidUsage, result);
}

// A compatible 2.30 request carries its requested ABI version into the returned
// device communicator; it is not silently stamped with the runtime's version.
TEST_F(GinMPIDeviceTests, DevComm_ReturnsRequestedVersion) {
  if (auto reason = cuMemReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.version = NCCL_VERSION(2, 30, 0);
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });
  ASSERT_MPI_EQ(reqs.version, devComm.version);
}

// Backend-independent ownership check retained for both proxy and rocSHMEM-GDA
// runs. The proxy-only test below adds functional queue/signal/counter coverage.
TEST_F(GinMPIDeviceTests, DevComm_PerInstanceGinHandlesAreDistinct) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginContextCount = 1;
  reqs.ginSignalCount = 1;

  ncclDevComm first{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &first));
  bool firstLive = true;
  auto firstCleanup = makeScopeGuard([&]() {
    if (firstLive) (void)ncclDevCommDestroy(comm, &first);
  });

  ncclDevComm second{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &second));
  bool secondLive = true;
  auto secondCleanup = makeScopeGuard([&]() {
    if (secondLive) (void)ncclDevCommDestroy(comm, &second);
  });

  ASSERT_MPI_GT(first.ginConnectionCount, 0);
  ASSERT_MPI_EQ(first.ginConnectionCount, second.ginConnectionCount);
  bool locallyDistinct = true;
  for (int connection = 0; connection < first.ginConnectionCount; ++connection) {
    locallyDistinct =
        locallyDistinct &&
        first.ginHandles[connection] != second.ginHandles[connection];
  }
  int allDistinct = locallyDistinct ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &allDistinct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (allDistinct == 0) secondLive = false;
  ASSERT_MPI_TRUE(allDistinct == 1);
}

__global__ void perDevCommIsolationProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin, size_t offset, size_t bytes,
    uint64_t expectedEpoch, uint64_t timeoutCycles, int* completed,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), /*peer=*/1,
            dstWin, offset, srcWin, offset, bytes,
            ncclGin_SignalInc{/*signal=*/0},
            ncclGin_CounterInc{/*counter=*/0});
    uint64_t start = clock64();
    while (gin.readCounter(/*counter=*/0) < expectedEpoch &&
           clock64() - start < timeoutCycles) {
    }
    *completed = gin.readCounter(/*counter=*/0) >= expectedEpoch ? 1 : 0;
  }
}

__global__ void perDevCommIsolationConsumerKernel(
    uint64_t expectedEpoch, uint64_t timeoutCycles, int* completed,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t start = clock64();
    while (gin.readSignal(/*signal=*/0) < expectedEpoch &&
           clock64() - start < timeoutCycles) {
    }
    *completed = gin.readSignal(/*signal=*/0) >= expectedEpoch ? 1 : 0;
  }
}

__global__ void readPerDevCommStateKernel(
    uint64_t* state, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    state[0] = gin.readSignal(/*signal=*/0);
    state[1] = gin.readCounter(/*counter=*/0);
  }
}

hipError_t runPerDevCommIsolationPhase(
    int rank, ncclWindow_t srcWin, ncclWindow_t dstWin,
    size_t offset, size_t bytes, uint64_t expectedEpoch,
    ncclDevComm devComm, hipStream_t stream, int* dCompleted,
    bool* completed) {
  constexpr uint64_t kTimeoutCycles = 5000000000ULL;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    perDevCommIsolationProducerKernel<<<kGinSingleThreadBlocks,
                                        kGinSingleThreadThreads, 0, stream>>>(
        srcWin, dstWin, offset, bytes, expectedEpoch, kTimeoutCycles,
        dCompleted, devComm);
  } else {
    perDevCommIsolationConsumerKernel<<<kGinSingleThreadBlocks,
                                        kGinSingleThreadThreads, 0, stream>>>(
        expectedEpoch, kTimeoutCycles, dCompleted, devComm);
  }
  hipError_t result = hipStreamSynchronize(stream);
  int hCompleted = 0;
  if (result == hipSuccess) {
    result = hipMemcpy(&hCompleted, dCompleted, sizeof(int), hipMemcpyDeviceToHost);
  }
  if (completed) *completed = hCompleted == 1;
  MPI_Barrier(MPI_COMM_WORLD);
  return result;
}

hipError_t readPerDevCommState(
    ncclDevComm devComm, hipStream_t stream, uint64_t state[2]) {
  uint64_t* dState = nullptr;
  hipError_t result = hipMalloc(&dState, 2 * sizeof(uint64_t));
  if (result != hipSuccess) return result;
  result = hipMemsetAsync(dState, 0, 2 * sizeof(uint64_t), stream);
  if (result == hipSuccess) {
    readPerDevCommStateKernel<<<kGinSingleThreadBlocks, kGinSingleThreadThreads, 0,
                                stream>>>(dState, devComm);
    result = hipStreamSynchronize(stream);
  }
  if (result == hipSuccess) {
    result = hipMemcpy(state, dState, 2 * sizeof(uint64_t), hipMemcpyDeviceToHost);
  }
  (void)hipFree(dState);
  return result;
}

// Exercise two device communicators backed by one host communicator, prove
// their signal/counter epochs are independent, destroy the first out of LIFO
// order, and then prove the second context and queue remain usable.
TEST_F(GinMPIDeviceTests, DevComm_PerInstanceGinResourcesRemainUsable) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (requestedGinType() != 2)
    GTEST_SKIP() << "Functional resource-isolation oracle requires GIN_IB_PROXY";
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();
  int rank = -1;
  ncclCommUserRank(comm, &rank);

  constexpr size_t kSlotBytes = 64;
  constexpr int kSlots = 3;
  constexpr size_t kBufferBytes = kSlots * kSlotBytes;
  void* dSrc = nullptr;
  void* dDst = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBufferBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
  });
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBufferBytes));

  ncclWindow_t srcWin = nullptr, dstWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBufferBytes, &srcWin,
                                       NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
  });
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBufferBytes, &dstWin,
                                       NCCL_WIN_COLL_SYMMETRIC));

  std::vector<uint8_t> hostSrc(kBufferBytes, 0);
  std::vector<uint8_t> hostDst(kBufferBytes, 0);
  for (int slot = 0; slot < kSlots; ++slot) {
    for (size_t i = 0; i < kSlotBytes; ++i) {
      hostSrc[slot * kSlotBytes + i] =
          static_cast<uint8_t>(0x20 + slot * 0x20 + (i & 0x1f));
    }
  }
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(dSrc, hostSrc.data(), kBufferBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess,
                hipMemcpy(dDst, hostDst.data(), kBufferBytes, hipMemcpyHostToDevice));

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginContextCount = 1;
  reqs.ginSignalCount = 1;
  reqs.ginCounterCount = 1;

  ncclDevComm first{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &first));
  bool firstLive = true;
  auto firstCleanup = makeScopeGuard([&]() {
    if (firstLive) (void)ncclDevCommDestroy(comm, &first);
  });

  ncclDevComm second{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &second));
  bool secondLive = true;
  auto secondCleanup = makeScopeGuard([&]() {
    if (secondLive) (void)ncclDevCommDestroy(comm, &second);
  });

  ASSERT_MPI_GT(first.ginConnectionCount, 0);
  ASSERT_MPI_EQ(first.ginConnectionCount, second.ginConnectionCount);
  bool handlesDistinct = true;
  for (int connection = 0; connection < first.ginConnectionCount; ++connection) {
    handlesDistinct =
        handlesDistinct &&
        first.ginHandles[connection] != second.ginHandles[connection];
  }
  int minHandlesDistinct = handlesDistinct ? 1 : 0;
  int maxHandlesDistinct = minHandlesDistinct;
  MPI_Allreduce(MPI_IN_PLACE, &minHandlesDistinct, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxHandlesDistinct, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  // Avoid a second destroy through an intentionally aliased handle in the
  // fail-before mutation; the process may exit with the leaked test context.
  if (minHandlesDistinct == 0) secondLive = false;
  ASSERT_MPI_TRUE(minHandlesDistinct == 1 && maxHandlesDistinct == 1);

  auto stateMatches = [&](ncclDevComm devComm,
                          uint64_t expectedSignal,
                          uint64_t expectedCounter) {
    uint64_t state[2] = {};
    if (readPerDevCommState(devComm, stream, state) != hipSuccess) return false;
    return state[0] == expectedSignal && state[1] == expectedCounter;
  };

  int* dPhaseCompleted = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dPhaseCompleted, sizeof(int)));
  auto phaseStatusCleanup = makeScopeGuard([&]() {
    if (dPhaseCompleted) (void)hipFree(dPhaseCompleted);
  });

  ASSERT_MPI_TRUE(stateMatches(first, 0, 0));
  ASSERT_MPI_TRUE(stateMatches(second, 0, 0));

  bool phaseCompleted = false;
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dPhaseCompleted, 0, sizeof(int)));
  ASSERT_MPI_EQ(hipSuccess,
                runPerDevCommIsolationPhase(rank, srcWin, dstWin,
                                            /*offset=*/0, kSlotBytes,
                                            /*expectedEpoch=*/1, first, stream,
                                            dPhaseCompleted,
                                            &phaseCompleted));
  ASSERT_MPI_TRUE(phaseCompleted);
  ASSERT_MPI_TRUE(stateMatches(first, rank == 1 ? 1 : 0, rank == 0 ? 1 : 0));
  ASSERT_MPI_TRUE(stateMatches(second, 0, 0));

  phaseCompleted = false;
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dPhaseCompleted, 0, sizeof(int)));
  ASSERT_MPI_EQ(hipSuccess,
                runPerDevCommIsolationPhase(rank, srcWin, dstWin,
                                            /*offset=*/kSlotBytes, kSlotBytes,
                                            /*expectedEpoch=*/1, second, stream,
                                            dPhaseCompleted,
                                            &phaseCompleted));
  ASSERT_MPI_TRUE(phaseCompleted);
  ASSERT_MPI_TRUE(stateMatches(first, rank == 1 ? 1 : 0, rank == 0 ? 1 : 0));
  ASSERT_MPI_TRUE(stateMatches(second, rank == 1 ? 1 : 0, rank == 0 ? 1 : 0));

  MPI_Barrier(MPI_COMM_WORLD);
  ncclResult_t firstDestroyResult = ncclDevCommDestroy(comm, &first);
  firstLive = false;
  ASSERT_MPI_EQ(ncclSuccess, firstDestroyResult);
  MPI_Barrier(MPI_COMM_WORLD);

  phaseCompleted = false;
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dPhaseCompleted, 0, sizeof(int)));
  ASSERT_MPI_EQ(hipSuccess,
                runPerDevCommIsolationPhase(rank, srcWin, dstWin,
                                            /*offset=*/2 * kSlotBytes, kSlotBytes,
                                            /*expectedEpoch=*/2, second, stream,
                                            dPhaseCompleted,
                                            &phaseCompleted));
  ASSERT_MPI_TRUE(phaseCompleted);
  ASSERT_MPI_TRUE(stateMatches(second, rank == 1 ? 2 : 0, rank == 0 ? 2 : 0));

  bool payloadMatches = true;
  if (rank == 1) {
    std::vector<uint8_t> hostResult(kBufferBytes, 0);
    payloadMatches =
        hipMemcpy(hostResult.data(), dDst, kBufferBytes, hipMemcpyDeviceToHost) ==
            hipSuccess &&
        hostSrc == hostResult;
  }
  ASSERT_MPI_TRUE(payloadMatches);

  MPI_Barrier(MPI_COMM_WORLD);
  ncclResult_t secondDestroyResult = ncclDevCommDestroy(comm, &second);
  secondLive = false;
  ASSERT_MPI_EQ(ncclSuccess, secondDestroyResult);
  MPI_Barrier(MPI_COMM_WORLD);
}

// ncclAlltoAll dispatches to ncclAllToAllGinSdma when eligible.
TEST_F(GinMPIDeviceTests, Alltoall_SdmaInternal) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (auto reason = sdmaInternalA2AEnvSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  if (auto reason = sdmaInternalA2ACommSkipReason(comm); !reason.empty())
    GTEST_SKIP() << reason;

  int rank = -1, nRanks = -1;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &nRanks);
  ASSERT_GE(nRanks, 2);
  ASSERT_LE(nRanks, 8);

  // Mirrors the NCCL_GIN_A2A_SDMA_MIN_BYTES default (enforced by the env skip).
  // Below it ncclAllToAllGinSdma runs ncclGinA2AKernel<false>, at or above <true>.
  constexpr size_t kSdmaMinBytesPerPeer = size_t{8} << 20;
  const std::vector<size_t> counts = {
      1, 1024, size_t{1} << 16,               // 4 B / 4 KiB / 256 KiB per peer -> LSA
      kSdmaMinBytesPerPeer / sizeof(float)};  // 8 MiB per peer -> SDMA engine

  // No ncclDevCommCreate here: the path lazily builds its own private devComm.
  const size_t maxBytes = counts.back() * static_cast<size_t>(nRanks) * sizeof(float);

  void* dSend = nullptr;
  void* dRecv = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, maxBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, maxBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSend) (void)ncclMemFree(dSend);
    if (dRecv) (void)ncclMemFree(dRecv);
  });

  // Registered once at the largest size and reused across the sweep.
  ncclWindow_t sendWin = nullptr, recvWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dSend, maxBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
      ncclCommWindowRegister(comm, dRecv, maxBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
    if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
  });

  for (size_t idx = 0; idx < counts.size(); idx++) {
    const size_t count = counts[idx];
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t totalElements = count * static_cast<size_t>(nRanks);
    const size_t sizeBytes     = totalElements * sizeof(float);

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

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAlltoAll(dSend, dRecv, count, ncclFloat, comm, stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Set by ncclGinA2AInitOnce and sticky for the comm's lifetime, so the first
    // dispatch is the only one that can fail this. Checking it here reports a
    // fallback before the sweep reaches the larger sizes, but it cannot tell the
    // LSA and SDMA tiers apart.
    if (idx == 0) ASSERT_MPI_TRUE(comm->ginA2AState.initialized);

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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
  const int    ctx       = (int)blockIdx.x % numContexts;
  const int    subSlotIx = (int)blockIdx.x / numContexts;
  const size_t off       = (size_t)ctx * slotStride + (size_t)subSlotIx * subSlotBytes;
  ncclGin gin{devComm, ctx};
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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
      putBasicProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          sz, kSigIdx, kPeer, devComm);
    } else {
      putBasicConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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

// Negative test: with NCCL_GIN_ENABLE=0, ncclDevCommCreate must fail.
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

  // ginSignalCount > 0 + ginConnectionType=FULL forces ncclDevCommCreate to
  // attempt GIN bring-up.
  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginSignalCount = 1;
  ncclDevComm devComm{};
  ncclResult_t r = ncclDevCommCreate(comm, &reqs, &devComm);

  ASSERT_TRUE(r == ncclInvalidUsage || r == ncclInvalidArgument)
      << "ncclDevCommCreate must fail with a request-rejected error when "
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
    ncclDevCommRequirements reqs = defaultGinReqs();
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
      putBasicProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
          srcWin, /*srcOff=*/0,
          dstWin, /*dstOff=*/0,
          kBufBytes, kSigIdx, kPeer,
          devComm);
    } else {
      putBasicConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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
  ncclDevCommRequirements reqs = defaultGinReqs();
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
    alltoallKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
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

// =====================================================================
// Coverage for the new NCCL 2.29.7 GIN device-API features. All run on the
// device-API GIN proxy path with cross-node placement (2 ranks on 2
// nodes). Tests for features that depend on topology/library support that
// may be unavailable skip gracefully rather than fail.
// =====================================================================

// nLsaTeams / ginType / railedGinType via ncclCommQueryProperties (2.29.7).
TEST_F(GinMPIDeviceTests, Properties_NLsaTeams) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/8))
    GTEST_SKIP() << "Requires 2-8 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  int nRanks = -1;
  ncclCommCount(comm, &nRanks);

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));

  // Backend type must match the requested NCCL_GIN_TYPE (proxy / rocshmem-gda / anvil-sdma).
  EXPECT_EQ(requestedGinType(), (int)props.ginType);

  // nLsaTeams = nRanks / lsaSize: >= 1 and must evenly divide nRanks.
  // Skip when the runtime reports nLsaTeams==0 on this configuration.
  if (props.nLsaTeams == 0)
    GTEST_SKIP() << "nLsaTeams reported 0 on this configuration";
  EXPECT_GE(props.nLsaTeams, 1);
  EXPECT_EQ(0, nRanks % props.nLsaTeams)
      << "nRanks=" << nRanks << " not divisible by nLsaTeams=" << props.nLsaTeams;
}

// Exclusive GIN contexts -- reqs.ginExclusiveContexts (2.29.7).
TEST_F(GinMPIDeviceTests, MultiContext_Exclusive) {
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

  constexpr int    kNumContexts   = 2;
  constexpr size_t kSlotStride    = 4 * 1024;
  constexpr size_t kTransferBytes = 1 * 1024;
  constexpr size_t kBufBytes      = kNumContexts * kSlotStride;
  constexpr int    kPeer          = 1;

  void* dSrc = nullptr; void* dDst = nullptr;
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

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginExclusiveContexts = true;          // <-- feature under test
  reqs.railGinBarrierCount  = 1;
  reqs.ginContextCount      = kNumContexts;
  reqs.ginSignalCount       = kNumContexts;
  ncclDevComm devComm{};
  // Exclusive contexts are carved from the same pool as shared contexts: shared
  // allocations grow from index 0 upward while exclusive ones are reserved from
  // the top down. The pool size is NCCL_GIN_NCONTEXTS (rounded up to the GIN
  // connection count). On a small topology (few connections) the default pool
  // can be fully consumed by shared allocations, leaving none to reserve
  // exclusively. Run with NCCL_GIN_NCONTEXTS large enough to leave headroom
  // (e.g. >= shared usage + ginContextCount; 8 is sufficient on a 2-rank/1-NIC
  // setup). Skip (rather than fail) when the pool has no room on this config.
  ncclResult_t cr = ncclDevCommCreate(comm, &reqs, &devComm);
  if (cr != ncclSuccess)
    GTEST_SKIP() << "Exclusive GIN contexts unavailable on this configuration "
                    "(increase NCCL_GIN_NCONTEXTS for more pool headroom); rc=" << (int)cr;
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  // The runtime rounds the requested context count up to a multiple of the GIN
  // connection count, so expect ROUNDUP(kNumContexts, ginConnectionCount) rather
  // than an exact match. Kernels below only drive [0, kNumContexts), so extras are harmless.
  const int connCount = (int)devComm.ginConnectionCount;
  const int expectedCtxs = connCount > 0
      ? ((kNumContexts + connCount - 1) / connCount) * connCount
      : kNumContexts;
  ASSERT_EQ(expectedCtxs, (int)devComm.ginContextCount)
      << "exclusive allocation returned " << (int)devComm.ginContextCount
      << " for " << connCount << " connection(s) (requested " << kNumContexts << ")";
  ASSERT_GE((int)devComm.ginContextCount, kNumContexts);

  std::vector<uint8_t> hostSrc(kBufBytes, 0), hostDst(kBufBytes, 0);
  for (int b = 0; b < kNumContexts; b++)
    std::fill_n(hostSrc.begin() + b * kSlotStride, kTransferBytes,
                static_cast<uint8_t>(0x20 + b));
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

  if (rank == 1) {
    std::vector<uint8_t> res(kBufBytes, 0);
    ASSERT_EQ(hipSuccess, hipMemcpy(res.data(), dDst, kBufBytes, hipMemcpyDeviceToHost));
    for (int b = 0; b < kNumContexts; b++) {
      const uint8_t pat = static_cast<uint8_t>(0x20 + b);
      const size_t base = (size_t)b * kSlotStride;
      for (size_t i = 0; i < kTransferBytes; i++)
        ASSERT_EQ(pat, res[base + i]) << "ctx " << b << " byte " << i;
    }
  }
}

// VA-based GIN signals -- ncclGin_VASignalInc + waitSignal(window, offset) (2.29.7).
// The other tests use the INDEXED signal API (ncclGin_SignalInc + a pool slot
// index). This one addresses the signal by its virtual address -- a (window,
// offset) pair backed by a user-registered symmetric window -- which drives the
// NCCL_GIN_SIGNAL_TYPE_VA descriptor path instead of NCCL_GIN_SIGNAL_TYPE_INDEXED.
// On the proxy backend the VA signal is emitted as a separate GFD after the
// data put (see gin/proxy/gin_proxy.h), so a successful payload+signal here
// proves the VA signal landed.
__global__ void putVASignalProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin, size_t bytes,
    ncclWindow_t sigWin, size_t sigOff, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, /*dstOff=*/0,
            srcWin, /*srcOff=*/0,
            bytes,
            ncclGin_VASignalInc{sigWin, sigOff});   // <-- VA signal (window,offset)
  }
  gin.flush(ncclCoopCta());
}

__global__ void putVASignalConsumerKernel(
    ncclWindow_t sigWin, size_t sigOff, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigWin, sigOff, expectedSignalValue);  // <-- VA wait
}

// waitSignal (at-least) then readSignal for exact post-wait value checks.
__global__ void indexedSignalWaitReadConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expected,
    uint64_t* outRead, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expected);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outRead = gin.readSignal(sigIdx);
  }
}

__global__ void vaSignalWaitReadConsumerKernel(
    ncclWindow_t sigWin, size_t sigOff, uint64_t expected,
    uint64_t* outRead, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigWin, sigOff, expected);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outRead = gin.readSignal(sigWin, sigOff);
  }
}

// Context-aware VA-signal producer/consumer for VASignal_Put: one block per
// GIN context (ctx = blockIdx.x). Each context owns a payload slot at
// ctx*payloadStride and a distinct VA signal cell at sigOff+ctx*sigStride
// inside the shared sigWin. Dedicated (not the shared putVASignal* kernels,
// which VASignal_StrictOrderingFlag reuses with a fixed signature). Launched
// with 1 block this matches the original single-context behavior.
__global__ void putVAMultiCtxProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin, size_t bytes, size_t payloadStride,
    ncclWindow_t sigWin, size_t sigOff, size_t sigStride, int peer,
    struct ncclDevComm devComm) {
  const int    ctx  = (int)blockIdx.x;
  const size_t base = (size_t)ctx * payloadStride;
  const size_t so   = sigOff + (size_t)ctx * sigStride;
  ncclGin gin{devComm, ctx};
  if (threadIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, base,
            srcWin, base,
            bytes,
            ncclGin_VASignalInc{sigWin, so});   // <-- VA signal (window,offset)
  }
  gin.flush(ncclCoopCta());
}

__global__ void putVAMultiCtxConsumerKernel(
    ncclWindow_t sigWin, size_t sigOff, size_t sigStride, uint64_t expectedSignalValue,
    struct ncclDevComm devComm) {
  const int ctx = (int)blockIdx.x;
  ncclGin gin{devComm, ctx};
  gin.waitSignal(ncclCoopCta(), sigWin, sigOff + (size_t)ctx * sigStride,
                 expectedSignalValue);          // <-- VA wait
}

void GinMPIDeviceTests::runVASignalPut(int nContexts) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = vaSignalTestSkipReason(); !reason.empty())
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

  constexpr size_t kSlotBytes = 4096;  // per-context payload slot
  constexpr size_t kSigStride = 64;    // per-context 8B VA signal cell
  constexpr size_t kSigOff    = 0;
  constexpr int    kPeer      = 1;
  constexpr uint64_t kExpectedSignal = 1;  // SignalInc from 0 on producer put
  const size_t kBytes    = (size_t)nContexts * kSlotBytes;
  const size_t kSigBytes = (size_t)nContexts * kSigStride;

  // Separate buffers for payload (src/dst) and the VA signal cells.
  void* dSrc = nullptr; void* dDst = nullptr; void* dSig = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSig, kSigBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
    if (dSig) (void)ncclMemFree(dSig);
  });

  // The VA signal is addressed through its own symmetric window; the payload
  // uses ordinary symmetric windows.
  ncclWindow_t srcWin = nullptr, dstWin = nullptr, sigWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSig, kSigBytes, &sigWin, NCCL_WIN_COLL_SYMMETRIC));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    if (sigWin) (void)ncclCommWindowDeregister(comm, sigWin);
  });

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginContextCount     = nContexts;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  if ((int)devComm.ginContextCount < nContexts) {
    GTEST_SKIP() << "Test requires >= " << nContexts << " GIN contexts, got "
                 << (int)devComm.ginContextCount
                 << " (set NCCL_GIN_NCONTEXTS=" << nContexts << ")";
  }

  // Stage per-context payload; zero the destination and every VA signal cell on
  // every rank (each signal is incremented from 0 -> 1 by its producer put).
  std::vector<uint8_t> hs(kBytes, 0), hd(kBytes, 0), hsig(kSigBytes, 0);
  for (int ctx = 0; ctx < nContexts; ctx++) {
    const size_t base = (size_t)ctx * kSlotBytes;
    for (size_t i = 0; i < kSlotBytes; i++)
      hs[base + i] = static_cast<uint8_t>(0x37 + ((i + ctx) & 0x1F));
  }
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hs.data(),   kBytes,    hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hd.data(),   kBytes,    hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSig, hsig.data(), kSigBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    putVAMultiCtxProducerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        srcWin, dstWin, kSlotBytes, kSlotBytes, sigWin, kSigOff, kSigStride, kPeer, devComm);
  else
    putVAMultiCtxConsumerKernel<<<nContexts, kGinKernelThreads, 0, stream>>>(
        sigWin, kSigOff, kSigStride, kExpectedSignal, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  {
    std::vector<uint8_t> r(kBytes, 0);
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(r.data(), dDst, kBytes, hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_BUFFER_EQ_ON_RANK(rank, 1, hs, r);
  }
}

TEST_F(GinMPIDeviceTests, VASignal_Put) {
  runVASignalPut(/*nContexts=*/1);
}

TEST_F(GinMPIDeviceTests, VASignal_Put_MultiContext) {
  int n = ginEnvContextCount();
  if (n == 0) GTEST_SKIP() << "Set NCCL_GIN_NCONTEXTS>1 to run the multi-context variant";
  runVASignalPut(n);
}

// ---------------------------------------------------------------------------
// Phase-1 signal coverage (NCCL device GIN API gaps vs GinDeviceMPITests)
// ---------------------------------------------------------------------------

__global__ void putVASignalAddProducerKernel(
    ncclWindow_t srcWin, ncclWindow_t dstWin, size_t bytes,
    ncclWindow_t sigWin, size_t sigOff, uint64_t addend, int peer,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.put(ncclTeamWorld(devComm), peer,
            dstWin, /*dstOff=*/0,
            srcWin, /*srcOff=*/0,
            bytes,
            ncclGin_VASignalAdd{sigWin, sigOff, addend});
  }
  gin.flush(ncclCoopCta());
}

__global__ void vaSignalOnlyIncProducerKernel(
    ncclWindow_t sigWin, size_t sigOff, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.signal(ncclTeamWorld(devComm), peer, ncclGin_VASignalInc{sigWin, sigOff});
  }
  gin.flush(ncclCoopCta());
}

__global__ void vaSignalOnlyAddProducerKernel(
    ncclWindow_t sigWin, size_t sigOff, uint64_t addend, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.signal(ncclTeamWorld(devComm), peer, ncclGin_VASignalAdd{sigWin, sigOff, addend});
  }
  gin.flush(ncclCoopCta());
}

__global__ void indexedSignalOnlyAddProducerKernel(
    ncclGinSignal_t sigIdx, uint64_t addend, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.signal(ncclTeamWorld(devComm), peer, ncclGin_SignalAdd{sigIdx, addend});
  }
  gin.flush(ncclCoopCta());
}

__global__ void vaSignalReadResetConsumerKernel(
    ncclWindow_t sigWin, size_t sigOff, uint64_t expected,
    uint64_t* outRead, uint64_t* outAfterReset, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigWin, sigOff, expected);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outRead = gin.readSignal(sigWin, sigOff);
    gin.resetSignal(sigWin, sigOff);
    *outAfterReset = gin.readSignal(sigWin, sigOff);
  }
}

__global__ void indexedSignalReadResetConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t expected,
    uint64_t* outRead, uint64_t* outAfterReset, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, expected);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outRead = gin.readSignal(sigIdx);
    gin.resetSignal(sigIdx);
    *outAfterReset = gin.readSignal(sigIdx);
  }
}

__global__ void putValueSignalAddProducerKernel(
    ncclWindow_t dstWin, size_t dstOff, uint64_t value,
    ncclGinSignal_t sigIdx, uint64_t addend, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.putValue<uint64_t>(ncclTeamWorld(devComm), peer,
                           dstWin, dstOff, value,
                           ncclGin_SignalAdd{sigIdx, addend});
  }
  gin.flush(ncclCoopCta());
}

__global__ void signalIncRepeatProducerKernel(
    ncclGinSignal_t sigIdx, int nSignals, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto team = ncclTeamWorld(devComm);
    for (int i = 0; i < nSignals; ++i) {
      gin.signal(team, peer, ncclGin_SignalInc{sigIdx});
    }
  }
  gin.flush(ncclCoopCta());
}

__global__ void signalAddOnceProducerKernel(
    ncclGinSignal_t sigIdx, uint64_t addend, int peer, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.signal(ncclTeamWorld(devComm), peer, ncclGin_SignalAdd{sigIdx, addend});
  }
  gin.flush(ncclCoopCta());
}

__global__ void signalWaitReadLow32BitsConsumerKernel(
    ncclGinSignal_t sigIdx, uint64_t least, int bits, uint64_t* outRead,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.waitSignal(ncclCoopCta(), sigIdx, least, bits);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outRead = gin.readSignal(sigIdx, bits);
  }
}

__global__ void signalShadowPtrKernel(
    ncclGinSignal_t sigIdx, uint64_t bumpAmount, uint64_t* outInitial, uint64_t* outAfterBump,
    struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t* shadow = gin.getSignalShadowPtr(sigIdx);
    *outInitial = *shadow;
    gin.increaseSignalShadow(sigIdx, bumpAmount);
    *outAfterBump = *shadow;
  }
}

__global__ void indexedSignalResetOnlyKernel(
    ncclGinSignal_t sigIdx, struct ncclDevComm devComm) {
  ncclGin gin{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gin.resetSignal(sigIdx);
  }
}

// gin.signal() with VA Inc/Add and indexed SignalAdd (no payload GFDs).
TEST_F(GinMPIDeviceTests, VASignal_NoPayload_IncAndAdd) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = vaSignalTestSkipReason(); !reason.empty())
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

  constexpr size_t kSigBytes = 64;
  constexpr size_t kSigOff   = 0;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr int kPeer = 1;
  constexpr uint64_t kPhaseAExpected = 1;   // VA SignalInc: 0 -> 1
  constexpr uint64_t kPhaseBAddend   = 7;
  constexpr uint64_t kPhaseCAddend   = 3;

  void* dSig = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSig, kSigBytes));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSig) (void)ncclMemFree(dSig);
  });

  ncclWindow_t sigWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSig, kSigBytes, &sigWin,
                                       NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
  auto winCleanup = makeScopeGuard([&]() {
    if (sigWin) (void)ncclCommWindowDeregister(comm, sigWin);
  });

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  std::vector<uint8_t> z(kSigBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSig, z.data(), kSigBytes, hipMemcpyHostToDevice));

  void* dOut = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOut, sizeof(uint64_t)));
  auto outCleanup = makeScopeGuard([&]() {
    if (dOut) (void)hipFree(dOut);
  });

  // Phase A: VA signal-only Inc (0 -> 1).
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    vaSignalOnlyIncProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(sigWin, kSigOff, kPeer, devComm);
  else
    vaSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        sigWin, kSigOff, kPhaseAExpected, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhaseAExpected, readVal);
  }

  ASSERT_MPI_EQ(hipSuccess, hipMemset(dSig, 0, kSigBytes));

  // Phase B: VA signal-only Add (0 -> kPhaseBAddend).
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    vaSignalOnlyAddProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        sigWin, kSigOff, kPhaseBAddend, kPeer, devComm);
  else
    vaSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        sigWin, kSigOff, kPhaseBAddend, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhaseBAddend, readVal);
  }

  // Phase C: indexed signal-only SignalAdd (pool cell 1: 0 -> kPhaseCAddend).
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    indexedSignalOnlyAddProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhaseCAddend, kPeer, devComm);
  else
    indexedSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhaseCAddend, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhaseCAddend, readVal);
  }
}

// readSignal / resetSignal for VA and indexed pool cells.
TEST_F(GinMPIDeviceTests, VASignal_ReadAndReset) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = vaSignalTestSkipReason(); !reason.empty())
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

  constexpr size_t kBytes    = 4096;
  constexpr size_t kSigBytes = 64;
  constexpr size_t kSigOff   = 8;   // non-zero, 8-byte aligned VA signal cell
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr uint64_t kVaAddend = 5;
  constexpr uint64_t kIdxAddend = 9;
  constexpr uint64_t kIdxPutIncAddend = kIdxAddend - 1;  // put SignalInc + one SignalAdd
  constexpr uint64_t kResetExpected = 0;
  constexpr int kPeer = 1;

  void* dSrc = nullptr;
  void* dDst = nullptr;
  void* dSig = nullptr;
  void* dOut = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSrc, kBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dDst, kBytes));
  ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSig, kSigBytes));
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOut, 2 * sizeof(uint64_t)));
  auto memCleanup = makeScopeGuard([&]() {
    if (dSrc) (void)ncclMemFree(dSrc);
    if (dDst) (void)ncclMemFree(dDst);
    if (dSig) (void)ncclMemFree(dSig);
    if (dOut) (void)hipFree(dOut);
  });

  ncclWindow_t srcWin = nullptr, dstWin = nullptr, sigWin = nullptr;
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSrc, kBytes, &srcWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dDst, kBytes, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_MPI_EQ(ncclSuccess,
                ncclCommWindowRegister(comm, dSig, kSigBytes, &sigWin,
                                       NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
  auto winCleanup = makeScopeGuard([&]() {
    if (srcWin) (void)ncclCommWindowDeregister(comm, srcWin);
    if (dstWin) (void)ncclCommWindowDeregister(comm, dstWin);
    if (sigWin) (void)ncclCommWindowDeregister(comm, sigWin);
  });

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  std::vector<uint8_t> hs(kBytes, 0), hd(kBytes, 0), hsig(kSigBytes, 0);
  for (size_t i = 0; i < kBytes; i++) hs[i] = static_cast<uint8_t>(0x11 + (i & 0x0F));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSrc, hs.data(), kBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hd.data(), kBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dSig, hsig.data(), kSigBytes, hipMemcpyHostToDevice));
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, 2 * sizeof(uint64_t)));

  // VA: put + VASignalAdd (STRICT_ORDERING, non-zero sigOff), read/reset, payload check.
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    putVASignalAddProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, dstWin, kBytes, sigWin, kSigOff, kVaAddend, kPeer, devComm);
  } else {
    vaSignalReadResetConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        sigWin, kSigOff, kVaAddend, (uint64_t*)dOut, (uint64_t*)dOut + 1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  {
    uint64_t hostOut[2] = {0, 0};
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(hostOut, dOut, sizeof(hostOut), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kVaAddend, hostOut[0]);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kResetExpected, hostOut[1]);
  }
  {
    std::vector<uint8_t> r(kBytes, 0);
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(r.data(), dDst, kBytes, hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_BUFFER_EQ_ON_RANK(rank, 1, hs, r);
  }

  // Indexed: put + SignalAdd, then read/reset on consumer.
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, 2 * sizeof(uint64_t)));
  if (rank == 0) {
    putBasicProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        srcWin, 0, dstWin, 0, kBytes, kSigIdx, kPeer, devComm);
    signalAddOnceProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kIdxPutIncAddend, kPeer, devComm);
  } else {
    indexedSignalReadResetConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kIdxAddend, (uint64_t*)dOut, (uint64_t*)dOut + 1, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  {
    uint64_t hostOut[2] = {0, 0};
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(hostOut, dOut, sizeof(hostOut), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kIdxAddend, hostOut[0]);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kResetExpected, hostOut[1]);
  }
}

// putValue + ncclGin_SignalAdd (variable addend, not Inc).
TEST_F(GinMPIDeviceTests, SignalAdd_PutValue) {
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

  constexpr size_t   kBufBytes = 4 * 1024;
  constexpr size_t   kDstOff   = 512;
  constexpr uint64_t kValue    = 0xA5A5A5A5A5A5A5A5ULL;
  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr uint64_t kAddend = 5;
  constexpr int kPeer = 1;

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

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  std::vector<uint8_t> hostDst(kBufBytes, 0);
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(dDst, hostDst.data(), kBufBytes, hipMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    putValueSignalAddProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        dstWin, kDstOff, kValue, kSigIdx, kAddend, kPeer, devComm);
  } else {
    putBasicConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kSigIdx, kAddend, devComm);
  }
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  {
    std::vector<uint8_t> hostResult(kBufBytes, 0);
    hipError_t hipErr = hipSuccess;
    uint64_t got = 0;
    if (rank == 1) {
      hipErr = hipMemcpy(hostResult.data(), dDst, kBufBytes, hipMemcpyDeviceToHost);
      std::memcpy(&got, hostResult.data() + kDstOff, sizeof(got));
    }
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kValue, got);
  }
}

// SignalInc repetition, optional Inc/Add mixing, recovery via resetSignal.
TEST_F(GinMPIDeviceTests, SignalInc_MixingRule) {
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
  constexpr int kPeer = 1;
  constexpr int kIncRepeatCount = 2;
  constexpr uint64_t kPhase1Expected = kIncRepeatCount;
  constexpr uint64_t kPhase2Addend   = 5;
  constexpr uint64_t kPhase2Expected = kPhase1Expected + kPhase2Addend;
  constexpr uint64_t kPhase3Addend   = 10;
  constexpr uint64_t kPhase3Expected = kPhase3Addend;
  constexpr int kResetKernelBlocks  = 1;
  constexpr int kResetKernelThreads = 1;

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  void* dOut = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOut, sizeof(uint64_t)));
  auto outCleanup = makeScopeGuard([&]() {
    if (dOut) (void)hipFree(dOut);
  });

  // P1: consecutive SignalInc -> signal cell reaches kPhase1Expected.
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    signalIncRepeatProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kIncRepeatCount, kPeer, devComm);
  else
    indexedSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhase1Expected, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhase1Expected, readVal);
  }

  // P2: SignalAdd without reset (NCCL spec forbids mixing; RCCL may allow).
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    signalAddOnceProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhase2Addend, kPeer, devComm);
  else
    indexedSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhase2Expected, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhase2Expected, readVal);
  }

  // P3: reset on consumer, then Add-only sequence must land at absolute kPhase3Expected.
  if (rank == 1)
    indexedSignalResetOnlyKernel<<<kResetKernelBlocks, kResetKernelThreads, 0, stream>>>(
        kSigIdx, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));
  if (rank == 0)
    signalAddOnceProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhase3Addend, kPeer, devComm);
  else
    indexedSignalWaitReadConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kPhase3Expected, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kPhase3Expected, readVal);
  }
}

// waitSignal/readSignal(..., bits=32): masked low-32-bit compare and readback.
TEST_F(GinMPIDeviceTests, Signal_WaitRead_Low32Bits) {
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
  constexpr int kPeer = 1;
  constexpr uint64_t kLargeAdd = 0x100000002ULL;
  constexpr uint64_t kLeast32  = 2;
  constexpr int kBits = 32;

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  void* dOut = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOut, sizeof(uint64_t)));
  auto outCleanup = makeScopeGuard([&]() {
    if (dOut) (void)hipFree(dOut);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, sizeof(uint64_t)));

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    signalAddOnceProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(kSigIdx, kLargeAdd, kPeer, devComm);
  else
    signalWaitReadLow32BitsConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
        kSigIdx, kLeast32, kBits, (uint64_t*)dOut, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  {
    uint64_t readVal = 0;
    hipError_t hipErr = hipSuccess;
    if (rank == 1)
      hipErr = hipMemcpy(&readVal, dOut, sizeof(readVal), hipMemcpyDeviceToHost);
    ASSERT_MPI_HIP_OK_ON_RANK(rank, 1, hipErr);
    constexpr uint64_t kExpectedMasked = kLargeAdd & ((1ULL << kBits) - 1);
    ASSERT_MPI_EQ_ON_RANK(rank, 1, kExpectedMasked, readVal);
  }
}

// NCCL docs: VA signals should use NCCL_WIN_STRICT_ORDERING on the signal window.
TEST_F(GinMPIDeviceTests, VASignal_StrictOrderingFlag) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (auto reason = vaSignalTestSkipReason(); !reason.empty())
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

  constexpr size_t kSigBytes = 64;
  constexpr size_t kSigOff = 0;
  constexpr int kPeer = 1;
  constexpr uint64_t kAddend = 1;
  constexpr uint64_t kExpected = kAddend;

  auto runSmoke = [&](int sigWinFlags) -> bool {
    void* dSig = nullptr;
    if (ncclMemAlloc(&dSig, kSigBytes) != ncclSuccess) return false;
    ncclWindow_t sigWin = nullptr;
    ncclResult_t reg =
      ncclCommWindowRegister(comm, dSig, kSigBytes, &sigWin, sigWinFlags);
    if (reg != ncclSuccess) {
      (void)ncclMemFree(dSig);
      return false;
    }

    ncclDevCommRequirements r = defaultGinReqs();
    r.railGinBarrierCount = 1;
    r.ginSignalCount = 1;
    ncclDevComm dc{};
    if (ncclDevCommCreate(comm, &r, &dc) != ncclSuccess) {
      (void)ncclCommWindowDeregister(comm, sigWin);
      (void)ncclMemFree(dSig);
      return false;
    }

    std::vector<uint8_t> z(kSigBytes, 0);
    (void)hipMemcpy(dSig, z.data(), kSigBytes, hipMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
      vaSignalOnlyAddProducerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
          sigWin, kSigOff, kAddend, kPeer, dc);
    else
      putVASignalConsumerKernel<<<kGinKernelBlocks, kGinKernelThreads, 0, stream>>>(
          sigWin, kSigOff, kExpected, dc);
    hipError_t hipRet = hipStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    (void)ncclDevCommDestroy(comm, &dc);
    (void)ncclCommWindowDeregister(comm, sigWin);
    (void)ncclMemFree(dSig);
    return hipRet == hipSuccess;
  };

  ASSERT_TRUE(runSmoke(NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING))
      << "VA signal with NCCL_WIN_STRICT_ORDERING must succeed";

  // RCCL currently registers all symmetric MRs with REMOTE_ATOMIC; this may
  // still pass without STRICT_ORDERING. Outcome is informational only.
  (void)runSmoke(NCCL_WIN_COLL_SYMMETRIC);
}

// getSignalShadowPtr + increaseSignalShadow on the indexed shadow table.
TEST_F(GinMPIDeviceTests, SignalShadow_GetPtr) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 2;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  auto devCommCleanup = makeScopeGuard([&]() {
    (void)ncclDevCommDestroy(comm, &devComm);
  });

  void* dOut = nullptr;
  ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dOut, 2 * sizeof(uint64_t)));
  auto outCleanup = makeScopeGuard([&]() {
    if (dOut) (void)hipFree(dOut);
  });
  ASSERT_MPI_EQ(hipSuccess, hipMemset(dOut, 0, 2 * sizeof(uint64_t)));

  constexpr ncclGinSignal_t kSigIdx = 1;
  constexpr uint64_t kInitialExpected = 0;
  constexpr uint64_t kShadowBump      = 42;
  constexpr int kSingleThreadBlocks  = 1;
  constexpr int kSingleThreadThreads = 1;
  signalShadowPtrKernel<<<kSingleThreadBlocks, kSingleThreadThreads, 0, stream>>>(
      kSigIdx, kShadowBump, (uint64_t*)dOut, (uint64_t*)dOut + 1, devComm);
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

  uint64_t hostOut[2] = {0, 0};
  ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hostOut, dOut, sizeof(hostOut), hipMemcpyDeviceToHost));
  ASSERT_MPI_EQ(kInitialExpected, hostOut[0]);
  ASSERT_MPI_EQ(kShadowBump, hostOut[1]);
}

// RAIL / no-cross-rail connection type -- ginConnectionType=RAIL (2.29.7).
TEST_F(GinMPIDeviceTests, RailConnection_Create) {
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;
  if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2))
    GTEST_SKIP() << "Requires exactly 2 ranks";

  // ginIsRailed reflects the communicator's connection mode (RAIL only when the
  // system cannot cross NICs). Opt in by disabling cross-NIC: NCCL_CROSS_NIC=0.
  const char* crossNic = std::getenv("NCCL_CROSS_NIC");
  if (!crossNic || std::strcmp(crossNic, "0") != 0)
    GTEST_SKIP() << "RAIL connection requires NCCL_CROSS_NIC=0 (rail-only mode)";

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  ncclComm_t comm = getActiveCommunicator();

  // RAIL is only valid when the communicator advertises a railed GIN type.
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));
  if (props.railedGinType == NCCL_GIN_TYPE_NONE)
    GTEST_SKIP() << "Communicator does not advertise railed GIN (railedGinType=NONE)";

  ncclDevCommRequirements reqs = defaultGinReqs();
  reqs.ginConnectionType   = NCCL_GIN_CONNECTION_RAIL;   // <-- feature under test
  reqs.railGinBarrierCount = 1;
  reqs.ginSignalCount      = 1;
  ncclDevComm devComm{};
  ASSERT_MPI_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));
  EXPECT_TRUE(devComm.ginConnectionsRailed) << "devComm should report railed GIN";
  (void)ncclDevCommDestroy(comm, &devComm);
}


std::string intraNodeSymReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize < 2)
    return "Symmetric ReduceScatter requires >=2 ranks per node";
  return "";
}

// End-to-end ReduceScatter through the public collective API on symmetric
// (NCCL_WIN_COLL_SYMMETRIC) buffers. Registering the buffers makes the
// symmetric-memory RS kernel (RailA2A_LsaLD) eligible, so this drives the
// production kernel path -- not a hand-written reference kernel.
TEST_F(GinMPIDeviceTests, ReduceScatter_Symmetric) {
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA)
    GTEST_SKIP() << "Skipping symmetric ReduceScatter (RailA2A_LsaLD) for rocSHMEM-GDA";
  // anvil-sdma is single-node. Symmetric ReduceScatter is a multi-node test.
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ANVIL_SDMA)
    GTEST_SKIP() << "Symmetric ReduceScatter not supported for anvil-sdma: single-node backend";
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (auto reason = intraNodeSymReason(); !reason.empty())
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

  // 1 (alignment/tail edges), 1024 (medium), 65536 (saturating).
  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    // Send buffer holds one block per rank; recv holds this rank's block.
    const size_t sendElems = count * static_cast<size_t>(nRanks);
    const size_t sendBytes = sendElems * sizeof(float);
    const size_t recvBytes = count * sizeof(float);

    void* dSend = nullptr;
    void* dRecv = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, sendBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, recvBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSend) (void)ncclMemFree(dSend);
      if (dRecv) (void)ncclMemFree(dRecv);
    });

    ncclWindow_t sendWin = nullptr, recvWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dSend, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dRecv, recvBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
      if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
    });

    // sendbuf[j] = j + rank. After RS-sum, rank r's output element i is
    // sum over all ranks s of ((r*count + i) + s) -- a closed form we check
    // exactly (values stay < 2^24, so float arithmetic is exact here).
    std::vector<float> hostSend(sendElems);
    for (size_t j = 0; j < sendElems; j++)
      hostSend[j] = static_cast<float>(j + static_cast<size_t>(rank));
    std::vector<float> hostRecv(count, -1.0f);
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dSend, hostSend.data(), sendBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dRecv, hostRecv.data(), recvBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(dSend, dRecv, count, ncclFloat32, ncclSum, comm, stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    // Expected[i] on rank r = nRanks*(r*count + i) + nRanks*(nRanks-1)/2.
    ASSERT_EQ(hipSuccess,
        hipMemcpy(hostRecv.data(), dRecv, recvBytes, hipMemcpyDeviceToHost));
    const double sumOfRanks = static_cast<double>(nRanks) * (nRanks - 1) / 2.0;
    for (size_t i = 0; i < count; i++) {
      const double base = static_cast<double>(static_cast<size_t>(rank) * count + i);
      const float expected = static_cast<float>(nRanks * base + sumOfRanks);
      ASSERT_EQ(expected, hostRecv[i]) << "rank=" << rank << " i=" << i;
    }
  }
}

// Same as ReduceScatter_Symmetric but exercises ncclAvg, which drives the
// FuncSumPostDiv post-divide path on the symmetric RS kernel (RailA2A_LsaLD).
TEST_F(GinMPIDeviceTests, ReduceScatter_Symmetric_Avg) {
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA)
    GTEST_SKIP() << "Skipping symmetric ReduceScatter (RailA2A_LsaLD) for rocSHMEM-GDA";
  // anvil-sdma is single-node. Symmetric ReduceScatter is a multi-node test.
  if (requestedGinType() == NCCL_NET_DEVICE_GIN_ANVIL_SDMA)
    GTEST_SKIP() << "Symmetric ReduceScatter not supported for anvil-sdma: single-node backend";
  if (auto reason = ginProxyTestSkipReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (auto reason = crossNodeReason(); !reason.empty())
    GTEST_SKIP() << reason;

  if (auto reason = intraNodeSymReason(); !reason.empty())
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

  const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};

  for (size_t count : counts) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);

    const size_t sendElems = count * static_cast<size_t>(nRanks);
    const size_t sendBytes = sendElems * sizeof(float);
    const size_t recvBytes = count * sizeof(float);

    void* dSend = nullptr;
    void* dRecv = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dSend, sendBytes));
    ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&dRecv, recvBytes));
    auto memCleanup = makeScopeGuard([&]() {
      if (dSend) (void)ncclMemFree(dSend);
      if (dRecv) (void)ncclMemFree(dRecv);
    });

    ncclWindow_t sendWin = nullptr, recvWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dSend, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, dRecv, recvBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
    auto winCleanup = makeScopeGuard([&]() {
      if (sendWin) (void)ncclCommWindowDeregister(comm, sendWin);
      if (recvWin) (void)ncclCommWindowDeregister(comm, recvWin);
    });

    std::vector<float> hostSend(sendElems);
    for (size_t j = 0; j < sendElems; j++)
      hostSend[j] = static_cast<float>(j + static_cast<size_t>(rank));
    std::vector<float> hostRecv(count, -1.0f);
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dSend, hostSend.data(), sendBytes, hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
        hipMemcpy(dRecv, hostRecv.data(), recvBytes, hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(dSend, dRecv, count, ncclFloat32, ncclAvg, comm, stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    // Avg = sum/nRanks; expected[i] on rank r = (r*count + i) + (nRanks-1)/2.
    // The kernel multiplies by a float 1/nRanks, so allow a small tolerance.
    ASSERT_EQ(hipSuccess,
        hipMemcpy(hostRecv.data(), dRecv, recvBytes, hipMemcpyDeviceToHost));
    const double sumOfRanks = static_cast<double>(nRanks) * (nRanks - 1) / 2.0;
    for (size_t i = 0; i < count; i++) {
      const double base = static_cast<double>(static_cast<size_t>(rank) * count + i);
      const double expected = (nRanks * base + sumOfRanks) / nRanks;
      const double tol = (expected < 0 ? -expected : expected) * 1e-5 + 1e-3;
      ASSERT_NEAR(expected, static_cast<double>(hostRecv[i]), tol)
          << "rank=" << rank << " i=" << i;
    }
  }
}

#endif  // MPI_TESTS_ENABLED
