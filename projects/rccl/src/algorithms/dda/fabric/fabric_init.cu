/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/fabric/fabric_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/alltoall/dda_alltoall_lazy.h"
#include "algorithms/dda/alltoall/alltoall_dda_watchdog.h"
#include "bootstrap.h"
#include "rccl_common.h"
#include "param.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

using dda::common::kDdaMaxNranks;
using nccl_dda_detail::ddaFabricMaxNBlocksForScratch;
using nccl_dda_detail::ddaFabricScratchSizing;
using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::DdaFabricBarrierState;
using nccl_dda_detail::DdaFabricMaxBlocksOverride;

RCCL_PARAM(DdaFabricBufferSizeForScratch, "DDA_FABRIC_BUFFER_SIZE", -1);
// DDA AllToAll device-side wait timeout in seconds; 0 (default) = never, as in NCCL: a slow peer is
// not an error by default (rank skew is normal in training), ncclCommAbort ends every wait instead.
// See alltoall_dda_watchdog.h.
RCCL_PARAM(DdaA2aTimeout, "DDA_A2A_TIMEOUT", 0);

dda::common::DdaA2aWdCtx ncclDdaA2aWdCtx(const ncclComm* comm) {
  return static_cast<dda::common::DdaA2aErr*>(comm->ddaA2aErrDev);
}

#if RCCL_DDA_A2A_TUNING
// Tuning builds only: a test hook, not API (not in rccl.h, not built in production). The device
// address of the comm's DdaA2aErr (uint32 site at byte 0, uint32 in-flight count at byte 4), or
// nullptr before fabric init. GPU test tools find it with dlsym(RTLD_DEFAULT, ...) to force the
// concurrent trip (write the in-flight count) or to check site == 0 and in-flight == 0 after a run.
extern "C" __attribute__((visibility("default"))) void* rcclDdaA2aTestErrWordDev(ncclComm* comm) {
  return comm != nullptr ? comm->ddaA2aErrDev : nullptr;
}
#endif

const char* ncclDdaA2aWdSiteName(uint32_t site) {
  using namespace dda::common;
  switch (site) {
  case kDdaWdPushEntry: return "push ENTRY (peer has not started the op)";
  case kDdaWdPushReady: return "push READY (peer data not arrived)";
  case kDdaWdPushCuEntry: return "push-CU ENTRY (peer has not started the op)";
  case kDdaWdPushCuReady: return "push-CU READY (peer data not arrived)";
  case kDdaWdDirectEntry: return "direct ENTRY (peer has not started the op)";
  case kDdaWdDirectDone: return "direct DONE (peer data not arrived)";
  case kDdaWdLL: return "LL line flag";
  case kDdaWdLL128V2: return "LL128 line flag";
  case kDdaA2aErrLaunch: return "push / Direct launch failed on the host (device epoch out of step with the peers)";
  case kDdaWdConcurrent: return "two AllToAll kernels of this communicator in flight at once (stream ordering broken)";
  case kDdaWdAborted: return "abort flag (ncclCommAbort)";
  default: return "unknown";
  }
}

void ncclDdaA2aPoison(ncclComm* comm, uint32_t site) {
  if (comm == nullptr || comm->ddaA2aErrHost == nullptr) return;
  rcclDdaA2aPoisonHostWord(comm->ddaA2aErrHost, site);
  WARN("DDA AllToAll: rank %d: %s; no further AllToAll can run on this communicator", comm->rank,
       ncclDdaA2aWdSiteName(site));
  // Device word, best effort and never waited for (a kernel of this comm may still be spinning): copy
  // the pinned host mirror on a private non-blocking stream, in relaxed capture mode (the failed call
  // may have been inside a capture). Kernels replayed from earlier graph captures then refuse at entry.
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  const bool modeSet = cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess;
  cudaStream_t stream = nullptr;
  if (modeSet && comm->ddaA2aErrDev != nullptr &&
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess) {
    (void)cudaMemcpyAsync(comm->ddaA2aErrDev, comm->ddaA2aErrHost, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    (void)cudaStreamDestroy(stream);
  }
  (void)cudaGetLastError();
  if (modeSet) (void)cudaThreadExchangeStreamCaptureMode(&mode);
}

bool ncclDdaUseFabricPath(ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  return comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
}

ncclResult_t ncclDdaFabricCommInit(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  if (comm->nRanks < 2 || comm->nRanks > kDdaMaxNranks || comm->bootstrap == nullptr) {
    return ncclSuccess;
  }

  // Fabric DDA assumes every rank shares one UALink clique; skip (fall back to
  // normal RCCL) if this comm spans multiple cliques -- e.g. multiple racks.
  if (comm->clique.size != comm->nRanks) {
    INFO(
      NCCL_INIT,
      "ncclDdaFabricCommInit: comm spans multiple fabric cliques (nRanks %d, clique.size %d); skipping fabric DDA path",
      comm->nRanks, comm->clique.size);
    return ncclSuccess;
  }

  const int nRanks = comm->nRanks;
  const int64_t llEnabled = rcclParamDdaLL();
  const int64_t ll128Enabled = rcclParamDdaLL128();
  // Size scratch to the largest DDA/CE-scratch table cap (all collectives,
  // including graph VMM), not AllReduce VMM alone. Otherwise AG CE-Scratch
  // or AR LL128 can win the selector and then fail the ddaScratchBytes check.
  // ddaVmmMaxGraph is in that max even if this comm never captures: scratch
  // is allocated once and exported across the clique, so a later graph AR
  // must still fit. On gfx1250 that is a silent 256 MiB tax vs eager VMM.
  // RCCL_DDA_FABRIC_BUFFER_SIZE overrides.
  const size_t llThresh = rcclDdaLLThreshold(comm, ncclFuncAllReduce);
  const size_t ll128Thresh = rcclDdaLL128Threshold(comm, ncclFuncAllReduce);
  const size_t simpleThresh = rcclDdaScratchPayloadCap(comm);
  const int64_t fabricScratchOverride = rcclParamDdaFabricBufferSizeForScratch();

  // Right-sized from the DDA thresholds and nRanks (env-overridable) instead of
  // a fixed 10 GiB. RCCL_DDA_FABRIC_BUFFER_SIZE=0 disables the fabric DDA path.
  size_t bytes = ddaFabricScratchSizing(nRanks, fabricScratchOverride, rcclParamDdaEnable(), simpleThresh, llEnabled,
                                        ll128Enabled, ll128Thresh);
  if (bytes == 0) {
    return ncclSuccess;
  }

  // Scratch (temp) buffer via VMM: the fabric path requires a fabric-capable
  // (cuMem) allocation so the handle can be exported across the clique. If VMM
  // is unavailable the fabric path is skipped (DDA disabled, normal RCCL path
  // used). ncclCuMemAlloc rounds size up to the allocation granularity.
  if (!ncclCuMemEnable()) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM unavailable; skipping fabric DDA path");
    return ncclSuccess;
  }

  const char* fabricMaxBlocksOverride = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
  DdaFabricMaxBlocksOverride parsedOverride;
  const int localBlocksMax =
    ddaFabricMaxNBlocksForScratch(comm->cuCount, fabricMaxBlocksOverride, &parsedOverride);

  if (parsedOverride.specified && !parsedOverride.valid) {
    WARN("ncclDdaFabricCommInit: Ignoring invalid RCCL_DDA_FABRIC_MAXBLOCKS='%s'; using CU-derived cap %d.",
         fabricMaxBlocksOverride, localBlocksMax);
  } else if (parsedOverride.valid && parsedOverride.requested < 1) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld is below the minimum; using 1.",
         parsedOverride.requested);
  } else if (parsedOverride.valid && parsedOverride.requested > localBlocksMax) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld exceeds CU-derived cap (%d); using %d. "
         "The override can only lower the block count, not raise it.",
         parsedOverride.requested, localBlocksMax, localBlocksMax);
  }

  // Per rank {CU-derived block cap, nonblocking config}.
  std::vector<int> blockCaps(2 * nRanks, 0);
  blockCaps[2 * comm->rank] = localBlocksMax;
  blockCaps[2 * comm->rank + 1] = comm->config.blocking ? 0 : 1;
  // Fabric cliques are normally homogeneous, but harvested or disabled CUs
  // can produce different local caps. Every rank must launch the same number
  // of blocks because the barrier pairs participants by blockIdx.
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, blockCaps.data(), 2 * sizeof(int)));

  int nBlocksMax = localBlocksMax;
  // AllToAll buffers now (eager) if any rank's comm is nonblocking (its API calls must not wait for
  // peers), else on first use; decided from gathered values so every rank makes the same choice.
  bool a2aEager = false;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (blockCaps[2 * rank] < nBlocksMax) {
      nBlocksMax = blockCaps[2 * rank];
    }
    a2aEager = a2aEager || blockCaps[2 * rank + 1] != 0;
  }

  const size_t eagerVmmAr = rcclDdaVmmThreshold(comm, ncclFuncAllReduce);
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: allocating %zu-byte scratch; payload cap %lld includes graph VMM "
       "(ddaVmmMaxGraph) even for eager-only comms; AR eager VMM=%zu. "
       "Override with RCCL_DDA_FABRIC_BUFFER_SIZE.",
       bytes, (long long)simpleThresh, eagerVmmAr);

  // Owned resources: handed to comm on success, freed at `fail` otherwise.
  // All declared before any goto so the cleanup label is reachable without
  // jumping over an initialization.
  void* scratch = nullptr;
  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  void* peerDev = nullptr;
  void** peerHost = nullptr;
  DdaFabricBarrierState* barrierState = nullptr;
  uint32_t* epochDev = nullptr;
  void* a2aErrDev = nullptr;
  uint32_t* a2aErrHost = nullptr;
  const int64_t a2aTimeoutSecs = rcclParamDdaA2aTimeout();
  std::vector<void*> h_ptrs(nRanks, nullptr);
  const size_t epochLen = ddaLLEpochCount(nRanks, nBlocksMax);
  ncclResult_t res = ncclSuccess;

  res = ncclCuMemAlloc(&scratch, &scratchHandle, ncclCuMemHandleType, bytes, comm->memManager);
  if (res != ncclSuccess || scratch == nullptr) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM scratch alloc failed; skipping fabric DDA path");
    scratch = nullptr;
    goto fail;
  }

  handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  if (handler == nullptr) {
    WARN("ncclDdaFabricCommInit: OOM allocating ncclFabricMemHandler");
    goto fail;
  }

  NCCLCHECKGOTO(handler->addSelfDeviceMem(scratch, scratchHandle, bytes), res, fail);
  NCCLCHECKGOTO(handler->exchangeMemPtrs(), res, fail);

  CUDACHECKGOTO(cudaMalloc(&peerDev, nRanks * sizeof(void*)), res, fail);

  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, fail);
  }

  CUDACHECKGOTO(cudaMemcpy(peerDev, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToDevice), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&peerHost, nRanks), res, fail);
  CUDACHECKGOTO(cudaMemcpy(peerHost, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToHost), res, fail);

  {
    auto barrierPair =
      dda::common::FabricGpuBarrier::mallocAndInit(nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager);
    if (!barrierPair.first) {
      WARN("ncclDdaFabricCommInit: FabricGpuBarrier malloc/init failed");
      goto fail;
    }
    // A local failure from here on is agreed below (no early exit).
    barrierState = new (std::nothrow) DdaFabricBarrierState();
    if (barrierState == nullptr) {
      WARN("ncclDdaFabricCommInit: OOM allocating DdaFabricBarrierState");
    } else {
      barrierState->resources = std::move(barrierPair.first);
      barrierState->barrierHost = barrierPair.second;
    }
  }

  {
    // Local steps after the last exchange; every rank that got here agrees on the outcome, so no
    // rank ends with the fabric path (and a pending AllToAll buffer agreement) while another has none.
    bool ok = barrierState != nullptr;
    // Zero the scratch once so the first epoch (>= 1) across all LL operations
    // never false-matches leftover flag words. Subsequent LL calls rely on
    // monotonic epochs + the 2-bank layout rather than re-zeroing.
    ok = ok && cudaMemset(scratch, 0, bytes) == cudaSuccess;
    // Device epoch cells for the LL collectives: zero-initialised
    // so the first device-derived flag is 1. Bumped on the device every LL launch.
    ok = ok && cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)) == cudaSuccess;
    ok = ok && cudaMemset(epochDev, 0, epochLen * sizeof(uint32_t)) == cudaSuccess;
    // AllToAll per-comm error word (device) + its host-pinned mirror (alltoall_dda_watchdog.h).
    // Its constant part (host mirror, abort flag, wall_clock64 tick counts at 100 MHz on gfx1250: the
    // opt-in timeout and the 1 ms abort-poll gate) is written here once, so kernels take one pointer.
    ok = ok && cudaMalloc(&a2aErrDev, sizeof(dda::common::DdaA2aErr)) == cudaSuccess;
    ok = ok && ncclCudaHostCalloc(&a2aErrHost, 1) == ncclSuccess;
    if (ok) {
      dda::common::DdaA2aErr init{};
      const double khz = GetDeviceWallClockRateInKhz(comm->cudaDev);
      init.errHost = a2aErrHost;
      init.abortFlag = comm->abortFlagDev;
      init.ticks = a2aTimeoutSecs > 0 ? (uint64_t)((double)a2aTimeoutSecs * khz * 1000.0) : 0;
      init.abortTicks = (uint64_t)khz;
      ok = cudaMemcpy(a2aErrDev, &init, sizeof(init), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    (void)cudaGetLastError();
    std::vector<int> oks(nRanks, 0);
    oks[comm->rank] = ok ? 1 : 0;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, oks.data(), sizeof(int)), res, fail);
    for (int x : oks) ok = ok && x != 0;
    if (!ok) {
      INFO(NCCL_INIT, "ncclDdaFabricCommInit: device setup failed on some rank; skipping fabric DDA path");
      goto fail;
    }
  }

  // Success: hand ownership of every resource to comm.
  comm->ddaFabricMemHandler = handler;
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaScratchIsVmm = true;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaPeerPtrsHost = peerHost;
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = (int)epochLen;
  comm->ddaA2aErrDev = a2aErrDev;
  comm->ddaA2aErrHost = a2aErrHost;
  if (a2aTimeoutSecs > 0) {
    INFO(NCCL_INIT, "DDA AllToAll: device wait timeout %lld s (RCCL_DDA_A2A_TIMEOUT)", (long long)a2aTimeoutSecs);
  }
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: nRanks %d, scratch %zu bytes (vmm, gfx1250 fabric path; derived from RCCL DDA params; "
       "RCCL_DDA_FABRIC_BUFFER_SIZE=%lld), LL enabled=%lld threshold=%lld, "
       "LL128 enabled=%lld threshold=%lld, Simple threshold=%lld, FabricGpuBarrier nBlocks=%d, peer table on device",
       nRanks, bytes, (long long)fabricScratchOverride, (long long)llEnabled, (long long)llThresh,
       (long long)ll128Enabled, (long long)ll128Thresh, (long long)simpleThresh, nBlocksMax);
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: CU count=%d, local max blocks=%d, communicator max blocks=%d",
       comm->cuCount, localBlocksMax, nBlocksMax);
  // AllToAll push staging buffer and Direct flags: created collectively by the first AllToAll that
  // can use them (dda_alltoall_fabric_lazy_host.h), so comms that never run one allocate nothing.
  // If any rank's comm is nonblocking they are created now: its API calls must not wait for peers.
  comm->ddaA2aLazy = dda_a2a_lazy::kUntried;
  if (a2aEager) NCCLCHECK(ncclDdaA2aLazyInit(comm));
  return ncclSuccess;

fail:
  if (a2aErrHost != nullptr) {
    (void)ncclCudaHostFree(a2aErrHost);
  }
  if (a2aErrDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(a2aErrDev));
  }
  if (barrierState != nullptr) {
    delete barrierState;
  }
  if (epochDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(epochDev));
  }
  if (peerHost != nullptr) {
    free(peerHost);
  }
  if (peerDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(peerDev));
  }
  if (handler != nullptr) {
    delete handler;
  }
  if (scratch != nullptr) {
    (void)ncclCuMemFree(scratch, comm->memManager);
  }
  return ncclSuccess;
}

ncclResult_t ncclDdaFabricCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->ddaA2aErrHost != nullptr) {
    const uint32_t site = __atomic_load_n(comm->ddaA2aErrHost, __ATOMIC_ACQUIRE);
    if (site != 0 && site != dda::common::kDdaWdAborted) {
      WARN("DDA AllToAll on rank %d failed earlier (wait site %u: %s); that call's results were invalid and the "
           "communicator reported an error from then on", comm->rank, site, ncclDdaA2aWdSiteName(site));
    }
  }
  (void)ncclDdaA2aPushCommFini(comm);
  (void)ncclDdaA2aDirectCommFini(comm);
  CUDACHECKIGNORE(cudaFree(comm->ddaA2aErrDev));
  comm->ddaA2aErrDev = nullptr;
  (void)ncclCudaHostFree(comm->ddaA2aErrHost);
  comm->ddaA2aErrHost = nullptr;
  if (comm->ddaFabricBarrierState != nullptr) {
    delete static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
    comm->ddaFabricBarrierState = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  // Destroying the fabric handler unmaps/frees the imported peer scratch buffers.
  if (comm->ddaFabricMemHandler != nullptr) {
    delete static_cast<ncclFabricMemHandler*>(comm->ddaFabricMemHandler);
    comm->ddaFabricMemHandler = nullptr;
  }
  // Free this rank's scratch buffer with the allocator that produced it.
  if (comm->ddaScratch != nullptr) {
    if (comm->ddaScratchIsVmm) {
      (void)ncclCuMemFree(comm->ddaScratch, comm->memManager);
    } else {
      CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
    }
  }
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  comm->ddaScratchIsVmm = false;
  return ncclSuccess;
}
