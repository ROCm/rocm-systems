/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/ipc/ipc_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/ipc/ipc_mem_handler.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// Relax the DDA IPC AllReduce eligibility beyond exactly kDdaNranks (8) ranks.
// When 0 (default) the classic 8-rank-only gate is enforced: collective selection,
// the 8-rank kernels' arithmetic and therefore results are unchanged from baseline.
// When 1, any single-node comm of 2..kDdaNranks
// ranks is eligible for the IPC path. Defined here alongside the DDA IPC comm-init
// gate so the host init unit tests link it without pulling in the all-reduce compute TU.
RCCL_PARAM(DdaNranksRelax, "DDA_NRANKS_RELAX", 0);

bool ncclDdaNranksRelaxEnabled() {
  return rcclParamDdaNranksRelax() != 0;
}

// Single source of truth for the supported DDA IPC participant counts. Both the
// comm-init gate below and the per-collective eligibility gates call this, so the
// two cannot drift apart and allocate IPC resources that eligibility then refuses.
bool ncclDdaIpcNranksSupported(int nRanks) {
  if (nRanks == kDdaNranks) {
    return true;
  }
  if (!ncclDdaNranksRelaxEnabled()) {
    return false;
  }
  return nRanks >= 2 && nRanks <= kDdaNranks;
}

bool ncclDdaNranksRelaxConsensusMatters(int nRanks) {
  return nRanks >= 2 && nRanks < kDdaNranks;
}

#define HIP_CALL(cmd) \
  do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " << __LINE__ << " in file " \
                << __FILE__ << "\n"; \
    } \
  } while (0)

namespace {

// The scratch comes from hipExtMallocWithFlags, so allocTracker is updated here, not by an allocator helper.
void freeDdaIpcScratch(ncclComm* comm, void* scratch, size_t bytes) {
  if (scratch == nullptr || comm == nullptr) return;
  ncclAllocUntrack(comm->cudaDev, bytes, "ncclDdaIpcCommInit");
  CUDACHECKIGNORE(cudaFree(scratch));
}

// Whether this communicator takes part in DDA IPC at all. The rank-count rule is
// ncclDdaIpcNranksSupported(): kDdaNranks by default, any 2..kDdaNranks when
// RCCL_DDA_NRANKS_RELAX=1.
bool ddaIpcCommEligible(const ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  const bool ddaArchSupported =
    comm->archName != nullptr && (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"));
  return ncclDdaIpcNranksSupported(comm->nRanks) && comm->nNodes == 1 && comm->bootstrap != nullptr &&
         !comm->directMode && !comm->MNNVL && ddaArchSupported;
}

// Setting DDA IPC up interleaves rank-local work (the P2P sweep, the scratch and
// peer-table allocations) with bootstrap collectives (ncclIpcMemHandler::
// exchangeMemPtrs, and the barrier's own internal exchange inside
// IpcGpuBarrier::mallocAndInit). Those two kinds of step do not mix safely: a
// rank that hit a local failure and simply returned would leave every peer that
// succeeded blocked in the next collective, waiting for a participant that is
// never coming. That is a hang, not the silent fall-back to the ring path the
// early returns look like they provide.
//
// So each rank carries a running "how much of this can I still do" level and, at
// every point where the next step is collective, all ranks exchange that level
// and act on it together: everyone proceeds at the lowest level any rank
// reported, or (level 0) everyone tears down and falls back. One int per rank on
// the bootstrap channel, only on the DDA IPC init path. Only the first
// comm->nRanks slots are exchanged and reduced; the clique can be smaller than
// kDdaNranks under RCCL_DDA_NRANKS_RELAX.
//
// A failed all-gather is returned as an error rather than folded into level 0:
// it is not an agreed verdict -- peers may have seen the exchange succeed -- so
// communicator init must fail instead of claiming a clean fall-back to the ring
// path.
ncclResult_t ddaIpcAgreeLevel(ncclComm* comm, int localLevel, int* agreedLevel, int* lowestRank) {
  *agreedLevel = 0;
  *lowestRank = comm->rank;
  std::vector<int> levels(static_cast<size_t>(comm->nRanks), 0);
  levels[static_cast<size_t>(comm->rank)] = localLevel;
  ncclResult_t res = bootstrapAllGather(comm->bootstrap, levels.data(), sizeof(int));
  if (res != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: readiness all-gather failed (%d)", res);
    return res;
  }
  const auto lowest = std::min_element(levels.begin(), levels.end());
  *agreedLevel = *lowest;
  *lowestRank = static_cast<int>(lowest - levels.begin());
  return ncclSuccess;
}

// Binary form: sets *allReady only when every rank reported ready.
ncclResult_t ddaIpcAllRanksReady(ncclComm* comm, bool localReady, bool* allReady) {
  int level = 0;
  int lowestRank = 0;
  *allReady = false;
  NCCLCHECK(ddaIpcAgreeLevel(comm, localReady ? 1 : 0, &level, &lowestRank));
  if (level == 0) {
    INFO(NCCL_INIT,
         "ncclDdaIpcCommInit: rank %d could not set up DDA IPC; disabling it on all %d ranks so none of them "
         "diverge into the DDA path",
         lowestRank, comm->nRanks);
    return ncclSuccess;
  }
  *allReady = true;
  return ncclSuccess;
}

} // namespace

ncclResult_t ncclDdaIpcCommInit(ncclComm* comm) {
  // Skip DDA if:
  // - nRanks is not a supported DDA IPC participant count (kDdaNranks by default;
  //   any 2..kDdaNranks when RCCL_DDA_NRANKS_RELAX=1)
  // - multi-node runs
  // - not using 1 process per GPU
  // - MNNVL (fabric-based P2P)
  // - the arch is not one the DDA algorithm actually runs on. The dispatch path
  //   (rcclDdaEnabled() in rccl_wrap.cc) only enables DDA on gfx942/gfx950; on
  //   every other arch the algorithm is never selected, so allocating the IPC
  //   scratch/barrier here is not necessary. On gfx12xx (RDNA4) the
  //   uncached-memory IPC export fails (hipIpcGetMemHandle -> hipErrorInvalidValue),
  //   which aborts comm init entirely. Gate init to match dispatch.
  if (!ddaIpcCommEligible(comm)) {
    return ncclSuccess;
  }

  // DDA IPC requires cross-GPU IPC memory mapping (hipIpcOpenMemHandle).
  // comm->isAllCudaP2p is set via ncclTopoCheckP2p which on AMD/HIP returns true
  // whenever ranks share a hostHash, regardless of actual P2P support (paths.cc).
  // Use hipDeviceCanAccessPeer directly — the authoritative runtime check for IPC
  // capability, and the same check used in init.cc for hasPeerAccess.
  //
  // Everything from here to the readiness check below is rank-local and can fail
  // on one rank while succeeding on its peers, so nothing in it returns early:
  // each step only narrows `localReady`, and the ranks agree on the outcome
  // before the first collective. See ddaIpcAllRanksReady() above.
  bool localReady = true;
  for (int i = 0; i < comm->nRanks && localReady; i++) {
    for (int j = i + 1; j < comm->nRanks; j++) {
      int canAccess = 0;
      hipError_t err = hipDeviceCanAccessPeer(&canAccess, comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
      if (err != hipSuccess || !canAccess) {
        INFO(NCCL_INIT, "ncclDdaIpcCommInit: no P2P between GPU %d and GPU %d, skipping DDA IPC",
             comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
        localReady = false;
        break;
      }
    }
  }

  // DDA_IPC_BUFFER_SIZE is a compile-time constant, so this is uniform across
  // ranks and returning directly cannot desynchronise them.
  size_t bytes = DDA_IPC_BUFFER_SIZE;
  if (bytes == 0) {
    return ncclSuccess;
  }

  void* scratch = nullptr;
  if (localReady) {
#if defined(HIP_UNCACHED_MEMORY)
    HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocUncached));
#else
    HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocFinegrained));
#endif
    // HIP_CALL only logs, so a failed allocation shows up as a null pointer.
    if (scratch == nullptr) {
      WARN("ncclDdaIpcCommInit: could not allocate the %zu-byte DDA IPC scratch buffer", bytes);
      localReady = false;
    } else {
      ncclAllocTrack(comm->cudaDev, bytes, "ncclDdaIpcCommInit");
      // Zero the scratch once so the LL all-gather's first epoch (>= 1) never
      // false-matches leftover flag words (mirrors the fabric path). Harmless for
      // the copy-based DDA collectives, which overwrite their staging area per op.
      HIP_CALL(hipMemset(scratch, 0, bytes));
    }
  }

  ncclIpcMemHandler* handler = nullptr;
  if (localReady) {
    handler = new (std::nothrow) ncclIpcMemHandler(comm->bootstrap, comm->rank, comm->nRanks);
    if (handler == nullptr) {
      WARN("ncclDdaIpcCommInit: OOM allocating ncclIpcMemHandler");
      localReady = false;
    }
  }

  if (localReady && handler->addSelfDeviceMemPtr(scratch) != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: addSelfDeviceMemPtr failed");
    localReady = false;
  }

  // First collective. Past this point every rank must be in lockstep.
  bool allReady = false;
  ncclResult_t readyRes = ddaIpcAllRanksReady(comm, localReady, &allReady);
  if (readyRes != ncclSuccess || !allReady) {
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    return readyRes;
  }

  // Its own all-gather has completed for everyone by the time this returns, but
  // the cudaIpcOpenMemHandle() loop after that all-gather is rank-local and can
  // still fail on a single rank. Record it rather than returning: the second
  // readiness check below is what every other rank is heading for, and a rank
  // that slipped out here would strand them in it.
  if (handler->exchangeMemPtrs() != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: exchangeMemPtrs failed");
    localReady = false;
  }

  // Peer table is sized for kDdaNranks (the max) but only comm->nRanks entries
  // are populated/copied when RCCL_DDA_NRANKS_RELAX shrinks the participant set.
  //
  // Everything down to the second readiness check is rank-local again, and the
  // step after it -- IpcGpuBarrier::mallocAndInit() -- runs its own bootstrap
  // exchange. Same rule as before the first collective: record failures, do not
  // return, and let all ranks decide together.
  const int nActiveRanks = comm->nRanks;
  void* peerDev = nullptr;
  DdaIpcBarrierState* barrierState = nullptr;

  cudaError_t ce = cudaMalloc(&peerDev, kDdaNranks * sizeof(void*));
  if (ce != cudaSuccess) {
    WARN("ncclDdaIpcCommInit: cudaMalloc(peer table) failed (%s)", cudaGetErrorString(ce));
    localReady = false;
  }

  // Zero the full peer table so any slot past the live prefix (nActiveRanks) reads
  // as null rather than uninitialized device memory when RCCL_DDA_NRANKS_RELAX
  // shrinks the participant set below kDdaNranks.
  if (localReady) {
    cudaError_t mce = cudaMemset(peerDev, 0, kDdaNranks * sizeof(void*));
    if (mce != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMemset(peer table) failed (%s)", cudaGetErrorString(mce));
      localReady = false;
    }
  }

  void* h_ptrs[kDdaNranks] = {};
  if (localReady) {
    for (int i = 0; i < nActiveRanks; ++i) {
      void* p = nullptr;
      if (handler->getPeerDeviceMemPtr(i, &p) != ncclSuccess) {
        WARN("ncclDdaIpcCommInit: getPeerDeviceMemPtr failed");
        localReady = false;
        break;
      }
      h_ptrs[i] = p;
    }
  }

  if (localReady) {
    ce = cudaMemcpy(peerDev, h_ptrs, nActiveRanks * sizeof(void*), cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMemcpy(peer table) failed (%s)", cudaGetErrorString(ce));
      localReady = false;
    }
  }

  if (localReady && ncclCalloc(&comm->ddaPeerPtrsHost, kDdaNranks) != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: OOM allocating host peer table");
    localReady = false;
  }

  // Only nActiveRanks entries of h_ptrs are populated; the calloc'd tail stays
  // null for <8-rank comms (the CE consumer reads comm->nRanks peer bases).
  if (localReady) {
    cudaError_t ddaCe = cudaMemcpy(comm->ddaPeerPtrsHost, h_ptrs, nActiveRanks * sizeof(void*), cudaMemcpyHostToHost);
    if (ddaCe != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMemcpy(host peer table) failed (%s)", cudaGetErrorString(ddaCe));
      localReady = false;
    }
  }

  // Allocated before the barrier rather than after it, so that this failure is
  // covered by the readiness check below too; it is only populated once
  // mallocAndInit() has returned.
  if (localReady) {
    barrierState = new (std::nothrow) DdaIpcBarrierState();
    if (barrierState == nullptr) {
      WARN("ncclDdaIpcCommInit: OOM allocating DdaIpcBarrierState");
      localReady = false;
    }
  }

  // Second collective (IpcGpuBarrier::mallocAndInit exchanges barrier pointers
  // over the bootstrap channel). Agree before entering it.
  readyRes = ddaIpcAllRanksReady(comm, localReady, &allReady);
  if (readyRes != ncclSuccess || !allReady) {
    delete barrierState;
    if (comm->ddaPeerPtrsHost != nullptr) {
      free(comm->ddaPeerPtrsHost);
      comm->ddaPeerPtrsHost = nullptr;
    }
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    return readyRes;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto barrierPair = dda::common::IpcGpuBarrier::mallocAndInit(nActiveRanks, nBlocksMax, comm->rank, comm->bootstrap,
                                                               dda::common::DdaKernelParams{comm->abortFlagDev});
  if (!barrierPair.first) {
    WARN("ncclDdaIpcCommInit: IpcGpuBarrier::mallocAndInit failed");
    localReady = false;
  }

  // LL epoch cells (DDA broadcast LL); optional, a failure only disables that tier.
  // Sized for the CU-derived cap so the Broadcast grid is not limited by the cells.
  // Rank-local, so a failure is folded into the agreed level below rather than
  // acted on here.
  const size_t epochLen = nccl_dda_detail::ddaLLEpochCount(
    kDdaNranks, nccl_dda_detail::ddaFabricMaxNBlocksForScratch(comm->cuCount, nullptr));
  uint32_t* epochDev = nullptr;
  if (localReady && (cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)) != cudaSuccess ||
                     cudaMemset(epochDev, 0, epochLen * sizeof(uint32_t)) != cudaSuccess)) {
    (void)cudaGetLastError(); // clear the sticky error
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
  }

  // Nothing collective follows, so a rank bailing out here would not hang comm
  // init -- but it would leave this rank with DDA IPC off while its peers have
  // it on, and they would then diverge at the first collective instead. The
  // enable/disable decision has to be unanimous, so agree one last time. The
  // outcome must be rank-uniform, else ranks pick different backends and hang.
  // Level 0: no DDA IPC, 1: no LL epoch cells, 2: all.
  //
  // A failure inside mallocAndInit cannot strand peers either: every rank
  // reaches its internal handle exchange, which reports a missing or
  // unexportable mailbox on any rank to all of them (ncclIpcMemHandler::
  // exchangeMemPtrs).
  int level = 0;
  int lowestRank = 0;
  readyRes = ddaIpcAgreeLevel(comm, localReady ? (epochDev == nullptr ? 1 : 2) : 0, &level, &lowestRank);
  if (readyRes != ncclSuccess || level == 0) {
    CUDACHECKIGNORE(cudaFree(epochDev));
    barrierPair.first.reset();
    delete barrierState;
    if (comm->ddaPeerPtrsHost != nullptr) {
      free(comm->ddaPeerPtrsHost);
      comm->ddaPeerPtrsHost = nullptr;
    }
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    if (readyRes == ncclSuccess) {
      ATTN("DDA IPC setup failed on rank %d; skipping the DDA IPC path on all %d ranks", lowestRank, comm->nRanks);
    }
    return readyRes;
  }
  if (level < 2) {
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
    ATTN("LL epoch cells unavailable on rank %d; DDA Broadcast disabled", lowestRank);
  }

  barrierState->resources = std::move(barrierPair.first);
  barrierState->barrierHost = barrierPair.second;

  comm->ddaIpcMemHandler = handler;
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaIpcBarrierState = barrierState;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = epochDev != nullptr ? (int)epochLen : 0;
  INFO(NCCL_INIT, "ncclDdaIpcCommInit: scratch %zu bytes, IpcGpuBarrier nBlocks=%d, peer IPC table on device", bytes,
       nBlocksMax);
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommClosePeerMappings(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  auto* barrierState = comm->ddaIpcBarrierState;
  const bool hasBarrierMappings = barrierState != nullptr && barrierState->resources != nullptr &&
                                  barrierState->resources->ipcMemHandler != nullptr;
  if ((hasBarrierMappings || comm->ddaIpcMemHandler != nullptr) && comm->doneEvent != nullptr) {
    // DDA kernels and CE-scratch copies run on the user stream, which the
    // hostStream/deviceStream syncs in commDestroySync do not cover. doneEvent
    // is recorded after every kernel and CE launch, so waiting on it keeps
    // in-flight work off unmapped peers.
    cudaError_t syncErr = cudaEventSynchronize(comm->doneEvent);
    if (syncErr != cudaSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: doneEvent sync failed (%s); continuing cleanup",
           cudaGetErrorString(syncErr));
    }
  }
  if (hasBarrierMappings) {
    ncclResult_t closeResult = barrierState->resources->ipcMemHandler->closePeerMappings();
    if (closeResult != ncclSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: barrier peer mapping close failed (%d); continuing cleanup",
           closeResult);
    }
  }
  if (comm->ddaIpcMemHandler != nullptr) {
    ncclResult_t closeResult = comm->ddaIpcMemHandler->closePeerMappings();
    if (closeResult != ncclSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: scratch peer mapping close failed (%d); continuing cleanup",
           closeResult);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  // commFree must remain non-collective. Destroy and abort close mappings in
  // commDestroySync and revoke closes them in commReclaim, before the serial
  // commFree loop; this idempotent call is the local backstop for init-failure
  // and other direct commFree paths.
  ncclResult_t closeResult = ncclSuccess;
  NCCLCHECKIGNORE(ncclDdaIpcCommClosePeerMappings(comm), closeResult);

  if (comm->ddaIpcBarrierState != nullptr) {
    delete comm->ddaIpcBarrierState;
    comm->ddaIpcBarrierState = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  if (comm->ddaIpcMemHandler != nullptr) {
    delete comm->ddaIpcMemHandler;
    comm->ddaIpcMemHandler = nullptr;
  }
  freeDdaIpcScratch(comm, comm->ddaScratch, comm->ddaScratchBytes);
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  return ncclSuccess;
}
