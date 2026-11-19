/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_RUNTIME_H_
#define NCCL_DEVICE_RUNTIME_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"
#include "multicast.h"

////////////////////////////////////////////////////////////////////////////////
// ncclDevr[_]: runtime implements for symmetric API.

struct ncclDevrMemory;

// No public NCCL_WIN_REGISTER_* flag means all capabilities. Specifying one or more
// registration flags selects only those capabilities. Add future public flags here.
enum ncclDevrRegisterCapability {
  ncclDevrRegisterGin = 1 << 0,
  ncclDevrRegisterLsa = 1 << 1,
  ncclDevrRegisterCft = 1 << 2,
  ncclDevrRegisterRma = 1 << 3,
  ncclDevrRegisterAll = ncclDevrRegisterGin | ncclDevrRegisterLsa | ncclDevrRegisterCft | ncclDevrRegisterRma,
};

bool ncclDevrWinRegEnabled(int winFlags, enum ncclDevrRegisterCapability capability);

struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset; // Offset in big VA space.
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem; // key for intrusive map
  struct ncclDevrWindow* next; // next for intrusive map
  struct ncclComm* comm; // comm for intrusive map window <> comm look up
  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]; // IB MR handles per GIN connection (proxy-only path)
  ncclGinWindow_t rmaDevWins[NCCL_GIN_MAX_CONNECTIONS]; // device-side GIN window handles (proxy-only path)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // RCCL: intra-node IPC peer table (NULL when inactive), sized to lsaSize.
  void** ipcPeerPtrs;
  void** ipcPeerPtrsAllocBase;
  // A peer mapping is either a legacy IPC handle or a cuMem import, and the two unmap differently.
  uint8_t* ipcPeerIsCuMem;
  int ipcPeerCount;
#endif
};

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// All-gathered by windowRegisterNonSym. One record per LSA rank. Host tests
// publish through this type, so a field added here cannot be reinterpreted by
// a second copy in another translation unit.
struct ncclDevrNonSymExchangeEntry {
  cudaIpcMemHandle_t handle;
  uint64_t cuMemHandle;
  size_t allocSize;
  int isCuMem;
  uint64_t hostHash;
  uint64_t pidHash;
  size_t userOffset; // userPtr - allocBase
  size_t userSize;
};
#endif

// winSorted[] entry, ordered by userAddr for ncclDevrFindWindow.
struct ncclDevrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclDevrWindow* win;
};
struct ncclDevrTeam;

struct ncclDevrRegTask {
  struct ncclDevrRegTask* next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;
};

struct ncclDevrCommCreateTask {
  struct ncclDevrCommCreateTask* next;
  struct ncclDevCommRequirements* reqs;
  struct ncclDevComm* outDevComm;
  uint32_t deviceCodeVersion;
  bool isInternal;
};

struct ncclDevrStateCftUc {
  ncclCftLeId baseId;
};

struct ncclDevrState {
  // A complete set of LSA ranks (with pointer access via CUDA P2P), in ascending communicator-rank order.
  // This is the local node normally, the MNNVL clique when MNNVL is active, and the full NVLink fabric domain when
  // cross-clique P2P is enabled.
  int lsaDomainSelf;
  int lsaDomainSize;
  int* lsaDomainRankList;

  // Like lsaDomainSelf/lsaDomainSize except the team ranks must be consecutive in the world
  // and all the lsa teams within a communicator have the same number of ranks. If any condition is
  // false then the lsa team is just the singleton of self.
  int lsaTeamSelf;
  int lsaTeamSize;
  int* lsaTeamRankList; // Part of lsaDomainRankList -- watch out when free()ing!
  int nLsaTeams;

  int cftSelf;
  int cftSize;
  int cftMcSelf;
  int cftMcSize;
  struct ncclDevrStateCftUc le[2]; // 0: UC LE ID base, 1: Counted UC LE ID base (rank_i le = base + i)

  int* lsaPeerDelta4GDev;

  size_t granularity; // cuMemGetAllocationGranularity
  bool ginEnabled;
  bool rmaProxyEnabled;
  struct ncclDevrMemory* memHead;
  uint64_t nextRegistryId; // next value for ncclDevrMemory::registryId
  struct ncclDevrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  struct ncclDevrTeam* teamHead;
  size_t bigSize; // size of our big logical space (128GB?)
  // bigSize slice of the NVLS transport's MC group, for a multimem team over exactly the comm's
  // local ranks. Empty on a comm that reuses a parent's NVLS resources instead of building the group.
  struct ncclMcPartition nvlsMcPartition;
  struct ncclSpace bigSpace; // allocates our big VA space.
  void* lsaFlatBase; // base ptr for all LSA-domain ranks' big VAs: size = lsaDomainSize*bigSize
  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
  struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};

struct ncclGinBackendState;

struct ncclDevCommCompat {
  int minVersion, maxVersion;
  ncclResult_t (*commPropertiesFilter)(ncclComm_t comm, struct ncclCommProperties* props);
  ncclResult_t (*devCommRequirementsFilter)(ncclComm_t comm, ncclDevCommRequirements_t* reqs);
  ncclResult_t (*devCommCopyNewToOld)(ncclComm_t comm, void* oldDevComm, struct ncclDevComm const* newDevComm);
  ncclResult_t (*devCommCopyOldToNew)(ncclComm_t comm, struct ncclDevComm* newDevComm, void const* oldDevComm);
  ncclResult_t (*ginSignalsPerBarrier)(ncclComm_t comm, struct ncclGinBackendState const* backend, int nRanks,
                                       int* outSlots);
};

// ginSignalsPerBarrier implementations. Device code before 2.32.4 strides by the team size; from 2.32.4 on it
// strides by ncclGinBarrierSlots() of the selected backend's barrier preference.
ncclResult_t ncclDevCommGinSignalsPerBarrier_v22902(ncclComm_t comm, struct ncclGinBackendState const* backend,
                                                    int nRanks, int* outSlots);
ncclResult_t ncclDevCommGinSignalsPerBarrier_v23204(ncclComm_t comm, struct ncclGinBackendState const* backend,
                                                    int nRanks, int* outSlots);

// One of NCCL's own GIN barrier requirements; ncclGinDevCommSetup re-sizes it for each backend it tries.
struct ncclGinBarrierReq {
  struct ncclDevResourceRequirements* req;
  struct ncclTeam team;
  int nBarriers;
};
ncclResult_t ncclGinBarrierSizeRequirement(struct ncclComm* comm, struct ncclGinBackendState const* backend,
                                           struct ncclDevCommCompat const* devCompat,
                                           struct ncclGinBarrierReq const* barrierReq);

// Check if GIN resources have been requested as part of `reqs`.
bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs);

int computeLsaTeamSize(struct ncclComm* comm);
// Compute the symmetric VA size (bigSize) from the comm's peer info without any side effects.
size_t computeBigSize(struct ncclComm* comm);

// Check if there is only one LSA team. This function uses the cached value of comm or computes the
// value from the comm topology.
bool ncclDevrIsOneLsaTeam(struct ncclComm* comm);

// Returns the CUDA version supported by CFT on this GPU, or 0 when CFT is unsupported.
ncclResult_t ncclGpuCftSupport(struct ncclComm* comm, int* gpuCftSupport, bool* gpuCftMulticastSupport,
                               bool* gpuCftCountedSupport);
ncclResult_t ncclGpuGetCliqueIds(CUdevice dev, uint32_t* unicastId, uint32_t* multicastId);

// We assume ncclComm has a `ncclDevrState symState` member.
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm);
ncclResult_t ncclDevrFinalize(struct ncclComm* comm);

// If found *outWinHost will be populated and *outWinId >= 0, otherwise *outWinId == -1
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* ptr, size_t size, int winFlags,
                                           ncclWindow_t* outWinDev);

ncclResult_t ncclDevrCommCreateInternal(struct ncclComm* comm, struct ncclDevCommRequirements* reqs,
                                        struct ncclDevComm* outDevComm, bool isInternal, uint32_t deviceCodeVersion);
ncclResult_t ncclDevrCommCreateAsync(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                     struct ncclDevComm* outDevComm, bool isInternal, uint32_t deviceCodeVersion);
void freeDevCommRequirements(struct ncclDevCommRequirements* reqs);

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win);
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win);
// RCCL: smallest registered size across this window's LSA team. Falls back to the
// local size for host-only test windows that do not carry runtime memory state.
size_t ncclDevrWindowLsaMinSize(struct ncclDevrWindow const* win);
// RCCL: bytes from userPtr that fit in this window and in every LSA peer allocation.
// lsaMinSize starts at the allocation base, so a window at memOffset only owns
// the tail of that minimum.
size_t ncclDevrWindowPeerSafeSize(struct ncclDevrWindow const* win);

// Get the corresponding pointer in another lsa rank's symmetric memory window
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank,
                                   void** outPtr);

// Convert a world rank to an LSA rank.
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm* comm, int peerWorldRank, int* peerLsaRank);

// Get the host RMA-proxy MR handle for a physical RMA connection.
void* ncclDevrGetRmaWin(struct ncclDevrWindow* winHost, int ctx);

// Get the byte offset of a window within its backing memory allocation.
size_t ncclDevrGetWinOffset(struct ncclDevrWindow* winHost);

// Get the multicast address for a given team
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset,
                                     struct ncclTeam lsaTeam, void** outPtr);

// Copies the devComm data from "rank" to "lsaBarrier".  Assumes the same memory layout at source and destination.
void ncclDevCommCopyLsaData(void* dstRankPtr, void const* srcRankPtr);

// RCCL: Get the LSA flat VA for self rank corresponding to a primary (ncclMemAlloc) address.
// If addr is already in the LSA flat range, returns addr unchanged.
// If addr matches a registered memory's primaryAddr, returns lsaFlatBase + lsaSelf*bigSize + bigOffset.
// outAddr is set to nullptr if addr cannot be resolved.
ncclResult_t ncclDevrGetLsaSelfAddr(struct ncclDevrState* devr, void* addr, void** outAddr);

// RCCL: LSA flat window base + stride for GIN Anvil symmetric registration (ncclGetLsaPointer layout).
ncclResult_t ncclDevrGetGinAnvilMemLayout(struct ncclDevrState* devr, void* addr, uintptr_t* outLsaFlatBase,
                                          uint32_t* outStride4G);

#endif
