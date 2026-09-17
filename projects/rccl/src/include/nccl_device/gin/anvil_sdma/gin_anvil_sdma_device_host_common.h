/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_

#include <stddef.h>
#include <stdint.h>

struct ncclGinAnvilIpcBufEntry;

#define NCCL_GIN_ANVIL_SDMA_NET_VERSION 117

/** Must match host plugin and device kernel build; checked on device. */
#define NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC 0xA6E17114u

/** Default SDMA threshold (bytes). Transfers of at most this size use inlined IPC flat stores;
 *  larger transfers use direct Anvil SDMA. */
#define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 128u

/** Default off until SDMA signal VA bind is stable on MI355; opt-in via NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=1. */
#define NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT 0u

/** sdmaDirty is a uint64_t bitmask; one bit per (peer, channel) slot. */
#define NCCL_GIN_ANVIL_SDMA_DIRTY_BITS 64

struct ncclGinAnvilSdmaGPUContext {
  uint32_t layoutMagic;  // NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC
  void** queueHandles;   // [local_pe * numChannels + ch] SdmaQueueDeviceHandle*
  uint64_t* sdmaDirty;   // GIN-owned dirty bitmask
  uint64_t* signals;
  uint64_t* counters;
  uint32_t nSignals;
  uint32_t nCounters;
  uint32_t sdmaThreshold;
  uint32_t fusedSdmaSignal;  // use COPY_LINEAR_WAIT_SIGNAL_MI4 for SignalInc SDMA puts
  int nRanks;
  int rank;
  int numChannels;
  int sdmaChannel;
  int sdmaChannelStride;
  const ncclGinAnvilIpcBufEntry* ipcTable;  // device pointer; fallback peer VA lookup
  int ipcTableCount;
  uintptr_t* signal_remote_addrs;  // [nRanks] peer signal region bases (GDA signal_raddrs pattern)
  uint32_t ipcAgentFence;          // 0=__threadfence_system on IPC (default), 1=agent-scope release
  uint32_t ipcSignalPeer;          // 1=shader IPC atomic signalPeer, 0=SDMA ATOMIC packet (default)

  // Private device-API DDA lane. Unlike ncclDevComm, this backend-owned
  // context can evolve without changing the public device-communicator ABI.
  void** fabricA2APeerScratch;
  // GIN-owned epoch cells for fabric LL A2A. Not comm->ddaLLEpochDev (host DDA).
  uint32_t* fabricA2ALlEpoch;
  size_t fabricA2AScratchBytes;
  size_t fabricA2ALlThreshold;
  int fabricA2ALlEpochLen;
  uint32_t fabricA2AEnabled;

  // Opt-in direct LSA-flat-from-fabric AllToAll (peer recv windows via fabric LSA map).
  uint32_t fabricA2ALsaEnabled;
  size_t fabricA2ALsaThreshold;

  // Device mutex for one in-flight fabric-LL AllToAll on this GIN context.
  // 0 = free; otherwise the owning kernel's launchId. Inflight counts LL CTAs.
  // A second overlapping launch must fail, not fall back to gin.put.
  uint32_t fabricA2ALlBusy;
  uint32_t fabricA2ALlInflight;
};

/** Current size of the context above, in bytes, on the LP64 targets RCCL builds for.
 *
 *  Every layout assertion in the tree compares NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC
 *  against itself, so a field added here without a magic bump would be invisible
 *  to both the device-side check and the host-side ones: an alltoall_perf built
 *  on the new header would read the new fields out of a context that an older
 *  librccl allocated and filled to the old size. Pinning the size makes that a
 *  compile error instead. When you change the layout, bump the magic and update
 *  this constant in the same commit.
 */
#define NCCL_GIN_ANVIL_SDMA_GPU_CONTEXT_BYTES 176

#if defined(__cplusplus)
static_assert(sizeof(struct ncclGinAnvilSdmaGPUContext) == NCCL_GIN_ANVIL_SDMA_GPU_CONTEXT_BYTES,
              "ncclGinAnvilSdmaGPUContext layout changed: bump NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC and "
              "NCCL_GIN_ANVIL_SDMA_GPU_CONTEXT_BYTES together");
static_assert(offsetof(struct ncclGinAnvilSdmaGPUContext, layoutMagic) == 0,
              "layoutMagic must stay first so a consumer can version-check before reading the rest");
#endif

struct ncclGinAnvilSdmaMemHandle {
  uintptr_t baseAddr;     // Symmetric LSA flat VA for this rank
  uintptr_t* remote_vas;  // [nRanks] precomputed peer VAs (GDA remote_vas pattern)
  ptrdiff_t vmmStride;    // LSA VMM stride; O(1) peer VA when non-zero
};

#endif
