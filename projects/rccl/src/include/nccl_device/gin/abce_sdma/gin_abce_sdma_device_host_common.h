/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ABCE_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ABCE_SDMA_DEVICE_HOST_COMMON_H_

#include <stddef.h>
#include <stdint.h>

// hip_ext_copy_engine's device detail header uses hipExtCopyEngine* types that
// live in hip/driver_types.h (pulled via hip_runtime.h), without including it
// itself. Include hip_runtime.h first so those types are declared regardless of
// include order in the aggregating TU (gin_device_api.h pulls this early).
#include <hip/hip_runtime.h>
#include <hip/hip_ext_copy_engine.h>

struct ncclGinAbceIpcBufEntry;

#define NCCL_GIN_ABCE_SDMA_NET_VERSION 115

/** Must match host plugin and device kernel build; checked on device. */
#define NCCL_GIN_ABCE_SDMA_LAYOUT_MAGIC 0xA6E17112u

/** Default SDMA threshold (bytes). Transfers of at most this size use inlined IPC flat stores;
 *  larger transfers use direct Abce SDMA. */
#define NCCL_GIN_ABCE_SDMA_THRESHOLD_DEFAULT 128u

/** Default off until SDMA signal VA bind is stable on MI355; opt-in via NCCL_GIN_ABCE_SDMA_FUSED_SIGNAL=1. */
#define NCCL_GIN_ABCE_SDMA_FUSED_SIGNAL_DEFAULT 0u

/** sdmaDirty is a uint64_t bitmask; one bit per (peer, channel) slot. */
#define NCCL_GIN_ABCE_SDMA_DIRTY_BITS 64

struct ncclGinAbceSdmaGPUContext {
  uint32_t layoutMagic;  // NCCL_GIN_ABCE_SDMA_LAYOUT_MAGIC
  // [local_pe * numChannels + ch] row of ABCE copy-engine handles, by value.
  // (anvil stored SdmaQueueDeviceHandle* pointers; ABCE handles are 16B values.)
  hipExtCopyEngineDeviceHandle* queueHandles;
  // Per-(peer*numChannels+ch) monotonic completion counters: each SDMA put
  // carries an Add-1 completion here, and flush/wait spins until the counter
  // reaches the number submitted. Replaces anvil's drainable quiet() queue.
  uint64_t* completion;      // device fine-grained; [nRanks*numChannels] words
  uint64_t* submitted;       // GIN-owned submit counts, same layout (device-written)
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
  const ncclGinAbceIpcBufEntry* ipcTable;  // device pointer; fallback peer VA lookup
  int ipcTableCount;
  uintptr_t* signal_remote_addrs;  // [nRanks] peer signal region bases (GDA signal_raddrs pattern)
  uint32_t ipcSignalPeer;          // 1=shader IPC atomic signalPeer, 0=SDMA ATOMIC packet (default)
};

struct ncclGinAbceSdmaMemHandle {
  uintptr_t baseAddr;     // Symmetric LSA flat VA for this rank
  uintptr_t* remote_vas;  // [nRanks] precomputed peer VAs (GDA remote_vas pattern)
  ptrdiff_t vmmStride;    // LSA VMM stride; O(1) peer VA when non-zero
};

#endif
