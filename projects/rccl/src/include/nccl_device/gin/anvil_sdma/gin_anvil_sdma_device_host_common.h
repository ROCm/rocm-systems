/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_

#include <stdint.h>

struct ncclGinAnvilIpcBufEntry;

#define NCCL_GIN_ANVIL_SDMA_NET_VERSION 115

/** Must match host plugin and device kernel build; checked on device. */
#define NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC 0xA6E17111u

/** Default SDMA threshold (bytes). Transfers of at most this size use inlined IPC flat stores;
 *  larger transfers use direct Anvil SDMA. */
#define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 128u

/** Default off: fused OSS7 copy+signal needs remote GPU signal VA; opt-in via env on MI355. */
#define NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT 0u

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
  const ncclGinAnvilIpcBufEntry* ipcTable;  // device pointer; peer VA lookup for IPC puts
  int ipcTableCount;
};

struct ncclGinAnvilSdmaMemHandle {
  uintptr_t baseAddr;  // Symmetric LSA flat VA; peer resolved via ginAnvilResolvePeerVa
};

#endif
