/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_

#include "../../hip_compat.h"

namespace nccl {
namespace gin {
namespace anvil {
namespace detail {

// LSA memory is VMM-mapped fine-grain (cache-coherent via Infinity Fabric).
// Plain stores are visible to all peers through the HW coherence domain.
// Ordering between data stores and signal atomics is provided by explicit
// fences in fenceBeforeSignal() / signalPeer(), not per-store ordering.

// Relaxed: ordering is provided by the caller's fence (signalPeer).
NCCL_DEVICE_INLINE void ipcFlatAtomicAddSys64(uint64_t* dst, uint64_t val) {
  __hip_atomic_fetch_add(reinterpret_cast<unsigned long long*>(dst), static_cast<unsigned long long>(val),
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace detail

NCCL_DEVICE_INLINE void ipcPutScalar(void* dst, const void* src, size_t bytes) {
  __builtin_memcpy(dst, src, bytes);
}

// IPC load/store path for transfers below the SDMA threshold.
NCCL_DEVICE_INLINE void ipcPut(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  __builtin_memcpy(dst, src, bytes);
}

}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
