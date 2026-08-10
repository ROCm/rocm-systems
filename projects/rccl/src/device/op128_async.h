/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OP128_ASYNC_H_
#define OP128_ASYNC_H_

#include "op128.h"

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)

#include "tdm/asyncCopy.h"

inline __device__ void asyncLoadGlobalToShmem(uint64_t* shmemDst, const uint64_t* globalSrc, int bytes) {
  asyncLoadToLDS<SyncPolicy::Async>(reinterpret_cast<const uint8_t*>(globalSrc),
                                    reinterpret_cast<uint8_t*>(shmemDst),
                                    static_cast<size_t>(bytes));
}

inline __device__ void asyncStoreShmemToGlobal(uint64_t* globalDst, const uint64_t* shmemSrc, int bytes) {
  asyncStoreFromLDS<SyncPolicy::Async>(reinterpret_cast<const uint8_t*>(shmemSrc),
                                       reinterpret_cast<uint8_t*>(globalDst),
                                       static_cast<size_t>(bytes));
}

inline __device__ void asyncWait() {
  asyncWait<0>();
}

#else

// Synchronous warp-collective fallback for targets without async-to-LDS builtins.
// ll128AlwaysShmem is only expected to be enabled on gfx1250.
inline __device__ void asyncLoadGlobalToShmem(uint64_t* shmemDst, const uint64_t* globalSrc, int bytes) {
  const unsigned lane = __lane_id();
  const unsigned warpSize = __builtin_amdgcn_wavefrontsize();
  const uint8_t* src = reinterpret_cast<const uint8_t*>(globalSrc);
  uint8_t* dst = reinterpret_cast<uint8_t*>(shmemDst);
  for (int i = lane; i < bytes; i += warpSize) {
    dst[i] = src[i];
  }
}

inline __device__ void asyncStoreShmemToGlobal(uint64_t* globalDst, const uint64_t* shmemSrc, int bytes) {
  const unsigned lane = __lane_id();
  const unsigned warpSize = __builtin_amdgcn_wavefrontsize();
  const uint8_t* src = reinterpret_cast<const uint8_t*>(shmemSrc);
  uint8_t* dst = reinterpret_cast<uint8_t*>(globalDst);
  for (int i = lane; i < bytes; i += warpSize) {
    dst[i] = src[i];
  }
}

inline __device__ void asyncWait() {}

#endif

#endif // OP128_ASYNC_H_
