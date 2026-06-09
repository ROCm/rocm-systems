/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ROCSHMEM_H_
#define _NCCL_DEVICE_GIN_ROCSHMEM_H_

#include "../gin_device_common.h"
#include "gin_rocshmem_device_host_common.h"
#include <rocshmem/rocshmem.hpp>

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;

    if (hasWins) {
      ncclGinRocshmemMemHandle* dstMh = (ncclGinRocshmemMemHandle*)dstWin;
      ncclGinRocshmemMemHandle* srcMh = (ncclGinRocshmemMemHandle*)srcWin;
      void* dst = (void*)(loadConst(&dstMh->baseAddr) + dstOff);
      void* src = (void*)(loadConst(&srcMh->baseAddr) + srcOff);

      rocshmem::rocshmem_putmem(dst, src, bytes, peer);
    }

    if (hasSignal || hasCounter) {
      if (hasCounter)
        rocshmem::rocshmem_quiet();  // counter only needs local completion (source consumed),
                                     // but shmem API only provides quiet (remote completion);
                                     // also sufficient to order data before signal
      else
        rocshmem::rocshmem_fence();  // lighter: just orders data before signal

      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        rocshmem::rocshmem_uint64_atomic_add(
          loadConst(&rsCtx->signals) + signal.indexedSignal.signalId, signalOpArg, peer);
      }
      if (hasCounter)
        atomicAdd((unsigned long long*)(loadConst(&rsCtx->counters) + counterId), 1ULL);
    }
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    ncclGinRocshmemMemHandle* dstMh = (ncclGinRocshmemMemHandle*)dstWin;
    T* dst = (T*)(loadConst(&dstMh->baseAddr) + dstOff);
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;

    // Use rocshmem_p (scalar put) — value is on stack, no symmetric src needed
    static_assert(sizeof(T) <= 8, "PutValue requires sizeof(T) <= 8");
    if constexpr (sizeof(T) == 8)
      rocshmem::rocshmem_longlong_p((long long*)dst, (long long)srcVal, peer);
    else if constexpr (sizeof(T) == 4)
      rocshmem::rocshmem_int_p((int*)dst, (int)srcVal, peer);
    else if constexpr (sizeof(T) == 2)
      rocshmem::rocshmem_short_p((short*)dst, (short)srcVal, peer);
    else if constexpr (sizeof(T) == 1)
      rocshmem::rocshmem_char_p((char*)dst, (char)srcVal, peer);

    if (hasSignal) {
      rocshmem::rocshmem_quiet();
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      rocshmem::rocshmem_uint64_atomic_add(
        loadConst(&rsCtx->signals) + signal.indexedSignal.signalId, signalOpArg, peer);
    }
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->counters) + counterId;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    nccl::utility::loadConst(&rsCtx->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->signals) + signalId;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    ncclGinRocshmemGPUContext* rsCtx = (ncclGinRocshmemGPUContext*)ctx.handle;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
      nccl::utility::loadConst(&rsCtx->signals)[signal.indexedSignal.signalId] = 0;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_API> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord) {
    rocshmem::rocshmem_quiet();
  }
};

#endif /* _NCCL_DEVICE_GIN_ROCSHMEM_H_ */
