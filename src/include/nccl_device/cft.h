/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT_H_
#define _NCCL_DEVICE_CFT_H_

#include "core.h"

#ifdef __CUDACC__

enum class ncclMemProxyType : uint32_t {
  Generic = 1u << 0,
  Fabric = 1u << 1,
};

enum class ncclMemFenceScope : uint32_t {
  Cta = 1u << 0,
  Sys = 1u << 1,
};

struct ncclCftSmem;

template <typename Coop>
struct ncclCft_internal;

template <typename Coop>
NCCL_DEVICE_INLINE void ncclMemFence(Coop coop, cuda::memory_order order, ncclMemProxyType producer,
                                     ncclMemProxyType consumer, ncclMemFenceScope scope);

template <typename Coop>
struct ncclCft : ncclCft_internal<Coop> {
  NCCL_DEVICE_INLINE ncclCft(Coop coop, ncclCftSmem& cftSmem);

  NCCL_DEVICE_INLINE ~ncclCft();

  NCCL_DEVICE_INLINE void submit(Coop coop);

  NCCL_DEVICE_INLINE void flushLocal(Coop coop);

  NCCL_DEVICE_INLINE void flush(Coop coop, bool* hasReport = nullptr, uint32_t* report = nullptr);

  NCCL_DEVICE_INLINE void put(Coop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes);

  NCCL_DEVICE_INLINE void putCpMask(Coop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes,
                                    uint16_t cpMask);

  NCCL_DEVICE_INLINE void putMultimem(Coop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes);

  NCCL_DEVICE_INLINE void putMultimemCpMask(Coop coop, ncclCftLeId leId, size_t leOffset, void* smemSource,
                                            uint32_t bytes, uint16_t cpMask);

  NCCL_DEVICE_INLINE void get(Coop coop, ncclCftLeId leId, size_t leOffset, void* smemDestination, uint32_t bytes);

  template <typename RedOp>
  NCCL_DEVICE_INLINE void red(Coop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource,
                              uint32_t bytes);

  template <typename RedOp>
  NCCL_DEVICE_INLINE void redCpMask(Coop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource,
                                    uint32_t bytes, uint16_t cpMask);

  template <typename RedOp>
  NCCL_DEVICE_INLINE void redMultimem(Coop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource,
                                      uint32_t bytes);

  template <typename RedOp>
  NCCL_DEVICE_INLINE void redMultimemCpMask(Coop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                            void* smemSource, uint32_t bytes, uint16_t cpMask);

  template <typename RedOp>
  NCCL_DEVICE_INLINE void pullRed(Coop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemDestination,
                                  uint32_t bytes);
};
#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT_H_
