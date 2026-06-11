/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Temporary glue header extracted during NCCL->RCCL merge.
// Provides the minimal subset of os.h needed by gin_v11.cc / gin_v12.cc
// until os.h is fully ported or replaced.

#ifndef NCCL_MERGE_STUBS_H_
#define NCCL_MERGE_STUBS_H_

#define NCCL_DESTROY NCCL_INIT
#include <dlfcn.h>
#include <unistd.h>

typedef void* ncclOsLibraryHandle;
static inline void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  return dlsym(handle, symbol);
}



struct ncclDevrWindow;
static inline bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* /*win*/) { return false; }
static inline bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* /*win*/) { return false; }


#include <atomic>
#define NCCL_CONVERT_ORDER(order) \
  ((order) == std::memory_order_relaxed ? __ATOMIC_RELAXED : \
   (order) == std::memory_order_consume ? __ATOMIC_CONSUME : \
   (order) == std::memory_order_acquire ? __ATOMIC_ACQUIRE : \
   (order) == std::memory_order_release ? __ATOMIC_RELEASE : \
   (order) == std::memory_order_acq_rel ? __ATOMIC_ACQ_REL : \
   (order) == std::memory_order_seq_cst ? __ATOMIC_SEQ_CST : \
   __ATOMIC_SEQ_CST)

#define COMPILER_ATOMIC_LOAD(ptr, order) \
  __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  __atomic_load((ptr), (dest), NCCL_CONVERT_ORDER(order)); \
} while(0)
#define COMPILER_ATOMIC_STORE(ptr, val, order) \
  __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_32(ptr, order) \
  __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_STORE_32(ptr, val, order) \
  __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) \
  __atomic_exchange_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  __atomic_compare_exchange_n((ptr), (expected), (desired), true, NCCL_CONVERT_ORDER(success_order), NCCL_CONVERT_ORDER(failure_order))
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) __atomic_add_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) __atomic_sub_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))

#define COMPILER_PREFETCH(addr) __builtin_prefetch((addr))
#define COMPILER_POPCOUNT32(x) __builtin_popcount(x)
#define COMPILER_POPCOUNT64(x) __builtin_popcountll(x)
#define COMPILER_EXPECT(x, v) __builtin_expect((x), (v))
#define COMPILER_FFS(x) __builtin_ffs(x)
#define COMPILER_FFSL(x) __builtin_ffsl(x)
#define COMPILER_FFSLL(x) __builtin_ffsll(x)
#define COMPILER_CLZ(x) __builtin_clz(x)
#define COMPILER_CLZL(x) __builtin_clzl(x)
#define COMPILER_CLZLL(x) __builtin_clzll(x)
#define COMPILER_BSWAP16(x) __builtin_bswap16(x)
#define COMPILER_BSWAP32(x) __builtin_bswap32(x)
#define COMPILER_BSWAP64(x) __builtin_bswap64(x)
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned((ptr), (alignment))
#define COMPILER_ATTRIBUTE_UNUSED __attribute__((unused))

#endif
