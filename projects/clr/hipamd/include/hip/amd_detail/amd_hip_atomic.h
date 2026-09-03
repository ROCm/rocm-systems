/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "amd_device_functions.h"
#endif

#if !defined(__HIP_ATOMIC_BACKWARD_COMPAT)
#define __HIP_ATOMIC_BACKWARD_COMPAT 1
#endif

#if defined(__has_extension) && __has_extension(clang_atomic_attributes) && __HIP_ATOMIC_BACKWARD_COMPAT
#define __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY [[clang::atomic(fine_grained_memory, remote_memory)]]
#else
#define __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY
#endif

// Annoyingly, unrecognized attributes are a warning and the scoped
// atomic builtins predate clang_atomic_attributes.
#if defined(__has_extension) && __has_extension(clang_atomic_attributes)
#define __HIP_ATOMIC_ATTR(...) [[clang::atomic(__VA_ARGS__)]]
#else
#define __HIP_ATOMIC_ATTR(...)
#endif

#if !__HIP_DEVICE_COMPILE__
// TODO: Remove this after compiler pre-defines the following Macros.
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_unsafe_atomics.h"
#endif

__device__ inline unsigned short int atomicCAS(unsigned short int* address,
                                               unsigned short int compare, unsigned short int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned short int atomicCAS_system(unsigned short int* address,
                                                      unsigned short int compare,
                                                      unsigned short int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline int atomicCAS(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline int atomicCAS_system(int* address, int compare, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned int atomicCAS(unsigned int* address, unsigned int compare,
                                         unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned int atomicCAS_system(unsigned int* address, unsigned int compare,
                                                unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned long atomicCAS(unsigned long* address, unsigned long compare,
                                          unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned long atomicCAS_system(unsigned long* address, unsigned long compare,
                                                 unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned long long atomicCAS(unsigned long long* address,
                                               unsigned long long compare, unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned long long atomicCAS_system(unsigned long long* address,
                                                      unsigned long long compare,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline float atomicCAS(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline float atomicCAS_system(float* address, float compare, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline double atomicCAS(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline double atomicCAS_system(double* address, double compare, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline int atomicAdd(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicAdd_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicAdd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicAdd_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicAdd_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicAdd(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicAdd_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicAdd(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicAdd_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

#if !defined(__HIPCC_RTC__)
HIP_DEPRECATED("use atomicAdd instead")
#endif  // !defined(__HIPCC_RTC__)
__device__ inline void atomicAddNoRet(float* address, float val) { unsafeAtomicAdd(address, val); }

__device__ inline double atomicAdd(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicAdd_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicSub(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicSub_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicSub(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicSub_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicSub(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicSub_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicSub(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicSub_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicSub(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicSub_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline double atomicSub(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicSub_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicExch(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicExch_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicExch(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicExch_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicExch(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicExch_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicExch(unsigned long long* address,
                                                unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicExch_system(unsigned long long* address,
                                                       unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicExch(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline float atomicExch_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline double atomicExch(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline double atomicExch_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicMin(int* address, int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicMin_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicMin(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicMin_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicMin(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicMin_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicMin(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicMin_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline long long atomicMin(long long* address, long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline long long atomicMin_system(long long* address, long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicMin(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicMin_system(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline double atomicMin(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicMin_system(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline int atomicMax(int* address, int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicMax_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicMax(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicMax_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicMax(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicMax_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicMax(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicMax_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
__device__ inline long long atomicMax(long long* address, long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline long long atomicMax_system(long long* address, long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicMax(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicMax_system(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline double atomicMax(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicMax_system(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline unsigned int atomicInc(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_ATTR(remote_memory, no_fine_grained_memory) {
    return __scoped_atomic_fetch_uinc(address, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

__device__ inline unsigned int atomicDec(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_ATTR(remote_memory, no_fine_grained_memory) {
    return __scoped_atomic_fetch_udec(address, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

__device__ inline int atomicAnd(int* address, int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicAnd_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicAnd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicAnd_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicAnd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicAnd_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicAnd(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicAnd_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicOr(int* address, int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicOr_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicOr(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicOr_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicOr(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicOr_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicOr_system(unsigned long long* address,
                                                     unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicXor(int* address, int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicXor_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicXor(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicXor_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicXor(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicXor_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicXor(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicXor_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
