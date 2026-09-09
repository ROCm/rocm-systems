/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_POISON_HIP_ATOMICS_H_
#define NCCL_POISON_HIP_ATOMICS_H_

// Force-included (-include) ahead of every RCCL translation unit when
// RCCL_POISON_HIP_ATOMICS=ON.  RCCL expresses device atomics with the Clang
// scoped-atomic builtins (__scoped_atomic_*, see device/op128.h and
// nccl_device/hip_compat.h) so that memory scope is explicit at every call
// site; the __hip_atomic_* builtins default their scope through HIP's headers
// instead.  Poisoning makes the compiler reject them rather than relying on a
// source grep, which macro expansion can hide.
//
// #pragma GCC poison rejects the identifier everywhere, including inside
// system headers, and HIP's own headers spell these builtins (amd_hip_atomic.h
// and amd_hip_unsafe_atomics.h implement atomicAdd() and friends with them;
// amd_hip_fp16.h and amd_hip_bf16.h carry their own half/bfloat16 atomicAdd
// overloads).  So HIP must be parsed before the poison takes effect, which is
// why those four headers are pulled in here -- hip_runtime.h alone does not
// reach the fp16/bf16 ones.  The consequence is that this covers RCCL's own
// sources only: HIP's atomicAdd()/atomicMax()/atomicExch() still reach the
// builtins through header code that was already parsed, and RCCL device code
// still calls those.
#if defined(__cplusplus) && (defined(__HIP__) || defined(__HIPCC__))
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#pragma GCC poison               \
    __hip_atomic_load            \
    __hip_atomic_store           \
    __hip_atomic_exchange        \
    __hip_atomic_compare_exchange_weak   \
    __hip_atomic_compare_exchange_strong \
    __hip_atomic_fetch_add       \
    __hip_atomic_fetch_sub       \
    __hip_atomic_fetch_and       \
    __hip_atomic_fetch_or        \
    __hip_atomic_fetch_xor       \
    __hip_atomic_fetch_min       \
    __hip_atomic_fetch_max

#endif
