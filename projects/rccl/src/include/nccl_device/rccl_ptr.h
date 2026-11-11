#pragma once

/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdint>

// Defines a series of global address space pointers.  Casting to these
// pointers in hot code paths should improve performance since global
// aperture vector instrutions like global_store_dwordx4 can be used.
// These are cheaper than flat loads and stores.
// Verify the intended effect by inspecting assembly.  If you see
// flat in the name of the emitted instruction, something is wrong.
using u64_gptr = __attribute__((address_space(1))) uint64_t*;
using u32_gptr = __attribute__((address_space(1))) uint32_t*;
using u16_gptr = __attribute__((address_space(1))) uint16_t*;
using u8_gptr = __attribute__((address_space(1))) uint8_t*;

#ifdef __HIP_DEVICE_COMPILE__
#if (defined(__gfx942__) || defined(__gfx950__) || (defined(__gfx1250__) || defined(__gfx1250_strict__))) && \
  __has_builtin(__builtin_amdgcn_global_load_b128) && __has_builtin(__builtin_amdgcn_global_store_b128) && \
  !defined(DWORDX4_INTRINSICS_FORCE_OFF)
#define RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS 1
// #pragma message "RCCL DWORDX4 Builtins Enabled on GFX942/GFX950/GFX1250"
#else
#define RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS 0
// #pragma message "RCCL DWORDX4 Builtins Disabled"
#endif
#else
#define RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS 0
#endif

#ifdef __HIP_DEVICE_COMPILE__
#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && \
  __has_builtin(__builtin_amdgcn_cooperative_atomic_load_8x16B) && \
  __has_builtin(__builtin_amdgcn_cooperative_atomic_store_8x16B) && \
  !defined(COOPERATIVE_ATOMIC_INTRINSICS_FORCE_OFF)
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 1
// #pragma message "RCCL Cooperative Atomic Builtins Enabled on GFX1250"
#else
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 0
// #pragma message "RCCL Cooperative Atomic Builtins Disabled"
#endif
#else
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 0
#endif

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u;
typedef __attribute__((address_space(1))) v4u* v4u_gptr;

typedef __attribute__((__vector_size__(4 * sizeof(int)))) int v4i;
typedef __attribute__((address_space(1))) v4i* v4i_gptr;

// "" means system scope, "agent" means device.  Adding this here because I don't think it's obvious otherwise that
// "" means system scope.
#define RCCL_SYSTEM_SYNCSCOPE ""

// System-scope 128-bit global load/store for the device-API headers below.
// __builtin_amdgcn_global_{load,store}_b128 lower to llvm.amdgcn.global.{load,store}.b128, which exist only in
// AMD's LLVM. Upstream LLVM has the same operation as llvm.amdgcn.av.{load,store}.b128, from
// __builtin_amdgcn_av_{load,store}_b128 (llvm/llvm-project#199176). Device code built from these headers is also
// compiled by upstream-LLVM consumers -- Triton kernels that link RCCL's GIN device API -- which abort on the AMD-only
// intrinsics ("Unknown type!"). So prefer the av builtins where the compiler has them; on gfx942/gfx950 both lower to
// the same global_store_dwordx4 / global_load_dwordx4 ... sc0 sc1.
#ifdef __HIP_DEVICE_COMPILE__
#if (defined(__gfx942__) || defined(__gfx950__) || (defined(__gfx1250__) || defined(__gfx1250_strict__))) && \
  __has_builtin(__builtin_amdgcn_av_load_b128) && __has_builtin(__builtin_amdgcn_av_store_b128) && \
  !defined(DWORDX4_INTRINSICS_FORCE_OFF)
#define RCCL_HAVE_AV_B128_BUILTINS 1
#else
#define RCCL_HAVE_AV_B128_BUILTINS 0
#endif
#else
#define RCCL_HAVE_AV_B128_BUILTINS 0
#endif

#define RCCL_HAVE_SYS_B128 (RCCL_HAVE_AV_B128_BUILTINS || RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS)
#if RCCL_HAVE_AV_B128_BUILTINS
#define RCCL_STORE_B128_SYS(ptr, vec) __builtin_amdgcn_av_store_b128((v4u_gptr)(ptr), (vec), __MEMORY_SCOPE_SYSTEM)
#define RCCL_LOAD_B128_SYS(ptr) __builtin_amdgcn_av_load_b128((v4u_gptr)(ptr), __MEMORY_SCOPE_SYSTEM)
#elif RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
#define RCCL_STORE_B128_SYS(ptr, vec) __builtin_amdgcn_global_store_b128((v4u_gptr)(ptr), (vec), RCCL_SYSTEM_SYNCSCOPE)
#define RCCL_LOAD_B128_SYS(ptr) __builtin_amdgcn_global_load_b128((v4u_gptr)(ptr), RCCL_SYSTEM_SYNCSCOPE)
#endif

// Observed on gfx1250: sibling DPX P2P into a cacheable comm FIFO is not
// coherent under plain/nontemporal load or store. Default hipMalloc and
// cuMem/VMM hung; uncached (hipDeviceMallocUncached) did not. System-scope
// b128 load *and* store both observe the peer. Restricted to gfx1250, the
// only arch measured. Following up with HIP/compilers.
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS && (defined(__gfx1250__) || defined(__gfx1250_strict__))
#define RCCL_LL_FIFO_SYS_SCOPE 1
#else
#define RCCL_LL_FIFO_SYS_SCOPE 0
#endif
// Deprecated alias for RCCL_LL_FIFO_SYS_SCOPE; kept for source-level backward
// compatibility with out-of-tree users of this installed header.
#define RCCL_LL_FIFO_SYS_SCOPE_LOAD RCCL_LL_FIFO_SYS_SCOPE
