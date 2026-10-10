#pragma once

// pragma once does not collapse the hipify copy and the build/debug/include
// copy of this header, which host TUs include through both paths. The macro
// guard does, so the named 16-byte pun is not redefined.
#ifndef RCCL_PTR_H_
#define RCCL_PTR_H_

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

// -DDWORDX4_INTRINSICS=OFF defines DWORDX4_INTRINSICS_FORCE_OFF. Cooperative
// 128-bit atomics are the same class of builtin, so that switch disables them
// too. Checked here so every translation unit that forces the dwordx4 path off
// (the library target and the unit-test fixtures) also drops the cooperative path.
#if defined(DWORDX4_INTRINSICS_FORCE_OFF) && !defined(COOPERATIVE_ATOMIC_INTRINSICS_FORCE_OFF)
#define COOPERATIVE_ATOMIC_INTRINSICS_FORCE_OFF
#endif

#ifdef __HIP_DEVICE_COMPILE__
#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && \
  __has_builtin(__builtin_amdgcn_cooperative_atomic_load_8x16B) && \
  __has_builtin(__builtin_amdgcn_cooperative_atomic_store_8x16B) && \
  !defined(COOPERATIVE_ATOMIC_INTRINSICS_FORCE_OFF)
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 1
#else
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 0
#endif
#else
#define RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS 0
#endif

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u;
typedef __attribute__((address_space(1))) v4u* v4u_gptr;

// v4i/v4i_gptr exist only for the cooperative_atomic_*_8x16B builtins, whose
// signature takes a signed vector. The global b128 intrinsics take v4u.
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int v4i;
typedef __attribute__((address_space(1))) v4i* v4i_gptr;

// One 16-byte pun for those two vector types and a pair of uint64 words.
union alignas(16) rcclB128 {
  v4u v;
  v4i vi;
  uint64_t u64[2];
};

// "" means system scope, "agent" means device.  Adding this here because I don't think it's obvious otherwise that
// "" means system scope.
#define RCCL_SYSTEM_SYNCSCOPE ""

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

#endif // RCCL_PTR_H_
