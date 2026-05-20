/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_ASSEMBLY_HPP_
#define LIBRARY_SRC_ASSEMBLY_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace rocshmem {

#define DO_PRAGMA(x) _Pragma(#x)
#define NOWARN(warnoption, ...)                 \
  DO_PRAGMA(GCC diagnostic push)                \
  DO_PRAGMA(GCC diagnostic ignored #warnoption) \
  __VA_ARGS__                                   \
  DO_PRAGMA(GCC diagnostic pop)

#define SFENCE() asm volatile("sfence" ::: "memory")

__device__ __forceinline__ int uncached_load_ubyte([[maybe_unused]] uint8_t* src) {
  int ret = 0;
#if defined(__gfx90a__)
  int16_t val16;
  asm volatile(
      "global_load_ubyte %0 %1 off glc slc \n"
      "s_waitcnt vmcnt(0)"
      : "=v"(val16)
      : "v"(src)
      : "memory");
  ret = static_cast<int>(val16);
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  int16_t val16;
  asm volatile(
      "global_load_ubyte %0 %1 off sc0 sc1 \n"
      "s_waitcnt vmcnt(0)"
      : "=v"(val16)
      : "v"(src)
      : "memory");
  ret = static_cast<int>(val16);
#endif
#if defined(__gfx1100__)
  asm volatile(
      "global_load_ubyte %0 %1 off glc slc \n"
      "s_waitcnt vmcnt(0)"
      : "=v"(ret)
      : "v"(src)
      : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile(
      "global_load_u8 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(ret)
      : "v"(src)
      : "memory");
#endif
  return ret;
}

__device__ __forceinline__ void refresh_volatile_sbyte([[maybe_unused]] volatile int *assigned_value,
                                                       [[maybe_unused]] volatile char *read_value) {
#if defined(__gfx90a__)
  int16_t val16;
  asm volatile(
    "global_load_sbyte %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(val16)
    : "v"(read_value)
    : "memory");
  *assigned_value = static_cast<int>(val16);
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  int16_t val16;
  asm volatile(
    "global_load_sbyte %0 %1 off sc0 sc1\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(val16)
    : "v"(read_value)
    : "memory");
  *assigned_value = static_cast<int>(val16);
#endif
#if defined(__gfx1100__)
  asm volatile(
    "global_load_sbyte %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value)
    : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile(
      "global_load_i8 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(*assigned_value)
      : "v"(read_value)
      : "memory");
#endif
}

__device__ __forceinline__ void refresh_volatile_dwordx2([[maybe_unused]] volatile uint64_t *assigned_value,
                                                         [[maybe_unused]] volatile uint64_t *read_value) {
#if defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile(
    "global_load_dwordx2 %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value)
    : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile(
    "global_load_dwordx2 %0 %1 off sc0 sc1\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value)
    : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile(
      "global_load_b64 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(*assigned_value)
      : "v"(read_value)
      : "memory");
#endif
}

template <typename T>
__device__ __forceinline__ T uncached_load([[maybe_unused]] T* src) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8 || sizeof(T) == 16,
                "uncached_load only supports 1/2/4/8/16-byte types");
  T ret{};
  switch (sizeof(T)) {
    case 1: {
#if defined(__gfx90a__)
    
      int16_t val16;
      asm volatile(
          "global_load_ubyte %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val16);
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int16_t val16;
      asm volatile(
          "global_load_ubyte %0 %1 off sc0 sc1 \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val16);
#endif
#if defined(__gfx1100__)
      int32_t val32;
      asm volatile(
          "global_load_ubyte %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val32);
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;
      asm volatile(
          "global_load_u8 %0 %1 off scope:SCOPE_SYS \n"
          "s_wait_loadcnt 0x0"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val32);
#endif
      break;
    }
    case 2: {
#if defined(__gfx90a__)
      asm volatile(
          "global_load_ushort %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx1100__)
      int32_t val32;
      asm volatile(
          "global_load_ushort %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val32);
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile(
          "global_load_ushort %0 %1 off sc0 sc1 \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;
      asm volatile(
          "global_load_u16 %0 %1 off scope:SCOPE_SYS \n"
          "s_wait_loadcnt 0x0"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      ret = static_cast<T>(val32);
#endif
      break;
    }
    case 4: {
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile(
          "global_load_dword %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile(
          "global_load_dword %0 %1 off sc0 sc1 \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile(
          "global_load_b32 %0 %1 off scope:SCOPE_SYS \n"
          "s_wait_loadcnt 0x0"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
      break;
    }
    case 8: {
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile(
          "global_load_dwordx2 %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile(
          "global_load_dwordx2 %0 %1 off sc0 sc1 \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile(
          "global_load_b64 %0 %1 off scope:SCOPE_SYS \n"
          "s_wait_loadcnt 0x0"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
      break;
    }
    case 16: {
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile(
          "global_load_dwordx4 %0 %1 off glc slc \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile(
          "global_load_dwordx4 %0 %1 off sc0 sc1 \n"
          "s_waitcnt vmcnt(0)"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile(
          "global_load_b128 %0 %1 off scope:SCOPE_SYS \n"
          "s_wait_loadcnt 0x0"
          : "=v"(ret)
          : "v"(src)
          : "memory");
#endif
      break;
    }
    default:
      break;
  }
  return ret;
}

__device__ __forceinline__ void __roc_flush() {
#if not defined USE_HDP_FLUSH
#if defined(__gfx906__)
#endif
#if defined(__gfx908__) || defined(__gfx1100__)
#endif
#if defined(__gfx90a__)
//  asm volatile("s_dcache_wb;");
//  asm volatile("buffer_wbl2;");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
//  asm volatile("s_dcache_wb;");
//  asm volatile("buffer_wbl2;");
#endif
#endif
}

__device__ __forceinline__ void put_asm([[maybe_unused]] uint8_t* src,
                                        [[maybe_unused]] uint8_t* dst,
                                        int size) {
  switch (size) {
    case 1: [[unlikely]] {
#if defined(__gfx90a__)
      int16_t val16{static_cast<int16_t>(*src)};
      asm volatile("flat_store_byte %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val16)
                   : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int16_t val16{static_cast<int16_t>(*src)};
      asm volatile("flat_store_byte %0, %1, sc0 sc1"
                   :
                   : "v"(dst), "v"(val16)
                   : "memory");
#endif
#if defined(__gfx1100__)
      int32_t val32{static_cast<int32_t>(*src)};
      asm volatile("flat_store_byte %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32{static_cast<int32_t>(*src)};
      asm volatile("flat_store_b8 %0, %1, scope:SCOPE_SYS"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
      break;
    }
    case 2: [[unlikely]] {
      [[maybe_unused]] int16_t val16{*(reinterpret_cast<int16_t*>(src))};
#if defined(__gfx90a__)
      asm volatile("flat_store_short %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val16)
                   : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_short %0, %1, sc0 sc1"
                   :
                   : "v"(dst), "v"(val16)
                   : "memory");
#endif
#if defined(__gfx1100__)
      int32_t val32{static_cast<int32_t>(val16)};
      asm volatile("flat_store_short %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32{static_cast<int32_t>(val16)};
      asm volatile("flat_store_b16 %0, %1, scope:SCOPE_SYS"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
      break;
    }
    case 4: [[unlikely]] {
      [[maybe_unused]] int32_t val32{*(reinterpret_cast<int32_t*>(src))};
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dword %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dword %0, %1, sc0 sc1"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SYS"
                   :
                   : "v"(dst), "v"(val32)
                   : "memory");
#endif
      break;
    }
    case 8: [[unlikely]] {
      [[maybe_unused]] int64_t val64{*(reinterpret_cast<int64_t*>(src))};
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dwordx2 %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val64)
                   : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dwordx2 %0, %1, sc0 sc1"
                   :
                   : "v"(dst), "v"(val64)
                   : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SYS"
                   :
                   : "v"(dst), "v"(val64)
                   : "memory");
#endif
      break;
    }
    case 16: [[likely]] {
      [[maybe_unused]] __int128_t val128{*(reinterpret_cast<__int128_t*>(src))};
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dwordx4 %0, %1, glc slc"
                   :
                   : "v"(dst), "v"(val128)
                   : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dwordx4 %0, %1, sc0 sc1"
                   :
                   : "v"(dst), "v"(val128)
                   : "memory");
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SYS"
                   :
                   : "v"(dst), "v"(val128)
                   : "memory");
#endif
      break;
    }
    default: [[unlikely]]
      break;
  }
}

__device__ __forceinline__ void get_asm([[maybe_unused]] uint8_t* src, 
                                        [[maybe_unused]] uint8_t* dst, 
                                        int size) {
  switch (size) {
    case 1: [[unlikely]] {
#if defined(__gfx90a__)
      int16_t val16;
      asm volatile(
          "flat_load_ubyte %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      *dst = static_cast<uint8_t>(val16);
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int16_t val16;
      asm volatile(
          "flat_load_ubyte %0, %1, sc0 sc1\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      *dst = static_cast<uint8_t>(val16);
#endif
#if defined(__gfx1100__)
      int32_t val32;
      asm volatile(
          "flat_load_ubyte %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *dst = static_cast<uint8_t>(val32);
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;
      asm volatile(
          "flat_load_u8 %0, %1, scope:SCOPE_SYS\n"
          "s_wait_loadcnt 0x0"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *dst = static_cast<uint8_t>(val32);
#endif
      break;
    }
    case 2: [[unlikely]] {
#if defined(__gfx90a__)
      int16_t val16;
      asm volatile(
          "flat_load_ushort %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int16_t*>(dst)) = val16;
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int16_t val16;
      asm volatile(
          "flat_load_ushort %0, %1, sc0 sc1\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val16)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int16_t*>(dst)) = val16;
#endif
#if defined(__gfx1100__)
      int32_t val32;
      asm volatile(
          "flat_load_ushort %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int16_t*>(dst)) = static_cast<int16_t>(val32);
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;
      asm volatile(
          "flat_load_u16 %0, %1, scope:SCOPE_SYS\n"
          "s_wait_loadcnt 0x0"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int16_t*>(dst)) = static_cast<int16_t>(val32);
#endif
      break;
    }
    case 4: [[unlikely]] {
#if defined(__gfx90a__) || defined(__gfx1100__)
      int32_t val32;
      asm volatile(
          "flat_load_dword %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int32_t*>(dst)) = val32;
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int32_t val32;
      asm volatile(
          "flat_load_dword %0, %1, sc0 sc1\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int32_t*>(dst)) = val32;
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;
      asm volatile(
          "flat_load_b32 %0, %1, scope:SCOPE_SYS\n"
          "s_wait_loadcnt 0x0"
          : "=v"(val32)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int32_t*>(dst)) = val32;
#endif
      break;
    }
    case 8: [[unlikely]] {
#if defined(__gfx90a__) || defined(__gfx1100__)
      int64_t val64;
      asm volatile(
          "flat_load_dwordx2 %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val64)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int64_t*>(dst)) = val64;
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int64_t val64;
      asm volatile(
          "flat_load_dwordx2 %0, %1, sc0 sc1\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val64)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int64_t*>(dst)) = val64;
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      int64_t val64;
      asm volatile(
          "flat_load_b64 %0, %1, scope:SCOPE_SYS\n"
          "s_wait_loadcnt 0x0"
          : "=v"(val64)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<int64_t*>(dst)) = val64;
#endif
      break;
    }
    case 16: [[likely]] {
#if defined(__gfx90a__) || defined(__gfx1100__)
      __int128_t val128;
      asm volatile(
          "flat_load_dwordx4 %0, %1, glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val128)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<__int128_t*>(dst)) = val128;
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      __int128_t val128;
      asm volatile(
          "flat_load_dwordx4 %0, %1, sc0 sc1\n"
          "s_waitcnt vmcnt(0)"
          : "=v"(val128)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<__int128_t*>(dst)) = val128;
#endif
#if defined(__gfx1201__) || defined(__gfx1250__)
      __int128_t val128;
      asm volatile(
          "flat_load_b128 %0, %1, scope:SCOPE_SYS\n"
          "s_wait_loadcnt 0x0"
          : "=v"(val128)
          : "v"(src)
          : "memory");
      *(reinterpret_cast<__int128_t*>(dst)) = val128;
#endif
      break;
    }
    default: [[unlikely]]
      break;
  }
}

// ==============================================================================
// BUFFER RESOURCE INTRINSICS
// Using LLVM raw-buffer intrinsics rather than inline asm "s" constraints:
//   - The resource descriptor must occupy 4 consecutive SGPRs (<4 x i32>).
//   - ext_vector_type(4) maps to <4 x i32> in LLVM IR; the compiler places
//     uniform values in SGPRs automatically when passed to these intrinsics.
//   - Inline asm "s" on HIP's struct int4 is UB (struct != ext_vector register
//     class); going through the intrinsics avoids the VGPR→SGPR hazard.
// ==============================================================================

using i32x4_t = int32_t __attribute__((ext_vector_type(4)));

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b128(
    __uint128_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i128");

// ==============================================================================
// CACHE POLICIES
// ==============================================================================
enum class CachePolicy {
  Standard,      // Normal C++ load/store (L1 and L2 cached)
  FlatCache,     // Flat load/store with L1 and L2 caching 
  BypassL1,      // Bypass L1 (sc0 / glc / scope:DEV)
  NonTemporal,   // Streaming data (nt / glc slc)
  SystemScope,   // Bypass L1 and L2 (sc0 sc1 / glc slc / scope:SYS)
  SystemScopeNT  // Bypass L1 and L2 + Streaming (sc0 sc1 nt)
};

template <int Size, CachePolicy LoadPolicy = CachePolicy::Standard,
          CachePolicy StorePolicy = LoadPolicy>
struct AsmAccess;

// ==============================================================================
// 16-BYTE ACCESS (128-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<16, LoadPolicy, StorePolicy> {
  using type = __int128_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      type val{};
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dwordx4 %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dwordx4 %0, %1, sc0" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_load_dwordx4 %0, %1, nt" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScope) {
        asm volatile("flat_load_dwordx4 %0, %1, sc0 sc1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_load_dwordx4 %0, %1, sc0 sc1 nt" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dwordx4 %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dwordx4 %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dwordx4 %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b128 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_b128 %0, %1, scope:SCOPE_DEV" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_b128 %0, %1, scope:SCOPE_SYS" : "=v"(val) : "v"(src) : "memory");
      }
#else
      val = *reinterpret_cast<type*>(src);
#endif
      return val;
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dwordx4 %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dwordx4 %0, %1, sc0" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_store_dwordx4 %0, %1, nt" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScope) {
        asm volatile("flat_store_dwordx4 %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_store_dwordx4 %0, %1, sc0 sc1 nt" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dwordx4 %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dwordx4 %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dwordx4 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }

  // --------------------------------------------------------------------------
  // Buffer instructions (MUBUF / MTBUF family)
  //
  // The buffer resource descriptor is four SGPRs (128 bits) laid out as:
  //   [63:0]  BASE_ADDRESS  – 64-bit byte address of the buffer base
  //   [95:64] NUM_RECORDS   – buffer size in bytes (range check limit)
  //   [127:96] CONFIG word:
  //     [18]    STRIDE == 0  (byte-addressed, no struct stride)
  //     [15:14] DST_SEL_*   (default 0 → pass-through)
  //     [21:20] INDEX_STRIDE (unused when STRIDE==0)
  //     Bit 17 set → "raw" (swizzle disabled), bit 16 = cache_swizzle
  //   We use 0x00020000 which sets bit 17 (DATA_FORMAT=BUF_DATA_FORMAT_INVALID
  //   in older ISAs, raw-buffer enable in GFX9+). This is the standard recipe
  //   used by every ROCm raw-buffer helper (see HipKittens, rocWMMA, etc.).
  //
  // The voffset (VGPR offset) is per-lane; soffset is a scalar added on top.
  // Together they address: BASE + voffset + soffset.
  //
  // Cache modifier encoding for buffer_load/store_b128 on GFX942/GFX950:
  //   bit 0 (sc0 / GLC) – bypass L1 (global coherence)
  //   bit 1 (sc1 / SLC) – bypass L2 (system coherence)
  //   bit 4 (NT)        – non-temporal / streaming hint
  //   0b00000 = L1+L2 cached (FlatCache)
  //   0b00001 = bypass L1    (BypassL1)
  //   0b00011 = bypass L1+L2 (SystemScope)
  //   0b10011 = bypass L1+L2 + NT (SystemScopeNT)
  //   0b10001 = NT only (NonTemporal)
  // --------------------------------------------------------------------------

  // Build a raw buffer resource descriptor for a contiguous byte buffer.
  // Returns i32x4_t (<4 x i32>) so the "s" asm constraint and the LLVM
  // raw-buffer intrinsics both receive the correct SGPR-friendly ext_vector.
  // Layout per GFX9+ ISA:
  //   [63:0]   BASE_ADDRESS
  //   [95:64]  NUM_RECORDS  (byte count; range-check limit when STRIDE == 0)
  //   [127:96] 0x00020000   (bit 17 = raw/unswizzled, STRIDE = 0)
  static __device__ __forceinline__ i32x4_t
  make_buffer_resource(const void *base, uint32_t num_bytes) {
    uint64_t addr = reinterpret_cast<uint64_t>(base);
    i32x4_t rsrc;
    rsrc[0] = static_cast<int32_t>(addr & 0xFFFFFFFFu);  // BASE_ADDRESS[31:0]
    rsrc[1] = static_cast<int32_t>(addr >> 32);          // BASE_ADDRESS[63:32]
    rsrc[2] = static_cast<int32_t>(num_bytes);           // NUM_RECORDS
    rsrc[3] = 0x00020000;  // raw, no stride/swizzle
    return rsrc;
  }

  // Load 16 bytes via llvm.amdgcn.raw.buffer.load.i128.
  // ptr      – base address of the buffer (alignment ≥ 16)
  // buf_size – byte size of the buffer (hardware range-check limit)
  // offset   – per-lane byte offset from ptr (voffset; added per-thread)
  //
  // aux cachepolicy bits: bit0=sc0(GLC), bit1=sc1(SLC), bit4=nt
  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux =
        (LoadPolicy == CachePolicy::Standard)     ? 0b00000 :
        (LoadPolicy == CachePolicy::FlatCache)    ? 0b00000 :
        (LoadPolicy == CachePolicy::BypassL1)     ? 0b00001 :
        (LoadPolicy == CachePolicy::NonTemporal)  ? 0b00010 :
        (LoadPolicy == CachePolicy::SystemScope)  ? 0b10001 :
      /*(LoadPolicy == CachePolicy::SystemScopeNT)*/ 0b10011;
    return static_cast<type>(
        llvm_amdgcn_raw_buffer_load_b128(rsrc, offset, 0, aux));
  }

  // Store 16 bytes via llvm.amdgcn.raw.buffer.store.i128.
  // ptr      – base address of the buffer
  // buf_size – byte size of the buffer (hardware range-check limit)
  // offset   – per-lane byte offset from ptr (voffset)
  // val      – 128-bit value to write
  //
  // aux cachepolicy bits: bit0=sc0(GLC), bit1=sc1(SLC), bit4=nt
  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux =
        (StorePolicy == CachePolicy::Standard)     ? 0b00000 :
        (StorePolicy == CachePolicy::FlatCache)    ? 0b00000 :
        (StorePolicy == CachePolicy::BypassL1)     ? 0b00001 :
        (StorePolicy == CachePolicy::NonTemporal)  ? 0b00010 :
        (StorePolicy == CachePolicy::SystemScope)  ? 0b10001 :
      /*(StorePolicy == CachePolicy::SystemScopeNT)*/ 0b10011;
    llvm_amdgcn_raw_buffer_store_b128(static_cast<__uint128_t>(val),
                                      rsrc, offset, 0, aux);
  }

  static __device__ __forceinline__ void load_store_buffer(void* __restrict__ src, 
                                                           void* __restrict__ dst,
                                                           uint32_t buf_size,
                                                           int offset, int tid,
                                                           int stride) {
    constexpr int Unroll = 16;
    type regs[Unroll];
    i32x4_t rsrc = make_buffer_resource(src, buf_size);
    i32x4_t rdst = make_buffer_resource(dst, buf_size);

    constexpr uint32_t aux_load =
        (LoadPolicy == CachePolicy::Standard)      ? 0b00000
        : (LoadPolicy == CachePolicy::FlatCache)   ? 0b00000
        : (LoadPolicy == CachePolicy::BypassL1)    ? 0b00001
        : (LoadPolicy == CachePolicy::NonTemporal) ? 0b00010
        : (LoadPolicy == CachePolicy::SystemScope) ? 0b10001
                                                   : 0b10011;
    constexpr uint32_t aux_store =
        (StorePolicy == CachePolicy::Standard)      ? 0b00000
        : (StorePolicy == CachePolicy::FlatCache)   ? 0b00000
        : (StorePolicy == CachePolicy::BypassL1)    ? 0b00001
        : (StorePolicy == CachePolicy::NonTemporal) ? 0b00010
        : (StorePolicy == CachePolicy::SystemScope) ? 0b10001
                                                    : 0b10011;

#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      regs[u] = llvm_amdgcn_raw_buffer_load_b128(rsrc, tid * 16,
        (offset + u * stride) * 16, aux_load);
    }
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      llvm_amdgcn_raw_buffer_store_b128(regs[u], rdst, tid * 16,
        (offset + u * stride) * 16, aux_store);
    }
  }
};

// ==============================================================================
// 8-BYTE ACCESS (64-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<8, LoadPolicy, StorePolicy> {
  using type = int64_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      type val{};
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dwordx2 %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dwordx2 %0, %1, sc0" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_load_dwordx2 %0, %1, nt" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScope) {
        asm volatile("flat_load_dwordx2 %0, %1, sc0 sc1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_load_dwordx2 %0, %1, sc0 sc1 nt" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dwordx2 %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dwordx2 %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dwordx2 %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b64 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_b64 %0, %1, scope:SCOPE_DEV" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_b64 %0, %1, scope:SCOPE_SYS" : "=v"(val) : "v"(src) : "memory");
      }
#else
      val = *reinterpret_cast<type*>(src);
#endif
      return val;
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dwordx2 %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dwordx2 %0, %1, sc0" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_store_dwordx2 %0, %1, nt" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScope) {
        asm volatile("flat_store_dwordx2 %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_store_dwordx2 %0, %1, sc0 sc1 nt" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dwordx2 %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dwordx2 %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dwordx2 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }
};

// ==============================================================================
// 4-BYTE ACCESS (32-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<4, LoadPolicy, StorePolicy> {
  using type = int32_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      type val{};
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dword %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dword %0, %1, sc0" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_load_dword %0, %1, nt" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScope) {
        asm volatile("flat_load_dword %0, %1, sc0 sc1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_load_dword %0, %1, sc0 sc1 nt" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_dword %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_dword %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dword %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b32 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_b32 %0, %1, scope:SCOPE_DEV" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_b32 %0, %1, scope:SCOPE_SYS" : "=v"(val) : "v"(src) : "memory");
      }
#else
      val = *reinterpret_cast<type*>(src);
#endif
      return val;
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dword %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dword %0, %1, sc0" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_store_dword %0, %1, nt" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScope) {
        asm volatile("flat_store_dword %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_store_dword %0, %1, sc0 sc1 nt" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx90a__) || defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_dword %0, %1" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_dword %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dword %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }
};

// ==============================================================================
// 2-BYTE ACCESS (16-bit / Widened)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<2, LoadPolicy, StorePolicy> {
  using type = int16_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx90a__)
      int16_t val{};  // Gfx9 supports native 16-bit vector registers
  #if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ushort %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ushort %0, %1, sc0" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_load_ushort %0, %1, nt" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScope) {
        asm volatile("flat_load_ushort %0, %1, sc0 sc1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_load_ushort %0, %1, sc0 sc1 nt" : "=v"(val) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ushort %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ushort %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ushort %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
  #endif
      return val;
#elif defined(__gfx1100__) || defined(__gfx1201__)
      int32_t val32;  // Gfx11/12 forces 16-bit ops into 32-bit registers
  #if defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ushort %0, %1" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ushort %0, %1, glc" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ushort %0, %1, glc slc" : "=v"(val32) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_u16 %0, %1, scope:SCOPE_SE" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_u16 %0, %1, scope:SCOPE_DEV" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_u16 %0, %1, scope:SCOPE_SYS" : "=v"(val32) : "v"(src) : "memory");
      }
  #endif
      return static_cast<type>(val32);
#else
      return *reinterpret_cast<type*>(src);
#endif
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx90a__)
      int16_t val16 = val;
  #if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_short %0, %1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_short %0, %1, sc0" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_store_short %0, %1, nt" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScope) {
        asm volatile("flat_store_short %0, %1, sc0 sc1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_store_short %0, %1, sc0 sc1 nt" : : "v"(dst), "v"(val16) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_short %0, %1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_short %0, %1, glc" : : "v"(dst), "v"(val16) : "memory");
      } else {
        asm volatile("flat_store_short %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
      }
  #endif
#elif defined(__gfx1100__) || defined(__gfx1201__)
      int32_t val32 = static_cast<int32_t>(val);
  #if defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_short %0, %1" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_short %0, %1, glc" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_short %0, %1, glc slc" : : "v"(dst), "v"(val32) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b16 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_b16 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_b16 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val32) : "memory");
      }
  #endif
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }
};

// ==============================================================================
// 1-BYTE ACCESS (8-bit / Widened)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<1, LoadPolicy, StorePolicy> {
  using type = uint8_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx90a__)
      int16_t val{};  // Gfx9 loads bytes into 16-bit registers minimum
  #if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ubyte %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ubyte %0, %1, sc0" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_load_ubyte %0, %1, nt" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScope) {
        asm volatile("flat_load_ubyte %0, %1, sc0 sc1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_load_ubyte %0, %1, sc0 sc1 nt" : "=v"(val) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ubyte %0, %1" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ubyte %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ubyte %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
  #endif
      return static_cast<type>(val);
#elif defined(__gfx1100__) || defined(__gfx1201__)
      int32_t val32{};  // Gfx11/12 forces 8-bit ops into 32-bit registers
  #if defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ubyte %0, %1" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_ubyte %0, %1, glc" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ubyte %0, %1, glc slc" : "=v"(val32) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_u8 %0, %1, scope:SCOPE_SE" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::BypassL1) {
        asm volatile("flat_load_u8 %0, %1, scope:SCOPE_DEV" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_u8 %0, %1, scope:SCOPE_SYS" : "=v"(val32) : "v"(src) : "memory");
      }
  #endif
      return static_cast<type>(val32);
#else
      return *reinterpret_cast<type*>(src);
#endif
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx90a__)
      int16_t val16 = static_cast<int16_t>(val);
  #if defined(__gfx942__) || defined(__gfx950__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_byte %0, %1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_byte %0, %1, sc0" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::NonTemporal) {
        asm volatile("flat_store_byte %0, %1, nt" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScope) {
        asm volatile("flat_store_byte %0, %1, sc0 sc1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::SystemScopeNT) {
        asm volatile("flat_store_byte %0, %1, sc0 sc1 nt" : : "v"(dst), "v"(val16) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_byte %0, %1" : : "v"(dst), "v"(val16) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_byte %0, %1, glc" : : "v"(dst), "v"(val16) : "memory");
      } else {
        asm volatile("flat_store_byte %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
      }
  #endif
#elif defined(__gfx1100__) || defined(__gfx1201__)
      int32_t val32 = static_cast<int32_t>(val);
  #if defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_byte %0, %1" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_byte %0, %1, glc" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_byte %0, %1, glc slc" : : "v"(dst), "v"(val32) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b8 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::BypassL1) {
        asm volatile("flat_store_b8 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_b8 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val32) : "memory");
      }
  #endif
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }
};

__device__ __forceinline__ void pipeline_wait_on_loads(int waits) {
#if defined(__gfx1201__)
  // GFX12 splits memory counters further; flat_load utilizes loadcnt
  switch (waits) {
    case 15: asm volatile("s_wait_loadcnt 15" ::: "memory"); break;
    case 14: asm volatile("s_wait_loadcnt 14" ::: "memory"); break;
    case 13: asm volatile("s_wait_loadcnt 13" ::: "memory"); break;
    case 12: asm volatile("s_wait_loadcnt 12" ::: "memory"); break;
    case 11: asm volatile("s_wait_loadcnt 11" ::: "memory"); break;
    case 10: asm volatile("s_wait_loadcnt 10" ::: "memory"); break;
    case 9:  asm volatile("s_wait_loadcnt 9"  ::: "memory"); break;
    case 8:  asm volatile("s_wait_loadcnt 8"  ::: "memory"); break;
    case 7:  asm volatile("s_wait_loadcnt 7"  ::: "memory"); break;
    case 6:  asm volatile("s_wait_loadcnt 6"  ::: "memory"); break;
    case 5:  asm volatile("s_wait_loadcnt 5"  ::: "memory"); break;
    case 4:  asm volatile("s_wait_loadcnt 4"  ::: "memory"); break;
    case 3:  asm volatile("s_wait_loadcnt 3"  ::: "memory"); break;
    case 2:  asm volatile("s_wait_loadcnt 2"  ::: "memory"); break;
    case 1:  asm volatile("s_wait_loadcnt 1"  ::: "memory"); break;
    default: asm volatile("s_wait_loadcnt 0"  ::: "memory"); break;
  }
#else
  // GFX9/10/11 use vmcnt
  switch (waits) {
    case 15: asm volatile("s_waitcnt vmcnt(15)" ::: "memory"); break;
    case 14: asm volatile("s_waitcnt vmcnt(14)" ::: "memory"); break;
    case 13: asm volatile("s_waitcnt vmcnt(13)" ::: "memory"); break;
    case 12: asm volatile("s_waitcnt vmcnt(12)" ::: "memory"); break;
    case 11: asm volatile("s_waitcnt vmcnt(11)" ::: "memory"); break;
    case 10: asm volatile("s_waitcnt vmcnt(10)" ::: "memory"); break;
    case 9:  asm volatile("s_waitcnt vmcnt(9)"  ::: "memory"); break;
    case 8:  asm volatile("s_waitcnt vmcnt(8)"  ::: "memory"); break;
    case 7:  asm volatile("s_waitcnt vmcnt(7)"  ::: "memory"); break;
    case 6:  asm volatile("s_waitcnt vmcnt(6)"  ::: "memory"); break;
    case 5:  asm volatile("s_waitcnt vmcnt(5)"  ::: "memory"); break;
    case 4:  asm volatile("s_waitcnt vmcnt(4)"  ::: "memory"); break;
    case 3:  asm volatile("s_waitcnt vmcnt(3)"  ::: "memory"); break;
    case 2:  asm volatile("s_waitcnt vmcnt(2)"  ::: "memory"); break;
    case 1:  asm volatile("s_waitcnt vmcnt(1)"  ::: "memory"); break;
    default: asm volatile("s_waitcnt vmcnt(0)"  ::: "memory"); break;
  }
#endif
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ASSEMBLY_HPP_
