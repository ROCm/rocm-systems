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
// ==============================================================================

using i32x4_t = int32_t __attribute__((ext_vector_type(4)));

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b128(
    __uint128_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i128");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b64(
    uint64_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b32(
    uint32_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b16(
    uint16_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ uint8_t llvm_amdgcn_raw_buffer_load_b8(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b8(
    uint8_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i8");

// ==============================================================================
// CACHE POLICIES
// ==============================================================================
enum class CachePolicy {
  Standard,      // Normal C++ load/store (L1 and L2 cached)
  FlatCache,     // Flat load/store with L1 and L2 caching 
  NonTemporal,   // Streaming data (nt / glc slc)
  DeviceScope,   // Bypass L1 (sc0 / glc / scope:DEV)
  SystemScope,   // Bypass L1 and L2 (sc0 sc1 / glc slc / scope:SYS)
  SystemScopeNT  // Bypass L1 and L2 + Streaming (sc0 sc1 nt)
};

// Maps a CachePolicy to the AMDGCN buffer instruction aux bits (sc0/sc1/nt).
__host__ __device__ constexpr uint32_t cache_policy_aux(CachePolicy p) {
#if defined(__gfx942__) || defined(__gfx950__)
  return p == CachePolicy::DeviceScope   ? 0b10000 :  // DEVICE_NT0: sc1
         p == CachePolicy::NonTemporal   ? 0b10010 :  // DEVICE_NT1: sc1|nt
         p == CachePolicy::SystemScope   ? 0b10001 :  // SYSTEM_NT0: sc1|sc0
         p == CachePolicy::SystemScopeNT ? 0b10011 :  // SYSTEM_NT1: sc1|sc0|nt
                                           0b00000;   // Standard / FlatCache (wave scope)
#elif defined(__gfx1201__) || defined(__gfx1250__)
  return p == CachePolicy::DeviceScope   ? 0b10000 :  // DEVICE_RT  = scope:DEV | temporal:RT
         p == CachePolicy::NonTemporal   ? 0b10001 :  // DEVICE_NT  = scope:DEV | temporal:NT
         p == CachePolicy::SystemScope   ? 0b11000 :  // SYSTEM_RT  = scope:SYS | temporal:RT
         p == CachePolicy::SystemScopeNT ? 0b11001 :  // SYSTEM_NT  = scope:SYS | temporal:NT
                                           0b00000;   // Standard / FlatCache (CU_RT)
#else
  return p == CachePolicy::DeviceScope   ? 0b01 :  // glc
         p == CachePolicy::NonTemporal   ? 0b11 :  // glc slc (closest available)
         p == CachePolicy::SystemScope   ? 0b11 :  // glc slc
         p == CachePolicy::SystemScopeNT ? 0b11 :  // glc slc
                                           0b00;   // Standard / FlatCache
#endif
}

__device__ __forceinline__ i32x4_t
make_buffer_resource(const void* base, uint32_t num_bytes) {
  uint64_t addr = reinterpret_cast<uint64_t>(base);
  i32x4_t rsrc;
  rsrc[0] = static_cast<int32_t>(addr & 0xFFFFFFFFu);
  rsrc[1] = static_cast<int32_t>(addr >> 32);
  rsrc[2] = static_cast<int32_t>(num_bytes);
#if defined(__gfx1201__) || defined(__gfx1100__)
  rsrc[3] = 0x31014000;  // raw buffer descriptor: no stride/swizzle
#else
  rsrc[3] = 0x00020000;  // raw buffer descriptor: no stride/swizzle
#endif
  return rsrc;
}

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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_dwordx4 %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dwordx4 %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b128 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_dwordx4 %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dwordx4 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b128(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b128(static_cast<__uint128_t>(val),
                                      rsrc, offset, 0, aux);
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_dwordx2 %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dwordx2 %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b64 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_dwordx2 %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dwordx2 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b64(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b64(static_cast<uint64_t>(val),
                                     rsrc, offset, 0, aux);
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_dword %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_dword %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_b32 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_dword %0, %1, glc" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_dword %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
      }
#elif defined(__gfx1201__) || defined(__gfx1250__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_DEV" : : "v"(dst), "v"(val) : "memory");
      } else {
        asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
      }
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b32(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b32(static_cast<uint32_t>(val),
                                     rsrc, offset, 0, aux);
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_ushort %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ushort %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
  #endif
      return val;
#elif defined(__gfx1100__) || defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32;  // Gfx11/12 forces 16-bit ops into 32-bit registers
  #if defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ushort %0, %1" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_ushort %0, %1, glc" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ushort %0, %1, glc slc" : "=v"(val32) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_u16 %0, %1, scope:SCOPE_SE" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_short %0, %1, glc" : : "v"(dst), "v"(val16) : "memory");
      } else {
        asm volatile("flat_store_short %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
      }
  #endif
#elif defined(__gfx1100__) || defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32 = static_cast<int32_t>(val);
  #if defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_short %0, %1" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_short %0, %1, glc" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_short %0, %1, glc slc" : : "v"(dst), "v"(val32) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b16 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return static_cast<type>(
        llvm_amdgcn_raw_buffer_load_b16(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b16(static_cast<uint16_t>(val),
                                     rsrc, offset, 0, aux);
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_ubyte %0, %1, glc" : "=v"(val) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ubyte %0, %1, glc slc" : "=v"(val) : "v"(src) : "memory");
      }
  #endif
      return static_cast<type>(val);
#elif defined(__gfx1100__) || defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32{};  // Gfx11/12 forces 8-bit ops into 32-bit registers
  #if defined(__gfx1100__)
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_ubyte %0, %1" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_load_ubyte %0, %1, glc" : "=v"(val32) : "v"(src) : "memory");
      } else {
        asm volatile("flat_load_ubyte %0, %1, glc slc" : "=v"(val32) : "v"(src) : "memory");
      }
  #else
      if constexpr (LoadPolicy == CachePolicy::FlatCache) {
        asm volatile("flat_load_u8 %0, %1, scope:SCOPE_SE" : "=v"(val32) : "v"(src) : "memory");
      } else if constexpr (LoadPolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_byte %0, %1, glc" : : "v"(dst), "v"(val16) : "memory");
      } else {
        asm volatile("flat_store_byte %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
      }
  #endif
#elif defined(__gfx1100__) || defined(__gfx1201__) || defined(__gfx1250__)
      int32_t val32 = static_cast<int32_t>(val);
  #if defined(__gfx1100__)
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_byte %0, %1" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
        asm volatile("flat_store_byte %0, %1, glc" : : "v"(dst), "v"(val32) : "memory");
      } else {
        asm volatile("flat_store_byte %0, %1, glc slc" : : "v"(dst), "v"(val32) : "memory");
      }
  #else
      if constexpr (StorePolicy == CachePolicy::FlatCache) {
        asm volatile("flat_store_b8 %0, %1, scope:SCOPE_SE" : : "v"(dst), "v"(val32) : "memory");
      } else if constexpr (StorePolicy == CachePolicy::DeviceScope) {
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

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return static_cast<type>(
        llvm_amdgcn_raw_buffer_load_b8(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    i32x4_t rsrc = make_buffer_resource(ptr, buf_size);
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b8(val, rsrc, offset, 0, aux);
  }
};

__device__ __forceinline__ void wait_on_vmem_loads([[maybe_unused]] int waits) {
#if defined(__gfx1201__) || defined(__gfx1250__)
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
#elif defined(__gfx90a__) || defined(__gfx942__) || \
      defined(__gfx950__) || defined(__gfx1100__)
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

__device__ __forceinline__ void wait_on_vmem_stores([[maybe_unused]] int waits) {
#if defined(__gfx1201__) || defined(__gfx1250__)
  switch (waits) {
    case 15: asm volatile("s_wait_storecnt 15" ::: "memory"); break;
    case 14: asm volatile("s_wait_storecnt 14" ::: "memory"); break;
    case 13: asm volatile("s_wait_storecnt 13" ::: "memory"); break;
    case 12: asm volatile("s_wait_storecnt 12" ::: "memory"); break;
    case 11: asm volatile("s_wait_storecnt 11" ::: "memory"); break;
    case 10: asm volatile("s_wait_storecnt 10" ::: "memory"); break;
    case 9:  asm volatile("s_wait_storecnt 9"  ::: "memory"); break;
    case 8:  asm volatile("s_wait_storecnt 8"  ::: "memory"); break;
    case 7:  asm volatile("s_wait_storecnt 7"  ::: "memory"); break;
    case 6:  asm volatile("s_wait_storecnt 6"  ::: "memory"); break;
    case 5:  asm volatile("s_wait_storecnt 5"  ::: "memory"); break;
    case 4:  asm volatile("s_wait_storecnt 4"  ::: "memory"); break;
    case 3:  asm volatile("s_wait_storecnt 3"  ::: "memory"); break;
    case 2:  asm volatile("s_wait_storecnt 2"  ::: "memory"); break;
    case 1:  asm volatile("s_wait_storecnt 1"  ::: "memory"); break;
    default: asm volatile("s_wait_storecnt 0"  ::: "memory"); break;
  }
#elif defined(__gfx90a__) || defined(__gfx942__) || \
      defined(__gfx950__) || defined(__gfx1100__)
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

__device__ __forceinline__ void wait_on_vmem([[maybe_unused]] int waits) {
#if defined(__gfx1201__) || defined(__gfx1250__)
  // GFX12 has no unified vmcnt; issue both load and store waits separately.
  wait_on_vmem_loads(waits);
  wait_on_vmem_stores(waits);
#elif defined(__gfx90a__) || defined(__gfx942__) || \
      defined(__gfx950__) || defined(__gfx1100__)
  wait_on_vmem_loads(waits);  // vmcnt covers both loads and stores on GFX9
#endif
}

// Waits for all in-flight vector memory operations and LDS operations to complete.
__device__ __forceinline__ void wait_on_vmem_and_lds([[maybe_unused]] int waits) {
#if defined(__gfx1201__) || defined(__gfx1250__)
  // GFX12 splits counters: loads, stores, and DS (LDS) are tracked separately.
  wait_on_vmem_loads(waits);
  wait_on_vmem_stores(waits);
  switch (waits) {
    case 15: asm volatile("s_wait_dscnt 15" ::: "memory"); break;
    case 14: asm volatile("s_wait_dscnt 14" ::: "memory"); break;
    case 13: asm volatile("s_wait_dscnt 13" ::: "memory"); break;
    case 12: asm volatile("s_wait_dscnt 12" ::: "memory"); break;
    case 11: asm volatile("s_wait_dscnt 11" ::: "memory"); break;
    case 10: asm volatile("s_wait_dscnt 10" ::: "memory"); break;
    case 9:  asm volatile("s_wait_dscnt 9"  ::: "memory"); break;
    case 8:  asm volatile("s_wait_dscnt 8"  ::: "memory"); break;
    case 7:  asm volatile("s_wait_dscnt 7"  ::: "memory"); break;
    case 6:  asm volatile("s_wait_dscnt 6"  ::: "memory"); break;
    case 5:  asm volatile("s_wait_dscnt 5"  ::: "memory"); break;
    case 4:  asm volatile("s_wait_dscnt 4"  ::: "memory"); break;
    case 3:  asm volatile("s_wait_dscnt 3"  ::: "memory"); break;
    case 2:  asm volatile("s_wait_dscnt 2"  ::: "memory"); break;
    case 1:  asm volatile("s_wait_dscnt 1"  ::: "memory"); break;
    default: asm volatile("s_wait_dscnt 0"  ::: "memory"); break;
  }
#elif defined(__gfx90a__) || defined(__gfx942__) || \
      defined(__gfx950__) || defined(__gfx1100__)
  // vmcnt covers both loads and stores; lgkmcnt covers LDS (DS) operations.
  switch (waits) {
    case 15: asm volatile("s_waitcnt vmcnt(15) lgkmcnt(15)" ::: "memory"); break;
    case 14: asm volatile("s_waitcnt vmcnt(14) lgkmcnt(14)" ::: "memory"); break;
    case 13: asm volatile("s_waitcnt vmcnt(13) lgkmcnt(13)" ::: "memory"); break;
    case 12: asm volatile("s_waitcnt vmcnt(12) lgkmcnt(12)" ::: "memory"); break;
    case 11: asm volatile("s_waitcnt vmcnt(11) lgkmcnt(11)" ::: "memory"); break;
    case 10: asm volatile("s_waitcnt vmcnt(10) lgkmcnt(10)" ::: "memory"); break;
    case 9:  asm volatile("s_waitcnt vmcnt(9)  lgkmcnt(9)"  ::: "memory"); break;
    case 8:  asm volatile("s_waitcnt vmcnt(8)  lgkmcnt(8)"  ::: "memory"); break;
    case 7:  asm volatile("s_waitcnt vmcnt(7)  lgkmcnt(7)"  ::: "memory"); break;
    case 6:  asm volatile("s_waitcnt vmcnt(6)  lgkmcnt(6)"  ::: "memory"); break;
    case 5:  asm volatile("s_waitcnt vmcnt(5)  lgkmcnt(5)"  ::: "memory"); break;
    case 4:  asm volatile("s_waitcnt vmcnt(4)  lgkmcnt(4)"  ::: "memory"); break;
    case 3:  asm volatile("s_waitcnt vmcnt(3)  lgkmcnt(3)"  ::: "memory"); break;
    case 2:  asm volatile("s_waitcnt vmcnt(2)  lgkmcnt(2)"  ::: "memory"); break;
    case 1:  asm volatile("s_waitcnt vmcnt(1)  lgkmcnt(1)"  ::: "memory"); break;
    default: asm volatile("s_waitcnt vmcnt(0)  lgkmcnt(0)"  ::: "memory"); break;
  }
#endif
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ASSEMBLY_HPP_
