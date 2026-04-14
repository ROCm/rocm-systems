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

__device__ __forceinline__ int uncached_load_ubyte(uint8_t* src) {
  int ret;
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile(
      "global_load_ubyte %0 %1 off glc slc \n"
      "s_waitcnt vmcnt(0)"
      : "=v"(ret)
      : "v"(src));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile(
      "global_load_ubyte %0 %1 off sc0 sc1 \n"
      "s_waitcnt vmcnt(0)"
      : "=v"(ret)
      : "v"(src));
#endif
#if defined(__gfx1201__)
  asm volatile(
      "global_load_u8 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(ret)
      : "v"(src));
#endif
  return ret;
}

__device__ __forceinline__ void refresh_volatile_sbyte(volatile int *assigned_value,
                                                       volatile char *read_value) {
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile(
    "global_load_sbyte %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile(
    "global_load_sbyte %0 %1 off sc0 sc1\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value));
#endif
#if defined(__gfx1201__)
  asm volatile(
      "global_load_i8 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(*assigned_value)
      : "v"(read_value));
#endif
}

__device__ __forceinline__ void refresh_volatile_dwordx2(volatile uint64_t *assigned_value,
                                                         volatile uint64_t *read_value) {
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile(
    "global_load_dwordx2 %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile(
    "global_load_dwordx2 %0 %1 off sc0 sc1\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(*assigned_value)
    : "v"(read_value));
#endif
 #if defined(__gfx1201__)
  asm volatile(
      "global_load_b64 %0 %1 off scope:SCOPE_SYS \n"
      "s_wait_loadcnt 0x0"
      : "=v"(*assigned_value)
      : "v"(read_value));
#endif
}

/* Ignore the warning about deprecated volatile.
 * The only usage of volatile is to force the compiler to generate
 * the assembly instruction. If volatile is omitted, the compiler
 * will NOT generate the non-temporal load or the waitcnt.
 */
// clang-format off
NOWARN(-Wdeprecated-volatile,
  template <typename T> __device__ __forceinline__ T uncached_load(T* src) {
    T ret;
    switch (sizeof(T)) {
      case 4:
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
        asm volatile(
            "global_load_dword %0 %1 off glc slc \n"
            "s_waitcnt vmcnt(0)"
            : "=v"(ret)
            : "v"(src));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
        asm volatile(
            "global_load_dword %0 %1 off sc0 sc1 \n"
            "s_waitcnt vmcnt(0)"
            : "=v"(ret)
            : "v"(src));
#endif
#if defined(__gfx1201__)
        asm volatile(
            "global_load_b32 %0 %1 off scope:SCOPE_SYS \n"
            "s_wait_loadcnt 0x0"
            : "=v"(ret)
            : "v"(src));
#endif
        break;
      case 8:
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
        asm volatile(
            "global_load_dwordx2 %0 %1 off glc slc \n"
            "s_waitcnt vmcnt(0)"
            : "=v"(ret)
            : "v"(src));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
        asm volatile(
            "global_load_dwordx2 %0 %1 off sc0 sc1 \n"
            "s_waitcnt vmcnt(0)"
            : "=v"(ret)
            : "v"(src));
#endif
#if defined(__gfx1201__)
        asm volatile(
            "global_load_b64 %0 %1 off scope:SCOPE_SYS \n"
            "s_wait_loadcnt 0x0"
            : "=v"(ret)
            : "v"(src));
#endif
        break;
      default:
        break;
    }
    return ret;
  }
)
// clang-format on

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

__device__ __forceinline__ void store_asm(uint8_t* val, uint8_t* dst,
                                          int size) {
  switch (size) {
    case 2: {
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__)
      int16_t val16{*(reinterpret_cast<int16_t*>(val))};
      asm volatile("flat_store_short %0 %1 glc slc" : : "v"(dst), "v"(val16));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      int16_t val16{*(reinterpret_cast<int16_t*>(val))};
      asm volatile("flat_store_short %0 %1 sc0 sc1" : : "v"(dst), "v"(val16));
#endif
#if defined(__gfx1100__)
      int32_t val32{*(reinterpret_cast<int32_t*>(val))};
      asm volatile("flat_store_short %0 %1 glc slc" : : "v"(dst), "v"(val32));
#endif
#if defined(__gfx1201__)
      int32_t val32{*(reinterpret_cast<int32_t*>(val))};
      asm volatile("flat_store_b16 %0 %1 scope:SCOPE_SYS" : : "v"(dst), "v"(val32));
#endif
      break;
    }
    case 4: {
      int32_t val32{*(reinterpret_cast<int32_t*>(val))};
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dword %0 %1 glc slc" : : "v"(dst), "v"(val32));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dword %0 %1 sc0 sc1" : : "v"(dst), "v"(val32));
#endif
#if defined(__gfx1201__)
      asm volatile("flat_store_b32 %0 %1 scope:SCOPE_SYS" : : "v"(dst), "v"(val32));
#endif
      break;
    }
    case 8: {
      int64_t val64{*(reinterpret_cast<int64_t*>(val))};
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dwordx2 %0 %1 glc slc" : : "v"(dst), "v"(val64));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dwordx2 %0 %1 sc0 sc1" : : "v"(dst), "v"(val64));
#endif
#if defined(__gfx1201__)
      asm volatile("flat_store_b64 %0 %1 scope:SCOPE_SYS" : : "v"(dst), "v"(val64));
#endif
      break;
    }
    default:
      break;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ASSEMBLY_HPP_
