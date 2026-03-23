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

__device__ __forceinline__ void refresh_volatile_sbyte([[maybe_unused]] volatile int *assigned_value,
                                                       [[maybe_unused]] volatile char *read_value) {
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

__device__ __forceinline__ void refresh_volatile_dwordx2([[maybe_unused]] volatile uint64_t *assigned_value,
                                                         [[maybe_unused]] volatile uint64_t *read_value) {
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
  template <typename T> __device__ __forceinline__ T uncached_load([[maybe_unused]] T* src) {
    T ret{};
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

// 128-bit (16-byte) vector type for use with flat_store_dwordx4 and buffer intrinsics
using i32x4 = int32_t __attribute__((ext_vector_type(4)));

struct buffer_resource {
  uint64_t ptr;
  uint32_t range;
  uint32_t config;
};

template <class To, class From>
__device__ To bit_cast_fallback(const From& src) noexcept {
  To dst;
  __builtin_memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
__device__ __forceinline__ buffer_resource make_buffer_resource(T* ptr, uint32_t buffer_size) {
  // ref: https://github.com/HazyResearch/HipKittens
  constexpr uint32_t config = 0x00020000;
  uint64_t as_u64 = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  return {as_u64, buffer_size, config};
}

// __device__ __forceinline__ buffer_resource& move_buffer_resource(buffer_resource& br, size_t size) {
//   br.ptr += static_cast<uint64_t>(size);
//   // br.range -= static_cast<uint32_t>(size);
//   return br;
// }

#if defined(__gfx942__) || defined(__gfx950__)

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset,
                                                         uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc,
                                                   uint32_t voffset, uint32_t soffset,
                                                   uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

#endif  // __gfx942__ || __gfx950__

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
      [[maybe_unused]] int32_t val32{*(reinterpret_cast<int32_t*>(val))};
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
      [[maybe_unused]] int64_t val64{*(reinterpret_cast<int64_t*>(val))};
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
    case 16: {
      __uint128_t val128{*(reinterpret_cast<__uint128_t*>(val))};
#if defined(__gfx906__)
#endif
#if defined(__gfx908__)
#endif
#if defined(__gfx90a__) || defined(__gfx1100__)
      asm volatile("flat_store_dwordx4 %0 %1 glc slc" : : "v"(dst), "v"(val128));
#endif
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("flat_store_dwordx4 %0 %1 sc0 sc1" : : "v"(dst), "v"(val128));
#endif
#if defined(__gfx1201__)
      asm volatile("flat_store_b128 %0 %1 scope:SCOPE_SYS" : : "v"(dst), "v"(val128));
#endif
      break;
    }

//     case 16: {
//       // 128-byte transfer: 8 x 16-byte iterations
// #if defined(__gfx906__)
// #endif
// #if defined(__gfx908__)
// #endif
// #if defined(__gfx90a__) || defined(__gfx1100__)
//       #pragma unroll
//       for (int i = 0; i < 1; i++) {
//         __uint128_t chunk = *(reinterpret_cast<__uint128_t*>(val) + i);
//         uint8_t* dst_i = dst + i * sizeof(__uint128_t);
//         asm volatile("flat_store_dwordx4 %0 %1 glc slc" : : "v"(dst_i), "v"(chunk));
//       }
// #endif
// #if defined(__gfx942__) || defined(__gfx950__)
//       {
//         constexpr int NUM_REG = 1;
//         __uint128_t regs[NUM_REG];
//         buffer_resource br_val = make_buffer_resource(val, size);
//         buffer_resource br_dst = make_buffer_resource(dst, size);

//         #pragma unroll
//         for (int i = 0; i < NUM_REG; i++) {
//           regs[i] = llvm_amdgcn_raw_buffer_load_b128(
//               *reinterpret_cast<i32x4*>(&br_val),
//               static_cast<uint32_t>(i) * 16u, 0u, 0b10001u);
//         }
//         #pragma unroll
//         for (int i = 0; i < NUM_REG; i++) {
//           llvm_amdgcn_raw_buffer_store_b128(
//               regs[i],
//               *reinterpret_cast<i32x4*>(&br_dst),
//               static_cast<uint32_t>(i) * 16u, 0u, 0b10001u);
//         }
//       }
// #endif
// #if defined(__gfx1201__)
//       #pragma unroll
//       for (int i = 0; i < 1; i++) {
//         __uint128_t chunk = *(reinterpret_cast<__uint128_t*>(val) + i);
//         uint8_t* dst_i = dst + i * sizeof(__uint128_t);
//         asm volatile("flat_store_b128 %0 %1 scope:SCOPE_SYS" : : "v"(dst_i), "v"(chunk));
//       }
// #endif
//       break;
//     }
    default:
      break;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ASSEMBLY_HPP_
