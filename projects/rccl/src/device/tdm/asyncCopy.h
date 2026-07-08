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

#ifndef __ASYNCCOPY_H
#define __ASYNCCOPY_H

#include "tdm.h"

using CachePolicy = uint32_t;

enum struct MemScope : uint32_t {
    WGP = 0,    // Workgroup processor scope - warps running on the same WGP should be able to see the effect of the operation
    SE,         // Shader engine a.k.a cluster scope
    DEV,        // Device scope
    SYS,        // System scope
};

enum struct TemporalHint : uint32_t {
    RT = 0, // Regular temporal (nothing special)
    NT,     // Not temporal 
    HT,     // High temporal 
    LU,     // Last use  
    NT_RT,
    RT_NT,
    NT_HT,
};

__host__ __device__ constexpr CachePolicy createCachePolicy(TemporalHint temporal, MemScope scope) noexcept {
    return static_cast<CachePolicy>(scope) << 3 | static_cast<CachePolicy>(temporal);
}

static_assert(createCachePolicy(TemporalHint::RT, MemScope::WGP) == 0);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::WGP) == 1);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::WGP) == 2);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::WGP) == 3);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::WGP) == 4);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::WGP) == 5);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::WGP) == 6);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::SE) == 8);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::SE) == 9);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::SE) == 10);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::SE) == 11);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::SE) == 12);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::SE) == 13);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::SE) == 14);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::DEV) == 16);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::DEV) == 17);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::DEV) == 18);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::DEV) == 19);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::DEV) == 20);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::DEV) == 21);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::DEV) == 22);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::SYS) == 24);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::SYS) == 25);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::SYS) == 26);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::SYS) == 27);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::SYS) == 28);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::SYS) == 29);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::SYS) == 30);

constexpr CachePolicy DEFAULT_CACHE_POLICY = createCachePolicy(TemporalHint::RT, MemScope::SYS);

// Used for setting TDM descriptor fields and arguments to the async load/store builtins
using __rccl_int32x2 = int32_t __attribute__((__vector_size__(8)));
using __rccl_int32x4 = int32_t __attribute__((__vector_size__(16)));
using __rccl_int32x8 = int32_t __attribute__((__vector_size__(32)));

// We don't currently support passing in per-lane offsets for addresses, as the pointers for src and dst can
// be updated per lane, which is what RCCL already does.  Additionally, the API requires the offset to be a
// compile-time constant, so we always pass in a zero offset.
namespace {
  constexpr int32_t ZERO_OFFSET = 0;
}

// Waits until at most WAIT_CNT outstanding async-to/from-LDS transfers remain in flight.  The count
// is baked into the hardware instruction, so it must be a compile-time constant.
template<int WAIT_CNT = 0>
__device__ void asyncWait(){
  __builtin_amdgcn_s_wait_asynccnt(WAIT_CNT);
}

/* Async load/Store APIs */
// The async-to/from-LDS builtins move sizeof(T) bytes per lane, with the access width selected at
// compile time from the element size.  Each lane passes its own pointers to LDS and global memory.
//
// The hardware only provides b8/b32/b64/b128 variants -- there is no 16-bit per lane instruction.
template<typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ void asyncLoadToLDS(const T* src, T* dst){
  if constexpr (sizeof(T) == 1) {
    __builtin_amdgcn_global_load_async_to_lds_b8(
        (__attribute__((address_space(1))) char*)src,
        (__attribute__((address_space(3))) char*)dst, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 4) {
    __builtin_amdgcn_global_load_async_to_lds_b32(
        (__attribute__((address_space(1))) int32_t*)src,
        (__attribute__((address_space(3))) int32_t*)dst, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 8) {
    __builtin_amdgcn_global_load_async_to_lds_b64(
        (__attribute__((address_space(1))) __rccl_int32x2*)src,
        (__attribute__((address_space(3))) __rccl_int32x2*)dst, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 16) {
    __builtin_amdgcn_global_load_async_to_lds_b128(
        (__attribute__((address_space(1))) __rccl_int32x4*)src,
        (__attribute__((address_space(3))) __rccl_int32x4*)dst, ZERO_OFFSET, cp);
  } else {
    static_assert(sizeof(T) == 0, "asyncLoadToLDS supports only 1, 4, 8, or 16 byte element types");
  }
}

template<typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ void asyncStoreFromLDS(const T* src, T* dst){
  if constexpr (sizeof(T) == 1) {
    __builtin_amdgcn_global_store_async_from_lds_b8(
        (__attribute__((address_space(1))) char*)dst,
        (__attribute__((address_space(3))) char*)src, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 4) {
    __builtin_amdgcn_global_store_async_from_lds_b32(
        (__attribute__((address_space(1))) int32_t*)dst,
        (__attribute__((address_space(3))) int32_t*)src, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 8) {
    __builtin_amdgcn_global_store_async_from_lds_b64(
        (__attribute__((address_space(1))) __rccl_int32x2*)dst,
        (__attribute__((address_space(3))) __rccl_int32x2*)src, ZERO_OFFSET, cp);
  } else if constexpr (sizeof(T) == 16) {
    __builtin_amdgcn_global_store_async_from_lds_b128(
        (__attribute__((address_space(1))) __rccl_int32x4*)dst,
        (__attribute__((address_space(3))) __rccl_int32x4*)src, ZERO_OFFSET, cp);
  } else {
    static_assert(sizeof(T) == 0, "asyncStoreFromLDS supports only 1, 4, 8, or 16 byte element types");
  }
}


/* TDM APIs */
// Higher productivity async copy wrappers on top of TDM header, which is lower level.
// The TDM API does not have a function to set the transfer size, so we need to do it manually.
__device__ static void setTransferSize(gfx1250_TDM_GROUP1& group1, int numElements){
  group1.tensorDim0(numElements);
  group1.tensorDim0Stride(numElements);
  group1.tileDim0(numElements);
}

template<typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
struct AsyncDataCopier{
  size_t size_{0};

  __device__ void loadTile(const T* src, T* dst, size_t size){
    size_ = size;
    asyncLoadToLDS<T, cp>(src, dst);
  }

  __device__ void storeTile(T* dst){}

  template<int WAIT_CNT = 0>
  __device__ void waitTile();
};

// Warp-level data copier.  Moves whole 1D tiles in between global memory and LDS using the TDM.  Both pointers ideally should be a multiple of 128-byte aligned 
// for maximum performance.
template<typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
struct TileMover{
  gfx1250_TDM_GROUP0 group0;
  gfx1250_TDM_GROUP1 group1;
  __device__ TileMover(){
    constexpr int log2DataSize = __builtin_ctzll(sizeof(T));
    static_assert(log2DataSize <= 3, "Datatype must be 1, 2, 4, or 8 bytes in width"); // TODO(breslow): Replace with C++ concepts when we migrate to C++20
    group1.dataSize(log2DataSize);
  }

  // An entire warp makes this call.  The shmemPtr and srcPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) {
    group0.ldsAddr((uintptr_t)shmemPtr);
    group0.globalAddr((uintptr_t)srcPtr);
    setTransferSize(group1, numElementsToProcess);
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, __rccl_int32x4{}, __rccl_int32x4{}, __rccl_int32x8{}, cp);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void storeTile(T* dstPtr) {
    group0.globalAddr((uintptr_t)dstPtr);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, __rccl_int32x4{}, __rccl_int32x4{}, __rccl_int32x8{}, cp);
  }

  template<int WAIT_CNT = 0> // Needs to be a template parameter because the count is baked into the hardware instruction
  __device__ void waitTile(){
    __builtin_amdgcn_s_wait_tensorcnt(WAIT_CNT);
  }
};


#endif // __ASYNCCOPY_H
