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

enum struct SyncPolicy : uint32_t {
    Async,
    Sync,
};

constexpr SyncPolicy DEFAULT_SYNC_POLICY = SyncPolicy::Async;

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

// The builtins take an immediate byte offset that is folded into the memory instruction, so it must be a
// compile-time constant.  We always pass a zero immediate and instead apply the per-lane offset directly to
// the src/dst pointers, which lets every lane target a different chunk of the transfer.
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
// The async-to/from-LDS builtins move a single b8/b32/b64/b128 access per lane between global memory and LDS.
// A whole warp issues one instruction, so a warp moves (warpSize * accessWidth) bytes at a time, with each
// lane targeting its own slice via a per-lane pointer offset.  The hardware only provides b8/b32/b64/b128
// variants -- there is no 16-bit per lane instruction.
namespace async_detail {

// Direction of the transfer.  In both cases the first pointer is the global-memory side and the second is the
// LDS side, which keeps the per-width helpers below symmetric.
enum struct AsyncDir { Load, Store };

using __rccl_global_char     = __attribute__((address_space(1))) char;
using __rccl_lds_char        = __attribute__((address_space(3))) char;
using __rccl_global_int32    = __attribute__((address_space(1))) int32_t;
using __rccl_lds_int32       = __attribute__((address_space(3))) int32_t;
using __rccl_global_int32x2  = __attribute__((address_space(1))) __rccl_int32x2;
using __rccl_lds_int32x2     = __attribute__((address_space(3))) __rccl_int32x2;
using __rccl_global_int32x4  = __attribute__((address_space(1))) __rccl_int32x4;
using __rccl_lds_int32x4     = __attribute__((address_space(3))) __rccl_int32x4;

// One b8 access (1 byte) for a single lane.
template<AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB8(const uint8_t* global, uint8_t* lds){
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b8(
        (__rccl_global_char*)global, (__rccl_lds_char*)lds, ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b8(
        (__rccl_global_char*)global, (__rccl_lds_char*)lds, ZERO_OFFSET, cp);
  }
}

// One b32 access (4 bytes) for a single lane.
template<AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB32(const uint8_t* global, uint8_t* lds){
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b32(
        (__rccl_global_int32*)global, (__rccl_lds_int32*)lds, ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b32(
        (__rccl_global_int32*)global, (__rccl_lds_int32*)lds, ZERO_OFFSET, cp);
  }
}

// One b64 access (8 bytes) for a single lane.
template<AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB64(const uint8_t* global, uint8_t* lds){
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b64(
        (__rccl_global_int32x2*)global, (__rccl_lds_int32x2*)lds, ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b64(
        (__rccl_global_int32x2*)global, (__rccl_lds_int32x2*)lds, ZERO_OFFSET, cp);
  }
}

// One b128 access (16 bytes) for a single lane.
template<AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB128(const uint8_t* global, uint8_t* lds){
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b128(
        (__rccl_global_int32x4*)global, (__rccl_lds_int32x4*)lds, ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b128(
        (__rccl_global_int32x4*)global, (__rccl_lds_int32x4*)lds, ZERO_OFFSET, cp);
  }
}

// Bytes moved by a single lane in the widest (b128) access.
constexpr size_t BYTES_PER_LANE_B128 = 16;
// The hot loop must start on a 128-byte boundary, so the peel aligns the global pointer to this.
constexpr size_t NATURAL_ALIGNMENT_BYTES = 128;
// How many b128 iterations the hot loop unrolls at a time.
constexpr int    HOT_LOOP_UNROLL     = 4;

// Warp-cooperative byte copy between global memory and LDS.  The whole warp must call this with identical
// `global`, `lds`, and `sizeInBytes` arguments -- the per-lane work distribution is derived internally.
//
// The copy is split into three phases:
//   1. Peel the leading bytes so the global pointer reaches 128-byte (cache-line) alignment: first with b8
//      accesses up to a 16-byte boundary, then with b128 accesses up to the 128-byte boundary.
//   2. Run a hot, unrolled loop of full-warp b128 transfers (warpSize * 16 bytes per iteration), starting on
//      the 128-byte aligned boundary produced by phase 1.
//   3. Drain the tail (< warpSize * 16 bytes) with a descending b128 -> b64 -> b32 -> b8 cascade.
//
// Alignment is peeled against the global-memory pointer; the LDS staging buffer is assumed to share the same
// alignment (RCCL allocates it 128-byte aligned), so applying the same offset keeps the LDS side aligned too.
template<AsyncDir DIR, CachePolicy cp>
__device__ inline void warpAsyncCopy(const uint8_t* global, uint8_t* lds, size_t sizeInBytes){
  const unsigned lane      = __lane_id();
  const size_t   laneCount = (size_t)__builtin_amdgcn_wavefrontsize();
  const size_t   laneByte  = (size_t)lane * BYTES_PER_LANE_B128;

  size_t offset = 0;

  // Phase 1a: peel leading bytes until `global` is 16-byte aligned, one byte per lane.
  const size_t misalign = (uintptr_t)global & (BYTES_PER_LANE_B128 - 1);
  if (misalign != 0) {
    const size_t head = (BYTES_PER_LANE_B128 - misalign) < sizeInBytes
                            ? (BYTES_PER_LANE_B128 - misalign)
                            : sizeInBytes;
    if (lane < head) {
      asyncCopyB8<DIR, cp>(global + lane, lds + lane);
    }
    offset += head;
  }

  // Phase 1b: peel whole 16-byte chunks until `global + offset` reaches 128-byte alignment, one b128 per
  // leading lane.  `global + offset` is already 16-byte aligned, so the number of bytes left to the next
  // 128-byte boundary is a multiple of 16 (at most 112 -> 7 chunks, well within a warp).
  {
    const size_t align128    = ((uintptr_t)global + offset) & (NATURAL_ALIGNMENT_BYTES - 1);
    const size_t bytesToLine = (NATURAL_ALIGNMENT_BYTES - align128) & (NATURAL_ALIGNMENT_BYTES - 1);
    size_t       peelChunks  = bytesToLine / BYTES_PER_LANE_B128;
    const size_t availChunks = (sizeInBytes - offset) / BYTES_PER_LANE_B128;
    if (peelChunks > availChunks) peelChunks = availChunks;
    if (lane < peelChunks) {
      const size_t laneOff = offset + laneByte;
      asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
    }
    offset += peelChunks * BYTES_PER_LANE_B128;
  }

  // Phase 2: hot unrolled loop of full-warp b128 transfers, now starting on a 128-byte boundary.
  const size_t bytesPerIter = laneCount * BYTES_PER_LANE_B128;
  size_t remaining          = sizeInBytes - offset;
  #pragma unroll HOT_LOOP_UNROLL
  while (remaining >= bytesPerIter) {
    const size_t laneOff = offset + laneByte;
    asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
    offset    += bytesPerIter;
    remaining -= bytesPerIter;
  }

  // Phase 3a: the remaining full 16-byte chunks, one b128 per leading lane.  `remaining < warpSize * 16`, so
  // `full128 < warpSize` and every chunk maps to a distinct lane.
  const size_t full128 = remaining / BYTES_PER_LANE_B128;
  if (lane < full128) {
    const size_t laneOff = offset + laneByte;
    asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
  }
  offset    += full128 * BYTES_PER_LANE_B128;
  remaining -= full128 * BYTES_PER_LANE_B128;  // now < 16 bytes

  // Phase 3b: 0..15 trailing bytes.  `offset` and `remaining` are uniform across the warp here, so the
  // descending cascade below is taken identically by every lane.  Because `offset` is 16-byte aligned, the
  // b64 (8-byte) and b32 (4-byte) accesses are naturally aligned.
  if (remaining >= 8) {
    if (lane == 0) {
      asyncCopyB64<DIR, cp>(global + offset, lds + offset);
    }
    offset    += 8;
    remaining -= 8;
  }
  if (remaining >= 4) {
    if (lane == 0) {
      asyncCopyB32<DIR, cp>(global + offset, lds + offset);
    }
    offset    += 4;
    remaining -= 4;
  }
  // Final 1..3 bytes, one b8 per lane.
  if (lane < remaining) {
    asyncCopyB8<DIR, cp>(global + offset + lane, lds + offset + lane);
  }
}

} // namespace async_detail

// Warp-level async copy from global memory into LDS.  The entire warp calls this with the same arguments.
template<SyncPolicy sp = SyncPolicy::Async, CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes){
  async_detail::warpAsyncCopy<async_detail::AsyncDir::Load, cp>(globalSrc, ldsDst, sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) {
    asyncWait<0>();
  }
}

// Warp-level async copy from LDS into global memory.  The entire warp calls this with the same arguments.
template<SyncPolicy sp = SyncPolicy::Async, CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ void asyncStoreFromLDS(const uint8_t* ldsSrc, uint8_t* globalDst, size_t sizeInBytes){
  async_detail::warpAsyncCopy<async_detail::AsyncDir::Store, cp>(globalDst, ldsSrc, sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) {
    asyncWait<0>();
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

template<CachePolicy cp = DEFAULT_CACHE_POLICY>
struct AsyncDataCopier{
  size_t size_{0};

  __device__ void loadTile(const uint8_t* src, uint8_t* dst, size_t size){
    size_ = size;
    asyncLoadToLDS<cp>(src, dst, size);
  }

  __device__ void storeTile(uint8_t* dst){
    asyncStoreFromLDS<cp>(dst, src);
  }

  template<int WAIT_CNT = 0>
  __device__ void waitTile(){
    asyncWait<WAIT_CNT>();
  }
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
