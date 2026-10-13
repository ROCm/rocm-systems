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

#include "cachePolicy.h"
#include "syncPolicy.h"   // shared SyncPolicy encoding (same as tdmCopy.h)

#ifndef __has_include
#define __has_include(x) 0
#endif
// The TileMover wrappers at the bottom of this header are built on the gfx1250 TDM
// descriptor types, so they need the SDK's descriptor header. Use the same
// host-evaluable detection as tdmCopy.h, guarded so the two can be included together.
// Unlike tdmCopy.h this keys the include on header availability rather than
// TDM_SUPPORTED, because the descriptor types must be nameable in every compilation
// pass that parses those wrappers, not just a gfx1250 device pass.
// Deliberately NOT gated on __gfx1250__: arch macros are undefined in the host pass,
// so adding one here would force this to 0 for every host caller and make the
// host-side capability queries report unsupported on real gfx1250 hardware.
#ifndef TDM_TOOLCHAIN_AVAILABLE
#if __has_include(<hip/amd_detail/amd_gfx1250_TDM.h>)
#define TDM_TOOLCHAIN_AVAILABLE 1
#else
#define TDM_TOOLCHAIN_AVAILABLE 0
#endif
#endif
#if TDM_TOOLCHAIN_AVAILABLE
#include <hip/amd_detail/amd_gfx1250_TDM.h>   // gfx1250_TDM_GROUP0/1 descriptor types
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// Capability macros for the two builtin families this header wraps. Unlike
// TDM_TOOLCHAIN_AVAILABLE above, these are device-pass only: each is 1 only when
// compiling for an arch that has the instructions AND whose compiler exposes the
// builtin. They are defined here, ahead of the wrappers below, because a
// multi-arch build (e.g. --amdgpu_targets gfx950 gfx1250) compiles this header
// once per target, and naming a gfx1250-only builtin on any other device pass is
// a hard error -- so every body that emits one has to be gated on these.
#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
                                  /* extend: || (defined(__gfxNNNN__) && ...) */
#define ASYNC_COPY_SUPPORTED 1
#else
#define ASYNC_COPY_SUPPORTED 0
#endif

// Same condition as tdmCopy.h's copy, kept #ifndef-guarded so the two headers can
// be included together in either order.
#ifndef TDM_SUPPORTED
#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && TDM_TOOLCHAIN_AVAILABLE
                                  /* extend: || (defined(__gfxNNNN__) && ...) */
#define TDM_SUPPORTED 1
#else
#define TDM_SUPPORTED 0
#endif
#endif

// ============================================================================
//  DROP-IN PARITY API (mirrors tdmCopy.h)
// ----------------------------------------------------------------------------
// The entry points below expose the EXACT same public surface as tdm/tdmCopy.h
// -- same names, signatures, defaults, and memcpy-style semantics -- so the two
// libraries are interchangeable. The only difference is internal: this
// implementation stages HBM->LDS->HBM through the async-to/from-LDS builtins
// (see warpAsyncCopy above) instead of the tensor-data-mover load/store
// instructions.
//
// AVAILABILITY: the async-to/from-LDS builtins are a gfx1250 feature. As with
// tdmCopy.h, ASYNC_COPY_SUPPORTED is 1 only for a device pass on a capable arch
// whose compiler exposes the builtin; otherwise each entry point is `= delete`d,
// so including the header is fine but CALLING one on an unsupported target is a
// compile-time error at the call site. async::IsTdmCopySupported() is always
// callable (host and device) and is the runtime/host-side guard.
// ============================================================================
// The pair is defined here, ahead of the first declaration that needs it, and applies to every
// namespace-scope entry point in this header -- the one-way asyncLoadToLDS()/asyncStoreFromLDS()
// primitives and asyncWait() as well as the async:: parity API.
#if ASYNC_COPY_SUPPORTED
#define ASYNC_API inline
#define ASYNC_DELETED
#else
#define ASYNC_API
#define ASYNC_DELETED = delete
#endif

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
template <int WAIT_CNT = 0>
__device__ ASYNC_API void asyncWait() ASYNC_DELETED;

/* Async load/Store APIs */
// The async-to/from-LDS builtins move a single b8/b32/b64/b128 access per lane between global memory and LDS.
// A whole warp issues one instruction, so a warp moves (warpSize * accessWidth) bytes at a time, with each
// lane targeting its own slice via a per-lane pointer offset.  The hardware only provides b8/b32/b64/b128
// variants -- there is no 16-bit per lane instruction.

// Warp-level async copy from global memory into LDS.  The entire warp calls this with the same arguments.
// Set Aligned=true only when globalSrc and ldsDst are known to be 128-byte aligned (skips the peel).
template <SyncPolicy sp = SyncPolicy::Async, CachePolicy cp = DEFAULT_CACHE_POLICY, bool Aligned = false>
__device__ ASYNC_API void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes) ASYNC_DELETED;

// Warp-level async copy from LDS into global memory.  The entire warp calls this with the same arguments.
// Set Aligned=true only when globalDst and ldsSrc are known to be 128-byte aligned (skips the peel).
template <SyncPolicy sp = SyncPolicy::Async, CachePolicy cp = DEFAULT_CACHE_POLICY, bool Aligned = false>
__device__ ASYNC_API void asyncStoreFromLDS(const uint8_t* ldsSrc, uint8_t* globalDst,
                                            size_t sizeInBytes) ASYNC_DELETED;

// ============================================================================
//  IMPLEMENTATION of the three primitives declared above
// ----------------------------------------------------------------------------
// Every body below names an async builtin, either directly or through a helper that does. Those
// names are non-dependent, so a device pass for an arch that does not expose them rejects the body
// when this header is PARSED -- being a template does not defer the lookup to instantiation. Since a
// multi-arch build (e.g. --amdgpu_targets gfx950 gfx1250) parses the header once per target, the
// implementation is compiled only on a capable pass; elsewhere the declarations above stand deleted,
// so a caller that slipped past its own guard fails at the call site instead of silently skipping
// the transfer. The async_detail helpers are internal, so they simply do not exist on those passes
// (same treatment as tdm::detail in tdmCopy.h) -- nothing outside this header can name them.
// ============================================================================
#if ASYNC_COPY_SUPPORTED

namespace async_detail {

// Direction of the transfer.  In both cases the first pointer is the global-memory side and the second is the
// LDS side, which keeps the per-width helpers below symmetric.
enum struct AsyncDir {
  Load,
  Store
};

using __rccl_global_char = __attribute__((address_space(1))) char;
using __rccl_lds_char = __attribute__((address_space(3))) char;
using __rccl_global_int32x4 = __attribute__((address_space(1))) __rccl_int32x4;
using __rccl_lds_int32x4 = __attribute__((address_space(3))) __rccl_int32x4;

// A generic-to-LDS pointer cast null-checks its operand, which costs a 64-bit compare and select per access. Every
// pointer passed here is into LDS, whose generic addresses carry the LDS offset in their low 32 bits.
template <typename T>
__device__ inline T* toLds(const void* p) {
  return (T*)static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

// One b8 access (1 byte) for a single lane.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB8(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b8((__rccl_global_char*)global, toLds<__rccl_lds_char>(lds), ZERO_OFFSET,
                                                 cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b8((__rccl_global_char*)global, toLds<__rccl_lds_char>(lds),
                                                    ZERO_OFFSET, cp);
  }
}

// One b128 access (16 bytes) for a single lane.  OFF is the instruction's immediate offset, which the hardware adds
// to BOTH the global and the LDS address, so an unrolled run can share a single pair of per-lane base pointers.
template <AsyncDir DIR, CachePolicy cp, int32_t OFF = ZERO_OFFSET>
__device__ inline void asyncCopyB128(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b128((__rccl_global_int32x4*)global, toLds<__rccl_lds_int32x4>(lds), OFF,
                                                   cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b128((__rccl_global_int32x4*)global, toLds<__rccl_lds_int32x4>(lds),
                                                      OFF, cp);
  }
}

// Every argument of the warp-level entry points is warp-uniform by contract, but the compiler cannot prove it when
// the caller derives it from threadIdx (e.g. a per-warp offset). Without that proof every loop bound, switch and
// wait count downstream becomes a per-lane value with EXEC-masked control flow. readfirstlane restores it.
__device__ inline uint32_t uniform(uint32_t v) {
  return __builtin_amdgcn_readfirstlane(v);
}
__device__ inline uint64_t uniform(uint64_t v) {
  const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
  const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v >> 32));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
template <typename T>
__device__ inline T* uniform(T* p) {
  return reinterpret_cast<T*>(uniform(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))));
}

// Bytes moved by a single lane in the widest (b128) access.
constexpr uint32_t BYTES_PER_LANE_B128 = 16;
// Bytes moved by one full-warp b128 instruction. gfx1250 is wave32-only; the unrolled runs below bake multiples of
// this into immediate offsets.
constexpr uint32_t WARP_BYTES_B128 = 32 * BYTES_PER_LANE_B128;
// The hot loop starts on a 128-byte boundary of its alignment anchor.
constexpr size_t NATURAL_ALIGNMENT_BYTES = 128;
// Longest straight-line run of b128 instructions: the hot loop's unroll factor and the largest pipeline chunk.
constexpr uint32_t MAX_RUN = 8;

// N back-to-back full-warp b128 transfers from the per-lane pointers `g`/`l`, as straight-line code: every access
// carries a compile-time immediate offset, so a run needs no per-access loop, branch or address math.
template <AsyncDir DIR, CachePolicy cp, uint32_t N, uint32_t I = 0>
__device__ inline void issueRunStatic(const uint8_t* g, uint8_t* l) {
  if constexpr (I < N) {
    asyncCopyB128<DIR, cp, static_cast<int32_t>(I * WARP_BYTES_B128)>(g, l);
    issueRunStatic<DIR, cp, N, I + 1>(g, l);
  }
}

// `count` (0..2*MAX_RUN-1) full-warp b128 transfers with a runtime count, advancing `g`/`l` past them. This is
// Duff's device recast for the GPU: AMDGPU has no jump tables, and the structurizer turns a fall-through switch
// into a chain of predicated blocks around every access, so the count is instead decomposed into power-of-two
// straight-line runs -- one uniform scalar branch per bit.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void issueRunB128(const uint8_t*& g, uint8_t*& l, uint32_t count) {
  if (count & 8) {
    issueRunStatic<DIR, cp, 8>(g, l);
    g += 8 * WARP_BYTES_B128;
    l += 8 * WARP_BYTES_B128;
  }
  if (count & 4) {
    issueRunStatic<DIR, cp, 4>(g, l);
    g += 4 * WARP_BYTES_B128;
    l += 4 * WARP_BYTES_B128;
  }
  if (count & 2) {
    issueRunStatic<DIR, cp, 2>(g, l);
    g += 2 * WARP_BYTES_B128;
    l += 2 * WARP_BYTES_B128;
  }
  if (count & 1) {
    issueRunStatic<DIR, cp, 1>(g, l);
    g += WARP_BYTES_B128;
    l += WARP_BYTES_B128;
  }
}
static_assert(MAX_RUN == 8, "issueRunB128 decomposes counts below 2 * MAX_RUN");

// Warp-cooperative byte copy between global memory and LDS.  The whole warp must call this with identical
// `global`, `lds`, and `sizeInBytes` arguments -- the per-lane work distribution is derived internally.
//
// The copy is split into three phases:
//   1. Peel the leading bytes so the alignment ANCHOR (see below) reaches a 128-byte boundary: one b8 instruction
//      up to the 16-byte boundary, then one partial b128 instruction up to the 128-byte boundary.
//   2. Stream full-warp b128 instructions (warpSize * 16 bytes each), MAX_RUN per iteration with immediate
//      offsets; the sub-MAX_RUN remainder is dispatched up front through issueRunB128 (Duff's device).
//   3. Drain the tail (< warpSize * 16 bytes) with one partial b128 instruction and one b8 instruction.
//
// Choice of anchor.  Both legs work at any alignment (every async width accepts unaligned global and LDS
// addresses), but throughput depends on which side is misaligned when the two pointers do not share their
// sub-16-byte offset.  Measured on gfx1250: a b128 STORE wants its global side aligned regardless; a b128 LOAD
// is nearly free with a 4/8/12-byte LDS misalignment but ~6x slower with an odd one, where a misaligned global
// address costs half as much instead. So loads anchor on LDS when the skew is not a multiple of 4 and on global
// otherwise; stores always anchor on global.
//
// `Aligned` fast path: when the caller has already proven that both `global` and `lds` start on a 128-byte
// boundary, the phase-1 peel is provably a no-op.  Compiling it out removes the peel branches and lets the hot
// loop start at offset 0.  Callers must only pass Aligned=true after checking the runtime pointers.
template <AsyncDir DIR, CachePolicy cp, bool Aligned = false>
__device__ inline void warpAsyncCopy(const uint8_t* global, uint8_t* lds, size_t sizeInBytes) {
  global = uniform(global);
  lds = uniform(lds);
  sizeInBytes = uniform(static_cast<uint64_t>(sizeInBytes));
  const uint32_t lane = __lane_id();
  size_t offset = 0;

  if constexpr (!Aligned) {
    const uintptr_t skew = ((uintptr_t)global - (uintptr_t)lds) & (BYTES_PER_LANE_B128 - 1);
    const uintptr_t anchor = (DIR == AsyncDir::Load && (skew & 3)) ? (uintptr_t)lds : (uintptr_t)global;

    // Phase 1a: bytes up to the anchor's next 16-byte boundary, one byte per lane.
    size_t head = (BYTES_PER_LANE_B128 - (anchor & (BYTES_PER_LANE_B128 - 1))) & (BYTES_PER_LANE_B128 - 1);
    if (head > sizeInBytes) head = sizeInBytes;
    if (lane < head) asyncCopyB8<DIR, cp>(global + lane, lds + lane);
    offset = head;

    // Phase 1b: whole 16-byte chunks up to the anchor's next 128-byte boundary (at most 7), one per lane.
    size_t peelChunks = ((NATURAL_ALIGNMENT_BYTES - ((anchor + offset) & (NATURAL_ALIGNMENT_BYTES - 1))) &
                         (NATURAL_ALIGNMENT_BYTES - 1)) /
                        BYTES_PER_LANE_B128;
    const size_t availChunks = (sizeInBytes - offset) / BYTES_PER_LANE_B128;
    if (peelChunks > availChunks) peelChunks = availChunks;
    if (lane < peelChunks) {
      const size_t laneOff = offset + lane * BYTES_PER_LANE_B128;
      asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
    }
    offset += peelChunks * BYTES_PER_LANE_B128;
  }

  // Phase 2: full-warp b128 instructions.  Remainder run first (Duff's device), then whole MAX_RUN runs.
  const size_t laneByte = (size_t)lane * BYTES_PER_LANE_B128;
  const uint8_t* g = global + offset + laneByte;
  uint8_t* l = lds + offset + laneByte;
  const size_t fullInstrs = (sizeInBytes - offset) / WARP_BYTES_B128;
  issueRunB128<DIR, cp>(g, l, static_cast<uint32_t>(fullInstrs % MAX_RUN));
  for (size_t runs = fullInstrs / MAX_RUN; runs != 0; --runs) {
    issueRunStatic<DIR, cp, MAX_RUN>(g, l);
    g += MAX_RUN * WARP_BYTES_B128;
    l += MAX_RUN * WARP_BYTES_B128;
  }
  offset += fullInstrs * WARP_BYTES_B128;

  // Phase 3a: the remaining whole 16-byte chunks (< warpSize of them), one per leading lane.
  const size_t tailChunks = (sizeInBytes - offset) / BYTES_PER_LANE_B128;
  if (lane < tailChunks) asyncCopyB128<DIR, cp>(g, l);
  offset += tailChunks * BYTES_PER_LANE_B128;

  // Phase 3b: the final 0..15 bytes, one byte per lane.
  if (lane < sizeInBytes - offset) asyncCopyB8<DIR, cp>(global + offset + lane, lds + offset + lane);
}

// Wait until at most N async ops are outstanding.  ASYNCcnt counts loads and stores together, and they retire in
// issue order among loads and among stores but not relative to each other, so this proves an op complete only if
// at least N ops of its own kind were issued after it.  Clamping to the 6-bit field only makes the wait stricter.
template <uint32_t N>
__device__ inline void asyncWaitAtMost() {
  __builtin_amdgcn_s_wait_asynccnt(N > 63 ? 63 : N);
}

template <uint32_t I, uint32_t N, typename F>
__device__ inline void staticFor(F&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<uint32_t, I>{});
    staticFor<I + 1, N>(f);
  }
}

} // namespace async_detail

template <int WAIT_CNT>
__device__ inline void asyncWait() {
  __builtin_amdgcn_s_wait_asynccnt(WAIT_CNT);
}

template <SyncPolicy sp, CachePolicy cp, bool Aligned>
__device__ inline void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes) {
  async_detail::warpAsyncCopy<async_detail::AsyncDir::Load, cp, Aligned>(globalSrc, ldsDst, sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) {
    asyncWait<0>();
  }
}

template <SyncPolicy sp, CachePolicy cp, bool Aligned>
__device__ inline void asyncStoreFromLDS(const uint8_t* ldsSrc, uint8_t* globalDst, size_t sizeInBytes) {
  // warpAsyncCopy shares one non-const `lds` parameter across load and store; the store path only reads from
  // it, so dropping const here is safe.
  async_detail::warpAsyncCopy<async_detail::AsyncDir::Store, cp, Aligned>(globalDst, const_cast<uint8_t*>(ldsSrc),
                                                                          sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) {
    asyncWait<0>();
  }
}

#endif // ASYNC_COPY_SUPPORTED

/* TDM APIs */
// setTransferSize and TileMover name the TDM descriptor types directly, so they only
// exist when the SDK ships the descriptor header. An SDK without it leaves them
// undeclared, making any use a compile error rather than a silent no-op copy;
// AsyncDataCopier below needs no descriptor and stays available either way.
#if TDM_TOOLCHAIN_AVAILABLE
// Higher productivity async copy wrappers on top of TDM header, which is lower level.
// The TDM API does not have a function to set the transfer size, so we need to do it manually.
__device__ static void setTransferSize(gfx1250_TDM_GROUP1& group1, int numElements) {
  group1.tensorDim0(numElements);
  group1.tensorDim0Stride(numElements);
  group1.tileDim0(numElements);
}
#endif // TDM_TOOLCHAIN_AVAILABLE

// Warp-level tile copier built on top of the async-to/from-LDS builtins.  Exposes the same interface as
// TileMover so the two can be used interchangeably as the TileMoverType of a copy kernel.  An entire warp
// makes each call, and the shmem/global pointers must be uniform across the warp (no per-lane offsets).
template <typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
struct AsyncDataCopier {
  size_t sizeInBytes_{0};
  T* shmemPtr_{nullptr};

// The bodies below issue the async-to/from-LDS builtins, so on a device pass for an
// arch without them the members are deleted rather than left empty: a caller that
// slipped through gets a compile error at the call site instead of a silent no-op copy.
#if ASYNC_COPY_SUPPORTED
  // An entire warp makes this call.  The shmemPtr and srcPtr should be the same across all lanes in the warp.
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) {
    sizeInBytes_ = static_cast<size_t>(numElementsToProcess) * sizeof(T);
    shmemPtr_ = shmemPtr;
    asyncLoadToLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(srcPtr),
                                          reinterpret_cast<uint8_t*>(shmemPtr), sizeInBytes_);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.
  __device__ void storeTile(T* dstPtr) {
    asyncStoreFromLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(shmemPtr_),
                                             reinterpret_cast<uint8_t*>(dstPtr), sizeInBytes_);
  }

  template <int WAIT_CNT = 0>
  __device__ void waitTile() {
    asyncWait<WAIT_CNT>();
  }
#else
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) = delete;
  __device__ void storeTile(T* dstPtr) = delete;
  template <int WAIT_CNT = 0>
  __device__ void waitTile() = delete;
#endif
};

#if TDM_TOOLCHAIN_AVAILABLE
// Warp-level data copier.  Moves whole 1D tiles in between global memory and LDS using the TDM.  Both pointers ideally should be a multiple of 128-byte aligned
// for maximum performance.
template <typename T, CachePolicy cp = DEFAULT_CACHE_POLICY>
struct TileMover {
  gfx1250_TDM_GROUP0 group0;
  gfx1250_TDM_GROUP1 group1;

// The descriptor members above only need the SDK header, but the bodies below emit the
// TDM instructions, so they are gated on the arch as well and deleted otherwise -- same
// reasoning as AsyncDataCopier. The enclosing TDM_TOOLCHAIN_AVAILABLE guard keeps this
// type nameable in the host pass, where TDM_SUPPORTED is always 0.
#if TDM_SUPPORTED
  __device__ TileMover() {
    constexpr int log2DataSize = __builtin_ctzll(sizeof(T));
    static_assert(
      log2DataSize <= 3,
      "Datatype must be 1, 2, 4, or 8 bytes in width"); // TODO(breslow): Replace with C++ concepts when we migrate to C++20
    group1.dataSize(log2DataSize);
  }

  // An entire warp makes this call.  The shmemPtr and srcPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) {
    group0.ldsAddr((uintptr_t)shmemPtr);
    group0.globalAddr((uintptr_t)srcPtr);
    setTransferSize(group1, numElementsToProcess);
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, __rccl_int32x4{}, __rccl_int32x4{},
                                        __rccl_int32x8{}, cp);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void storeTile(T* dstPtr) {
    group0.globalAddr((uintptr_t)dstPtr);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, __rccl_int32x4{}, __rccl_int32x4{},
                                           __rccl_int32x8{}, cp);
  }

  template <
    int WAIT_CNT = 0> // Needs to be a template parameter because the count is baked into the hardware instruction
  __device__ void waitTile() {
    __builtin_amdgcn_s_wait_tensorcnt(WAIT_CNT);
  }
#else
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) = delete;
  __device__ void storeTile(T* dstPtr) = delete;
  template <int WAIT_CNT = 0>
  __device__ void waitTile() = delete;
#endif
};
#endif // TDM_TOOLCHAIN_AVAILABLE

namespace async {

/// \brief Report whether async-to/from-LDS copies are usable. Always available
///        (never deleted), so it is safe to call on any target as a guard.
/// \see tdm::IsTdmCopySupported for the identical query on the TDM library.
__host__ __device__ inline bool IsTdmCopySupported(int deviceId = 0) {
#if defined(__HIP_DEVICE_COMPILE__)
  (void)deviceId;
  return ASYNC_COPY_SUPPORTED; // compile-time constant for this arch pass
#else
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, deviceId) != hipSuccess) return false;
    // gcnArchName looks like "gfx1250:sramecc+:xnack-"; match the arch prefix.
  const char* arch = prop.gcnArchName;
  const char* p = "gfx1250";
  while (*p && *arch == *p) {
    ++arch;
    ++p;
  }
  return (*p == '\0');
#endif
}

#if ASYNC_COPY_SUPPORTED

namespace detail {

// LDS is subdivided into per-warp staging windows; each window is a multiple of
// this many bytes (matching the widest async b128 access / 128B cache line).
constexpr uint32_t WINDOW_GRAIN = 128;
// The copy's bulk starts on, and is split across warps in, multiples of this many destination bytes: one
// full-warp b128 instruction. Async accesses run markedly slower when a wave's 512B footprint is not 512B aligned
// in global memory.
constexpr uint32_t ANCHOR_GRAIN = async_detail::WARP_BYTES_B128;
// Smallest per-warp window worth splitting LDS into when the load leg runs in the source's frame.
constexpr uint32_t RING_MIN_WINDOW = 8192;
// Fewest warps worth narrowing the team to for those windows; with less LDS, more warps on plain slots win.
constexpr uint32_t RING_MIN_ISSUERS = 3;

// Direct global->global byte copy by one warp, used only when the LDS staging
// area is too small to hold even a single window (the vector fallback, mirroring
// tdm::detail::warpVecCopy's role). Byte granularity keeps it correct for any
// source/destination alignment.
template <CachePolicy cp>
__device__ inline void warpGlobalCopy(const uint8_t* s, uint8_t* d, size_t n, uint32_t warpThread,
                                      uint32_t warpThreads) {
  for (size_t i = warpThread; i < n; i += warpThreads) d[i] = s[i];
}

// One chunk of `len` bytes (a multiple of 16, below 2 * MAX_RUN full-warp instructions) between the per-lane
// pointers `g` (global) and `l` (LDS), for chunks whose length is only known at run time.
template <async_detail::AsyncDir DIR, CachePolicy cp>
__device__ inline void issueChunk(const uint8_t* g, uint8_t* l, uint32_t len, uint32_t lane) {
  using namespace async_detail;
  issueRunB128<DIR, cp>(g, l, len / WARP_BYTES_B128);
  if (lane < (len % WARP_BYTES_B128) / BYTES_PER_LANE_B128) asyncCopyB128<DIR, cp>(g, l);
}

// issueChunk for any `len` (a multiple of 16).
template <async_detail::AsyncDir DIR, CachePolicy cp>
__device__ inline void issueSpan(const uint8_t* g, uint8_t* l, uint32_t len, uint32_t lane) {
  using namespace async_detail;
  constexpr uint32_t RUN = MAX_RUN * WARP_BYTES_B128;
  for (; len >= RUN; len -= RUN, g += RUN, l += RUN) issueRunStatic<DIR, cp, MAX_RUN>(g, l);
  issueChunk<DIR, cp>(g, l, len, lane);
}

// Load-leg layouts of the pipeline (see pipelineStage).
enum class LoadFrame {
  Slot, // each chunk loads exactly its own slot
  Ring, // source-aligned, t0 a nonzero multiple of 16: lanes stay 16B aligned in LDS
  Ring4, // source-aligned, t0 % 16 in {4, 8, 12}: lanes are offset in LDS, one of them straddles each seam
  Shift, // source-aligned, t0 % 16 != 0: lanes stay 16B aligned in LDS, each slot is realigned before its store
};

// How far before its slot a chunk's LDS image starts in a ring frame: t0, or for Shift t0 rounded down to 16, which
// leaves the image t0 % 16 bytes PAST the destination frame.
template <LoadFrame F>
__device__ inline uint32_t ldsLead(uint32_t t0) {
  return F == LoadFrame::Shift ? (t0 & ~(async_detail::BYTES_PER_LANE_B128 - 1)) : t0;
}

using LdsWord = __attribute__((address_space(3))) uint32_t;
using Word4 = uint32_t __attribute__((ext_vector_type(4)));
using LdsWord4 = __attribute__((address_space(3))) Word4;

// Shift frame: move slot `slot` of the ring back by r = t0 % 16 bytes, in place, into the destination frame. Each
// lane rebuilds its 16 bytes from the five words starting at its r & ~3 (the next lane's leading ones included),
// and an instruction's reads come before its writes and after the previous instruction's, so every read precedes
// the write that clobbers it. The chunk's last r bytes are in the next slot's first lane group, which the next
// chunk's first load writes; for the window's last slot that is slot 0's, which the load leg mirrors into a 16-byte
// pad past the window so the reads never wrap. The word offset is folded into the LDS address and the byte offset
// into v_alignbyte, so the code has no data-dependent selects for the compiler to multiply.
template <uint32_t U>
__device__ inline void realignSlot(uint8_t* win, uint32_t slot, uint32_t t0) {
  using namespace async_detail;
  constexpr uint32_t B = U * WARP_BYTES_B128;
  constexpr uint32_t G = U < 2 ? U : 2; // instructions per batch
  const uint32_t r = t0 & (BYTES_PER_LANE_B128 - 1);
  uint8_t* p = win + slot * B;
  const uint8_t* q = p + (r & ~3u);
#pragma unroll 1
  for (uint32_t i = 0; i < U; i += G) {
    uint32_t w[G][5];
#pragma unroll
    for (uint32_t k = 0; k < G; ++k) {
#pragma unroll
      for (uint32_t j = 0; j < 5; ++j) w[k][j] = *toLds<const LdsWord>(q + (i + k) * WARP_BYTES_B128 + 4 * j);
    }
#pragma unroll
    for (uint32_t k = 0; k < G; ++k) {
      Word4 o;
      o.x = __builtin_amdgcn_alignbyte(w[k][1], w[k][0], r);
      o.y = __builtin_amdgcn_alignbyte(w[k][2], w[k][1], r);
      o.z = __builtin_amdgcn_alignbyte(w[k][3], w[k][2], r);
      o.w = __builtin_amdgcn_alignbyte(w[k][4], w[k][3], r);
      *toLds<LdsWord4>(p + (i + k) * WARP_BYTES_B128) = o;
    }
  }
  // The async store reads LDS behind the DS counter's back.
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

// Load leg of pipeline chunk c into `slot` of the window, from per-lane pointers `g` (the chunk's source) and `win`
// (the window base). In the Slot frame a chunk is U full-warp instructions into its own slot.
//
// In the Ring frames the source sits `t0` bytes past a 512B boundary and the load leg runs in the SOURCE's aligned
// frame instead: chunk c's U instructions cover the 512B-aligned source blocks starting t0 bytes before the chunk
// (`g` is that aligned address), so their LDS image starts t0 bytes before the slot -- the tail of the previous
// slot, or for slot 0 of the window's last slot, the one lane group that wraps around. Every global access of both
// legs is then a whole aligned block; only the LDS side of the loads is offset, which the hardware barely notices.
// Chunk 0 masks off the lanes wholly ahead of the copy.
//
// Ring4 also has a lane that straddles each slot seam. The window is padded by 16 bytes at both ends to absorb
// the parts of a straddling lane that fall outside it, and since the wrapped straddler's second half lands in the
// end pad rather than at the window start, chunks into slot 0 re-issue that one lane unwrapped first.
//
// Shift starts the image only t0 rounded down to 16 bytes before the slot, so its lanes stay 16B aligned and the
// image sits t0 % 16 bytes past the destination frame until realignSlot moves it back. Its window has a 16-byte
// pad at the end, into which chunks into slot 0 re-issue the lane that writes the window's first 16 bytes.
template <CachePolicy cp, LoadFrame F, uint32_t U, uint32_t S, bool FIRST = false>
__device__ inline void loadFull(const uint8_t* g, uint8_t* win, uint32_t slot, uint32_t pos, uint32_t t0) {
  using namespace async_detail;
  constexpr uint32_t B = U * WARP_BYTES_B128;
  if constexpr (F == LoadFrame::Slot) {
    issueRunStatic<AsyncDir::Load, cp, U>(g, win + slot * B);
  } else {
    const uint32_t lt = ldsLead<F>(t0);
    uint8_t* l = win + slot * B - lt;
    if constexpr (FIRST) {
      if (pos + BYTES_PER_LANE_B128 > t0) asyncCopyB128<AsyncDir::Load, cp>(g, l);
    } else {
      if (F == LoadFrame::Ring4 && slot == 0 && pos < t0 && pos + BYTES_PER_LANE_B128 > t0)
        asyncCopyB128<AsyncDir::Load, cp>(g, l);
      if (F == LoadFrame::Shift && slot == 0 && pos == lt) asyncCopyB128<AsyncDir::Load, cp>(g, l + S * B);
      asyncCopyB128<AsyncDir::Load, cp>(g, (slot == 0 && pos < lt) ? l + S * B : l);
    }
    issueRunStatic<AsyncDir::Load, cp, U - 1>(g + WARP_BYTES_B128, l + WARP_BYTES_B128);
  }
}

// Load leg of the short last chunk, `len` bytes of the copy (at least one instruction is issued either way).
template <CachePolicy cp, LoadFrame F, uint32_t U, uint32_t S>
__device__ inline void loadPartial(const uint8_t* g, uint8_t* win, uint32_t slot, uint32_t len, uint32_t lane,
                                   uint32_t t0, bool first) {
  using namespace async_detail;
  constexpr uint32_t B = U * WARP_BYTES_B128;
  if constexpr (F == LoadFrame::Slot) {
    issueChunk<AsyncDir::Load, cp>(g, win + slot * B, len, lane);
  } else {
    const uint32_t pos = lane * BYTES_PER_LANE_B128;
    const uint32_t lt = ldsLead<F>(t0);
    uint8_t* l = win + slot * B - lt;
    const uint32_t hi = t0 + len;
    const uint32_t n = (hi + WARP_BYTES_B128 - 1) / WARP_BYTES_B128;
    if (first) {
      if (pos + BYTES_PER_LANE_B128 > t0 && pos < hi) asyncCopyB128<AsyncDir::Load, cp>(g, l);
    } else {
      const bool wrap = slot == 0 && pos < lt;
      if (F == LoadFrame::Ring4 && wrap && pos + BYTES_PER_LANE_B128 > t0) asyncCopyB128<AsyncDir::Load, cp>(g, l);
      if (F == LoadFrame::Shift && slot == 0 && pos == lt) asyncCopyB128<AsyncDir::Load, cp>(g, l + S * B);
      if (pos < hi) asyncCopyB128<AsyncDir::Load, cp>(g, wrap ? l + S * B : l);
    }
    for (uint32_t j = 1; j < n; ++j) {
      if (j * WARP_BYTES_B128 + pos < hi)
        asyncCopyB128<AsyncDir::Load, cp>(g + j * WARP_BYTES_B128, l + j * WARP_BYTES_B128);
    }
  }
}

// Stage [0, bytes) to `d` through this warp's LDS window `win` as a software pipeline. `d` and `win` are PER-LANE
// pointers (lane * 16 already added) and 16-byte aligned; `bytes` is a nonzero multiple of 16. `s` is the
// per-lane source, moved back by `t0` to its aligned frame when SHIFT (see loadFull).
//
// The LDS image follows the DESTINATION's alignment, so every store-from-LDS is aligned on both its global and its
// LDS side -- the combination the hardware is most sensitive to. The load leg reads the source at whatever
// alignment it has; when src and dst agree mod 16 it is shifted so every load is 512B aligned too.
//
// The window is cut into S slots of B = U * 512 bytes, and chunk c of the range cycles through slot c % S. The
// chunks go P at a time: each phase stores the chunks loaded so far and loads the next P, so a phase's loads and
// stores are in flight together. ASYNCcnt counts both kinds, and loads retire in order among loads and stores
// among stores but not relative to each other, so a partial wait only proves an op complete if that many ops of
// its own kind follow it -- and while the other kind is in flight it has to drain anyway. Each phase therefore
// starts with a full drain, which settles both dependencies:
//   * RAW: a phase stores chunks loaded in earlier phases. In the ring frames a chunk's last t0 bytes come from
//     the next chunk's load, so each phase holds back its last loaded chunk (L = 1) until the next;
//   * WAR: a phase loads into slots whose previous chunks were stored in earlier phases, since 2P + L <= S.
//
// Returns with the final stores still in flight; the caller drains with asyncWait<0>().
template <CachePolicy cp, uint32_t U, uint32_t S, LoadFrame F>
__device__ __forceinline__ void pipelineCore(const uint8_t* s, uint8_t* d, uint8_t* win, size_t bytes, uint32_t lane,
                                             uint32_t t0) {
  using namespace async_detail;
  constexpr uint32_t B = U * WARP_BYTES_B128;
  constexpr uint32_t L = F == LoadFrame::Slot ? 0 : 1;
  constexpr uint32_t P = (S - L) / 2;
  static_assert(P >= 1 && U <= MAX_RUN, "pipeline needs >= 2 slots of at most MAX_RUN instructions");

  const uint32_t pos = lane * BYTES_PER_LANE_B128;
  const size_t K = (bytes + B - 1) / B;
  const uint32_t lastLen = static_cast<uint32_t>(bytes - (K - 1) * B);
  auto loadF = [&](size_t c, uint32_t slot) { loadFull<cp, F, U, S>(s + c * B, win, slot, pos, t0); };
  auto load = [&](size_t c, uint32_t slot) {
    if (c == K - 1) loadPartial<cp, F, U, S>(s + c * B, win, slot, lastLen, lane, t0, c == 0);
    else if (c == 0) loadFull<cp, F, U, S, true>(s, win, 0, pos, t0);
    else loadF(c, slot);
  };
  auto realign = [&](uint32_t slot) {
    if constexpr (F == LoadFrame::Shift) realignSlot<U>(win, slot, t0);
  };
  auto storeF = [&](size_t c, uint32_t slot) {
    realign(slot);
    issueRunStatic<AsyncDir::Store, cp, U>(d + c * B, win + slot * B);
  };
  auto store = [&](size_t c, uint32_t slot) {
    if (c != K - 1) return storeF(c, slot);
    realign(slot);
    issueChunk<AsyncDir::Store, cp>(d + c * B, win + slot * B, lastLen, lane);
  };

  if (K <= S) { // fits in the window: no slot is reused, so fill everything, then drain everything
    for (size_t c = 0; c < K; ++c) load(c, static_cast<uint32_t>(c));
    asyncWaitAtMost<0>();
    for (size_t c = 0; c < K; ++c) store(c, static_cast<uint32_t>(c));
    return;
  }
  // K > S > P, so the first phase's chunks are all full.
  staticFor<0, P>([&](auto c) { loadFull<cp, F, U, S, c == 0>(s + c * B, win, c, pos, t0); });

  // st and ld are the next chunks to store and to load. A phase starts with st = ld - P - L (0 in the first), stores
  // up to ld - L and loads [ld, ld + P), so the chunks it touches span at most 2P + L <= S slots.
  uint32_t storeSlot = 0;
  uint32_t loadSlot = P;
  size_t st = 0;
  size_t ld = P;
  auto storeNext = [&](auto&& storeFn) {
    storeFn(st++, storeSlot);
    storeSlot = (storeSlot + 1 == S) ? 0 : storeSlot + 1;
  };
  auto loadNext = [&](auto&& loadFn) {
    loadFn(ld++, loadSlot);
    loadSlot = (loadSlot + 1 == S) ? 0 : loadSlot + 1;
  };
  // Phases whose chunks are all full (the last chunk is the only short one). The first stores P - L chunks and each
  // later one P, so every count is static: issue time after the drain is on the critical path.
  if (ld + P < K) {
    asyncWaitAtMost<0>();
    staticFor<0, P - L>([&](auto) { storeNext(storeF); });
    staticFor<0, P>([&](auto) { loadNext(loadF); });
  }
  while (ld + P < K) {
    asyncWaitAtMost<0>();
    staticFor<0, P>([&](auto) { storeNext(storeF); });
    staticFor<0, P>([&](auto) { loadNext(loadF); });
  }
  while (st < K) {
    asyncWaitAtMost<0>();
    const size_t storeEnd = ld < K ? ld - L : K;
    while (st < storeEnd) storeNext(store);
    const size_t loadEnd = ld + P < K ? ld + P : K;
    while (ld < loadEnd) loadNext(load);
  }
}

// pipelineCore, plus a constraint of the Ring4 and Shift frames: the last lane of their load leg writes up to 15
// bytes past the copy's LDS image, which must stay inside the last chunk's slot. When the range ends exactly on a
// slot boundary, its final 16 bytes are moved through registers instead.
template <CachePolicy cp, uint32_t U, uint32_t S, LoadFrame F>
__device__ __forceinline__ void pipelineStage(const uint8_t* s, uint8_t* d, uint8_t* win, size_t bytes, uint32_t lane,
                                              uint32_t t0) {
  using namespace async_detail;
  constexpr uint32_t B = U * WARP_BYTES_B128;
  if constexpr (F == LoadFrame::Ring4 || F == LoadFrame::Shift) {
    if (bytes % B == 0) {
      const size_t staged = bytes - BYTES_PER_LANE_B128;
      const uint32_t pos = lane * BYTES_PER_LANE_B128;
      const uint8_t* tailSrc = s - pos + t0 + staged;
      uint8_t* tailDst = d - pos + staged;
      const uint32_t tailByte = tailSrc[lane % BYTES_PER_LANE_B128];
      pipelineCore<cp, U, S, F>(s, d, win, staged, lane, t0);
      if (lane < BYTES_PER_LANE_B128) tailDst[lane] = static_cast<uint8_t>(tailByte);
      return;
    }
  }
  pipelineCore<cp, U, S, F>(s, d, win, bytes, lane, t0);
}

// A window too small for two full-warp slots: the whole window is one (partial-warp) chunk, staged serially.
template <CachePolicy cp>
__device__ __forceinline__ void serialStage(const uint8_t* s, uint8_t* d, uint8_t* win, size_t bytes, uint32_t window,
                                            uint32_t lane) {
  using namespace async_detail;
  for (size_t o = 0; o < bytes; o += window) {
    const uint32_t len = static_cast<uint32_t>(bytes - o < window ? bytes - o : window);
    issueChunk<AsyncDir::Load, cp>(s + o, win, len, lane);
    asyncWaitAtMost<0>();
    issueChunk<AsyncDir::Store, cp>(d + o, win, len, lane);
    asyncWaitAtMost<0>();
  }
}

// A copy that fits in the window, in one round: load everything, then store everything. `s`, `d`, `win` are NOT
// offset per lane. When the source sits t0 (a nonzero multiple of 16) bytes past a 512B boundary, the loads run in
// its aligned frame as in the Ring pipeline, with the lanes ahead of the copy masked off.
template <CachePolicy cp>
__device__ __forceinline__ void wholeStage(const uint8_t* s, uint8_t* d, uint8_t* win, uint32_t bytes, uint32_t lane,
                                           uint32_t t0) {
  using namespace async_detail;
  const uint32_t pos = lane * BYTES_PER_LANE_B128;
  if (t0 != 0 && (t0 & (BYTES_PER_LANE_B128 - 1)) == 0) {
    const uint8_t* g = s - t0 + pos;
    uint8_t* l = win - t0 + pos;
    const uint32_t hi = t0 + bytes;
    if (pos + BYTES_PER_LANE_B128 > t0 && pos < hi) asyncCopyB128<AsyncDir::Load, cp>(g, l);
    if (hi > WARP_BYTES_B128)
      issueSpan<AsyncDir::Load, cp>(g + WARP_BYTES_B128, l + WARP_BYTES_B128, hi - WARP_BYTES_B128, lane);
  } else {
    issueSpan<AsyncDir::Load, cp>(s + pos, win + pos, bytes, lane);
  }
  asyncWaitAtMost<0>();
  issueSpan<AsyncDir::Store, cp>(d + pos, win + pos, bytes, lane);
}

// Slot geometry by window size, at least 2KB (Ring: 4KB, Ring4 and Shift: 7.5KB after their padding). The ring frames
// hold one chunk back per phase, so they favor more, smaller slots.
//
// Out of line: the tiers are up to ~85KB of code per frame, and inlined they scatter the setup and wholeStage paths
// across it, which costs ~1.5us of instruction fetch on a cold launch. Everything above this (down to the public entry
// points) is force-inlined into the kernel instead, so the call saves no registers and the kernel needs no scratch,
// whose setup costs about as much again per launch.
//
// Keep each instantiation well under 128KB: past that, the AMDGPU backend's long-branch relaxation borrows the
// return address s[30:31] without restoring it, and the function returns into itself.
template <CachePolicy cp, LoadFrame F>
__device__ __attribute__((noinline)) void pipelineByWindow(const uint8_t* s, uint8_t* d, uint8_t* win, size_t bytes,
                                                           uint32_t window, uint32_t lane, uint32_t t0) {
  // Arguments arrive in VGPRs; restore their uniformity so the tier choice and the counts stay scalar.
  bytes = async_detail::uniform(static_cast<uint64_t>(bytes));
  window = async_detail::uniform(window);
  t0 = async_detail::uniform(t0);
  if constexpr (F == LoadFrame::Slot) {
    // Tiers at 1.5x each power of two, so that a window of LDS / 8 warps from an uneven budget isn't a third idle.
    if (window >= 65536) pipelineStage<cp, 8, 16, F>(s, d, win, bytes, lane, t0);
    else if (window >= 49152) pipelineStage<cp, 8, 12, F>(s, d, win, bytes, lane, t0);
    else if (window >= 32768) pipelineStage<cp, 4, 16, F>(s, d, win, bytes, lane, t0);
    else if (window >= 24576) pipelineStage<cp, 4, 12, F>(s, d, win, bytes, lane, t0);
    else if (window >= 16384) pipelineStage<cp, 4, 8, F>(s, d, win, bytes, lane, t0);
    else if (window >= 12288) pipelineStage<cp, 4, 6, F>(s, d, win, bytes, lane, t0);
    else if (window >= 8192) pipelineStage<cp, 4, 4, F>(s, d, win, bytes, lane, t0);
    else if (window >= 6144) pipelineStage<cp, 2, 6, F>(s, d, win, bytes, lane, t0);
    else if (window >= 4096) pipelineStage<cp, 2, 4, F>(s, d, win, bytes, lane, t0);
    else if (window >= 3072) pipelineStage<cp, 1, 6, F>(s, d, win, bytes, lane, t0);
    else pipelineStage<cp, 1, 4, F>(s, d, win, bytes, lane, t0);
  } else if constexpr (F == LoadFrame::Ring) {
    if (window >= 65536) pipelineStage<cp, 8, 16, F>(s, d, win, bytes, lane, t0);
    else if (window >= 32768) pipelineStage<cp, 4, 16, F>(s, d, win, bytes, lane, t0);
    else if (window >= 16384) pipelineStage<cp, 2, 16, F>(s, d, win, bytes, lane, t0);
    else if (window >= 8192) pipelineStage<cp, 1, 16, F>(s, d, win, bytes, lane, t0);
    else pipelineStage<cp, 1, 8, F>(s, d, win, bytes, lane, t0);
  } else { // padded windows: 15 slots, so a power-of-two window less its padding loses only one
    if (window >= 15 * 4096) pipelineStage<cp, 8, 15, F>(s, d, win, bytes, lane, t0);
    else if (window >= 15 * 2048) pipelineStage<cp, 4, 15, F>(s, d, win, bytes, lane, t0);
    else if (window >= 15 * 1024) pipelineStage<cp, 2, 15, F>(s, d, win, bytes, lane, t0);
    else pipelineStage<cp, 1, 15, F>(s, d, win, bytes, lane, t0);
  }
}

// Pick the load-leg mode and slot geometry for a `window`-byte LDS buffer and run the pipeline. `d` and `win`
// must be 16B aligned; `s`, `d`, `win` are NOT yet offset per lane.
template <CachePolicy cp>
__device__ __forceinline__ void pipelinedStage(const uint8_t* s, uint8_t* d, uint8_t* win, size_t bytes,
                                               uint32_t window) {
  using namespace async_detail;
  if (bytes == 0) return;
  const uint32_t lane = __lane_id();
  const size_t laneByte = (size_t)lane * BYTES_PER_LANE_B128;
  const uint32_t t0 = static_cast<uint32_t>((uintptr_t)s & (WARP_BYTES_B128 - 1));
  // A copy that fits the window is one load-all / store-all round; Ring4's straddling lanes need the pipeline.
  if (bytes <= window && ((t0 & 3) != 0 || (t0 & (BYTES_PER_LANE_B128 - 1)) == 0)) {
    wholeStage<cp>(s, d, win, static_cast<uint32_t>(bytes), lane, t0);
    return;
  }
  if (window < 1024) {
    serialStage<cp>(s + laneByte, d + laneByte, win + laneByte, bytes, window, lane);
  } else if (window < 2048) {
    pipelineStage<cp, 1, 2, LoadFrame::Slot>(s + laneByte, d + laneByte, win + laneByte, bytes, lane, 0);
  } else {
    constexpr uint32_t PAD = BYTES_PER_LANE_B128;
    // Below these windows the source-aligned frames lose to plain slots: with only 512B slots, waiting for the next
    // load (and Shift's realign) leaves too little in flight.
    constexpr uint32_t RING_MIN = 8 * 512, SHIFT_MIN = 15 * 512 + PAD, RING4_MIN = 15 * 512 + 2 * PAD;
    if (t0 != 0 && (t0 & (BYTES_PER_LANE_B128 - 1)) == 0 && window >= RING_MIN) {
      pipelineByWindow<cp, LoadFrame::Ring>(s - t0 + laneByte, d + laneByte, win + laneByte, bytes, window, lane, t0);
    } else if ((t0 & 3) != 0 && window >= SHIFT_MIN) {
      pipelineByWindow<cp, LoadFrame::Shift>(s - t0 + laneByte, d + laneByte, win + laneByte, bytes, window - PAD, lane,
                                             t0);
    } else if ((t0 & 3) == 0 && (t0 & (BYTES_PER_LANE_B128 - 1)) != 0 && window >= RING4_MIN) {
      pipelineByWindow<cp, LoadFrame::Ring4>(s - t0 + laneByte, d + laneByte, win + PAD + laneByte, bytes,
                                             window - 2 * PAD, lane, t0);
    } else {
      pipelineByWindow<cp, LoadFrame::Slot>(s + laneByte, d + laneByte, win + laneByte, bytes, window, lane, 0);
    }
  }
}

using GlobalByte = __attribute__((address_space(1))) uint8_t;

template <uint32_t W>
struct HeadElem;
template <>
struct HeadElem<1> {
  using T = uint8_t;
};
template <>
struct HeadElem<2> {
  using T = uint16_t;
};
template <>
struct HeadElem<4> {
  using T = uint32_t;
};
template <>
struct HeadElem<8> {
  using T = uint32_t __attribute__((ext_vector_type(2)));
};
template <>
struct HeadElem<16> {
  using T = uint32_t __attribute__((ext_vector_type(4)));
};

// A copy's head (< ANCHOR_GRAIN bytes, a multiple of W) moved through registers in W-byte accesses, W a common
// alignment of src and dst: at most 16 accesses per lane, 64 bytes in `v`. Global address space, or as possible
// LDS reads the loads would have to complete before the pipeline's first write to LDS. The loads are clamped
// rather than guarded: a guarded load merged into a packed register makes the compiler wait on each in turn.
constexpr uint32_t HEAD_WORDS = ANCHOR_GRAIN / 32;
template <uint32_t W>
__device__ inline void headLoad(const GlobalByte* s, uint32_t head, uint32_t lane, uint32_t (&v)[HEAD_WORDS]) {
  using E = typename HeadElem<W>::T;
  constexpr uint32_t PER = ANCHOR_GRAIN / (32 * W);
  constexpr uint32_t WORDS = W < 4 ? 1 : W / 4;
#pragma unroll
  for (uint32_t k = 0; k < PER; ++k) {
    if (k * 32 * W < head) {
      const uint32_t i = (lane + 32 * k) * W;
      const E x = *reinterpret_cast<const __attribute__((address_space(1))) E*>(s + (i < head ? i : 0));
      if constexpr (W <= 4) {
        v[k] = x;
      } else {
#pragma unroll
        for (uint32_t j = 0; j < WORDS; ++j) v[k * WORDS + j] = x[j];
      }
    }
  }
}
template <uint32_t W>
__device__ inline void headStore(GlobalByte* d, uint32_t head, uint32_t lane, const uint32_t (&v)[HEAD_WORDS]) {
  using E = typename HeadElem<W>::T;
  constexpr uint32_t PER = ANCHOR_GRAIN / (32 * W);
  constexpr uint32_t WORDS = W < 4 ? 1 : W / 4;
#pragma unroll
  for (uint32_t k = 0; k < PER; ++k) {
    const uint32_t i = (lane + 32 * k) * W;
    if (k * 32 * W < head && i < head) {
      E x;
      if constexpr (W <= 4) {
        x = static_cast<E>(v[k]);
      } else {
#pragma unroll
        for (uint32_t j = 0; j < WORDS; ++j) x[j] = v[k * WORDS + j];
      }
      *reinterpret_cast<__attribute__((address_space(1))) E*>(d + i) = x;
    }
  }
}

// core: partition + issue the whole copy for the team [start, stop). Work + LDS
// partition by rank within the team (warpId - start), so each team indexes its
// own ldsBuffer from zero. Safe to call collectively (warps outside the range
// return immediately) or only from the team's warps. NO final wait: the last
// pipeline chunks (and the register-staged edge stores) are left in flight for
// async::tdmWait() to drain, so the asynchronous forms genuinely overlap.
//
// LAYOUT OF A COPY
//   [ head (registers) ][ ---- dst-aligned bulk (async pipeline per warp) ---- ][ residual < 16B (registers) ]
//   * The LDS image follows the destination: misaligned async stores cost far more than misaligned loads, and
//     when src and dst agree mod 4 the pipeline's ring-framed load leg keeps the loads 512B aligned as well.
//   * `head` brings the DESTINATION up to an ANCHOR_GRAIN boundary. It is < 512 bytes, so the team's first warp
//     copies it through registers: its loads are issued before that warp's pipeline starts and its stores after
//     the pipeline is primed, so it adds no latency and stays off ASYNCcnt.
//   * The bulk is split across issuers in ANCHOR_GRAIN grains, so every warp's destination start stays aligned.
//   * The last issuer's final < 16 bytes are copied the same way as the head.
template <CachePolicy cp>
__device__ __forceinline__ void issue(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                      size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  using async_detail::uniform;
  const uint8_t* s = uniform(reinterpret_cast<const uint8_t*>(src));
  uint8_t* d = uniform(reinterpret_cast<uint8_t*>(dst));
  uint8_t* lds = uniform(reinterpret_cast<uint8_t*>(ldsBuffer));
  sizeBytes = uniform(static_cast<uint64_t>(sizeBytes));
  const uint32_t ldsBytes = uniform(static_cast<uint32_t>(ldsBufferBytes)); // LDS is small
  startWarpId = uniform(startWarpId);
  stopWarpId = uniform(stopWarpId);

  const uint32_t W = warpSize;
  const uint32_t nThreads = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t warpThread = tid % W; // thread index within its warp
  const uint32_t warpId = async_detail::uniform(tid / W);
  const uint32_t nWarps = (nThreads + W - 1) / W;

  // --- team membership: this warp participates iff in [start, stop) --------
  const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
  if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop) return; // not on this team
  const uint32_t rank = warpId - startWarpId; // rank within the team

  // active threads in THIS warp (handles partial final warp); stride for the
  // byte fallback below.
  const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

  if (sizeBytes == 0) return;

  // --- byte fallback: not enough LDS for a single window, or no full warp ---
  // The async path spreads each instruction across all lanes of the wave, so only full warps issue it; a partial
  // final warp sits out unless the team has nothing else.
  const uint32_t fullWarps = nThreads / W;
  const uint32_t issueStop = teamStop < fullWarps ? teamStop : fullWarps;
  if (ldsBytes < WINDOW_GRAIN || startWarpId >= issueStop) {
    if (rank == 0) warpGlobalCopy<cp>(s, d, sizeBytes, warpThread, warpThreads);
    return;
  }
  if (warpId >= issueStop) return;
  const uint32_t teamWarps = issueStop - startWarpId; // >= 1

  // --- [head][bulk]: head brings dst to an ANCHOR_GRAIN boundary --------------
  size_t head = (ANCHOR_GRAIN - ((uintptr_t)d & (ANCHOR_GRAIN - 1))) & (ANCHOR_GRAIN - 1);
  if (head > sizeBytes) head = sizeBytes;
  const size_t bulk = sizeBytes - head;

  // --- give each issuing warp a 128B-multiple LDS window -------------------
  // Every warp's source start has the same offset t0 within a 512B block. When the pipeline's load leg has to run
  // in the source's frame (t0 a nonzero multiple of 4), deeper windows matter more than more warps.
  const uint32_t t0 = static_cast<uint32_t>((uintptr_t)(s + head) & (ANCHOR_GRAIN - 1));
  const bool ring = t0 != 0 && (t0 & 3) == 0;
  uint32_t maxIssuers = ldsBytes / WINDOW_GRAIN; // #warps we can give a window
  if (ring && ldsBytes >= RING_MIN_ISSUERS * RING_MIN_WINDOW) maxIssuers = ldsBytes / RING_MIN_WINDOW;
  uint32_t issuers = teamWarps < maxIssuers ? teamWarps : maxIssuers;
  uint32_t window = (ldsBytes / issuers) & ~(WINDOW_GRAIN - 1); // per-warp 128B-multiple
  if (rank >= issuers) return; // this warp doesn't issue

  // Distribute the bulk across issuers in contiguous, dst-aligned blocks of whole grains. The sub-grain
  // remainder is appended to the last issuing warp so coverage remains contiguous and gap-free.
  const size_t grain = ANCHOR_GRAIN;
  const size_t nGrains = bulk / grain;
  const size_t tail = bulk - nGrains * grain;

  const size_t baseG = nGrains / issuers;
  const size_t extraG = nGrains % issuers;
  const size_t myG = baseG + (rank < extraG ? 1u : 0u);
  const size_t startG = rank * baseG + (rank < extraG ? rank : extraG);

  const size_t myStart = head + startG * grain;
  size_t myBytes = myG * grain;
  if (rank == issuers - 1) myBytes += tail;
  const size_t staged = myBytes & ~static_cast<size_t>(async_detail::BYTES_PER_LANE_B128 - 1);
  const uint32_t resid = static_cast<uint32_t>(myBytes - staged); // nonzero only on the last issuer

  // Register-staged edges: load now, store once the pipeline is primed.
  const GlobalByte* gs = (const GlobalByte*)s;
  GlobalByte* gd = (GlobalByte*)d;
  uint32_t headRegs[HEAD_WORDS];
  const bool hasHead = rank == 0 && head != 0;
  // log2 of the widest access (up to 16B) both sides of the head are aligned to and its length (clamped to a short
  // copy) is a multiple of.
  const uint32_t headLen = static_cast<uint32_t>(head);
  const uint32_t headLog = __builtin_ctz(static_cast<uint32_t>((uintptr_t)s | (uintptr_t)d) | headLen | 16u);
  if (hasHead) {
    switch (headLog) {
    case 4:
      headLoad<16>(gs, headLen, warpThread, headRegs);
      break;
    case 3:
      headLoad<8>(gs, headLen, warpThread, headRegs);
      break;
    case 2:
      headLoad<4>(gs, headLen, warpThread, headRegs);
      break;
    case 1:
      headLoad<2>(gs, headLen, warpThread, headRegs);
      break;
    default:
      headLoad<1>(gs, headLen, warpThread, headRegs);
      break;
    }
  }
  uint32_t residByte = 0;
  if (resid != 0) residByte = gs[myStart + staged + (warpThread < resid ? warpThread : 0)];

  pipelinedStage<cp>(s + myStart, d + myStart, lds + static_cast<size_t>(rank) * window, staged, window);

  if (hasHead) {
    switch (headLog) {
    case 4:
      headStore<16>(gd, headLen, warpThread, headRegs);
      break;
    case 3:
      headStore<8>(gd, headLen, warpThread, headRegs);
      break;
    case 2:
      headStore<4>(gd, headLen, warpThread, headRegs);
      break;
    case 1:
      headStore<2>(gd, headLen, warpThread, headRegs);
      break;
    default:
      headStore<1>(gd, headLen, warpThread, headRegs);
      break;
    }
  }
  if (warpThread < resid) gd[myStart + staged + warpThread] = static_cast<uint8_t>(residByte);
}

} // namespace detail

#endif // ASYNC_COPY_SUPPORTED

/// \brief Drain the CALLING WARP's outstanding async-to/from-LDS ops.
/// \see tdm::tdmWait. The blocking copy forms already drain internally.
__device__ ASYNC_API void tdmWait() ASYNC_DELETED;

/// \brief Non-blocking block-collective copy: issue and return.
/// \see tdm::tdmCopyAsync.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                       size_t ldsBufferBytes) ASYNC_DELETED;

/// \brief Blocking block-collective copy of [0, sizeBytes): dst <- src.
/// \see tdm::tdmCopy.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                  size_t ldsBufferBytes) ASYNC_DELETED;

/// \brief Non-blocking WARP-SPECIALIZED copy by one contiguous warp team.
/// \see tdm::tdmCopyAsyncByTeam.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                             size_t ldsBufferBytes, uint32_t startWarpId,
                                             uint32_t stopWarpId) ASYNC_DELETED;

/// \brief Blocking WARP-SPECIALIZED copy by one contiguous warp team.
/// \see tdm::tdmCopyByTeam.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                        size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) ASYNC_DELETED;

#if ASYNC_COPY_SUPPORTED

__device__ inline void tdmWait() {
  asyncWait<0>();
  // The copy's register-staged head/residual are ordinary global stores; drain those too. There is no builtin
  // for STOREcnt.
  __asm__ volatile("s_wait_storecnt 0x0" ::: "memory");
}

template <CachePolicy cp>
__device__ __forceinline__ void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                             size_t ldsBufferBytes) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
}

template <CachePolicy cp>
__device__ __forceinline__ void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                        size_t ldsBufferBytes) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
  tdmWait();
}

template <CachePolicy cp>
__device__ __forceinline__ void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                                   size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
}

template <CachePolicy cp>
__device__ __forceinline__ void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                              size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
  tdmWait(); // no-op on any warp that issued nothing / is off-team
}

#endif // ASYNC_COPY_SUPPORTED

} // namespace async

#undef ASYNC_API
#undef ASYNC_DELETED

#endif // __ASYNCCOPY_H
