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
#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
                                  /* extend: || (defined(__gfxNNNN__) && ...) */
#define ASYNC_COPY_SUPPORTED 1
#else
#define ASYNC_COPY_SUPPORTED 0
#endif

// Same condition as tdmCopy.h's copy, kept #ifndef-guarded so the two headers can
// be included together in either order.
#ifndef TDM_SUPPORTED
#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && TDM_TOOLCHAIN_AVAILABLE
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
using __rccl_global_int32 = __attribute__((address_space(1))) int32_t;
using __rccl_lds_int32 = __attribute__((address_space(3))) int32_t;
using __rccl_global_int32x2 = __attribute__((address_space(1))) __rccl_int32x2;
using __rccl_lds_int32x2 = __attribute__((address_space(3))) __rccl_int32x2;
using __rccl_global_int32x4 = __attribute__((address_space(1))) __rccl_int32x4;
using __rccl_lds_int32x4 = __attribute__((address_space(3))) __rccl_int32x4;

// One b8 access (1 byte) for a single lane.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB8(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b8((__rccl_global_char*)global, (__rccl_lds_char*)lds, ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b8((__rccl_global_char*)global, (__rccl_lds_char*)lds, ZERO_OFFSET,
                                                    cp);
  }
}

// One b32 access (4 bytes) for a single lane.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB32(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b32((__rccl_global_int32*)global, (__rccl_lds_int32*)lds, ZERO_OFFSET,
                                                  cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b32((__rccl_global_int32*)global, (__rccl_lds_int32*)lds, ZERO_OFFSET,
                                                     cp);
  }
}

// One b64 access (8 bytes) for a single lane.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB64(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b64((__rccl_global_int32x2*)global, (__rccl_lds_int32x2*)lds, ZERO_OFFSET,
                                                  cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b64((__rccl_global_int32x2*)global, (__rccl_lds_int32x2*)lds,
                                                     ZERO_OFFSET, cp);
  }
}

// One b128 access (16 bytes) for a single lane.
template <AsyncDir DIR, CachePolicy cp>
__device__ inline void asyncCopyB128(const uint8_t* global, uint8_t* lds) {
  if constexpr (DIR == AsyncDir::Load) {
    __builtin_amdgcn_global_load_async_to_lds_b128((__rccl_global_int32x4*)global, (__rccl_lds_int32x4*)lds,
                                                   ZERO_OFFSET, cp);
  } else {
    __builtin_amdgcn_global_store_async_from_lds_b128((__rccl_global_int32x4*)global, (__rccl_lds_int32x4*)lds,
                                                      ZERO_OFFSET, cp);
  }
}

// Bytes moved by a single lane in the widest (b128) access.
constexpr size_t BYTES_PER_LANE_B128 = 16;
// The hot loop must start on a 128-byte boundary, so the peel aligns the global pointer to this.
constexpr size_t NATURAL_ALIGNMENT_BYTES = 128;
// How many b128 iterations the hot loop unrolls at a time.
constexpr int HOT_LOOP_UNROLL = 4;

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
//
// `Aligned` fast path: when the caller has already proven that both `global` and `lds` start on a 128-byte
// boundary, the phase-1 peel is provably a no-op.  Compiling it out removes the peel branches and lets the
// hot loop start at offset 0 with folded, compile-time-known strides.  Callers must only pass Aligned=true
// after checking the runtime pointers (see async::detail::issue).
template <AsyncDir DIR, CachePolicy cp, bool Aligned = false>
__device__ inline void warpAsyncCopy(const uint8_t* global, uint8_t* lds, size_t sizeInBytes) {
  const unsigned lane = __lane_id();
  const size_t laneCount = (size_t)__builtin_amdgcn_wavefrontsize();
  const size_t laneByte = (size_t)lane * BYTES_PER_LANE_B128;

  size_t offset = 0;

  if constexpr (!Aligned) {
    // Phase 1a: peel leading bytes until `global` is 16-byte aligned, one byte per lane.
    const size_t misalign = (uintptr_t)global & (BYTES_PER_LANE_B128 - 1);
    if (misalign != 0) {
      const size_t head =
        (BYTES_PER_LANE_B128 - misalign) < sizeInBytes ? (BYTES_PER_LANE_B128 - misalign) : sizeInBytes;
      if (lane < head) {
        asyncCopyB8<DIR, cp>(global + lane, lds + lane);
      }
      offset += head;
    }

    // Phase 1b: peel whole 16-byte chunks until `global + offset` reaches 128-byte alignment, one b128 per
    // leading lane.  `global + offset` is already 16-byte aligned, so the number of bytes left to the next
    // 128-byte boundary is a multiple of 16 (at most 112 -> 7 chunks, well within a warp).
    const size_t align128 = ((uintptr_t)global + offset) & (NATURAL_ALIGNMENT_BYTES - 1);
    const size_t bytesToLine = (NATURAL_ALIGNMENT_BYTES - align128) & (NATURAL_ALIGNMENT_BYTES - 1);
    size_t peelChunks = bytesToLine / BYTES_PER_LANE_B128;
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
  size_t remaining = sizeInBytes - offset;
#pragma unroll HOT_LOOP_UNROLL
  while (remaining >= bytesPerIter) {
    const size_t laneOff = offset + laneByte;
    asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
    offset += bytesPerIter;
    remaining -= bytesPerIter;
  }

  // Phase 3a: the remaining full 16-byte chunks, one b128 per leading lane.  `remaining < warpSize * 16`, so
  // `full128 < warpSize` and every chunk maps to a distinct lane.
  const size_t full128 = remaining / BYTES_PER_LANE_B128;
  if (lane < full128) {
    const size_t laneOff = offset + laneByte;
    asyncCopyB128<DIR, cp>(global + laneOff, lds + laneOff);
  }
  offset += full128 * BYTES_PER_LANE_B128;
  remaining -= full128 * BYTES_PER_LANE_B128;  // now < 16 bytes

  if (remaining == 0) return;   // common case after a 16B-multiple chunk: nothing left to drain

  // Phase 3b: 0..15 trailing bytes.  `offset` and `remaining` are uniform across the warp here, so the
  // descending cascade below is taken identically by every lane.  Because `offset` is 16-byte aligned, the
  // b64 (8-byte) and b32 (4-byte) accesses are naturally aligned.
  if (remaining >= 8) {
    if (lane == 0) {
      asyncCopyB64<DIR, cp>(global + offset, lds + offset);
    }
    offset += 8;
    remaining -= 8;
  }
  if (remaining >= 4) {
    if (lane == 0) {
      asyncCopyB32<DIR, cp>(global + offset, lds + offset);
    }
    offset += 4;
    remaining -= 4;
  }
  // Final 1..3 bytes, one b8 per lane.
  if (lane < remaining) {
    asyncCopyB8<DIR, cp>(global + offset + lane, lds + offset + lane);
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
// The two policies are named for the calls they govern: loadCp applies to loadTile()'s read of global
// memory and storeCp to storeTile()'s write. Each leg is driven separately here, so which one faces a
// peer is the caller's business.
template <typename T, CachePolicy loadCp = DEFAULT_CACHE_POLICY, CachePolicy storeCp = DEFAULT_CACHE_POLICY>
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
    asyncLoadToLDS<SyncPolicy::Async, loadCp>(reinterpret_cast<const uint8_t*>(srcPtr),
                                              reinterpret_cast<uint8_t*>(shmemPtr), sizeInBytes_);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.
  __device__ void storeTile(T* dstPtr) {
    asyncStoreFromLDS<SyncPolicy::Async, storeCp>(reinterpret_cast<const uint8_t*>(shmemPtr_),
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
// for maximum performance.  loadCp governs loadTile()'s read of global memory, storeCp storeTile()'s write to it
// (same naming as AsyncDataCopier, so the two stay interchangeable).
template <typename T, CachePolicy loadCp = DEFAULT_CACHE_POLICY, CachePolicy storeCp = DEFAULT_CACHE_POLICY>
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
                                        __rccl_int32x8{}, loadCp);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void storeTile(T* dstPtr) {
    group0.globalAddr((uintptr_t)dstPtr);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, __rccl_int32x4{}, __rccl_int32x4{},
                                           __rccl_int32x8{}, storeCp);
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

// Direct global->global byte copy by one warp, used only when the LDS staging
// area is too small to hold even a single window (the vector fallback, mirroring
// tdm::detail::warpVecCopy's role). Byte granularity keeps it correct for any
// source/destination alignment.
template <CachePolicy cp>
__device__ inline void warpGlobalCopy(const uint8_t* s, uint8_t* d, size_t n, uint32_t warpThread,
                                      uint32_t warpThreads) {
  for (size_t i = warpThread; i < n; i += warpThreads) d[i] = s[i];
}

// Stage [myStart, myStart+myBytes) HBM->LDS->HBM through this warp's single `window`-byte LDS buffer.
// LoadAligned/StoreAligned pick the peel-free warpAsyncCopy fast path for the src/dst side respectively;
// they are loop-invariant (myStart and window are both 128B multiples), so the choice is made once by the
// caller and templated in here rather than branched per iteration. The Sync policy waits after the load
// (RAW: the store must see the filled LDS) and after the store (WAR: the next load must not overwrite LDS
// that is still draining).
//
// Each leg carries its own cache policy: localCp on the read of `s`, remoteCp on the write to `d`.
template <CachePolicy localCp, CachePolicy remoteCp, bool LoadAligned, bool StoreAligned>
__device__ inline void stageLoop(const uint8_t* s, uint8_t* d, uint8_t* myLds, size_t myStart, size_t myBytes,
                                 uint32_t window) {
  for (size_t o = 0; o < myBytes; o += window) {
    const size_t chunk = (myBytes - o < window) ? (myBytes - o) : window;
    asyncLoadToLDS<SyncPolicy::Sync, localCp, LoadAligned>(s + myStart + o, myLds, chunk);
    asyncStoreFromLDS<SyncPolicy::Sync, remoteCp, StoreAligned>(myLds, d + myStart + o, chunk);
  }
}

// core: partition + issue the whole copy for the team [start, stop). Work + LDS
// partition by rank within the team (warpId - start), so each team indexes its
// own ldsBuffer from zero. Safe to call collectively (warps outside the range
// return immediately) or only from the team's warps. NO final wait -- each async
// tile already drains its own load/store (RAW/WAR) inside the loop, so nothing
// is left in flight on return; async::tdmWait() below is therefore a no-op guard
// kept for API parity with tdm::tdmWait().
template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void issue(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  uint8_t* d = reinterpret_cast<uint8_t*>(dst);
  uint8_t* lds = reinterpret_cast<uint8_t*>(ldsBuffer);
  const uint32_t ldsBytes = static_cast<uint32_t>(ldsBufferBytes); // LDS is small

  const uint32_t W = warpSize;
  const uint32_t nThreads = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t warpThread = tid % W; // thread index within its warp
  const uint32_t warpId = tid / W;
  const uint32_t nWarps = (nThreads + W - 1) / W;

  // --- team membership: this warp participates iff in [start, stop) --------
  const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
  if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop) return; // not on this team
  const uint32_t rank = warpId - startWarpId; // rank within the team
  const uint32_t teamWarps = teamStop - startWarpId; // >= 1

  // active threads in THIS warp (handles partial final warp); stride for the
  // byte fallback below.
  const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

  if (sizeBytes == 0) return;

  // --- tiny-LDS fallback: not enough LDS for a single window ---------------
  if (ldsBytes < WINDOW_GRAIN) {
    if (rank == 0) warpGlobalCopy<localCp>(s, d, sizeBytes, warpThread, warpThreads);
    return;
  }

  // --- give each issuing warp a 128B-multiple LDS window -------------------
  uint32_t maxIssuers = ldsBytes / WINDOW_GRAIN; // #warps we can give a window
  uint32_t issuers = teamWarps < maxIssuers ? teamWarps : maxIssuers;
  uint32_t window = (ldsBytes / issuers) & ~(WINDOW_GRAIN - 1); // per-warp 128B-multiple
  if (rank >= issuers) return; // this warp doesn't issue

  // Distribute the byte range across issuers in contiguous, 128B-aligned blocks. Splitting on whole
  // WINDOW_GRAIN (128B) grains -- rather than raw sizeBytes/issuers -- keeps every issuing warp's start a
  // multiple of 128B, so `s + myStart` and `d + myStart` inherit the base pointers' alignment and stay
  // aligned across chunks (the stride `window` is a 128B multiple). The sub-128B remainder is appended to
  // the last issuing warp so coverage remains contiguous and gap-free.
  const size_t grain = WINDOW_GRAIN;
  const size_t nGrains = sizeBytes / grain;
  const size_t tail = sizeBytes - nGrains * grain; // 0..127 trailing bytes

  const size_t baseG = nGrains / issuers;
  const size_t extraG = nGrains % issuers;
  const size_t myG = baseG + (rank < extraG ? 1u : 0u);
  const size_t startG = rank * baseG + (rank < extraG ? rank : extraG);

  size_t myStart = startG * grain; // 128B-aligned
  size_t myBytes = myG * grain;
  if (rank == issuers - 1) myBytes += tail; // last warp mops up the < 128B tail
  if (myBytes == 0) return;

  uint8_t* myLds = lds + static_cast<size_t>(rank) * window;

  // Pick the peel-free fast path per direction (load reads src, store writes dst). myStart and window are
  // both 128B multiples, so each side's alignment is loop-invariant and computed once here.
  const uintptr_t mask = WINDOW_GRAIN - 1;
  const bool ldsAligned = ((uintptr_t)myLds & mask) == 0;
  const bool loadAligned = ldsAligned && (((uintptr_t)(s + myStart) & mask) == 0);
  const bool storeAligned = ldsAligned && (((uintptr_t)(d + myStart) & mask) == 0);

  if (loadAligned && storeAligned) stageLoop<localCp, remoteCp, true, true>(s, d, myLds, myStart, myBytes, window);
  else if (loadAligned) stageLoop<localCp, remoteCp, true, false>(s, d, myLds, myStart, myBytes, window);
  else if (storeAligned) stageLoop<localCp, remoteCp, false, true>(s, d, myLds, myStart, myBytes, window);
  else stageLoop<localCp, remoteCp, false, false>(s, d, myLds, myStart, myBytes, window);
}

} // namespace detail

#endif // ASYNC_COPY_SUPPORTED

/// \brief Drain the CALLING WARP's outstanding async-to/from-LDS ops.
/// \see tdm::tdmWait. The blocking copy forms already drain internally.
__device__ ASYNC_API void tdmWait() ASYNC_DELETED;

/// \brief Non-blocking block-collective copy: issue and return.
/// \see tdm::tdmCopyAsync.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                       size_t ldsBufferBytes) ASYNC_DELETED;

/// \brief Blocking block-collective copy of [0, sizeBytes): dst <- src.
/// \see tdm::tdmCopy, including the localCp (read leg) / remoteCp (write leg) split.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                  size_t ldsBufferBytes) ASYNC_DELETED;

/// \brief Non-blocking WARP-SPECIALIZED copy by one contiguous warp team.
/// \see tdm::tdmCopyAsyncByTeam.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                             size_t ldsBufferBytes, uint32_t startWarpId,
                                             uint32_t stopWarpId) ASYNC_DELETED;

/// \brief Blocking WARP-SPECIALIZED copy by one contiguous warp team.
/// \see tdm::tdmCopyByTeam.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ ASYNC_API void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                        size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) ASYNC_DELETED;

#if ASYNC_COPY_SUPPORTED

__device__ inline void tdmWait() {
  asyncWait<0>();
}

template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                    size_t ldsBufferBytes) {
  detail::issue<localCp, remoteCp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
}

template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes) {
  detail::issue<localCp, remoteCp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
  tdmWait();
}

template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                          size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<localCp, remoteCp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
}

template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                     size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<localCp, remoteCp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
  tdmWait(); // no-op on any warp that issued nothing / is off-team
}

#endif // ASYNC_COPY_SUPPORTED

} // namespace async

#undef ASYNC_API
#undef ASYNC_DELETED

#endif // __ASYNCCOPY_H
