/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
/// \file tdmCopy.h
/// \brief Helper functions to perform HBM to HBM copies via the Tensor Data Mover
///        which copies HBM to a LDS staging buffer, then from LDS out to HBM
///        all without touching cache
///
///        This is introduced via __device__ level memcpy-like API which can be
///        either blocking or asynchronous, and utilize all warps, or a team of
///        contiguous warps, to allow for other warps to do other tasks
///
/// \par Quick start
/// \code
///   #include "tdmCopy.h"
///   __shared__ uint8_t staging[N];                 // or dynamic extern __shared__
///   tdm::tdmCopy(dst, src, bytes, staging, N);     // block-collective, blocking
///   __syncthreads();                               // block-wide visibility
/// \endcode
///
/// \par Two axes of control
/// - Completion: tdmCopy() (blocking) vs. tdmCopyAsync() + tdmWait() (deferred).
/// - Participation: block-collective (all warps) vs. tdmCopyByTeam() (a
///   contiguous warp range, leaving the other warps free to compute).
///
/// \par One-way transfers
/// asyncLoadToLDS() / asyncStoreFromLDS() move a single leg (global -> LDS or
/// LDS -> global) for one warp, for callers that want to keep or produce the
/// data in LDS rather than round-trip it. They mirror the identically named
/// primitives in asyncCopy.h, which implement the same interface on top of the
/// async-to/from-LDS builtins instead of the tensor data mover.
///
/// \par Availability
/// TDM is a hardware feature present on some architectures only. On a target
/// without it, every entry point is `= delete`d: including the header is always
/// fine, but calling any tdm:: function is a hard compile-time error at the call
/// site. See the AVAILABILITY block below. For a runtime/host check (e.g. to pick
/// a kernel before launch), use IsTdmCopySupported(), which is always callable.
///
/// The design rationale, hardware model, and visibility rules live at the bottom
/// of this file under "IMPLEMENTATION NOTES"; the public API is right here.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "cachePolicy.h"   // shared CachePolicy encoding (same as asyncCopy.h)
#include "syncPolicy.h"    // shared SyncPolicy encoding  (same as asyncCopy.h)
#include "uniform.h"       // shared warp-uniformity hints (same as asyncCopy.h)

// ============================================================================
//  AVAILABILITY
// ----------------------------------------------------------------------------
// TDM is supported on a subset of architectures. Detection is centralized in the
// single macro TDM_SUPPORTED, which is 1 only when we are compiling a device pass
// for a TDM-capable arch AND the compiler exposes the TDM builtin AND the arch's
// descriptor header is on the include path. Keying on __has_builtin / __has_include
// (not just the arch macro) means an older toolchain that predates the builtin, or
// a build without the descriptor header, degrades gracefully instead of failing;
// a new TDM-capable arch works as soon as its target macro is added below.
//
// Each public entry point is individually guarded. When TDM_SUPPORTED is 1,
// the real implementation is compiled. Otherwise every entry point is declared
// `= delete`, so including this header is always fine but CALLING any tdm::
// function on an unsupported target is a hard compile-time error at the call
// site ("call to deleted function"). The TDM_API / TDM_DELETED macro pair below
// applies that guard uniformly to every declaration.
// ============================================================================
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#ifndef __has_include
#define __has_include(x) 0
#endif
// Host-evaluable toolchain capability. TDM_SUPPORTED (below) keys on device arch
// macros (e.g. __gfx1250__) that are never defined during the host pass, so it is
// always 0 in host code and cannot gate the host-side IsTdmCopySupported() check.
// The descriptor header's presence is the reliable host-visible proxy for toolchain
// support (__has_builtin for the amdgcn intrinsic is unreliable in the host/x86
// pass), and it is also a prerequisite of TDM_SUPPORTED, so it is factored out here.
// Without this host gate, a build whose device pass fell back to the no-op TDM stub
// would still report support on gfx1250 hardware and silently dispatch a no-op copy.
#if __has_include(<hip/amd_detail/amd_gfx1250_TDM.h>)
#define TDM_TOOLCHAIN_AVAILABLE 1
#else
#define TDM_TOOLCHAIN_AVAILABLE 0
#endif

#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && TDM_TOOLCHAIN_AVAILABLE
                                  /* extend: || (defined(__gfxNNNN__) && ...) */
#define TDM_SUPPORTED 1
#else
#define TDM_SUPPORTED 0
#endif

#if TDM_SUPPORTED
#include <hip/amd_detail/amd_gfx1250_TDM.h>   // D# descriptor types for the target's TDM
#define TDM_API inline     // normal inline declaration (defined below)
#define TDM_DELETED             //   ... and not deleted
#else
#define TDM_API                 // no linkage keyword on a deleted declaration
#define TDM_DELETED = delete   // unsupported target: any call is a compile error
#endif

namespace tdm {

// ============================================================================
//  PUBLIC API
// ============================================================================

/// \brief Report whether TDM copies are usable. Always available (never deleted),
///        so it is safe to call on any target as a guard before the copy fns.
///
/// The answer differs by compilation pass, because TDM availability is a property
/// of the specific GPU arch:
/// - DEVICE code: returns the compile-time constant TDM_SUPPORTED for the arch
///   this device pass was built for. It is a constant expression, so it folds away
///   and can drive `if constexpr` / dead-code elimination of the copy calls.
/// - HOST code: there is no single compile-time answer (a build may target many
///   archs), so it queries the given device's architecture via the HIP runtime and
///   reports whether it is TDM-capable. Returns false if the query fails.
///
/// \param deviceId HIP device to query (host only; ignored in device code).
/// \return true if tdm:: copies will run on the target/device in question.
///
/// \code
///   // Host: pick an implementation before launching.
///   if (tdm::IsTdmCopySupported(dev)) launchTdmKernel(...);
///   else                              launchFallbackKernel(...);
/// \endcode
__host__ __device__ inline bool IsTdmCopySupported(int deviceId = 0) {
#if defined(__HIP_DEVICE_COMPILE__)
  (void)deviceId;
  return TDM_SUPPORTED;                 // compile-time constant for this arch pass
#else
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, deviceId) != hipSuccess) return false;
    // gcnArchName looks like "gfx1250:sramecc+:xnack-"; match the arch prefix.
    // Keep this list in sync with the TDM_SUPPORTED arch condition above.
  const char* arch = prop.gcnArchName;
  const char* p = "gfx1250";
  while (*p && *arch == *p) {
    ++arch;
    ++p;
  }
    // Require BOTH a TDM-capable arch AND a toolchain that actually built TDM
    // (otherwise the device pass emitted a no-op stub and enabling the path here
    // would silently produce wrong results).
  return TDM_TOOLCHAIN_AVAILABLE && (*p == '\0');
#endif
}

/// \brief Blocking block-collective copy of [0, sizeBytes): dst <- src.
///
/// Call from ALL threads of the block with identical arguments (memcpy-style
/// order). Work is partitioned across every warp in the block. On return, the
/// calling warp's TDM ops are complete.
///
/// \param dst            Destination in global memory (HBM).
/// \param src            Source in global memory (HBM).
/// \param sizeBytes      Number of bytes to copy.
/// \param ldsBuffer      Per-block LDS staging area (shared memory).
/// \param ldsBufferBytes Size of \p ldsBuffer in bytes; subdivided among warps.
///
/// \tparam cp Compile-time cache policy (temporal hint + memory scope) baked into
///            the tensor load/store as their immediate cpol operand. Build one
///            with createCachePolicy(); see cachePolicy.h. Must be a constant
///            because the hardware instruction takes an immediate.
///
/// \note For BLOCK-wide visibility, follow with __syncthreads() (and see the
///       visibility notes at the bottom of the file).
/// \warning \p src and \p dst must be GLOBAL (HBM) pointers. Passing a shared /
///          LDS pointer compiles (it decays to void*) but is undefined at run
///          time -- the address is programmed into the descriptor's global field.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                size_t ldsBufferBytes) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopy(): issue and return.
///
/// Identical partitioning to tdmCopy(), but does NOT drain on return. The last
/// few TDM ops (bounded by the per-wave queue depth) stay in flight so they
/// overlap with whatever the calling warp does next. Pair with tdmWait().
///
/// \see tdmCopy for parameter meanings (including the \p cp cache policy).
/// \see tdmWait to complete the copy.
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                     size_t ldsBufferBytes) TDM_DELETED;

/// \brief Blocking WARP-SPECIALIZED copy performed by one contiguous warp team.
///
/// Only warps in the half-open range [\p startWarpId, \p stopWarpId) participate;
/// all other warps return immediately and are free to do compute. Work and LDS
/// are partitioned by RANK WITHIN THE TEAM (warpId - startWarpId), so several
/// disjoint teams can each run a different copy concurrently, each with its own
/// dst/src and its own \p ldsBuffer region.
///
/// \param dst            Destination in global memory (HBM).
/// \param src            Source in global memory (HBM).
/// \param sizeBytes      Number of bytes to copy.
/// \param ldsBuffer      THIS team's LDS region (distinct per team).
/// \param ldsBufferBytes Size of this team's LDS region, split among its warps.
/// \param startWarpId    First warp of the team (inclusive).
/// \param stopWarpId     One past the last warp (clamped to nWarps; ~0u = end).
///
/// \note This does NOT synchronize the block -- you choose the barrier. A single
///       __syncthreads() after the copy/compute branches is enough for
///       independent compute; use named/arrive-wait barriers for a pipelined
///       producer/consumer so the copy team can run ahead.
/// \warning Give each team (and the compute warps) a NON-OVERLAPPING LDS region;
///          the library trusts the pointer/size you pass.
///
/// \code
///   const uint32_t warpId = threadIdx.x / warpSize;   // (1D block)
///   if (warpId < COPY_WARPS)
///       tdm::tdmCopyByTeam(dst, src, n, staging, teamLdsBytes, 0u, COPY_WARPS);
///   else
///       /* compute -- use LDS past the team's window */;
///   __syncthreads();
/// \endcode
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                      size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopyByTeam(): issue and return.
/// \see tdmCopyByTeam for parameter meanings and team semantics (incl. \p cp).
/// \see tdmWait to complete the copy (called by each participating warp).
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                           size_t ldsBufferBytes, uint32_t startWarpId,
                                           uint32_t stopWarpId) TDM_DELETED;

/// \brief Drain the CALLING WARP's outstanding TDM ops (TENSORcnt -> 0).
///
/// TENSORcnt is a per-wave counter, so this waits only on the ops this warp
/// issued -- nothing else.
///
/// \note Multiple teams need no special handling: each participating warp calls
///       tdmWait() to drain its own ops, and one team's wait has no effect on
///       another's (there is no shared counter). Every ISSUING warp must call it
///       -- a warp cannot drain its teammates' ops. tdmCopy*()/*ByTeam() blocking
///       forms already do this internally.
/// \note For BLOCK-wide visibility follow with __syncthreads(); if copied data is
///       consumed within the block AND the vector head ran, also
///       __threadfence_block() so those ordinary global stores are observed.
__device__ TDM_API void tdmWait() TDM_DELETED;

// ============================================================================
//  ONE-WAY GLOBAL <-> LDS TRANSFERS
// ----------------------------------------------------------------------------
// The copies above stage HBM->LDS->HBM and hand the LDS window back untouched.
// The two entry points below expose a single leg of that journey, so the caller
// can keep (or produce) the data in LDS. They carry the SAME names, signatures
// and semantics as the global-scope asyncLoadToLDS()/asyncStoreFromLDS() in
// asyncCopy.h -- these are simply the tensor-data-mover implementation of the
// same primitive, so the two are drop-in replacements for one another.
// ============================================================================

/// \brief Warp-level transfer of \p sizeInBytes from global memory into LDS.
///
/// The ENTIRE warp calls this with identical arguments; the descriptor lives in
/// SGPRs, so the pointers and size must be wave-uniform (no per-lane offsets).
///
/// \param globalSrc   Source in global memory (HBM).
/// \param ldsDst      Destination in LDS (shared memory).
/// \param sizeInBytes Number of bytes to transfer.
///
/// \tparam sp      SyncPolicy::Sync drains before returning; SyncPolicy::Async
///                 leaves the transfer in flight to overlap with the caller's
///                 next work, to be completed with tdmWait().
/// \tparam cp      Compile-time cache policy baked into the tensor load as its
///                 immediate cpol operand; see cachePolicy.h.
/// \tparam Aligned Pass true only when \p globalSrc and \p ldsDst are both known
///                 to be 128-byte aligned; this compiles out the head peel.
///
/// \note Alignment is a THROUGHPUT concern here, not a correctness one. Any
///       alignment works on either side: the head is peeled against the global
///       pointer at byte granularity, and the bulk rows carry an unconstrained
///       LDS byte address. Verified over 448 mismatched global/LDS sub-128B
///       combinations, and down to a 1-byte LDS offset under the 4-byte bulk
///       descriptor -- all byte-correct. What matching sub-128B offsets buy is
///       bulk rows that land on 128B natural alignment, and what \p Aligned buys is one
///       descriptor instead of head/bulk/tail.
template <SyncPolicy sp = DEFAULT_SYNC_POLICY, CachePolicy cp = DEFAULT_CACHE_POLICY, bool Aligned = false>
__device__ TDM_API void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes) TDM_DELETED;

/// \brief Warp-level transfer of \p sizeInBytes from LDS out to global memory.
/// \see asyncLoadToLDS for the calling convention, template parameters and the
///      alignment note (peeled against \p globalDst here).
template <SyncPolicy sp = DEFAULT_SYNC_POLICY, CachePolicy cp = DEFAULT_CACHE_POLICY, bool Aligned = false>
__device__ TDM_API void asyncStoreFromLDS(const uint8_t* ldsSrc, uint8_t* globalDst, size_t sizeInBytes) TDM_DELETED;

} // namespace tdm

#undef TDM_API
#undef TDM_DELETED

// ############################################################################
// #                                                                          #
// #                         IMPLEMENTATION NOTES                             #
// #                                                                          #
// ############################################################################
//
// LAYOUT OF A COPY
//   [ head (VECTOR) ][ chunk ][ chunk ] ... [ chunk ]   (each chunk: TDM 1-D load + store)
//   * `head` brings the DESTINATION up to a 256B boundary (one more 256B when
//     the source lead below would start before src); being the unaligned
//     remainder, it stays a cooperative vector copy done by the team's first warp.
//   * The rest is split into 256B-aligned runs, one contiguous run per issuing
//     warp, and each run into chunks of up to a slot's size. Every chunk is one
//     byte-granular 1-D load and one 1-D store, so the last needs no separate tail.
//   * A chunk's store starts on a 256B destination boundary; its load starts
//     (src - dst) % 256 bytes earlier, on a 256B source boundary, and the offset is
//     absorbed in LDS (see issue()). Neither global leg is ever misaligned.
//
// HARDWARE MODEL (measured on gfx1250)
//   * TDM engines: 1 per SIMD-pair -> 2 per WGP; a warp runs on 1 SIMD32 and
//     shares its pair's engine. Bandwidth needs >=2 issuing warps/block (both
//     engines) and many blocks (many WGPs); a lone warp is latency-bound.
//   * Throughput depends on alignment, not on data_size or row width. Stores
//     need a 128B-aligned global address (8-64B: ~4.5x slower, 1-2B: ~60x) and
//     an 8B-aligned LDS address. Loads are fastest from 256B-aligned global
//     addresses (128B: 10-25% slower, 4-64B: ~2x, 1-2B: ~4x) and need a
//     4B-aligned LDS address (otherwise ~7x slower). 512B buys nothing more.
//   * A wave's TDM ops (loads and stores alike) COMPLETE in issue order, so
//     TENSORcnt <= n waits for every op but the last n issued. Their memory
//     effects still overlap, so an op that reuses LDS another op touches must
//     wait for that op to complete. TDM ops are unordered with other memory
//     instructions, including the warp's own LDS accesses.
//   * Each warp's window holds up to two slots. A phase waits for TENSORcnt to
//     reach zero, then loads the next chunk into one slot and stores the previous
//     one from the other, so a load always overlaps a store (and the shift).
//     Zero is also the exact wait: refilling a slot needs its store complete,
//     and that store was issued after the other slot's load. Deeper rings that
//     keep loads in flight across the wait measured slower (up to 45% with few
//     warps per block): they split the same window into smaller TDM ops, and
//     bytes per op matter more than overlap.
//     Unless the copy shifts in LDS, windows under TWO_SLOT_WINDOW use one slot
//     instead, with a wait after each load and each store: at that size a chunk
//     twice as large is worth more than the overlap.
//
// ONE-WAY LEGS (asyncLoadToLDS / asyncStoreFromLDS)
//   * A line-aligned call of whole rows is one descriptor behind a single check:
//     back-to-back sub-KB calls are issue-bound, and the symmetric kernels'
//     reduce loops on 2-byte types are latency-bound per tile.
//   * The head and tail stay TDM byte ops. Per-lane async b8/b128 ops move
//     misaligned edges faster in isolation (up to ~2x at odd offsets), but
//     inlined into the symmetric kernels they cost 12-15% on 2-byte AllReduce
//     even when they never execute, and out of line the call costs more. Those
//     kernels only pass 16B-aligned tiles anyway.
//
// VISIBILITY
//   tdmWait() drains only the calling wave's TDM ops. Cross-warp / block-wide
//   visibility is the caller's __syncthreads(); the vector head's ordinary
//   global stores additionally want __threadfence_block() if consumed within the
//   block. Host-after-kernel and grid dependencies are handled by the stream.
//
// BUILTINS: verify names against your tree:
//   grep -iE 'tensor_(load|store)|tensorcnt' \
//        <llvm>/clang/include/clang/Basic/BuiltinsAMDGPU.def
// ============================================================================

namespace tdm {

#if TDM_SUPPORTED            // ===== real TDM implementation ================

namespace detail {

constexpr uint32_t WIDTH = 256;          // bytes per TDM row (first tile dim)
constexpr uint32_t ELT = 4;            // dword
constexpr uint32_t DS4 = 2;            // data_size code for 4-byte
constexpr uint32_t DS1 = 0;            // data_size code for 1-byte
constexpr uint32_t TD0 = WIDTH / ELT;  // 64 elements per row

// ---- instruction emission (the only arch-specific piece) -------------------
// The tensor DMA is a single builtin taking the FULL descriptor: five register
// groups plus a constant cache policy. Per the clang reference
// (https://clang.llvm.org/docs/AMDGPUBuiltinReference.html):
//   void __builtin_amdgcn_tensor_load_to_lds  (v4u32 D0, v8i32 D1, v4i32 D2,
//                                              v4i32 D3, v8i32 D4, int cpol);
//   void __builtin_amdgcn_tensor_store_from_lds(<same signature>);
// D0=GROUP0 (addresses), D1=GROUP1 (2D shape). D2/D3/D4 carry the higher tensor
// dimensions; for a <=2D copy they are simply ZERO vectors ("unused"). This
// mirrors known-good example usage, which passes the group m_bitfields straight
// through (no signed cast) and zero raw vectors for the unused higher dims -- so
// we depend only on GROUP0/GROUP1 existing, not on GROUP2/3/4 by name.
using u32x4 = __attribute__((ext_vector_type(4))) uint32_t;   // D0, D2, D3
using u32x8 = __attribute__((ext_vector_type(8))) uint32_t;   // D1, D4

// The cache policy is baked into the instruction as its immediate cpol operand,
// so it must be a compile-time constant -- hence a template parameter (mirroring
// how asyncCopy.h passes its cache policy to the async builtins).
template <CachePolicy cp>
__device__ inline void load(const gfx1250_TDM_GROUP0& g0, const gfx1250_TDM_GROUP1& g1) {
  __builtin_amdgcn_tensor_load_to_lds(g0.m_bitfield,               // D0  addresses
                                      g1.m_bitfield,               // D1  2D shape
                                      u32x4{}, u32x4{}, u32x8{},   // D2/D3/D4  higher dims: unused (zero)
                                      /*cpol=*/cp);
}
template <CachePolicy cp>
__device__ inline void store(const gfx1250_TDM_GROUP0& g0, const gfx1250_TDM_GROUP1& g1) {
  __builtin_amdgcn_tensor_store_from_lds(g0.m_bitfield, g1.m_bitfield, u32x4{}, u32x4{}, u32x8{},
                                         /*cpol=*/cp);
}
__device__ inline void waitTensor0() {
  __builtin_amdgcn_s_wait_tensorcnt(0);
}
using wave::uniform;
// ---- cooperative vector copy of a small byte range, by one warp ------------
// All threads of the calling warp participate. Dword-wide where possible; the
// (<4 byte) ragged end is finished by the warp's first thread.
__device__ inline void warpVecCopy(const uint8_t* s, uint8_t* d, size_t n, uint32_t warpThread, uint32_t warpThreads) {
  size_t nd = n >> 2;
  const uint32_t* s32 = reinterpret_cast<const uint32_t*>(s);
  uint32_t* d32 = reinterpret_cast<uint32_t*>(d);
  for (size_t i = warpThread; i < nd; i += warpThreads) d32[i] = s32[i];
  uint32_t rem = static_cast<uint32_t>(n & 3u);
  if (rem && warpThread == 0)
    for (uint32_t b = 0; b < rem; ++b) d[nd * 4 + b] = s[nd * 4 + b];
}

// ---- one 1-D tile of `nbytes` (<= TILE_DIM_MAX) at BYTE granularity. -------
// Exact length, so the copy's chunks need no separate sub-row tail. Higher dims
// (D2/D3/D4) are unused and passed as zero inside load()/store().
template <bool Store, CachePolicy cp>
__device__ inline void tile1d(uint64_t global, uint32_t lds, uint32_t nbytes) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS1);
  g1.tileDim0(nbytes);
  g1.tileDim1(1);
  g1.tensorDim0(nbytes);
  g1.tensorDim1(1);
  g1.tensorDim0Stride(nbytes);
  gfx1250_TDM_GROUP0 g0(lds, global);
  if constexpr (Store) store<cp>(g0, g1);
  else load<cp>(g0, g1);
}

using LdsU32 = __attribute__((address_space(3))) uint32_t;
using LdsU32x4 = __attribute__((address_space(3))) u32x4;

constexpr uint32_t SHIFT_GROUP = 4;   // lane-strides of reads issued before their writes
constexpr uint32_t SHIFT_SLACK = 32;  // bytes shiftDown() may touch past lds + n

// ---- move `n` bytes at lds + r down to lds (0 < r < 4, lds 16B aligned), in place, by one warp.
// Each lane rebuilds 16 bytes from the four words at its own offset and the next
// lane's first word. A group's reads all precede its writes, and the next group
// reads only above them, so every read precedes the write that clobbers it.
// Reads and writes stay below lds + n + SHIFT_SLACK.
__device__ inline void shiftDown(uint32_t lds, uint32_t n, uint32_t r, uint32_t warpThread, uint32_t warpThreads) {
  const uint32_t stride = 16 * warpThreads;
#pragma unroll 1
  for (uint32_t base = 16 * warpThread; base < n; base += SHIFT_GROUP * stride) {
    u32x4 w[SHIFT_GROUP];
    uint32_t next[SHIFT_GROUP];
#pragma unroll
    for (uint32_t k = 0; k < SHIFT_GROUP; ++k) {
      const uint32_t p = lds + base + k * stride;
      if (base + k * stride < n) {
        w[k] = *(const LdsU32x4*)p;
        next[k] = *(const LdsU32*)(p + 16);
      }
    }
#pragma unroll
    for (uint32_t k = 0; k < SHIFT_GROUP; ++k) {
      if (base + k * stride < n) {
        u32x4 v;
        v.x = __builtin_amdgcn_alignbyte(w[k].y, w[k].x, r);
        v.y = __builtin_amdgcn_alignbyte(w[k].z, w[k].y, r);
        v.z = __builtin_amdgcn_alignbyte(w[k].w, w[k].z, r);
        v.w = __builtin_amdgcn_alignbyte(next[k], w[k].w, r);
        *(LdsU32x4*)(lds + base + k * stride) = v;
      }
    }
  }
  // The tensor store reads LDS behind the DS counter's back.
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

// ---- one-way global <-> LDS transfer of a byte range, by ONE warp -----------
// Direction of a single leg: the first pointer is always the global-memory side
// and the second the LDS side, which keeps the tile helpers below symmetric.
enum struct LdsDir {
  Load,
  Store
};

constexpr uint32_t TILE_DIM_MAX = 0xFFFFu;   // tile_dim0/tile_dim1 are 16-bit fields
constexpr uint32_t LINE = 128;       // alignment the head peel targets
// Round-trip copy chunking. Stores run at full rate from 128B destination
// boundaries, loads only from 256B source boundaries (about 10-25% faster than
// 128B); 512B buys nothing more on either side.
constexpr uint32_t ALIGN = 256;
constexpr uint32_t MAX_CHUNK = (TILE_DIM_MAX - (ALIGN - 1)) & ~(ALIGN - 1); // a load is chunk + delta bytes
constexpr uint32_t MIN_SLOT = 2 * ALIGN + SHIFT_SLACK; // room for o <= ALIGN, one ALIGN chunk and the shift
// Below this per-warp window, one large slot outruns two half-size ones (3-10% at
// 4-8KB): bytes per TDM op matter more there than overlapping a load with a store.
// A copy that shifts in LDS keeps two slots, so each shift overlaps a load.
constexpr uint32_t TWO_SLOT_WINDOW = 16384;
// Whole 256B rows one 1-D bulk tile can carry: tile_dim0 now holds the entire length
// in elements, not one row's worth, so the 16-bit field is the limit (1023 x 256B).
constexpr uint32_t MAX_ROWS = TILE_DIM_MAX / TD0;

template <LdsDir DIR, CachePolicy cp>
__device__ inline void ldsIssue(const gfx1250_TDM_GROUP0& g0, const gfx1250_TDM_GROUP1& g1) {
  if constexpr (DIR == LdsDir::Load) load<cp>(g0, g1);
  else store<cp>(g0, g1);
}

// One 1-D tile at BYTE granularity, for the ragged head/tail. Exact length, so
// it works at any alignment; `nbytes` must be <= TILE_DIM_MAX.
template <LdsDir DIR, CachePolicy cp>
__device__ inline void ldsTileBytes(uint64_t global, uint32_t lds, uint32_t nbytes) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS1);
  g1.tileDim0(nbytes);
  g1.tensorDim0(nbytes);
  gfx1250_TDM_GROUP0 g0(lds, global);
  ldsIssue<DIR, cp>(g0, g1);
}

// One 1-D tile covering `rows` back-to-back 256B rows -- the bulk shape. The rows are
// contiguous (the 2-D form this replaced used stride == TD0), so a flat run of
// rows*TD0 elements describes exactly the same bytes and the second dimension bought
// nothing. `rows` must be <= MAX_ROWS.
template <LdsDir DIR, CachePolicy cp>
__device__ inline void ldsTileRows(uint64_t global, uint32_t lds, uint32_t rows) {
  const uint32_t elems = rows * TD0;
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS4);
  g1.tileDim0(elems);
  g1.tensorDim0(elems);
  gfx1250_TDM_GROUP0 g0(lds, global);
  ldsIssue<DIR, cp>(g0, g1);
}

// Split [0, sizeInBytes) into [head][ aligned 256B rows ][tail], mirroring the
// layout detail::issue() uses for the round trip. The pieces target disjoint LDS
// and global ranges, so unlike the round trip they need NO waits between them --
// the whole range is issued back-to-back and the caller drains once (or not at
// all, under SyncPolicy::Async).
//
// `head` brings the GLOBAL pointer up to a 128B boundary and the same offset is
// applied to the LDS side. When the two pointers start with the same sub-128B
// offset the bulk rows land on a 128B natural alignment on both ends; when they
// do not, the rows are still valid -- the LDS address field is an unconstrained
// byte address -- they just straddle lines. A line-aligned global pointer with a
// whole number of rows is one descriptor; `Aligned` asserts both offsets are zero
// and compiles the head peel out.
template <LdsDir DIR, CachePolicy cp, bool Aligned>
__device__ inline void warpLdsCopy(uint64_t global, uint32_t lds, size_t sizeInBytes) {
  global = uniform(global);
  lds = uniform(lds);
  sizeInBytes = uniform(static_cast<uint64_t>(sizeInBytes));
  if (sizeInBytes == 0) return;            // a zero-extent tile is not a valid descriptor

  if ((((static_cast<uint32_t>(global) & (LINE - 1)) | (static_cast<uint32_t>(sizeInBytes) & (WIDTH - 1))) == 0) &&
      sizeInBytes <= static_cast<size_t>(MAX_ROWS) * WIDTH) {
    ldsTileRows<DIR, cp>(global, lds, static_cast<uint32_t>(sizeInBytes / WIDTH));
    return;
  }

  size_t off = 0;
  if constexpr (!Aligned) {
    uint32_t head = static_cast<uint32_t>((LINE - (global & (LINE - 1))) & (LINE - 1));
    if (head > sizeInBytes) head = static_cast<uint32_t>(sizeInBytes);
    if (head) {
      ldsTileBytes<DIR, cp>(global, lds, head);
      off = head;
    }
  }

    // Bulk: whole 256B rows, split across as many tiles as the 16-bit length field needs.
  size_t rows = (sizeInBytes - off) / WIDTH;
  while (rows) {
    const uint32_t chunk = rows < MAX_ROWS ? static_cast<uint32_t>(rows) : MAX_ROWS;
    ldsTileRows<DIR, cp>(global + off, lds + static_cast<uint32_t>(off), chunk);
    off += static_cast<size_t>(chunk) * WIDTH;
    rows -= chunk;
  }

    // Tail: the < 256B sub-row remainder, again at byte granularity.
  const uint32_t tail = static_cast<uint32_t>(sizeInBytes - off);
  if (tail) ldsTileBytes<DIR, cp>(global + off, lds + static_cast<uint32_t>(off), tail);
}

// ---- core: partition + issue the whole copy for the team [start, stop). -----
// NO final wait. Work + LDS partition by rank within the team (warpId - start),
// so each team indexes its own `ldsBuffer` from zero. Safe to call collectively
// (warps outside the range return immediately) or only from the team's warps.
template <CachePolicy cp>
__device__ inline void issue(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
  const uint8_t* s = uniform(reinterpret_cast<const uint8_t*>(src));
  uint8_t* d = uniform(reinterpret_cast<uint8_t*>(dst));
  const uint32_t ldsBase = uniform(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ldsBuffer)));
  const uint32_t ldsBytes = uniform(static_cast<uint32_t>(ldsBufferBytes)); // LDS is small
  sizeBytes = uniform(static_cast<uint64_t>(sizeBytes));
  startWarpId = uniform(startWarpId);
  stopWarpId = uniform(stopWarpId);

  const uint32_t W = warpSize;
  const uint32_t nThreads = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t warpThread = tid % W;               // thread index within its warp
  const uint32_t warpId = uniform(tid / W);
  const uint32_t nWarps = (nThreads + W - 1) / W;

    // --- team membership: this warp participates iff in [start, stop) --------
  const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
  if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop) return; // not on this team
  const uint32_t rank = warpId - startWarpId; // rank within the team
  const uint32_t teamWarps = teamStop - startWarpId; // >= 1

  // active threads in THIS warp (handles partial final warp); stride for vector.
  const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

  // --- split the range: [head (vector)][ chunks (TDM) ] --------------------
  // Every chunk starts at an ALIGN boundary of the destination, and its load
  // starts delta = (src - dst) % ALIGN bytes earlier, at an ALIGN boundary of the
  // source. The difference is absorbed in LDS, where the store reads 16B-aligned
  // at slot + o: its multiple-of-4 part `sigma` by loading to slot + o - sigma
  // (4B aligned), and its sub-dword part `r` by the warp shifting the chunk down
  // in place before storing it. The extra delta bytes each load reads belong to
  // the previous chunk and are never stored from this one.
  const uint64_t sAddr = reinterpret_cast<uint64_t>(s);
  const uint64_t dAddr = reinterpret_cast<uint64_t>(d);
  const uint32_t delta = static_cast<uint32_t>(sAddr - dAddr) & (ALIGN - 1);
  const uint32_t r = delta & 3u;
  const uint32_t sigma = delta - r;
  const uint32_t o = (sigma + 15u) & ~15u;
  size_t head = (ALIGN - (dAddr & (ALIGN - 1))) & (ALIGN - 1);
  if (head < delta) head += ALIGN; // the first load must not start before src
  if (head > sizeBytes) head = sizeBytes;

  if (rank == 0 && head) warpVecCopy(s, d, head, warpThread, warpThreads);
  const size_t bulk = sizeBytes - head;
  if (bulk == 0) return;

  // --- LDS: one window per issuing warp, two slots per window --------------
  const uint32_t ldsPad = (16u - (ldsBase & 15u)) & 15u; // slots must be 16B aligned
  const uint32_t ldsUsable = ldsBytes > ldsPad ? ldsBytes - ldsPad : 0;
  const uint32_t maxByLds = ldsUsable / MIN_SLOT;
  if (maxByLds == 0) { // LDS too small to stage a chunk: vector fallback
    if (rank == 0) warpVecCopy(s + head, d + head, bulk, warpThread, warpThreads);
    return;
  }
  const size_t units = (bulk + ALIGN - 1) / ALIGN;
  uint32_t issuers = teamWarps < maxByLds ? teamWarps : maxByLds;
  if (units < issuers) issuers = static_cast<uint32_t>(units);
  if (rank >= issuers) return; // this warp doesn't issue

  const uint32_t window = (ldsUsable / issuers) & ~15u;
  const uint32_t nSlots = window >= 2 * MIN_SLOT && (r || window >= TWO_SLOT_WINDOW) ? 2 : 1;
  const uint32_t slotBytes = (window / nSlots) & ~15u;
  // slot + o + T (plus the shift's slack) fits; MIN_SLOT keeps T >= ALIGN
  uint32_t T = (slotBytes - o - (r ? SHIFT_SLACK : 0)) & ~(ALIGN - 1);
  if (T > MAX_CHUNK) T = MAX_CHUNK;

  // distribute the ALIGN-byte units across issuers by team rank (contiguous)
  const size_t base = units / issuers;
  const size_t extra = units % issuers;
  const size_t myStart = rank * base + (rank < extra ? rank : extra);
  const size_t myUnits = base + (rank < extra ? 1u : 0u);
  const size_t begin = head + myStart * ALIGN;
  const size_t end = begin + myUnits * ALIGN < sizeBytes ? begin + myUnits * ALIGN : sizeBytes;

  const uint32_t win = ldsBase + ldsPad + rank * window;
  const uint64_t sB = sAddr + begin - delta;
  const uint64_t dB = dAddr + begin;
  const size_t len = end - begin;
  auto storeChunk = [&](uint32_t slot, uint64_t global, uint32_t n) {
    if (r) shiftDown(slot + o, n, r, warpThread, warpThreads);
    tile1d<true, cp>(global, slot + o, n);
  };

  if (nSlots == 1) {
    // One slot: each chunk's store waits for its load, and the next load for the store.
    for (size_t off = 0; off < len; off += T) {
      const uint32_t n = len - off < T ? static_cast<uint32_t>(len - off) : T;
      tile1d<false, cp>(sB + off, win + o - sigma, n + delta);
      waitTensor0();
      storeChunk(win, dB + off, n);
      waitTensor0();
    }
    return;
  }

  // Two slots: each phase loads the next chunk into one slot, then stores (and,
  // when shifting, first realigns) the chunk the previous phase loaded into the
  // other. Refilling a slot needs its store complete, and that store was issued
  // after the other slot's load, so in-order completion leaves nothing to keep
  // in flight across the wait: it drains to zero.
  uint32_t slot = 0;
  for (size_t off = 0; off < len; off += T) {
    const uint32_t n = len - off < T ? static_cast<uint32_t>(len - off) : T;
    if (off) waitTensor0();
    tile1d<false, cp>(sB + off, win + slot + o - sigma, n + delta);
    if (off) storeChunk(win + (slot ^ slotBytes), dB + off - T, T);
    slot ^= slotBytes;
  }
  waitTensor0();
  const size_t lastOff = (len - 1) / T * T;
  storeChunk(win + (slot ^ slotBytes), dB + lastOff, static_cast<uint32_t>(len - lastOff));
}

} // namespace detail

// ---- public API definitions (declared at the top of this file) -------------

__device__ inline void tdmWait() {
  detail::waitTensor0();
}

template <CachePolicy cp>
__device__ inline void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                    size_t ldsBufferBytes) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
}

template <CachePolicy cp>
__device__ inline void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
  tdmWait();
}

template <CachePolicy cp>
__device__ inline void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                          size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
}

template <CachePolicy cp>
__device__ inline void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                     size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) {
  detail::issue<cp>(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
  tdmWait(); // no-op on any warp that issued nothing / is off-team
}

// The TDM descriptor holds the LDS side as a 32-bit address. A generic pointer
// into shared memory carries the LDS offset in its low 32 bits (the aperture
// occupies the high half), so the truncation below is the same conversion the
// copy path uses for its ldsBuffer argument.
namespace detail {
__device__ inline uint32_t ldsAddrOf(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}
} // namespace detail

template <SyncPolicy sp, CachePolicy cp, bool Aligned>
__device__ inline void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes) {
  detail::warpLdsCopy<detail::LdsDir::Load, cp, Aligned>(reinterpret_cast<uint64_t>(globalSrc),
                                                         detail::ldsAddrOf(ldsDst), sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) tdmWait();
}

template <SyncPolicy sp, CachePolicy cp, bool Aligned>
__device__ inline void asyncStoreFromLDS(const uint8_t* ldsSrc, uint8_t* globalDst, size_t sizeInBytes) {
  detail::warpLdsCopy<detail::LdsDir::Store, cp, Aligned>(reinterpret_cast<uint64_t>(globalDst),
                                                          detail::ldsAddrOf(ldsSrc), sizeInBytes);
  if constexpr (sp == SyncPolicy::Sync) tdmWait();
}

#endif // TDM_SUPPORTED
// On an unsupported target the entry points were declared `= delete` at the top,
// so there is nothing to define here -- any call is a compile-time error.

} // namespace tdm
