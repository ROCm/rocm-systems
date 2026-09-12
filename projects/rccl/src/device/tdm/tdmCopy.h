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

#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && TDM_TOOLCHAIN_AVAILABLE
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
/// \tparam localCp  Compile-time cache policy (temporal hint + memory scope) for the
///                  READ leg, i.e. the tensor load that pulls \p src into LDS. Named
///                  for the common case where the source is local HBM.
/// \tparam remoteCp Same, for the WRITE leg: the tensor store that pushes LDS out to
///                  \p dst, typically a peer's memory across the interconnect.
///
/// Each policy becomes the immediate cpol operand of its own instruction, so both
/// must be constants. Build them with createCachePolicy(); see cachePolicy.h. The
/// unaligned head (and the LDS-too-small fallback) is an ordinary vector copy and
/// carries neither policy.
///
/// \note For BLOCK-wide visibility, follow with __syncthreads() (and see the
///       visibility notes at the bottom of the file).
/// \warning \p src and \p dst must be GLOBAL (HBM) pointers. Passing a shared /
///          LDS pointer compiles (it decays to void*) but is undefined at run
///          time -- the address is programmed into the descriptor's global field.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopy(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                size_t ldsBufferBytes) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopy(): issue and return.
///
/// Identical partitioning to tdmCopy(), but does NOT drain on return. The last
/// few TDM ops (bounded by the per-wave queue depth) stay in flight so they
/// overlap with whatever the calling warp does next. Pair with tdmWait().
///
/// \see tdmCopy for parameter meanings (including the two cache policies).
/// \see tdmWait to complete the copy.
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
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
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
__device__ TDM_API void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer,
                                      size_t ldsBufferBytes, uint32_t startWarpId, uint32_t stopWarpId) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopyByTeam(): issue and return.
/// \see tdmCopyByTeam for parameter meanings and team semantics (incl. the cache policies).
/// \see tdmWait to complete the copy (called by each participating warp).
template <CachePolicy localCp = DEFAULT_CACHE_POLICY, CachePolicy remoteCp = DEFAULT_CACHE_POLICY>
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
///                 immediate cpol operand; see cachePolicy.h. One leg touches one
///                 global buffer, so there is a single policy here -- the caller
///                 passes whichever of its local/remote policies applies (the
///                 round-trip copies above take both).
/// \tparam Aligned Pass true only when \p globalSrc and \p ldsDst are both known
///                 to be 128-byte aligned; this compiles out the head peel.
///
/// \warning \p globalSrc and \p ldsDst must share the same sub-128B alignment
///          (the head is peeled against the global pointer and the same offset
///          is applied to the LDS side). RCCL's LDS staging buffers are
///          128-byte aligned, which satisfies this for an aligned source.
template <SyncPolicy sp = DEFAULT_SYNC_POLICY, CachePolicy cp = DEFAULT_CACHE_POLICY, bool Aligned = false>
__device__ TDM_API void asyncLoadToLDS(const uint8_t* globalSrc, uint8_t* ldsDst, size_t sizeInBytes) TDM_DELETED;

/// \brief Warp-level transfer of \p sizeInBytes from LDS out to global memory.
/// \see asyncLoadToLDS for the calling convention, template parameters and the
///      matching-alignment requirement (peeled against \p globalDst here).
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
//   [ head (VECTOR) ][ ---- aligned 256B rows (TDM) ---- ][ tail (TDM 1-D) ]
//   * Fixed choices for bandwidth: 4-byte data_size, 256B TDM row width.
//   * `head` brings the SOURCE up to a 128B boundary (the direct-copy
//     requirement); being the unaligned remainder, it stays a cooperative
//     vector copy done by the team's first warp.
//   * The aligned bulk is 2D TDM tiles (64 dwords x N rows of 256B).
//   * `tail` (< 256B sub-row remainder) is a separate 1-D TDM op at byte
//     granularity. TDM's out-of-bounds clamp is per-dimension (rectangular) and
//     cannot express "N full rows + a partial row", so the partial row must be
//     its own tile rather than riding the bulk descriptor's clamp.
//
// HARDWARE MODEL (why the per-tile waits are REQUIRED)
//   * TDM engines: 1 per SIMD-pair -> 2 per WGP; a warp runs on 1 SIMD32 and
//     shares its pair's engine. Bandwidth needs >=2 issuing warps/block (both
//     engines) and many blocks (many WGPs); a lone warp is latency-bound.
//   * Same-wave TDM ops ISSUE in order but their memory effects OVERLAP: up to 3
//     ops are outstanding per wave (that is what TENSORcnt counts). In-order issue
//     does NOT serialize completion, so it does NOT make LDS reuse hazard-free.
//   * This copy is single-buffered (one LDS window per warp), so each tile is a
//     load->store dependency chain on that window: the store reads what the load
//     wrote (RAW), and the next tile's load overwrites the window the store is
//     still draining (WAR). Both edges need an s_wait_tensorcnt, so issueRows()/
//     issueRow1d() wait after the load and after the store. Consequence: with a
//     single window the copy is effectively serialized; tdmCopyAsync() therefore
//     overlaps very little today. Regaining overlap needs DOUBLE-BUFFERING (>=2
//     LDS windows per warp) so a load into window B runs while window A stores.
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
// how asyncCopy.h passes its cache policy to the async builtins). Each instruction
// carries one cpol, which is what lets a round trip give its read and write legs
// different policies (see issueRows()).
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

// ---- issue one chunk of whole 256B rows (2D tile) through ONE LDS window. ---
// This staging window is single-buffered, so the two TDM ops form a dependency
// chain that MUST be enforced with TENSORcnt waits -- up to 3 TDM ops are
// outstanding per wave (they overlap), so "same-wave in-order issue" does NOT
// serialize their memory effects:
//   * load -> wait: the store reads the LDS the load just wrote (RAW hazard).
//   * store -> wait: the caller reuses this same window next iteration; the next
//     load must not overwrite LDS the store is still draining (WAR hazard).
//
// The load reads `src` (the near side, localCp) and the store writes `dst` (the far
// side, remoteCp); the LDS window in between is untouched by either policy.
template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void issueRows(uint64_t src, uint32_t lds, uint64_t dst, uint32_t rows) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS4);
  g1.tileDim0(TD0);
  g1.tileDim1(rows);
  g1.tensorDim0(TD0);
  g1.tensorDim1(rows);
  g1.tensorDim0Stride(TD0);                // rows back-to-back (contiguous)

    // Higher dims (D2/D3/D4) are unused for this 2D tile -> passed as zero inside
    // load()/store(), matching known-good example usage.
  gfx1250_TDM_GROUP0 g0l(lds, src);
  load<localCp>(g0l, g1);
  waitTensor0();       // RAW: fill LDS before store reads it
  gfx1250_TDM_GROUP0 g0s(lds, dst);
  store<remoteCp>(g0s, g1);
  waitTensor0();       // WAR: drain store before window reuse
}

// ---- issue a sub-row tail (<256B) as a 1-D tile at BYTE granularity. ---------
// Same single-buffered LDS window and the same required RAW/WAR waits as above.
template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void issueRow1d(uint64_t src, uint32_t lds, uint64_t dst, uint32_t nbytes) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS1);                        // 1-byte elements: exact length
  g1.tileDim0(nbytes);
  g1.tileDim1(1);
  g1.tensorDim0(nbytes);
  g1.tensorDim1(1);
  g1.tensorDim0Stride(nbytes);

  gfx1250_TDM_GROUP0 g0l(lds, src);        // unused higher dims -> zero (see load())
  load<localCp>(g0l, g1);
  waitTensor0();       // RAW: fill LDS before store reads it
  gfx1250_TDM_GROUP0 g0s(lds, dst);
  store<remoteCp>(g0s, g1);
  waitTensor0();       // WAR: drain store before window reuse
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
  g1.tileDim1(1);
  g1.tensorDim0(nbytes);
  g1.tensorDim1(1);
  g1.tensorDim0Stride(nbytes);
  gfx1250_TDM_GROUP0 g0(lds, global);
  ldsIssue<DIR, cp>(g0, g1);
}

// One 2-D tile of `rows` back-to-back 256B rows -- the bulk shape, chosen for
// bandwidth exactly as in issueRows(). `rows` must be <= TILE_DIM_MAX.
template <LdsDir DIR, CachePolicy cp>
__device__ inline void ldsTileRows(uint64_t global, uint32_t lds, uint32_t rows) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(DS4);
  g1.tileDim0(TD0);
  g1.tileDim1(rows);
  g1.tensorDim0(TD0);
  g1.tensorDim1(rows);
  g1.tensorDim0Stride(TD0);                // rows back-to-back (contiguous)
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
// applied to the LDS side, so the bulk's 4-byte elements stay naturally aligned
// only when the two pointers start with the same sub-128B offset -- the contract
// documented on the public entry points. `Aligned` asserts that offset is zero
// for both and compiles the peel out entirely.
template <LdsDir DIR, CachePolicy cp, bool Aligned>
__device__ inline void warpLdsCopy(uint64_t global, uint32_t lds, size_t sizeInBytes) {
  if (sizeInBytes == 0) return;            // a zero-extent tile is not a valid descriptor

  size_t off = 0;
  if constexpr (!Aligned) {
    uint32_t head = static_cast<uint32_t>((LINE - (global & (LINE - 1))) & (LINE - 1));
    if (head > sizeInBytes) head = static_cast<uint32_t>(sizeInBytes);
    if (head) {
      ldsTileBytes<DIR, cp>(global, lds, head);
      off = head;
    }
  }

    // Bulk: whole 256B rows, split across as many tiles as the 16-bit row count needs.
  size_t rows = (sizeInBytes - off) / WIDTH;
  while (rows) {
    const uint32_t chunk = rows < TILE_DIM_MAX ? static_cast<uint32_t>(rows) : TILE_DIM_MAX;
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
template <CachePolicy localCp, CachePolicy remoteCp>
__device__ inline void issue(void* dst, const void* src, size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  uint8_t* d = reinterpret_cast<uint8_t*>(dst);
  const uint32_t ldsBase = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ldsBuffer));
  const uint32_t ldsBytes = static_cast<uint32_t>(ldsBufferBytes);  // LDS is small

  const uint32_t W = warpSize;
  const uint32_t nThreads = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t warpThread = tid % W;               // thread index within its warp
  const uint32_t warpId = tid / W;
  const uint32_t nWarps = (nThreads + W - 1) / W;

    // --- team membership: this warp participates iff in [start, stop) --------
  const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
  if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop) return; // not on this team
  const uint32_t rank = warpId - startWarpId; // rank within the team
  const uint32_t teamWarps = teamStop - startWarpId; // >= 1

  // active threads in THIS warp (handles partial final warp); stride for vector.
  const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

  // --- split the range: [head][ aligned 256B rows ][tail] ------------------
  uint64_t sAddr = reinterpret_cast<uint64_t>(s);
  uint32_t head = static_cast<uint32_t>((128u - (sAddr & 127u)) & 127u);
  if (head > sizeBytes) head = static_cast<uint32_t>(sizeBytes);
  size_t bulk = sizeBytes - head; // starts 128B-aligned
  size_t rows = bulk / WIDTH; // whole 256B rows
  size_t tdmBytes = rows * WIDTH;
  size_t tailOff = head + tdmBytes;
  uint32_t tail = static_cast<uint32_t>(sizeBytes - tailOff); // < 256B

  // --- edges (team's FIRST warp = rank 0): vector head, TDM tail -----------
  if (rank == 0 && head) warpVecCopy(s, d, head, warpThread, warpThreads);
  if (rank == 0 && tail) {
    if (ldsBytes >= tail) // stage tail in rank 0's window
      issueRow1d<localCp, remoteCp>(reinterpret_cast<uint64_t>(s + tailOff), ldsBase,
                                    reinterpret_cast<uint64_t>(d + tailOff), tail);
    else warpVecCopy(s + tailOff, d + tailOff, tail, warpThread, warpThreads);
  }

  // --- aligned bulk via TDM ------------------------------------------------
  if (rows == 0) return; // no aligned bulk (edges done)
  uint32_t maxByLds = ldsBytes / WIDTH; // #warps we can give a window
  if (maxByLds == 0) { // LDS < 256B: vector fallback
    if (rank == 0) warpVecCopy(s + head, d + head, tdmBytes, warpThread, warpThreads);
    return;
  }
  uint32_t issuers = teamWarps < maxByLds ? teamWarps : maxByLds;
  uint32_t window = (ldsBytes / issuers) & ~(WIDTH - 1); // per-warp 256B-multiple
  uint32_t rowsPerChunk = window / WIDTH;

  if (rank >= issuers) return; // this warp doesn't issue

  // distribute `rows` across issuers by team rank (contiguous row blocks)
  size_t base = rows / issuers;
  size_t extra = rows % issuers;
  size_t myRows = base + (rank < extra ? 1u : 0u);
  size_t myStart = rank * base + (rank < extra ? rank : extra);
  if (myRows == 0) return;

  uint32_t myLds = ldsBase + rank * window;
  uint64_t sBase = reinterpret_cast<uint64_t>(s + head) + (uint64_t)myStart * WIDTH;
  uint64_t dBase = reinterpret_cast<uint64_t>(d + head) + (uint64_t)myStart * WIDTH;

  for (size_t r = 0; r < myRows; r += rowsPerChunk) {
    uint32_t chunkRows = (myRows - r < rowsPerChunk) ? static_cast<uint32_t>(myRows - r) : rowsPerChunk;
    uint64_t off = (uint64_t)r * WIDTH;
    issueRows<localCp, remoteCp>(sBase + off, myLds, dBase + off, chunkRows);
  }
}

} // namespace detail

// ---- public API definitions (declared at the top of this file) -------------

__device__ inline void tdmWait() {
  detail::waitTensor0();
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
