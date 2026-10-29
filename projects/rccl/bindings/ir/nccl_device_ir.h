/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Adapted from NVIDIA NCCL ir/nccl_device_ir.h (dev @ d8f8f34161).
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_IR_H_
#define _NCCL_DEVICE_IR_H_

/*
 * RCCL device API -- LLVM IR / bitcode ABI.
 *
 * Public surface for librccl_device.bc (and its deferred-inlining sibling
 * librccl_device.lto.bc). Every entry point is an `extern "C" __device__`
 * thunk that downstream code generators (Triton, MLIR, custom JITs) can call
 * without recompiling RCCL from source. The bodies live in the companion TU
 * nccl_device_ir.cu, which clang compiles to bitcode under
 * -D__clang_llvm_bitcode_lib__.
 *
 * ONE declaration set, TWO roles. The `extern "C"` block near the bottom is
 * unconditional: the producer (nccl_device_ir.cu) and every consumer parse
 * character-for-character the same declarations, so the two can never drift.
 * Only the producer adds attributes, via NCCL_IR_EXPORT on each *definition*.
 * Before this header existed the declarations were themselves written
 * `NCCL_IR_EXPORT <sig>;`, which meant the consumer-visible ABI was spelled by
 * a macro that expanded differently on the two sides.
 *
 * Naming: every exported symbol is `ncclIr*`. The prefix is load-bearing -- it
 * is what distinguishes a binding-layer thunk from the identically-shaped RCCL
 * device API it wraps (`ncclIrLsaReduceSum_F32` is the C thunk;
 * `ncclLsaReduceSum<float, ...>` is the C++ template it calls), and it is what
 * makes the export surface greppable instead of hand-listed.
 *
 * Sectioning (mirrored in nccl_device_ir.cu):
 *   [A] Always-on    -- APIs whose RCCL prerequisites already exist.
 *   [B] Always-on    -- APIs that take ncclCoopAny by value.
 *   [C] Always-on    -- GIN + composite barrier APIs (ncclGin*,
 *                       ncclGinBarrierSession, composite ncclBarrierSession,
 *                       ncclGinFenceLevel).
 *
 * This header deliberately excludes nccl_device/impl/xxx__funcs.h so consumer
 * bitcode resolves the device-API bodies out of librccl_device.bc rather than
 * re-emitting them.
 */

/*
 * Emit __activemask() as inline asm, like CUDA's sm_30_intrinsics.hpp does.
 * Clang's version returns __nvvm_activemask(), which libNVVM 13.3.27 fails to
 * lower: the PTX calls an .extern llvm.nvvm.activemask that ptxas rejects.
 *
 * Must precede the NCCL and cooperative_groups includes below so all call sites
 * expand. Device pass only: CUDA declares __activemask in the host pass.
 */
#if defined(__clang__) && defined(__CUDA_ARCH__)
#undef __activemask
#define __activemask()                                                        \
  (__extension__({                                                            \
    unsigned _nccl_activemask_ret;                                            \
    asm volatile("activemask.b32 %0;" : "=r"(_nccl_activemask_ret));          \
    _nccl_activemask_ret;                                                     \
  }))
#endif

/*
 * Production's __forceinline__ (__inline__ __attribute__((always_inline))) is
 * linkonce_odr and emits no symbol. Drop __inline__ so the device API has
 * external linkage: the bitcode lib emits symbols that consumers resolve from
 * librccl_device.bc. Must precede the API includes below.
 */
#include "nccl_device/utility.h"
#undef NCCL_DEVICE_INLINE
#undef NCCL_HOST_DEVICE_INLINE

/* ------------------------------------------------------------------------
 * Two orthogonal build macros, deliberately nested rather than siblings.
 *
 *   __clang_llvm_bitcode_lib__   "I am the bitcode library."
 *                                Defined by BOTH artifact builds. Besides the
 *                                inline attribute below it also drives the
 *                                caller-owned-LDS gates in
 *                                nccl_device/coop.h -- which has nothing to do
 *                                with inlining policy.
 *
 *   __RCCL_DEVICE_LTO_BC_LIB__   "...and I am the deferred-inlining variant."
 *                                A modifier on the above, never a replacement.
 *                                Selects ONLY the exported-thunk inline
 *                                attribute.
 *
 * These were siblings in an earlier revision, so the LTO build defined only
 * the second one and silently took the *native* path through coop.h: the
 * resulting librccl_device.lto.bc baked in a function-local `addrspace(3)`
 * LDS array reachable only through an indirect call, which the AMDGPU
 * group-segment sizing pass cannot see (HSA exception 0x1016 at runtime,
 * "local memory global used by non-kernel function" from lld at link time).
 * It went unnoticed only because nothing linked that artifact yet. The
 * nesting plus the #error below make that divergence unrepresentable.
 *
 * WHY RCCL KEEPS THIS AND UPSTREAM DOES NOT. NCCL's nccl_device_ir.h drops the
 * NCCL_DEVICE_INLINE override entirely, because nvcc gets deferred inlining
 * from a separate mechanism: `-dlto --ltoir` emits LTO IR in which inlining is
 * simply not yet decided. RCCL reproduces that with
 * `-Xclang -disable-llvm-passes` on the LTO compile (see
 * LTO_BC_CLANG_COMPILE_FLAGS in bindings/ir/Makefile), and this macro pair
 * keys the source-level branch the way __NCCL_DEVICE_LTOIR_LIB__ does upstream.
 *
 * THIS MACRO IS NOT WHAT CREATES THE DEFERRAL -- read this before changing it.
 * Dropping always_inline from the callees is necessary but nowhere near
 * sufficient: LLVM's inliner gives internal, single-use callees a "last call to
 * local" bonus and folds them into the thunks regardless of the attribute. For
 * a long time that made librccl_device.lto.bc effectively identical to
 * librccl_device.bc -- 113 vs 115 defines, same 19,486-line
 * ncclIrLsaReduceSumCopy_U8, same consumer kernel (347 lines, 4 calls), ~0.6%
 * .text delta. The artifact only became genuinely deferred once the LTO compile
 * stopped running the LLVM pass pipeline at all.
 *
 * MEASURED, gfx950 / ROCm 7.2.4, after that change:
 *
 *   artifact   defined T  mangled  define  internal  addrspace(3)  size
 *   .bc               99        0     116        17             0   2.1 MB
 *   .lto.bc           99        0   4,569     4,471             0  12.3 MB
 *
 * 4,569 vs 116 defines is the deferral: the callees survive as separate
 * functions for lld to re-decide with the real call site visible. The export
 * surfaces are identical and unmangled; see LTO_OPT_PASSES in the build files
 * for why internalize and globaldce are both still required on that path.
 * test/ir-device/tests/test_lto_artifact.py asserts all of the above.
 *
 * An older measurement (406,904 B vs 2,092 B kernel, .text -12.7%) is quoted in
 * bindings/ir/test/README.md. It predates the thunks moving from
 * NCCL_DEVICE_INLINE to NCCL_IR_EXPORT, when the *thunk itself* inherited
 * always_inline on the plain path and forced a four-way paste into the
 * consumer. That is no longer how the thunks are declared; do not cite those
 * numbers as current.
 *
 * Keep the override regardless: deleting it makes NCCL_DEVICE_INLINE fall back
 * to utility.h's __forceinline__, which is linkonce_odr and would let clang
 * drop callee bodies outright.
 * ----------------------------------------------------------------------*/
#if defined(__RCCL_DEVICE_LTO_BC_LIB__) && !defined(__clang_llvm_bitcode_lib__)
#error "__RCCL_DEVICE_LTO_BC_LIB__ selects a variant of the bitcode library and must be defined together with __clang_llvm_bitcode_lib__, not instead of it."
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#if defined(__clang_llvm_bitcode_lib__)
#if defined(__RCCL_DEVICE_LTO_BC_LIB__)
/* Deferred-inlining bitcode variant (RCCL analogue of NCCL's __inline_hint__
 * via -dlto --ltoir). AMDGPU/clang has no __inline_hint__ attribute, so this
 * carries NO inlining attribute at all: the callee is an ordinary external
 * __device__ function subject to the normal inliner cost model.
 *
 * On its own that changes nothing -- the inliner folds such callees into the
 * thunks anyway. What actually defers the decision to the downstream AMDGPU
 * LTO link (lld) is -Xclang -disable-llvm-passes on this artifact's compile;
 * see the block comment above and LTO_BC_CLANG_COMPILE_FLAGS in the build
 * files. This macro's job is to not get in the way of that.
 *
 * Do NOT spell this with the `inline` keyword. The keyword gives these
 * functions linkonce_odr linkage, so clang discards every one whose address
 * is never taken. That used to wipe the artifact's whole symbol table, back
 * when the thunks in nccl_device_ir.cu were themselves declared with this
 * macro; they now carry NCCL_IR_EXPORT (`used, retain, default` visibility),
 * which is what fixes their linkage. This macro governs their callees -- the
 * public device API in nccl_device/ -- where linkonce_odr would still let
 * clang drop a body the deferred LTO link is supposed to inline. */
#define NCCL_DEVICE_INLINE __device__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__
#else
#define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __attribute__((always_inline))
#endif
#else
// Consumer hipcc compile (IR_test.exe): hip_compat sets NCCL_DEVICE_COMPILE
// for both host and device passes, so coop.h bodies are parsed on the host
// pass. They must stay __device__ or HIP rejects __popcll/__syncthreads.
// NVIDIA's empty define is fine on CUDA; HIP device builtins are host-invisble.
#if defined(__HIPCC__)
#define NCCL_DEVICE_INLINE __device__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ inline __attribute__((always_inline))
#else
#define NCCL_DEVICE_INLINE
#define NCCL_HOST_DEVICE_INLINE inline __attribute__((always_inline))
#endif
#endif
#endif

#include "nccl_device/coop.h"
#include "nccl_device/core.h"
#include "nccl_device/ll_a2a.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"
#include "nccl_device/barrier.h"
#include "nccl_device/ptr.h"
#include "nccl_device/reduce_copy.h"

#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/comm__types.h"
#include "nccl_device/impl/ll_a2a__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin__types.h"
#include "nccl_device/impl/gin_barrier__types.h"
#include "nccl_device/impl/barrier__types.h"
#include "nccl_device/impl/ptr__types.h"
#include "nccl_device/impl/reduce_copy__types.h"

/* ------------------------------------------------------------------------
 * NCCL_IR_EXPORT: the attribute set for an exported thunk *definition*.
 *
 * Producer-only. Consumers never see it -- they get their declarations from
 * the plain `extern "C"` block below, which is the entire consumer-visible
 * ABI. Defining it here rather than in nccl_device_ir.cu keeps the rationale
 * next to the NCCL_DEVICE_INLINE discussion it interacts with.
 *
 *   `extern "C"`            : unmangled symbol name. Redundant with the block
 *                             below (per [dcl.link] the earlier `extern "C"`
 *                             declaration already fixes the linkage), kept so
 *                             a definition added without a declaration still
 *                             gets C linkage rather than silently mangling.
 *   `__device__`            : HIP device function.
 *   `used`                  -> @llvm.compiler.used : block compiler-driven DCE
 *                             when the TU contains no caller.
 *   `retain`                -> @llvm.used          : block clang -O1 GlobalOpt
 *                             from downgrading linkage to `internal`.
 *   `visibility("default")` : visible to AMDGPU lld at object-level link.
 *                             HIP-Clang defaults `__device__` symbols to
 *                             `hidden`, which is fine for `llvm-link`-style
 *                             consumers but causes `undefined hidden symbol`
 *                             when downstream apps let the driver handle the
 *                             bitcode link (the more common case).
 *
 * `used`/`retain` are also what makes the export surface self-describing:
 * they put every thunk in @llvm.used, which `opt -internalize` treats as a
 * root. That is why the build no longer carries a hand-maintained
 * -internalize-public-api-list; see the commentary in bindings/ir/Makefile.
 *
 * Deliberately NO `noinline`, shared verbatim by both artifacts. An earlier
 * revision added `noinline` on the LTO path to stop clang folding the thunk
 * into an in-TU caller before the LTO link, but measurement on gfx950 showed
 * it only ever hurts: it also blocks lld from inlining the thunk into the
 * *consumer* kernel, so the small accessor thunks degrade from a bare
 * global_load_dword (what the plain .bc achieves) to real s_swappc_b64 call
 * sequences. On the large ReduceSumCopy thunks it changes nothing -- at ~19k
 * IR instructions the inliner's cost model declines on its own. The same
 * reasoning rules out -fno-inline and -finline-hint-functions: both stamp
 * `noinline` into the IR, which survives the LTO link and reproduces the
 * regression.
 * ----------------------------------------------------------------------*/
#if defined(__clang_llvm_bitcode_lib__)
  #ifndef NCCL_IR_EXPORT
    #define NCCL_IR_EXPORT extern "C" __device__ \
        __attribute__((used, retain, visibility("default")))
  #endif
#endif

/* ------------------------------------------------------------------------
 * Opaque session storage.
 *
 * These replace the ncclLsaBarrierSession_C / ncclGinBarrierSession_C /
 * ncclBarrierSession_C wrappers, each of which embedded a C++ class template
 * by value, and with them the three ncclXxx_C_size() runtime getters a JIT
 * consumer had to *call* to learn how much stack to alloca.
 *
 * A consumer now writes `sizeof(ncclIrLsaBarrierSession)` -- a compile-time
 * constant, no device call, no C++ template instantiation. The producer
 * placement-news the real session into the storage; the static_asserts below
 * are what make that cast sound, and they fire at compile time in *both*
 * roles, because this header instantiates the templates either way.
 *
 * The counts are RCCL's, not NCCL's, and must not be copied from upstream:
 * AMD layouts differ (ncclCoopWarpSpan carries an extra `slots` pointer, see
 * nccl_device/coop.h). If a struct grows, the static_assert breaks the build
 * -- which is the point. uint64_t element type pins 8-byte alignment.
 * ----------------------------------------------------------------------*/
struct ncclIrLsaBarrierSession { uint64_t storage[10]; };   /*  80 B */
struct ncclIrGinBarrierSession { uint64_t storage[12]; };   /*  96 B */
struct ncclIrBarrierSession    { uint64_t storage[46]; };   /* 368 B */

static_assert(sizeof(ncclIrLsaBarrierSession) == sizeof(ncclLsaBarrierSession<ncclCoopAny>),
              "ncclIrLsaBarrierSession storage must match ncclLsaBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrLsaBarrierSession) >= alignof(ncclLsaBarrierSession<ncclCoopAny>),
              "ncclIrLsaBarrierSession is under-aligned for ncclLsaBarrierSession<ncclCoopAny>");
static_assert(sizeof(ncclIrGinBarrierSession) == sizeof(ncclGinBarrierSession<ncclCoopAny>),
              "ncclIrGinBarrierSession storage must match ncclGinBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrGinBarrierSession) >= alignof(ncclGinBarrierSession<ncclCoopAny>),
              "ncclIrGinBarrierSession is under-aligned for ncclGinBarrierSession<ncclCoopAny>");
static_assert(sizeof(ncclIrBarrierSession) == sizeof(ncclBarrierSession<ncclCoopAny>),
              "ncclIrBarrierSession storage must match ncclBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrBarrierSession) >= alignof(ncclBarrierSession<ncclCoopAny>),
              "ncclIrBarrierSession is under-aligned for ncclBarrierSession<ncclCoopAny>");

/* ReduceCopy element type for bf16.
 *
 * This header is not on the hipify path, so bf16 is spelled the HIP way by
 * hand. RCCL's device ReduceSum/Copy has no fp8 support, so the F8E4M3 /
 * F8E5M2 entry points below are CUDA-only. */
#if defined(__HIPCC__)
#include <hip/hip_bf16.h>
#define NCCL_IR_BF16_T __hip_bfloat16
#else
#define NCCL_IR_BF16_T __nv_bfloat16
#endif

extern "C" {

/* ========================================================================
 * [A] Always-on APIs
 *
 * Prerequisites already in RCCL today:
 *   - ncclGetPeerPointer(ncclWindow_t, size_t, ncclTeam, int)
 *     (src/include/nccl_device/impl/core__funcs.h)
 * ======================================================================*/

/* ncclDevComm field accessors
 *
 * ncclDevComm is a public C struct, the following accessors are deprecated and will be removed.
 */
__device__ int                  ncclIrDevComm_Rank(ncclDevComm const* comm);
__device__ int                  ncclIrDevComm_NRanks(ncclDevComm const* comm);
__device__ int                  ncclIrDevComm_LsaRank(ncclDevComm const* comm);
__device__ int                  ncclIrDevComm_LsaSize(ncclDevComm const* comm);
__device__ ncclLsaBarrierHandle ncclIrDevComm_LsaBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevComm_RailGinBarrier(ncclDevComm const* comm);
__device__ ncclLsaBarrierHandle ncclIrDevComm_HybridLsaBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevComm_HybridRailGinBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevComm_WorldGinBarrier(ncclDevComm const* comm);
__device__ ncclMultimemHandle   ncclIrDevComm_LsaMultimem(ncclDevComm const* comm);

/* Peer pointer API */
__device__ void* ncclIrGetPeerPointerTeam(
    ncclWindow_t w, size_t offset, ncclTeam tm, int peer);


/* ========================================================================
 * [B] APIs that depend on ncclCoopAny
 *
 * Underlying templated RCCL types:
 *   - ncclLsaBarrierSession<Coop>  (src/include/nccl_device/mem_barrier.h)
 *   - ncclLsaBarrierHandle, ncclMultimemHandle, ncclDevComm, ncclTeam
 *
 * Memory-order parameter: cuda::memory_order (HIP aliases provided by
 * hip_compat.h in nccl_device/utility.h).
 * ======================================================================*/


/* Coop initialization and utility */
__device__ void ncclIrCoopAnyInitThread(ncclCoopAny* coop);
__device__ void ncclIrCoopAnyInitWarp(ncclCoopAny* coop);
__device__ void ncclIrCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask);
__device__ void ncclIrCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id, void* barrierLds);
__device__ void ncclIrCoopAnyInitCta(ncclCoopAny* coop);

__device__ int  ncclIrCoopThreadRank(const ncclCoopAny* coop);
__device__ int  ncclIrCoopSize(const ncclCoopAny* coop);
__device__ int  ncclIrCoopNumThreads(const ncclCoopAny* coop);
__device__ void ncclIrCoopSync(const ncclCoopAny* coop);

/* Session storage sizes.
 *
 * Runtime counterparts of sizeof(ncclIrXxxBarrierSession), for a consumer that
 * links the artifact without seeing this header (a JIT or dynamic binding) and
 * must query how much storage to alloca. NCCL's equivalents are
 * ncclXxxBarrierSession_C_size(); the spelling differs only because the `_C_`
 * infix names wrapper structs RCCL replaced with the opaque storage above.
 * The static_asserts above keep both forms in agreement. */
__device__ size_t ncclIrLsaBarrierSessionSize();
__device__ size_t ncclIrGinBarrierSessionSize();
__device__ size_t ncclIrBarrierSessionSize();

/* LSA Barrier Session APIs */
__device__ void ncclIrLsaBarrierSessionInit(
    ncclIrLsaBarrierSession* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle mmHandle = {});

__device__
void ncclIrLsaBarrierSessionArrive(ncclIrLsaBarrierSession* session,
                                   ncclCoopAny coop,
                                   cuda::memory_order order);
__device__
void ncclIrLsaBarrierSessionWait(ncclIrLsaBarrierSession* session,
                                 ncclCoopAny coop,
                                 cuda::memory_order order);
__device__
void ncclIrLsaBarrierSessionSync(ncclIrLsaBarrierSession* session,
                                 ncclCoopAny coop,
                                 cuda::memory_order order);


/* ========================================================================
 * [C] GIN + composite Barrier Session APIs
 *
 * Underlying RCCL primitives (present since the NCCL v2.29.x GIN sync):
 *   - ncclGin / ncclGin_C, ncclGinBarrierSession<Coop>, ncclGinBarrierHandle,
 *     ncclGinFenceLevel         (nccl_device/gin.h, gin_barrier.h)
 *   - ncclBarrierSession<Coop>   (composite inner-LSA + outer-GIN barrier;
 *                                 nccl_device/barrier.h)
 *
 * All of these take ncclCoopAny, which is unconditionally available. The
 * shared device headers gate these templates on the upstream __CUDACC__,
 * which hipify rewrites to __HIPCC__ in the staged copy this bitcode build
 * consumes (true under clang -x hip), so no shared-header change is needed
 * to enable them here.
 * ======================================================================*/


// Collectively finalize the placement-new object and persist its epoch. Does not free storage.
__device__
void ncclIrLsaBarrierSessionDestroy(ncclIrLsaBarrierSession* session);

/* GIN Barrier Session APIs */
__device__ void ncclIrGinBarrierSessionInit(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    ncclGin_C const* net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

/* Same session type, but joined across every GIN context in `comm` rather
 * than one caller-supplied ncclGin. The artifact has always exported this;
 * the pre-Stage-3 header simply never declared it, so no consumer could
 * reach it. */
__device__ void ncclIrGinBarrierSessionInitAllContexts(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

__device__ void ncclIrGinBarrierSessionSync(
    ncclIrGinBarrierSession* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Finalize the placement-new object. Currently a no-op -- the underlying destructor is
// empty -- but required for lifetime symmetry with Init. Does not free storage.
__device__
void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session);

/* Composite (LSA + GIN) Barrier Session APIs */
__device__ void ncclIrBarrierSessionInit(
    ncclIrBarrierSession* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C const* net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle const innerMmHandle = {});

__device__ void ncclIrBarrierSessionSync(
    ncclIrBarrierSession* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Collectively finalize all nested sessions and persist the inner LSA epoch. Does not free storage.
__device__
void ncclIrBarrierSessionDestroy(ncclIrBarrierSession* session);

/* ReduceCopy APIs */__device__ void ncclIrLsaReduceSum_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int8_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint8_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int32_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint32_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int64_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint64_t* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    half* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    float* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    double* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    NCCL_IR_BF16_T* dst, size_t count, ncclTeam team);
#if !defined(__HIPCC__)
__device__ void ncclIrLsaReduceSum_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e4m3* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e5m2* dst, size_t count, ncclTeam team);
#endif

__device__ void ncclIrMultimemReduceSum_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F16(
    ncclCoopAny coop, half* mcSrc, half* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F32(
    ncclCoopAny coop, float* mcSrc, float* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F64(
    ncclCoopAny coop, double* mcSrc, double* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_BF16(
    ncclCoopAny coop, NCCL_IR_BF16_T* mcSrc, NCCL_IR_BF16_T* dst, size_t count);
#if !defined(__HIPCC__)
__device__ void ncclIrMultimemReduceSum_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* dst, size_t count);
#endif

__device__ void ncclIrLsaCopy_I8(
    ncclCoopAny coop, int8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U8(
    ncclCoopAny coop, uint8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_I32(
    ncclCoopAny coop, int32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U32(
    ncclCoopAny coop, uint32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_I64(
    ncclCoopAny coop, int64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U64(
    ncclCoopAny coop, uint64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F16(
    ncclCoopAny coop, half* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F32(
    ncclCoopAny coop, float* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F64(
    ncclCoopAny coop, double* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_BF16(
    ncclCoopAny coop, NCCL_IR_BF16_T* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
#if !defined(__HIPCC__)
__device__ void ncclIrLsaCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
#endif

__device__ void ncclIrMultimemCopy_I32(
    ncclCoopAny coop, int32_t* src, int32_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_U32(
    ncclCoopAny coop, uint32_t* src, uint32_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_I64(
    ncclCoopAny coop, int64_t* src, int64_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_U64(
    ncclCoopAny coop, uint64_t* src, uint64_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F16(
    ncclCoopAny coop, half* src, half* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F32(
    ncclCoopAny coop, float* src, float* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F64(
    ncclCoopAny coop, double* src, double* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_BF16(
    ncclCoopAny coop, NCCL_IR_BF16_T* src, NCCL_IR_BF16_T* mcDst, size_t count);
#if !defined(__HIPCC__)
__device__ void ncclIrMultimemCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, __nv_fp8_e4m3* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, __nv_fp8_e5m2* mcDst, size_t count);
#endif

__device__ void ncclIrLsaReduceSumCopy_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
#if !defined(__HIPCC__)
__device__ void ncclIrLsaReduceSumCopy_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
#endif

__device__ void ncclIrMultimemReduceSumCopy_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F16(
    ncclCoopAny coop, half* mcSrc, half* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F32(
    ncclCoopAny coop, float* mcSrc, float* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F64(
    ncclCoopAny coop, double* mcSrc, double* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_BF16(
    ncclCoopAny coop, NCCL_IR_BF16_T* mcSrc, NCCL_IR_BF16_T* mcDst, size_t count);
#if !defined(__HIPCC__)
__device__ void ncclIrMultimemReduceSumCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* mcDst, size_t count);
#endif

__device__ void ncclIrLocalReduceSumCopy_I8(
    ncclCoopAny coop, int nSrc, int8_t* srcBase, size_t srcDispl,
    int nDst, int8_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U8(
    ncclCoopAny coop, int nSrc, uint8_t* srcBase, size_t srcDispl,
    int nDst, uint8_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_I32(
    ncclCoopAny coop, int nSrc, int32_t* srcBase, size_t srcDispl,
    int nDst, int32_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U32(
    ncclCoopAny coop, int nSrc, uint32_t* srcBase, size_t srcDispl,
    int nDst, uint32_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_I64(
    ncclCoopAny coop, int nSrc, int64_t* srcBase, size_t srcDispl,
    int nDst, int64_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U64(
    ncclCoopAny coop, int nSrc, uint64_t* srcBase, size_t srcDispl,
    int nDst, uint64_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F16(
    ncclCoopAny coop, int nSrc, half* srcBase, size_t srcDispl,
    int nDst, half* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F32(
    ncclCoopAny coop, int nSrc, float* srcBase, size_t srcDispl,
    int nDst, float* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F64(
    ncclCoopAny coop, int nSrc, double* srcBase, size_t srcDispl,
    int nDst, double* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_BF16(
    ncclCoopAny coop, int nSrc, NCCL_IR_BF16_T* srcBase, size_t srcDispl,
    int nDst, NCCL_IR_BF16_T* dstBase, size_t dstDispl, size_t count);
#if !defined(__HIPCC__)
__device__ void ncclIrLocalReduceSumCopy_F8E4M3(
    ncclCoopAny coop, int nSrc, __nv_fp8_e4m3* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e4m3* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F8E5M2(
    ncclCoopAny coop, int nSrc, __nv_fp8_e5m2* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e5m2* dstBase, size_t dstDispl, size_t count);
#endif

}  /* extern "C" */

#endif  /* _NCCL_DEVICE_IR_H_ */
