/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Adapted from NVIDIA NCCL ir/nccl_device_wrapper.h (v2.29.2-1).
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_H_
#define _NCCL_DEVICE_WRAPPER_H_

/*
 * RCCL Device API C-style wrapper functions.
 *
 * Public surface for the LLVM IR / bitcode artifact (librccl_device.bc).
 * Each entry point is an extern "C" __device__ thunk that downstream code
 * generators (Triton, MLIR, custom JITs, etc.) can call into without
 * recompiling RCCL from source. The thunks themselves are defined in the
 * companion translation unit nccl_device_wrapper__impl.h, which clang
 * compiles to bitcode under -D__clang_llvm_bitcode_lib__.
 *
 * Sectioning of this header (mirrored in __impl.h):
 *   [A] Always-on    — APIs whose RCCL prerequisites already exist.
 *   [B] Always-on    — APIs that take ncclCoopAny by value.
 *   [C] Always-on    — GIN + composite barrier APIs (ncclGin*,
 *                      ncclGinBarrierSession, composite ncclBarrierSession,
 *                      ncclGinFenceLevel). Enabled now that the NCCL
 *                      v2.29.x GIN sync landed these primitives in RCCL.
 */

/*
 * Declaration/type-only view of the NCCL Device API for LLVM IR users.
 *
 * This header intentionally excludes nccl_device/impl/xxx__funcs.h so user IR
 * bitcode can resolve NCCL Device API implementations from libnccl_device.bc.
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
 * libnccl_device.bc. Must precede the API includes below.
 */
#include "nccl_device/utility.h"
#undef NCCL_DEVICE_INLINE
#undef NCCL_HOST_DEVICE_INLINE
#if defined(__CUDACC__) || defined(__HIPCC__)
#if defined(__NCCL_DEVICE_LTOIR_LIB__)
#define NCCL_DEVICE_INLINE __device__ __inline_hint__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __inline_hint__
#elif defined(__clang_llvm_bitcode_lib__)
#define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __attribute__((always_inline))
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
 * NCCL_IR_EXPORT: full attribute set for an exported C-ABI thunk in the
 * bitcode artifact.
 *
 *   - `extern "C"`              : unmangled symbol name in the .bc.
 *   - `__device__`              : HIP device function.
 *   - `__attribute__((used))`   : prevent clang -O1 from DCE'ing the body
 *                                 when there is no caller in this TU.
 *   - NO `always_inline`/`inline` : the thunks are the public ABI; they
 *                                 must retain external linkage so opt's
 *                                 `internalize` pass keeps them on the
 *                                 public-api list. With `always_inline`
 *                                 (as `NCCL_DEVICE_INLINE` expands to in
 *                                 bitcode mode), clang's GlobalOpt
 *                                 downgrades the thunk to `internal`
 *                                 linkage before opt even runs.
 *
 * Callees of the thunk (e.g. ncclGetPeerPointer from core__funcs.h) still
 * carry `NCCL_DEVICE_INLINE` and therefore still inline into the thunk
 * body; this macro only controls the linkage of the thunk itself.
 * ----------------------------------------------------------------------*/
#ifndef NCCL_IR_EXPORT
  #ifdef __clang_llvm_bitcode_lib__
    /* Building the bitcode artifact: applied to the definitions in
     * nccl_device_wrapper__impl.h. Three attributes are needed:
     *   `used`                -> @llvm.compiler.used : block compiler-driven DCE.
     *   `retain`              -> @llvm.used          : block clang -O1 GlobalOpt
     *                                                  from downgrading linkage
     *                                                  to `internal`. Without it
     *                                                  the thunk arrives at opt
     *                                                  already internal and the
     *                                                  public-api-list cannot
     *                                                  promote it back.
     *   `visibility("default")` -> visible to AMDGPU lld at object-level link.
     *                              HIP-Clang defaults `__device__` symbols to
     *                              `hidden` visibility, which is fine for
     *                              `llvm-link`-style consumers but causes
     *                              `undefined hidden symbol` errors when
     *                              downstream apps let the driver handle
     *                              the bitcode link (the more common case).
     */
    #define NCCL_IR_EXPORT extern "C" __device__ \
        __attribute__((used, retain, visibility("default")))
  #else
    /* Downstream consumer use of this header: applied to declarations
     * only. Must be `extern "C" __device__` so a __global__ kernel can
     * call the thunk directly; the body is supplied at link time by
     * linking librccl_device.bc into the device-side image. */
    #define NCCL_IR_EXPORT extern "C" __device__
  #endif
#endif

/* ========================================================================
 * [A] Always-on APIs
 *
 * Prerequisites already in RCCL today:
 *   - ncclGetPeerPointer(ncclWindow_t, size_t, ncclTeam, int)
 *     (src/include/nccl_device/impl/core__funcs.h)
 * ======================================================================*/

/* Session struct size getters
 *
 * Used by the Python device API to allocate session storage with the correct
 * size via llvm.alloca, without duplicating the C++ struct layout in Python.
 */
NCCL_IR_EXTERN_C __device__ size_t ncclLsaBarrierSession_C_size();
NCCL_IR_EXTERN_C __device__ size_t ncclGinBarrierSession_C_size();
NCCL_IR_EXTERN_C __device__ size_t ncclBarrierSession_C_size();

/* ncclDevComm field accessors
 *
 * ncclDevComm is a public C struct, the following accessors are deprecated and will be removed.
 */
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_Rank(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_NRanks(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_LsaRank(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_LsaSize(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclLsaBarrierHandle ncclDevComm_LsaBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_RailGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclLsaBarrierHandle ncclDevComm_HybridLsaBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_HybridRailGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_WorldGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclMultimemHandle   ncclDevComm_LsaMultimem(ncclDevComm const* comm);

/* Peer pointer API */
NCCL_IR_EXPORT void* ncclGetPeerPointerTeam(
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

/* Struct definitions */
struct ncclLsaBarrierSession_C {
  ncclLsaBarrierSession<ncclCoopAny> bar;
};

/* Coop initialization and utility */
NCCL_IR_EXPORT void ncclCoopAnyInitThread(ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopAnyInitWarp(ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask);
NCCL_IR_EXPORT void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id, void* barrierLds);
NCCL_IR_EXPORT void ncclCoopAnyInitCta(ncclCoopAny* coop);

NCCL_IR_EXPORT int  ncclCoopThreadRank(const ncclCoopAny* coop);
NCCL_IR_EXPORT int  ncclCoopSize(const ncclCoopAny* coop);
NCCL_IR_EXPORT int  ncclCoopNumThreads(const ncclCoopAny* coop);
NCCL_IR_EXPORT void ncclCoopSync(const ncclCoopAny* coop);

/* LSA Barrier Session APIs */
NCCL_IR_EXPORT void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle mmHandle = {});

NCCL_IR_EXPORT
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session,
                                 ncclCoopAny coop,
                                 cuda::memory_order order);
NCCL_IR_EXPORT
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session,
                               ncclCoopAny coop,
                               cuda::memory_order order);
NCCL_IR_EXPORT
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session,
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

/* Struct definitions */
struct ncclGinBarrierSession_C {
  ncclGinBarrierSession<ncclCoopAny> bar;
};

struct ncclBarrierSession_C {
  ncclBarrierSession<ncclCoopAny> bar;
};
// Collectively finalize the placement-new object and persist its epoch. Does not free storage.
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionDestroy(ncclLsaBarrierSession_C* session);

/* GIN Barrier Session APIs */
NCCL_IR_EXPORT void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C const* net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

NCCL_IR_EXPORT void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Finalize the placement-new object. Currently a no-op -- the underlying destructor is
// empty -- but required for lifetime symmetry with Init. Does not free storage.
NCCL_IR_EXTERN_C __device__
void ncclGinBarrierSessionDestroy(ncclGinBarrierSession_C* session);

/* Composite (LSA + GIN) Barrier Session APIs */
NCCL_IR_EXPORT void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C const* net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle const innerMmHandle = {});

NCCL_IR_EXPORT void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Collectively finalize all nested sessions and persist the inner LSA epoch. Does not free storage.
NCCL_IR_EXTERN_C __device__
void ncclBarrierSessionDestroy(ncclBarrierSession_C* session);

/* ReduceCopy APIs */
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int8_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint8_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int32_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint32_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int64_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint64_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    half* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    float* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    double* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_bfloat16* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e4m3* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e5m2* dst, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F16(
    ncclCoopAny coop, half* mcSrc, half* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F32(
    ncclCoopAny coop, float* mcSrc, float* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F64(
    ncclCoopAny coop, double* mcSrc, double* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_BF16(
    ncclCoopAny coop, __nv_bfloat16* mcSrc, __nv_bfloat16* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* dst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I8(
    ncclCoopAny coop, int8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U8(
    ncclCoopAny coop, uint8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I32(
    ncclCoopAny coop, int32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U32(
    ncclCoopAny coop, uint32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I64(
    ncclCoopAny coop, int64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U64(
    ncclCoopAny coop, uint64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F16(
    ncclCoopAny coop, half* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F32(
    ncclCoopAny coop, float* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F64(
    ncclCoopAny coop, double* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_I32(
    ncclCoopAny coop, int32_t* src, int32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_U32(
    ncclCoopAny coop, uint32_t* src, uint32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_I64(
    ncclCoopAny coop, int64_t* src, int64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_U64(
    ncclCoopAny coop, uint64_t* src, uint64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F16(
    ncclCoopAny coop, half* src, half* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F32(
    ncclCoopAny coop, float* src, float* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F64(
    ncclCoopAny coop, double* src, double* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* src, __nv_bfloat16* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, __nv_fp8_e4m3* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, __nv_fp8_e5m2* mcDst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F16(
    ncclCoopAny coop, half* mcSrc, half* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F32(
    ncclCoopAny coop, float* mcSrc, float* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F64(
    ncclCoopAny coop, double* mcSrc, double* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* mcSrc, __nv_bfloat16* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* mcDst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I8(
    ncclCoopAny coop, int nSrc, int8_t* srcBase, size_t srcDispl,
    int nDst, int8_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U8(
    ncclCoopAny coop, int nSrc, uint8_t* srcBase, size_t srcDispl,
    int nDst, uint8_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I32(
    ncclCoopAny coop, int nSrc, int32_t* srcBase, size_t srcDispl,
    int nDst, int32_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U32(
    ncclCoopAny coop, int nSrc, uint32_t* srcBase, size_t srcDispl,
    int nDst, uint32_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I64(
    ncclCoopAny coop, int nSrc, int64_t* srcBase, size_t srcDispl,
    int nDst, int64_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U64(
    ncclCoopAny coop, int nSrc, uint64_t* srcBase, size_t srcDispl,
    int nDst, uint64_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F16(
    ncclCoopAny coop, int nSrc, half* srcBase, size_t srcDispl,
    int nDst, half* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F32(
    ncclCoopAny coop, int nSrc, float* srcBase, size_t srcDispl,
    int nDst, float* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F64(
    ncclCoopAny coop, int nSrc, double* srcBase, size_t srcDispl,
    int nDst, double* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_BF16(
    ncclCoopAny coop, int nSrc, __nv_bfloat16* srcBase, size_t srcDispl,
    int nDst, __nv_bfloat16* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F8E4M3(
    ncclCoopAny coop, int nSrc, __nv_fp8_e4m3* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e4m3* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F8E5M2(
    ncclCoopAny coop, int nSrc, __nv_fp8_e5m2* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e5m2* dstBase, size_t dstDispl, size_t count);

#endif  /* _NCCL_DEVICE_WRAPPER_H_ */
