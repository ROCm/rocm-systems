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

/**
 * @file rocshmem.cpp
 * @brief Public header for rocSHMEM device and host libraries.
 *
 * This is the implementation for the public rocshmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */

#include <hip/hip_runtime.h>

#include <cstdlib>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "backend_bc.hpp"
#include "context_incl.hpp"
#include "team.hpp"
#include "templates.hpp"
#include "util.hpp"

#if defined(USE_GDA)
#if defined (ENABLE_IBGDA_BITCODE)
#  include "gda/backend_gda.hpp"
#endif
#include "gda/context_gda_tmpl_device.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/context_ro_tmpl_device.hpp"
#endif
#if defined(USE_IPC)
# if defined(ENABLE_IPC_BITCODE)
#  include "ipc/backend_ipc.hpp"
# endif
#include "ipc/context_ipc_tmpl_device.hpp"
#endif

/******************************************************************************
 **************************** Device Vars And Init ****************************
 *****************************************************************************/

namespace rocshmem {

__device__  rocshmem_ctx_t __attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT{};

__constant__ Backend *device_backend_proxy;

__constant__ rocshmem_ctx_t ROCSHMEM_CTX_INVALID = {nullptr, nullptr};

#if defined(ENABLE_IPC_BITCODE)
  typedef IPCContext ContextTy;
#elif defined(ENABLE_IBGDA_BITCODE)
  typedef GDAContext ContextTy;
#else
  typedef Context ContextTy;
#endif

__device__ void rocshmem_wg_init() {
  int provided;

  /*
   * Non-threaded init is allowed to select any thread mode, so don't worry
   * if provided is different.
   */
  rocshmem_wg_init_thread(ROCSHMEM_THREAD_WG_FUNNELED, &provided);
}

__device__ void rocshmem_wg_init_thread([[maybe_unused]] int requested,
                                         int *provided) {
  rocshmem_query_thread(provided);
}

__device__ void rocshmem_query_thread(int *provided) {
#ifdef USE_THREADS
  *provided = ROCSHMEM_THREAD_MULTIPLE;
#else
  *provided = ROCSHMEM_THREAD_WG_FUNNELED;
#endif
}

__device__ void rocshmem_wg_finalize() {}


/******************************************************************************
* These host APIs use Device side symbol - ROCSHMEM_CTX_DEFAULT so it needs
* to stay here to avoid getting pulled into other places in compilation
******************************************************************************/

__host__ void * rocshmem_get_device_ctx() {
  rocshmem_ctx_t ctx;

  CHECK_HIP(hipMemcpyFromSymbol(&ctx, HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT),
                             sizeof(rocshmem_ctx_t)));
  return ctx.ctx_opaque;

}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

__device__ void rocshmem_putmem(void *dest, const void *source, size_t nelems,
                                 int pe) {
  rocshmem_ctx_putmem(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_p(T *dest, T value, int pe) {
  rocshmem_p(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T rocshmem_g(const T *source, int pe) {
  return rocshmem_g(ROCSHMEM_CTX_DEFAULT, source, pe);
}

__device__ void rocshmem_getmem(void *dest, const void *source, size_t nelems,
                                 int pe) {
  rocshmem_ctx_getmem(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_get(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_putmem_nbi(void *dest, const void *source,
                                     size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi(T *dest, const T *source, size_t nelems,
                                  int pe) {
  rocshmem_put_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_getmem_nbi(void *dest, const void *source,
                                     size_t nelems, int pe) {
  rocshmem_ctx_getmem_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi(T *dest, const T *source, size_t nelems,
                                  int pe) {
  rocshmem_get_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_fence() {
  rocshmem_ctx_fence(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_fence(int pe) {
  rocshmem_ctx_fence(ROCSHMEM_CTX_DEFAULT, pe);
}

__device__ void rocshmem_quiet() {
  rocshmem_ctx_quiet(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_pe_quiet(const int *target_pes, size_t npes) {
  rocshmem_ctx_pe_quiet(ROCSHMEM_CTX_DEFAULT, target_pes, npes);
}

__device__ void rocshmem_threadfence_system() {
  rocshmem_ctx_threadfence_system(ROCSHMEM_CTX_DEFAULT);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_add(T *dest, T val, int pe) {
  return rocshmem_atomic_fetch_add(ROCSHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return rocshmem_atomic_compare_swap(ROCSHMEM_CTX_DEFAULT, dest, cond, val,
                                       pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_inc(T *dest, int pe) {
  return rocshmem_atomic_fetch_inc(ROCSHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch(T *source, int pe) {
  return rocshmem_atomic_fetch(ROCSHMEM_CTX_DEFAULT, source, pe);
}

template <typename T>
__device__ void rocshmem_atomic_add(T *dest, T val, int pe) {
  rocshmem_atomic_add(ROCSHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_inc(T *dest, int pe) {
  rocshmem_atomic_inc(ROCSHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ void rocshmem_atomic_set(T *dest, T value, int pe) {
  rocshmem_atomic_set(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T rocshmem_atomic_swap(T *dest, T value, int pe) {
  return rocshmem_atomic_swap(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_and(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_and(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void rocshmem_atomic_and(T *dest, T value, int pe) {
  rocshmem_atomic_and(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_or(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_or(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void rocshmem_atomic_or(T *dest, T value, int pe) {
  rocshmem_atomic_or(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_xor(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_xor(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void rocshmem_atomic_xor(T *dest, T value, int pe) {
  rocshmem_atomic_xor(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

#define ROCSHMEM_PUTMEM_SIGNAL_DEF(SUFFIX)                                                      \
  __device__ void rocshmem_putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems, \
                                                  uint64_t *sig_addr, uint64_t signal,           \
                                                  int sig_op, int pe) {                          \
    rocshmem_ctx_putmem_signal##SUFFIX(ROCSHMEM_CTX_DEFAULT,                                   \
                                        dest, source, nelems,                                    \
                                        sig_addr, signal, sig_op, pe);                           \
  }                                                                                              \
                                                                                                 \
  template <typename T>                                                                          \
  __device__ void rocshmem_put_signal##SUFFIX(T *dest, const T *source, size_t nelems,          \
                                               uint64_t *sig_addr, uint64_t signal,              \
                                               int sig_op, int pe) {                             \
    rocshmem_ctx_put_signal##SUFFIX(ROCSHMEM_CTX_DEFAULT,                                      \
                                     dest, source, nelems,                                       \
                                     sig_addr, signal, sig_op, pe);                              \
  }

ROCSHMEM_PUTMEM_SIGNAL_DEF()
ROCSHMEM_PUTMEM_SIGNAL_DEF(_wg)
ROCSHMEM_PUTMEM_SIGNAL_DEF(_wave)
ROCSHMEM_PUTMEM_SIGNAL_DEF(_nbi)
ROCSHMEM_PUTMEM_SIGNAL_DEF(_nbi_wg)
ROCSHMEM_PUTMEM_SIGNAL_DEF(_nbi_wave)

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__device__ int translate_pe(rocshmem_ctx_t ctx, int pe) {
  if (ctx.team_opaque) {
    TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
    return (tinfo->pe_start + tinfo->stride * pe);
  } else {
    return pe;
  }
}

__host__ void set_internal_ctx(rocshmem_ctx_t *ctx) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT), ctx,
                              sizeof(rocshmem_ctx_t), 0,
                              hipMemcpyHostToDevice));
}

__device__ ContextTy *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<ContextTy *>(ctx.ctx_opaque);
}

__device__ int rocshmem_wg_ctx_create(long options, rocshmem_ctx_t *ctx) {
  GPU_DPRINTF("Function: rocshmem_wg_ctx_create (options=%ld)\n", options);
  bool result{true};
  if (get_flat_block_id() == 0) {
    ctx->team_opaque = reinterpret_cast<TeamInfo *>(ROCSHMEM_CTX_DEFAULT.team_opaque);
    result = device_backend_proxy->create_ctx(options, ctx);
    if (!result) {
      *ctx = ROCSHMEM_CTX_INVALID;
    }
  }
  __syncthreads();
  return result == true ? 0 : -1;
}

__device__ int rocshmem_wg_team_create_ctx(rocshmem_team_t team, long options,
                                           rocshmem_ctx_t *ctx) {
  GPU_DPRINTF("Function: rocshmem_wg_team_create_ctx (team=%zd, options=%ld)\n",
    (intptr_t)team, options);
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  }

  bool result{true};
  if (get_flat_block_id() == 0) {
    Team *team_obj{get_internal_team(team)};
    TeamInfo *info_wrt_world = team_obj->tinfo_wrt_world;
    ctx->team_opaque = info_wrt_world;
    result = device_backend_proxy->create_ctx(options, ctx);
    if (!result) {
      *ctx = ROCSHMEM_CTX_INVALID;
    }
  }
  __syncthreads();

  return result == true ? 0 : -1;
}

__device__ void rocshmem_wg_ctx_destroy(
    [[maybe_unused]] rocshmem_ctx_t *ctx) {
  GPU_DPRINTF("Function: rocshmem_wg_ctx_destroy (ctx=%zd)\n",
    ctx->ctx_opaque);

  if (get_flat_block_id() == 0 && *ctx != ROCSHMEM_CTX_INVALID) {
    device_backend_proxy->destroy_ctx(ctx);
  }
}

__device__ void rocshmem_ctx_threadfence_system(rocshmem_ctx_t ctx) {
  GPU_DPRINTF("Function: rocshmem_ctx_threadfence_system (ctx=%zd)\n",
    ctx.ctx_opaque);

  get_internal_ctx(ctx)->threadfence_system();
}

__device__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_ctx_putmem (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_put (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->put(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_p (ctx=%zd, dest=%p, value=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, value, pe, pe_in_world);

  get_internal_ctx(ctx)->p(dest, value, pe_in_world);
}

template <typename T>
__device__ T rocshmem_g(rocshmem_ctx_t ctx, const T *source, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_g (ctx=%zd, source=%p, pe=%d w%d)\n",
    ctx.ctx_opaque, source, pe, pe_in_world);

  return get_internal_ctx(ctx)->g(source, pe_in_world);
}

__device__ void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_ctx_getmem (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->getmem(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_get(rocshmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_get (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->get(dest, source, nelems, pe_in_world);
}

__device__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_ctx_putmem_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_put_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe_in_world);
}

__device__ void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_ctx_getmem_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_get_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_get_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe_in_world);
}

__device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx) {
  GPU_DPRINTF("Function: rocshmem_ctx_fence (ctx=%zd)\n", ctx.ctx_opaque);

  get_internal_ctx(ctx)->fence();
}

__device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe) {

  int pe_in_world = translate_pe(ctx, pe);
  GPU_DPRINTF("Function: rocshmem_ctx_fence (ctx=%zd, pe=%d w%d))\n",
    ctx.ctx_opaque, pe, pe_in_world);

  get_internal_ctx(ctx)->fence(pe_in_world);
}

__device__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  GPU_DPRINTF("Function: rocshmem_ctx_quiet (ctx=%zd)\n",
    ctx.ctx_opaque);

  get_internal_ctx(ctx)->quiet();
}

__device__ void rocshmem_ctx_pe_quiet(rocshmem_ctx_t ctx, const int *target_pes, size_t npes) {
  GPU_DPRINTF("Function: %s (ctx=%zd)\n", __func__, ctx.ctx_opaque);

  ContextTy *internal_ctx = get_internal_ctx(ctx);

  for (int i = 0; i < npes;  i++) {
    internal_ctx->pe_quiet(translate_pe(ctx, target_pes[i]));
  }
}

__device__ void *rocshmem_ptr(const void *dest, int pe) {
  GPU_DPRINTF("Function: rocshmem_ptr (dest=%p, pe=%d w%d\n",
    dest, pe, pe);

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->shmem_ptr(dest, pe);
}

template <typename T, ROCSHMEM_OP Op>
__device__ int rocshmem_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                   T *dest, const T *source, int nreduce) {
  GPU_DPRINTF("Function: rocshmem_reduce_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nreduce=%d\n",
    ctx.ctx_opaque, team, dest, source, nreduce);

  return get_internal_ctx(ctx)->reduce<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__device__ void rocshmem_broadcast_wg(rocshmem_ctx_t ctx,
                                       rocshmem_team_t team, T *dest,
                                       const T *source, int nelem,
                                       int pe_root) {
  GPU_DPRINTF("Function: Team-based rocshmem_broadcast_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelem=%d, root=%d)\n",
    ctx.ctx_opaque, team, dest, source, nelem, pe_root);

  get_internal_ctx(ctx)->broadcast<T>(team, dest, source, nelem, pe_root);
}

template <typename T>
__device__ void rocshmem_ctx_alltoall_wg(rocshmem_ctx_t ctx,
                                         rocshmem_team_t team, T *dest,
                                         const T *source, int nelem) {
  GPU_DPRINTF("Function: rocshmem_ctx_alltoall_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelem=%d\n",
              ctx.ctx_opaque, team, dest, source, nelem);

  get_internal_ctx(ctx)->alltoall<T>(team, dest, source, nelem);
}

template <typename T>
__device__ void rocshmem_alltoall_wg(rocshmem_team_t team, T *dest,
                                     const T *source, int nelem) {
  GPU_DPRINTF("Function: rocshmem_alltoall_wg (team=%zd, dest=%p, source=%p, nelem=%d\n",
              team, dest, source, nelem);

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->alltoall<T>(team, dest, source, nelem);
}

template <typename T>
__device__ void rocshmem_alltoallv_wg(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
  GPU_DPRINTF("Function: rocshmem_alltoallv_wg(team=%zd, dest=%p, source=%p\n",
              team, dest, source);

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->alltoallv<T>(team,
                                                       dest, dest_nelems, dest_displs,
                                                       source, source_nelems, source_displs);
}

template <typename T>
__device__ void rocshmem_fcollect_wg(rocshmem_ctx_t ctx,
                                      rocshmem_team_t team, T *dest,
                                      const T *source, int nelem) {
  GPU_DPRINTF("Function: rocshmem_fcollect_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelem=%d\n",
    ctx.ctx_opaque, team, dest, source, nelem);

  get_internal_ctx(ctx)->fcollect<T>(team, dest, source, nelem);
}

template <typename T>
__device__ void rocshmem_wait_until(T *ivars, int cmp, T val) {
  GPU_DPRINTF("Function: rocshmem_wait_until (ivars=%p, cmp=%d, val=%g)\n",
    ivars, cmp, (double)val);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL);
  ctx_internal->wait_until(ivars, cmp, val);
}

template <typename T>
__device__ void rocshmem_wait_until_all(T *ivars, size_t nelems, const int* status,
                                         int cmp, T val) {
  GPU_DPRINTF("Function: rocshmem_wait_until_all (ivars=%p, nelems=%zd cmp=%d, val=%g)\n",
    ivars, nelems, cmp, (double)val);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_ALL);
  ctx_internal->wait_until_all(ivars, nelems, status, cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_any(T *ivars, size_t nelems, const int* status,
                                           int cmp, T val) {
  GPU_DPRINTF("Function: rocshmem_wait_until_any (ivars=%p, nelems=%zd cmp=%d, val=%g)\n",
    ivars, nelems, cmp, (double)val);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_ANY);
  return ctx_internal->wait_until_any(ivars, nelems, status, cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_some(T *ivars, size_t nelems, size_t* indices,
                                          const int* status, int cmp,
                                          T val) {
  DPRINTF("Function: rocshmem_wait_until_some (ivars=%p, nelems=%zd cmp=%d, val=%g)\n",
    ivars, nelems, cmp, (double)val);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_SOME);
  return ctx_internal->wait_until_some(ivars, nelems, indices, status, cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_any_vector(T *ivars, size_t nelems, const int* status,
                                                  int cmp, T* vals) {
  DPRINTF("Function: rocshmem_wait_until_any_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)\n",
    ivars, nelems, cmp, vals);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_ANY_VECTOR);
  return ctx_internal->wait_until_any_vector(ivars, nelems, status, cmp, vals);
}

template <typename T>
__device__ void rocshmem_wait_until_all_vector(T *ivars, size_t nelems, const int* status,
                                                int cmp, T* vals) {
  DPRINTF("Function: rocshmem_wait_until_all_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)\n",
    ivars, nelems, cmp, vals);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_ALL_VECTOR);
  ctx_internal->wait_until_all_vector(ivars, nelems, status, cmp, vals);
}

template <typename T>
__device__ size_t rocshmem_wait_until_some_vector(T *ivars, size_t nelems,
                                                 size_t* indices,
                                                 const int* status,
                                                 int cmp, T* vals) {
  DPRINTF("Function: rocshmem_wait_until_some_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)\n",
    ivars, nelems, cmp, vals);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL_SOME_VECTOR);
  return ctx_internal->wait_until_some_vector(ivars, nelems, indices, status, cmp, vals);
}

template <typename T>
__device__ int rocshmem_test(T *ivars, int cmp, T val) {
  GPU_DPRINTF("Function: rocshmem_test (ivars=%p, cmp=%d, val=%g)\n",
    ivars, cmp, (double)val);

  Context *ctx_internal = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_TEST);

  return ctx_internal->test(ivars, cmp, val);
}

__global__ ATTR_NO_INLINE void rocshmem_barrier_all_kernel(){
  rocshmem_barrier_all();
}

__global__ ATTR_NO_INLINE void rocshmem_alltoallmem_kernel(rocshmem_team_t team,
                                                           void *dest,
                                                           const void *source,
                                                           size_t size) {
  // Create a context for this workgroup to avoid contention on default context
  // This allows parallel execution across multiple streams without serialization
  __shared__ rocshmem_ctx_t ctx;
  __shared__ int ctx_result;

  ctx_result = rocshmem_wg_team_create_ctx(team, 0, &ctx);

  // If context creation failed, fall back to default context
  if (ctx_result != 0) {
    ctx = ROCSHMEM_CTX_DEFAULT;
    __syncthreads();
  }

  // Call device alltoall function with created context and provided team
  // Using char type since size is in bytes (1 byte per element)
  rocshmem_ctx_alltoall_wg<char>(ctx, team, (char *) dest,
                                 (const char *) source, (int) size);

  if (ctx_result == 0) {
    rocshmem_wg_ctx_destroy(&ctx);
  }
}

__global__ ATTR_NO_INLINE void rocshmem_broadcastmem_kernel(
    rocshmem_team_t team, void *dest, const void *source, size_t nelems,
    int pe_root) {
  __shared__ rocshmem_ctx_t ctx;
  __shared__ int ctx_result;

  ctx_result = rocshmem_wg_team_create_ctx(team, 0, &ctx);

  // If context creation failed, fall back to default context
  if (ctx_result != 0) {
    ctx = ROCSHMEM_CTX_DEFAULT;
    __syncthreads();
  }

  // Call device broadcast function with created context and provided team
  // Using char type since nelems is in bytes (1 byte per element)
  rocshmem_broadcast_wg<char>(ctx, team, (char *) dest, (const char *) source,
                              (int) nelems, pe_root);

  if (ctx_result == 0) {
    rocshmem_wg_ctx_destroy(&ctx);
  }
}

__global__ ATTR_NO_INLINE void rocshmem_getmem_kernel(void *dest,
                                                      const void *source,
                                                      size_t nelems, int pe) {
  // Use work-group collective getmem with default context
  rocshmem_getmem_wg(dest, source, nelems, pe);
}

__global__ ATTR_NO_INLINE void rocshmem_putmem_kernel(void *dest,
                                                      const void *source,
                                                      size_t nelems, int pe) {
  // Use work-group collective putmem with default context
  rocshmem_putmem_wg(dest, source, nelems, pe);
}

__global__ ATTR_NO_INLINE void rocshmem_putmem_signal_kernel(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe) {
  // Use work-group collective putmem_signal with default context
  rocshmem_putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__global__ ATTR_NO_INLINE void rocshmem_signal_wait_until_kernel(
    uint64_t *sig_addr, int cmp, uint64_t cmp_value) {
  // Use default context to wait on signal
  rocshmem_uint64_wait_until(sig_addr, cmp, cmp_value);
}

__device__ void rocshmem_barrier_all() {
  GPU_DPRINTF("Function: rocshmem_barrier_all (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->barrier_all();
}

__device__ void rocshmem_barrier_all_wave() {
  GPU_DPRINTF("Function: rocshmem_barrier_all_wave (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->barrier_all_wave();
}

__device__ void rocshmem_barrier_all_wg() {
  GPU_DPRINTF("Function: rocshmem_barrier_all_wg (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->barrier_all_wg();
}

__device__ void rocshmem_ctx_barrier(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  GPU_DPRINTF("Function: rocshmem_ctx_barrier (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

  get_internal_ctx(ctx)->barrier(team);
}

__device__ void rocshmem_ctx_barrier_wave(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  GPU_DPRINTF("Function: rocshmem_ctx_barrier_wave (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

  get_internal_ctx(ctx)->barrier_wave(team);
}

__device__ void rocshmem_ctx_barrier_wg(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  GPU_DPRINTF("Function: rocshmem_ctx_barrier_wg (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

  get_internal_ctx(ctx)->barrier_wg(team);
}

__device__ void rocshmem_sync_all() {
  GPU_DPRINTF("Function: rocshmem_sync_all (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->sync_all();
}

__device__ void rocshmem_sync_all_wave() {
  GPU_DPRINTF("Function: rocshmem_sync_all_wave (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->sync_all_wave();
}

__device__ void rocshmem_sync_all_wg() {
  GPU_DPRINTF("Function: rocshmem_sync_all_wg (ctx=%zd)\n",
    get_internal_ctx(ROCSHMEM_CTX_DEFAULT));

  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->sync_all_wg();
}

__device__ void rocshmem_ctx_sync(rocshmem_ctx_t ctx,
                                  rocshmem_team_t team) {
  GPU_DPRINTF("Function: rocshmem_ctx_sync (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

  get_internal_ctx(ctx)->sync_wg(team);
}

__device__ void rocshmem_ctx_sync_wave(rocshmem_ctx_t ctx,
                                       rocshmem_team_t team) {
GPU_DPRINTF("Function: rocshmem_ctx_sync_wave (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

get_internal_ctx(ctx)->sync_wg(team);
}

__device__ void rocshmem_ctx_sync_wg(rocshmem_ctx_t ctx,
                                     rocshmem_team_t team) {
GPU_DPRINTF("Function: rocshmem_ctx_sync_wg (ctx=%zd, team=%zd)\n",
    ctx.ctx_opaque, team);

get_internal_ctx(ctx)->sync_wg(team);
}

__device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx) {
  GPU_DPRINTF("Function: rocshmem_ctx_n_pes (ctx=%zd)\n",
    ctx.ctx_opaque);

  TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
  return tinfo->size;
}

__device__ int rocshmem_n_pes() {
  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->num_pes;
}

__device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx) {
  GPU_DPRINTF("Function: rocshmem_ctx_my_pe (ctx=%zd)\n",
    ctx.ctx_opaque);

  TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
  int my_pe{get_internal_ctx(ctx)->my_pe};
  int pe_start{tinfo->pe_start};
  int stride{tinfo->stride};
  int size{tinfo->size};

  int translated_pe = (my_pe - pe_start) / stride;

  if ((my_pe < pe_start) ||
     ((my_pe - pe_start) % stride) ||
     (translated_pe >= size)) {
    translated_pe = -1;
  }

  return translated_pe;
}

__device__ int rocshmem_my_pe() {
  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->my_pe;
}

template <typename T>
__device__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch_add (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                           T val, int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_compare_swap (ctx=%zd, dest=%p, cond=%g, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, cond, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch_inc (ctx=%zd, dest=%p, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, 1, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch(rocshmem_ctx_t ctx, T *source, int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch (ctx=%zd, source=%p, pe=%d w%d)\n",
    ctx.ctx_opaque, source, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_add<T>(source, 0, pe);
}

template <typename T>
__device__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_add (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_inc (ctx=%zd, dest=%p, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_add<T>(dest, 1, pe);
}

template <typename T>
__device__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_set (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_swap (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_and(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch_and (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_and(dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_and(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_and (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_and(dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_or(rocshmem_ctx_t ctx, T *dest, T val,
                                       int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch_or (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_or(dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_or(rocshmem_ctx_t ctx, T *dest, T val,
                                    int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_or (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_or(dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_fetch_xor (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return get_internal_ctx(ctx)->amo_fetch_xor(dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: rocshmem_atomic_xor (ctx=%zd, dest=%p, val=%g, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->amo_xor(dest, val, pe);
}

/**
 *      SHMEM X RMA API for WG and Wave level
 */
__device__ void rocshmem_ctx_putmem_wave(rocshmem_ctx_t ctx, void *dest,
                                          const void *source, size_t nelems,
                                          int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_putmem_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->putmem_wave(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_wg(rocshmem_ctx_t ctx, void *dest,
                                        const void *source, size_t nelems,
                                        int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_putmem_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->putmem_wg(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_nbi_wave(rocshmem_ctx_t ctx, void *dest,
                                              const void *source,
                                              size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_putmem_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->putmem_nbi_wave(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_nbi_wg(rocshmem_ctx_t ctx, void *dest,
                                            const void *source, size_t nelems,
                                            int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_putmem_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->putmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wave(rocshmem_ctx_t ctx, T *dest,
                                   const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_put_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->put_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                 size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_put_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->put_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                       const T *source, size_t nelems,
                                       int pe) {
  GPU_DPRINTF("Function: rocshmem_put_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->put_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                     const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_put_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->put_nbi_wg(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_wg(rocshmem_ctx_t ctx, void *dest,
                                        const void *source, size_t nelems,
                                        int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_getmem_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->getmem_wg(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_wave(rocshmem_ctx_t ctx, void *dest,
                                          const void *source, size_t nelems,
                                          int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_getmem_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->getmem_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                 size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_get_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->get_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wave(rocshmem_ctx_t ctx, T *dest,
                                   const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_get_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->get_wave(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_nbi_wg(rocshmem_ctx_t ctx, void *dest,
                                            const void *source, size_t nelems,
                                            int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_getmem_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->getmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                     const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_get_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->get_nbi_wg(dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_nbi_wave(rocshmem_ctx_t ctx, void *dest,
                                              const void *source,
                                              size_t nelems, int pe) {
  GPU_DPRINTF("Function: rocshmem_ctx_getmem_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->getmem_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                       const T *source, size_t nelems,
                                       int pe) {
  GPU_DPRINTF("Function: rocshmem_get_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)\n",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  get_internal_ctx(ctx)->get_nbi_wave(dest, source, nelems, pe);
}

#define ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(SUFFIX)                                             \
  __device__ void rocshmem_ctx_putmem_signal##SUFFIX(rocshmem_ctx_t ctx,                   \
                                                      void *dest, const void *source,      \
                                                      size_t nelems,                       \
                                                      uint64_t *sig_addr, uint64_t signal, \
                                                      int sig_op,                          \
                                                      int pe) {                            \
    GPU_DPRINTF("Function: rocshmem_ctx_putmem_signal##SUFFIX (ctx=%zd, dest=%p, "         \
      "source=%p, nelems=%d, sig_addr=%p, signal=%ld, sig_op=%d, pe=%d w%d)\n",            \
      ctx.ctx_opaque, dest, source, nelems,                                                \
      sig_addr, signal, sig_op, pe, translate_pe(ctx, pe));                                \
                                                                                           \
    get_internal_ctx(ctx)->putmem_signal##SUFFIX(dest, source, nelems,                     \
                                                 sig_addr, signal, sig_op, pe);            \
  }                                                                                        \
                                                                                           \
  template <typename T>                                                                    \
  __device__ void rocshmem_ctx_put_signal##SUFFIX(rocshmem_ctx_t ctx,                      \
                                                   T *dest, const T *source,               \
                                                   size_t nelems,                          \
                                                   uint64_t *sig_addr, uint64_t signal,    \
                                                   int sig_op, int pe) {                   \
    GPU_DPRINTF("Function: rocshmem_ctx_put_signal##SUFFIX (ctx=%zd, dest=%p, "            \
      "source=%p, nelems=%d, sig_addr=%p, signal=%ld, sig_op=%d, pe=%d w%d)\n",            \
      ctx.ctx_opaque, dest, source, nelems,                                                \
      sig_addr, signal, sig_op, pe, translate_pe(ctx, pe));                                \
                                                                                           \
    get_internal_ctx(ctx)->put_signal##SUFFIX(dest, source, nelems,                        \
                                              sig_addr, signal, sig_op, pe);               \
  }

ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF()
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_wg)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_wave)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi_wg)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi_wave)

#define ROCSHMEM_SIGNAL_FETCH_DEF(SUFFIX)                                          \
  __device__ uint64_t rocshmem_signal_fetch##SUFFIX(const uint64_t *sig_addr) {    \
    return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->signal_fetch##SUFFIX(sig_addr); \
  }

ROCSHMEM_SIGNAL_FETCH_DEF()
ROCSHMEM_SIGNAL_FETCH_DEF(_wg)
ROCSHMEM_SIGNAL_FETCH_DEF(_wave)

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__device__ int rocshmem_team_translate_pe(rocshmem_team_t src_team,
                                           int src_pe,
                                           rocshmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************* Template Generation Macros *************************
 *****************************************************************************/

/**
 * Template generator for reductions
 */
#define REDUCTION_GEN(T, Op)                                                   \
  template __device__ int rocshmem_reduce_wg<T, Op>(                           \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nreduce);

/**
 * Declare templates for the required datatypes (for the compiler)
 */
#define RMA_GEN(T)                                                             \
  template __device__ void rocshmem_put<T>(                                    \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_put_nbi<T>(                                \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_p<T>(rocshmem_ctx_t ctx, T * dest,         \
                                          T value, int pe);                    \
  template __device__ void rocshmem_get<T>(                                    \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_get_nbi<T>(                                \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ T rocshmem_g<T>(rocshmem_ctx_t ctx, const T *source,     \
                                       int pe);                                \
  template __device__ void rocshmem_put<T>(T * dest, const T *source,          \
                                            size_t nelems, int pe);            \
  template __device__ void rocshmem_put_nbi<T>(T * dest, const T *source,      \
                                                size_t nelems, int pe);        \
  template __device__ void rocshmem_p<T>(T * dest, T value, int pe);           \
  template __device__ void rocshmem_get<T>(T * dest, const T *source,          \
                                            size_t nelems, int pe);            \
  template __device__ void rocshmem_get_nbi<T>(T * dest, const T *source,      \
                                                size_t nelems, int pe);        \
  template __device__ T rocshmem_g<T>(const T *source, int pe);                \
  template __device__ void rocshmem_broadcast_wg<T>(                           \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelem, int pe_root);                                                 \
  template __device__ void rocshmem_ctx_alltoall_wg<T>(                        \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelem);                                                              \
  template __device__ void rocshmem_alltoall_wg<T>(                            \
      rocshmem_team_t team, T * dest, const T *source,                         \
      int nelem);                                                              \
  template __device__ void rocshmem_alltoallv_wg<T>(                           \
                                      rocshmem_team_t team,                    \
                                      T *dest, const size_t dest_nelems[],     \
                                      const size_t dest_displs[],              \
                                      T *source, const size_t source_nelems[], \
                                      const size_t source_displs[]);           \
  template __device__ void rocshmem_fcollect_wg<T>(                            \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelem);                                                              \
  template __device__ void rocshmem_put_wave<T>(                               \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_put_wg<T>(                                 \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_put_wave<T>(T * dest, const T *source,     \
                                                 size_t nelems, int pe);       \
  template __device__ void rocshmem_put_wg<T>(T * dest, const T *source,       \
                                               size_t nelems, int pe);         \
  template __device__ void rocshmem_put_nbi_wave<T>(                           \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_put_nbi_wg<T>(                             \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_put_nbi_wave<T>(                           \
      T * dest, const T *source, size_t nelems, int pe);                       \
  template __device__ void rocshmem_put_nbi_wg<T>(T * dest, const T *source,   \
                                                   size_t nelems, int pe);     \
  template __device__ void rocshmem_get_wave<T>(                               \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_get_wg<T>(                                 \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_get_wave<T>(T * dest, const T *source,     \
                                                 size_t nelems, int pe);       \
  template __device__ void rocshmem_get_wg<T>(T * dest, const T *source,       \
                                               size_t nelems, int pe);         \
  template __device__ void rocshmem_get_nbi_wave<T>(                           \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_get_nbi_wg<T>(                             \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);   \
  template __device__ void rocshmem_get_nbi_wave<T>(                           \
      T * dest, const T *source, size_t nelems, int pe);                       \
  template __device__ void rocshmem_get_nbi_wg<T>(T * dest, const T *source,   \
                                                   size_t nelems, int pe);

/**
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                    \
  template __device__ T rocshmem_atomic_compare_swap<T>(                       \
      rocshmem_ctx_t ctx, T * dest, T cond, T value, int pe);                  \
  template __device__ T rocshmem_atomic_compare_swap<T>(T * dest, T cond,      \
                                                         T value, int pe);     \
  template __device__ T rocshmem_atomic_fetch_inc<T>(rocshmem_ctx_t ctx,       \
                                                      T * dest, int pe);       \
  template __device__ T rocshmem_atomic_fetch_inc<T>(T * dest, int pe);        \
  template __device__ void rocshmem_atomic_inc<T>(rocshmem_ctx_t ctx,          \
                                                   T * dest, int pe);          \
  template __device__ void rocshmem_atomic_inc<T>(T * dest, int pe);           \
  template __device__ T rocshmem_atomic_fetch_add<T>(                          \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                          \
  template __device__ T rocshmem_atomic_fetch_add<T>(T * dest, T value,        \
                                                      int pe);                 \
  template __device__ void rocshmem_atomic_add<T>(rocshmem_ctx_t ctx,          \
                                                   T * dest, T value, int pe); \
  template __device__ void rocshmem_atomic_add<T>(T * dest, T value, int pe);

/**
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                    \
  template __device__ T rocshmem_atomic_fetch<T>(rocshmem_ctx_t ctx,           \
                                                  T * dest, int pe);           \
  template __device__ T rocshmem_atomic_fetch<T>(T * dest, int pe);            \
  template __device__ void rocshmem_atomic_set<T>(rocshmem_ctx_t ctx,          \
                                                   T * dest, T value, int pe); \
  template __device__ void rocshmem_atomic_set<T>(T * dest, T value, int pe);  \
  template __device__ T rocshmem_atomic_swap<T>(rocshmem_ctx_t ctx,            \
                                                 T * dest, T value, int pe);   \
  template __device__ T rocshmem_atomic_swap<T>(T * dest, T value, int pe);

/**
 * Declare templates for the bitwise amo types
 */
#define AMO_BITWISE_GEN(T)                                                     \
  template __device__ T rocshmem_atomic_fetch_and<T>(                          \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                          \
  template __device__ T rocshmem_atomic_fetch_and<T>(T * dest, T value,        \
                                                      int pe);                 \
  template __device__ void rocshmem_atomic_and<T>(rocshmem_ctx_t ctx,          \
                                                   T * dest, T value, int pe); \
  template __device__ void rocshmem_atomic_and<T>(T * dest, T value, int pe);  \
  template __device__ T rocshmem_atomic_fetch_or<T>(                           \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                          \
  template __device__ T rocshmem_atomic_fetch_or<T>(T * dest, T value,         \
                                                     int pe);                  \
  template __device__ void rocshmem_atomic_or<T>(rocshmem_ctx_t ctx,           \
                                                  T * dest, T value, int pe);  \
  template __device__ void rocshmem_atomic_or<T>(T * dest, T value, int pe);   \
  template __device__ T rocshmem_atomic_fetch_xor<T>(                          \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                          \
  template __device__ T rocshmem_atomic_fetch_xor<T>(T * dest, T value,        \
                                                      int pe);                 \
  template __device__ void rocshmem_atomic_xor<T>(rocshmem_ctx_t ctx,          \
                                                   T * dest, T value, int pe); \
  template __device__ void rocshmem_atomic_xor<T>(T * dest, T value, int pe);

/**
 * Declare templates for the wait types
 */
#define WAIT_GEN(T)                                                            \
  template __device__ void rocshmem_wait_until<T>(T *ivars,                    \
                                                   int cmp, T val);            \
  template __device__ size_t rocshmem_wait_until_any<T>(T *ivars,              \
                                      size_t nelems, const int* status,        \
                                      int cmp, T val);                         \
  template __device__ void rocshmem_wait_until_all<T>(T *ivars,                \
                                      size_t nelems, const int* status,        \
                                      int cmp, T val);                         \
  template __device__ size_t rocshmem_wait_until_some<T>(T *ivars,             \
                                      size_t nelems, size_t* indices,          \
                                      const int* status,                       \
                                      int cmp, T val);                         \
  template __device__ size_t rocshmem_wait_until_any_vector<T>(T *ivars,       \
                                      size_t nelems, const int* status,        \
                                      int cmp, T* vals);                       \
  template __device__ void rocshmem_wait_until_all_vector<T>(T *ivars,         \
                                      size_t nelems, const int* status,        \
                                      int cmp, T* vals);                       \
  template __device__ size_t rocshmem_wait_until_some_vector<T>(T *ivars,      \
                                      size_t nelems, size_t* indices,          \
                                      const int* status, int cmp,              \
                                      T* vals);                                \
  template __device__ int rocshmem_test<T>(T *ivars, int cmp,                  \
                                            T val);                            \
  template __device__ void Context::wait_until<T>(T *ivars, int cmp,           \
                                                  T val);                      \
  template __device__ size_t Context::wait_until_any<T>(T *ivars,              \
                                      size_t nelems, const int* status,        \
                                      int cmp, T val);                         \
  template __device__ void Context::wait_until_all<T>(T *ivars,                \
                                      size_t nelems, const int* status,        \
                                      int cmp, T val);                         \
  template __device__ size_t Context::wait_until_some<T>(T *ivars,             \
                                      size_t nelems,                           \
                                      size_t* indices, const int* status,      \
                                      int cmp, T val);                         \
  template __device__ size_t Context::wait_until_any_vector<T>(T *ivars,       \
                                      size_t nelems, const int* status,        \
                                      int cmp, T* vals);                       \
  template __device__ void Context::wait_until_all_vector<T>(T *ivars,         \
                                      size_t nelems, const int* status,        \
                                      int cmp, T* vals);                       \
  template __device__ size_t Context::wait_until_some_vector<T>(T *ivars,      \
                                      size_t nelems, size_t* indices,          \
                                      const int* status, int cmp,              \
                                      T* vals);                                \
  template __device__ int Context::test<T>(T *ivars, int cmp, T val);

#define ARITH_REDUCTION_GEN(T)    \
  REDUCTION_GEN(T, ROCSHMEM_SUM) \
  REDUCTION_GEN(T, ROCSHMEM_MIN) \
  REDUCTION_GEN(T, ROCSHMEM_MAX) \
  REDUCTION_GEN(T, ROCSHMEM_PROD)

#define BITWISE_REDUCTION_GEN(T)  \
  REDUCTION_GEN(T, ROCSHMEM_OR)  \
  REDUCTION_GEN(T, ROCSHMEM_AND) \
  REDUCTION_GEN(T, ROCSHMEM_XOR)

#define INT_REDUCTION_GEN(T) \
  ARITH_REDUCTION_GEN(T)     \
  BITWISE_REDUCTION_GEN(T)

#define FLOAT_REDUCTION_GEN(T) ARITH_REDUCTION_GEN(T)

/**
 * Define APIs to call the template functions
 **/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                               \
  __device__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce_wg(                 \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nreduce) {                                                          \
    return rocshmem_reduce_wg<T, Op>(ctx, team, dest, source, nreduce);       \
  }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME)         \
  REDUCTION_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM) \
  REDUCTION_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN) \
  REDUCTION_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX) \
  REDUCTION_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME)       \
  REDUCTION_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)   \
  REDUCTION_DEF_GEN(T, TNAME, and, ROCSHMEM_AND) \
  REDUCTION_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME) \
  ARITH_REDUCTION_DEF_GEN(T, TNAME)     \
  BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) ARITH_REDUCTION_DEF_GEN(T, TNAME)

#define RMA_DEF_GEN(T, TNAME)                                                 \
  __device__ void rocshmem_ctx_##TNAME##_put(                                 \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_put_nbi(                             \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_p(rocshmem_ctx_t ctx, T *dest,       \
                                            T value, int pe) {                \
    rocshmem_p<T>(ctx, dest, value, pe);                                      \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get(                                 \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_g(rocshmem_ctx_t ctx, const T *source,  \
                                         int pe) {                            \
    return rocshmem_g<T>(ctx, source, pe);                                    \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get_nbi(                             \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put(T *dest, const T *source,            \
                                          size_t nelems, int pe) {            \
    rocshmem_put<T>(dest, source, nelems, pe);                                \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put_nbi(T *dest, const T *source,        \
                                              size_t nelems, int pe) {        \
    rocshmem_put_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_p(T *dest, T value, int pe) {            \
    rocshmem_p<T>(dest, value, pe);                                           \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get(T *dest, const T *source,            \
                                          size_t nelems, int pe) {            \
    rocshmem_get<T>(dest, source, nelems, pe);                                \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get_nbi(T *dest, const T *source,        \
                                              size_t nelems, int pe) {        \
    rocshmem_get_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_g(const T *source, int pe) {                \
    return rocshmem_g<T>(source, pe);                                         \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_put_wave(                            \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_wave<T>(ctx, dest, source, nelems, pe);                      \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_put_wg(                              \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_wg<T>(ctx, dest, source, nelems, pe);                        \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put_wave(T *dest, const T *source,       \
                                               size_t nelems, int pe) {       \
    rocshmem_put_wave<T>(dest, source, nelems, pe);                           \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put_wg(T *dest, const T *source,         \
                                             size_t nelems, int pe) {         \
    rocshmem_put_wg<T>(dest, source, nelems, pe);                             \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_put_nbi_wave(                        \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_nbi_wave<T>(ctx, dest, source, nelems, pe);                  \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_put_nbi_wg(                          \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_nbi_wg<T>(ctx, dest, source, nelems, pe);                    \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put_nbi_wave(T *dest, const T *source,   \
                                                   size_t nelems, int pe) {   \
    rocshmem_put_nbi_wave<T>(dest, source, nelems, pe);                       \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_put_nbi_wg(T *dest, const T *source,     \
                                                 size_t nelems, int pe) {     \
    rocshmem_put_nbi_wg<T>(dest, source, nelems, pe);                         \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get_wave(                            \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_wave<T>(ctx, dest, source, nelems, pe);                      \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get_wg(                              \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_wg<T>(ctx, dest, source, nelems, pe);                        \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get_wave(T *dest, const T *source,       \
                                               size_t nelems, int pe) {       \
    rocshmem_get_wave<T>(dest, source, nelems, pe);                           \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get_wg(T *dest, const T *source,         \
                                             size_t nelems, int pe) {         \
    rocshmem_get_wg<T>(dest, source, nelems, pe);                             \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get_nbi_wave(                        \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_nbi_wave<T>(ctx, dest, source, nelems, pe);                  \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_get_nbi_wg(                          \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_nbi_wg<T>(ctx, dest, source, nelems, pe);                    \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get_nbi_wave(T *dest, const T *source,   \
                                                   size_t nelems, int pe) {   \
    rocshmem_get_nbi_wave<T>(dest, source, nelems, pe);                       \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_get_nbi_wg(T *dest, const T *source,     \
                                                 size_t nelems, int pe) {     \
    rocshmem_get_nbi_wg<T>(dest, source, nelems, pe);                         \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_broadcast_wg(                        \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelem, int pe_root) {                                               \
    rocshmem_broadcast_wg<T>(ctx, team, dest, source, nelem, pe_root);        \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_alltoall_wg(                         \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelem) {                                                            \
    rocshmem_ctx_alltoall_wg<T>(ctx, team, dest, source, nelem);              \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_alltoall_wg(                             \
      rocshmem_team_t team, T *dest, const T *source,                         \
      int nelem) {                                                            \
    rocshmem_alltoall_wg<T>(team, dest, source, nelem);                       \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_alltoallv_wg(                            \
                                      rocshmem_team_t team,                   \
                                      T *dest, const size_t dest_nelems[],    \
                                      const size_t dest_displs[],             \
                                      T *source, const size_t source_nelems[],\
                                      const size_t source_displs[]) {         \
    rocshmem_alltoallv_wg<T>(team,                                            \
                             dest, dest_nelems, dest_displs,                  \
                             source, source_nelems, source_displs);           \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_fcollect_wg(                         \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelem) {                                                            \
    rocshmem_fcollect_wg<T>(ctx, team, dest, source, nelem);                  \
  }

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                        \
  __device__ T rocshmem_ctx_##TNAME##_atomic_compare_swap(                    \
      rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe) {                 \
    return rocshmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);       \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_compare_swap(T *dest, T cond,        \
                                                       T value, int pe) {     \
    return rocshmem_atomic_compare_swap<T>(dest, cond, value, pe);            \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_inc(rocshmem_ctx_t ctx,    \
                                                        T *dest, int pe) {    \
    return rocshmem_atomic_fetch_inc<T>(ctx, dest, pe);                       \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) {         \
    return rocshmem_atomic_fetch_inc<T>(dest, pe);                            \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_inc(rocshmem_ctx_t ctx,       \
                                                     T *dest, int pe) {       \
    rocshmem_atomic_inc<T>(ctx, dest, pe);                                    \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_inc(T *dest, int pe) {            \
    rocshmem_atomic_inc<T>(dest, pe);                                         \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_add(                       \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_add<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch_add(T *dest, T value,          \
                                                    int pe) {                 \
    return rocshmem_atomic_fetch_add<T>(dest, value, pe);                     \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_add(                          \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    rocshmem_atomic_add<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_add(T *dest, T value, int pe) {   \
    rocshmem_atomic_add<T>(dest, value, pe);                                  \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                        \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch(rocshmem_ctx_t ctx,        \
                                                    T *source, int pe) {      \
    return rocshmem_atomic_fetch<T>(ctx, source, pe);                         \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch(T *source, int pe) {           \
    return rocshmem_atomic_fetch<T>(source, pe);                              \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_set(                          \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    rocshmem_atomic_set<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_set(T *dest, T value, int pe) {   \
    rocshmem_atomic_set<T>(dest, value, pe);                                  \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_atomic_swap(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) {\
    return rocshmem_atomic_swap<T>(ctx, dest, value, pe);                     \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {     \
    return rocshmem_atomic_swap<T>(dest, value, pe);                          \
  }

#define AMO_BITWISE_DEF_GEN(T, TNAME)                                         \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_and(                       \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_and<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch_and(T *dest, T value,          \
                                                    int pe) {                 \
    return rocshmem_atomic_fetch_and<T>(dest, value, pe);                     \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_and(                          \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    rocshmem_atomic_and<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_and(T *dest, T value, int pe) {   \
    rocshmem_atomic_and<T>(dest, value, pe);                                  \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_or(                        \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_or<T>(ctx, dest, value, pe);                 \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch_or(T *dest, T value, int pe) { \
    return rocshmem_atomic_fetch_or<T>(dest, value, pe);                      \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_or(                           \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    rocshmem_atomic_or<T>(ctx, dest, value, pe);                              \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_or(T *dest, T value, int pe) {    \
    rocshmem_atomic_or<T>(dest, value, pe);                                   \
  }                                                                           \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_xor(                       \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_xor<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __device__ T rocshmem_##TNAME##_atomic_fetch_xor(T *dest, T value,          \
                                                    int pe) {                 \
    return rocshmem_atomic_fetch_xor<T>(dest, value, pe);                     \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_atomic_xor(                          \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    rocshmem_atomic_xor<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_atomic_xor(T *dest, T value, int pe) {   \
    rocshmem_atomic_xor<T>(dest, value, pe);                                  \
  }

#define WAIT_DEF_GEN(T, TNAME)                                                \
  __device__ void rocshmem_##TNAME##_wait_until(T *ivars, int cmp,            \
                                                 T val) {                     \
    rocshmem_wait_until<T>(ivars, cmp, val);                                  \
  }                                                                           \
  __device__ size_t rocshmem_##TNAME##_wait_until_any(T *ivars, size_t nelems,\
                                                     const int* status,       \
                                                     int cmp,                 \
                                                     T val) {                 \
    return rocshmem_wait_until_any<T>(ivars, nelems, status, cmp, val);       \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_wait_until_all(T *ivars, size_t nelems,  \
                                                   const int* status,         \
                                                   int cmp,                   \
                                                   T val) {                   \
    rocshmem_wait_until_all<T>(ivars, nelems, status, cmp, val);              \
  }                                                                           \
  __device__ size_t rocshmem_##TNAME##_wait_until_some(T *ivars,              \
                                                    size_t nelems,            \
                                                    size_t* indices,          \
                                                    const int* status,        \
                                                    int cmp,                  \
                                                    T val) {                  \
    return rocshmem_wait_until_some<T>(ivars, nelems, indices, status, cmp,   \
                                        val);                                 \
  }                                                                           \
  __device__ size_t rocshmem_##TNAME##_wait_until_any_vector(T *ivars,        \
                                                          size_t nelems,      \
                                                          const int* status,  \
                                                          int cmp,            \
                                                          T* vals) {          \
    return rocshmem_wait_until_any_vector<T>(ivars, nelems, status, cmp,      \
                                              vals);                          \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_wait_until_all_vector(T *ivars,          \
                                                          size_t nelems,      \
                                                          const int* status,  \
                                                          int cmp,            \
                                                          T* vals) {          \
    rocshmem_wait_until_all_vector<T>(ivars, nelems, status, cmp, vals);      \
  }                                                                           \
  __device__ size_t rocshmem_##TNAME##_wait_until_some_vector(T *ivars,       \
                                                           size_t nelems,     \
                                                           size_t* indices,   \
                                                           const int* status, \
                                                           int cmp,           \
                                                           T* vals) {         \
    return rocshmem_wait_until_some_vector<T>(ivars, nelems, indices,         \
        status, cmp, vals);                                                   \
  }                                                                           \
  __device__ int rocshmem_##TNAME##_test(T *ivars, int cmp, T val) {          \
    return rocshmem_test<T>(ivars, cmp, val);                                 \
  }

#define RMA_SIGNAL_SUFFIX_DEC(SUFFIX)                                                    \
  template <typename T>                                                                  \
  __device__ void rocshmem_ctx_put_signal##SUFFIX(rocshmem_ctx_t ctx,                 \
                                                    T *dest, const T *source,            \
                                                    size_t nelems,                       \
                                                    uint64_t *sig_addr, uint64_t signal, \
                                                    int sig_op, int pe);                 \
                                                                                         \
  template <typename T>                                                                  \
  __device__ void rocshmem_put_signal##SUFFIX(T *dest, const T *source, size_t nelems, \
                                                uint64_t *sig_addr, uint64_t signal,     \
                                                int sig_op, int pe);                     \

#define RMA_SIGNAL_SUFFIX_DEF(T, TNAME, SUFFIX)                                                   \
  __device__ void rocshmem_ctx_##TNAME##_put_signal##SUFFIX(rocshmem_ctx_t ctx,                 \
                                                             T *dest, const T *source,            \
                                                             size_t nelems,                       \
                                                             uint64_t *sig_addr, uint64_t signal, \
                                                             int sig_op, int pe) {                \
    rocshmem_ctx_put_signal##SUFFIX<T>(ctx, dest, source, nelems, sig_addr, signal, sig_op, pe); \
  }                                                                                               \
                                                                                                  \
  __device__ void rocshmem_##TNAME##_put_signal##SUFFIX(T *dest, const T *source, size_t nelems, \
                                                         uint64_t *sig_addr, uint64_t signal,     \
                                                         int sig_op, int pe) {                    \
    rocshmem_put_signal##SUFFIX(dest, source, nelems, sig_addr, signal, sig_op, pe);             \
  }

#define RMA_SIGNAL_GEN(SUFFIX)                                 \
  RMA_SIGNAL_SUFFIX_DEC(SUFFIX)                                \
  RMA_SIGNAL_SUFFIX_DEF(float, float, SUFFIX)                  \
  RMA_SIGNAL_SUFFIX_DEF(double, double, SUFFIX)                \
  RMA_SIGNAL_SUFFIX_DEF(char, char, SUFFIX)                    \
  RMA_SIGNAL_SUFFIX_DEF(signed char, schar, SUFFIX)            \
  RMA_SIGNAL_SUFFIX_DEF(short, short, SUFFIX)                  \
  RMA_SIGNAL_SUFFIX_DEF(int, int, SUFFIX)                      \
  RMA_SIGNAL_SUFFIX_DEF(long, long, SUFFIX)                    \
  RMA_SIGNAL_SUFFIX_DEF(long long, longlong, SUFFIX)           \
  RMA_SIGNAL_SUFFIX_DEF(unsigned char, uchar, SUFFIX)          \
  RMA_SIGNAL_SUFFIX_DEF(unsigned short, ushort, SUFFIX)        \
  RMA_SIGNAL_SUFFIX_DEF(unsigned int, uint, SUFFIX)            \
  RMA_SIGNAL_SUFFIX_DEF(unsigned long, ulong, SUFFIX)          \
  RMA_SIGNAL_SUFFIX_DEF(unsigned long long, ulonglong, SUFFIX) \
  RMA_SIGNAL_SUFFIX_DEF(int8_t, int8, SUFFIX)                  \
  RMA_SIGNAL_SUFFIX_DEF(int16_t, int16, SUFFIX)                \
  RMA_SIGNAL_SUFFIX_DEF(int32_t, int32, SUFFIX)                \
  RMA_SIGNAL_SUFFIX_DEF(int64_t, int64, SUFFIX)                \
  RMA_SIGNAL_SUFFIX_DEF(uint8_t, uint8, SUFFIX)                \
  RMA_SIGNAL_SUFFIX_DEF(uint16_t, uint16, SUFFIX)              \
  RMA_SIGNAL_SUFFIX_DEF(uint32_t, uint32, SUFFIX)              \
  RMA_SIGNAL_SUFFIX_DEF(uint64_t, uint64, SUFFIX)              \
  RMA_SIGNAL_SUFFIX_DEF(size_t, size, SUFFIX)                  \
  RMA_SIGNAL_SUFFIX_DEF(ptrdiff_t, ptrdiff, SUFFIX)

RMA_SIGNAL_GEN(_wg)
RMA_SIGNAL_GEN()
RMA_SIGNAL_GEN(_wave)
RMA_SIGNAL_GEN(_nbi)
RMA_SIGNAL_GEN(_nbi_wg)
RMA_SIGNAL_GEN(_nbi_wave)

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

// clang-format off
INT_REDUCTION_GEN(int)
INT_REDUCTION_GEN(short)
INT_REDUCTION_GEN(long)
INT_REDUCTION_GEN(long long)
FLOAT_REDUCTION_GEN(float)
FLOAT_REDUCTION_GEN(double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_GEN(long double)

RMA_GEN(float)
RMA_GEN(double)
// RMA_GEN(long double)
RMA_GEN(char)
RMA_GEN(signed char)
RMA_GEN(short)
RMA_GEN(int)
RMA_GEN(long)
RMA_GEN(long long)
RMA_GEN(unsigned char)
RMA_GEN(unsigned short)
RMA_GEN(unsigned int)
RMA_GEN(unsigned long)
RMA_GEN(unsigned long long)

AMO_STANDARD_GEN(int)
AMO_STANDARD_GEN(long)
AMO_STANDARD_GEN(long long)
AMO_STANDARD_GEN(unsigned int)
AMO_STANDARD_GEN(unsigned long)
AMO_STANDARD_GEN(unsigned long long)

AMO_EXTENDED_GEN(float)
AMO_EXTENDED_GEN(double)
AMO_EXTENDED_GEN(int)
AMO_EXTENDED_GEN(long)
AMO_EXTENDED_GEN(long long)
AMO_EXTENDED_GEN(unsigned int)
AMO_EXTENDED_GEN(unsigned long)
AMO_EXTENDED_GEN(unsigned long long)

AMO_BITWISE_GEN(unsigned int)
AMO_BITWISE_GEN(unsigned long)
AMO_BITWISE_GEN(unsigned long long)

/* Supported synchronization types */
WAIT_GEN(float)
WAIT_GEN(double)
// WAIT_GEN(long double)
WAIT_GEN(char)
WAIT_GEN(unsigned char)
WAIT_GEN(unsigned short)
WAIT_GEN(signed char)
WAIT_GEN(short)
WAIT_GEN(int)
WAIT_GEN(long)
WAIT_GEN(long long)
WAIT_GEN(unsigned int)
WAIT_GEN(unsigned long)
WAIT_GEN(unsigned long long)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
// RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)
RMA_DEF_GEN(int8_t, int8)
RMA_DEF_GEN(int16_t, int16)
RMA_DEF_GEN(int32_t, int32)
RMA_DEF_GEN(int64_t, int64)
RMA_DEF_GEN(uint8_t, uint8)
RMA_DEF_GEN(uint16_t, uint16)
RMA_DEF_GEN(uint32_t, uint32)
RMA_DEF_GEN(uint64_t, uint64)
RMA_DEF_GEN(size_t, size)
RMA_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_STANDARD_DEF_GEN(int, int)
AMO_STANDARD_DEF_GEN(long, long)
AMO_STANDARD_DEF_GEN(long long, longlong)
AMO_STANDARD_DEF_GEN(unsigned int, uint)
AMO_STANDARD_DEF_GEN(unsigned long, ulong)
AMO_STANDARD_DEF_GEN(unsigned long long, ulonglong)
AMO_STANDARD_DEF_GEN(int32_t, int32)
AMO_STANDARD_DEF_GEN(int64_t, int64)
AMO_STANDARD_DEF_GEN(uint32_t, uint32)
AMO_STANDARD_DEF_GEN(uint64_t, uint64)
AMO_STANDARD_DEF_GEN(size_t, size)
AMO_STANDARD_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
AMO_EXTENDED_DEF_GEN(int32_t, int32)
AMO_EXTENDED_DEF_GEN(int64_t, int64)
AMO_EXTENDED_DEF_GEN(uint32_t, uint32)
AMO_EXTENDED_DEF_GEN(uint64_t, uint64)
AMO_EXTENDED_DEF_GEN(size_t, size)
AMO_EXTENDED_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_BITWISE_DEF_GEN(unsigned int, uint)
AMO_BITWISE_DEF_GEN(unsigned long, ulong)
AMO_BITWISE_DEF_GEN(unsigned long long, ulonglong)
AMO_BITWISE_DEF_GEN(int32_t, int32)
AMO_BITWISE_DEF_GEN(int64_t, int64)
AMO_BITWISE_DEF_GEN(uint32_t, uint32)
AMO_BITWISE_DEF_GEN(uint64_t, uint64)

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
// WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(char, char)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)
WAIT_DEF_GEN(uint64_t, uint64)
// clang-format on

}  // namespace rocshmem
