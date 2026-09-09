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
#include "constmem.hpp"
#include "context_incl.hpp"
#include "team.hpp"
#include "templates.hpp"
#include "log.hpp"
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

/*
 * Unconditional backend includes so that the direct-dispatch helpers below
 * can name GDABackend/ROBackend/IPCBackend directly, bypassing
 * Backend::create_ctx/destroy_ctx's runtime switch(this->type). This mirrors
 * the include pattern already used by backend_bc.cpp and rocshmem.cpp.
 */
#if defined(USE_GDA) && !defined(ENABLE_IBGDA_BITCODE)
#include "gda/backend_gda.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/backend_ro.hpp"
#endif
#if defined(USE_IPC) && !defined(ENABLE_IPC_BITCODE)
#include "ipc/backend_ipc.hpp"
#endif

/******************************************************************************
 **************************** Device Vars And Init ****************************
 *****************************************************************************/

namespace rocshmem {

__device__  rocshmem_ctx_t
__attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT{};

__constant__  rocshmem_ctx_t *rocshmem_ctx_array;

__constant__ Backend *device_backend_proxy;

__constant__ constmem_t constmem;

__constant__ rocshmem_ctx_t ROCSHMEM_CTX_INVALID = {nullptr, nullptr};

__constant__ struct logd_constants logd_constants;

namespace device {
    extern "C" __constant__ rocshmem_team_t
    __attribute__((visibility("default"))) ROCSHMEM_TEAM_WORLD = nullptr;
    extern "C" __constant__ rocshmem_team_t
    __attribute__((visibility("default"), used)) ROCSHMEM_TEAM_SHARED = nullptr;
}

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
* These host API use Device side symbol - ROCSHMEM_CTX_DEFAULT so it needs
* to stay here to avoid getting pulled into other places in compilation
******************************************************************************/

__host__ void * rocshmem_get_device_ctx() {
  rocshmem_ctx_t ctx = {nullptr, nullptr};
  CHECK_HIP(hipMemcpyFromSymbol(&ctx, HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT),
                                sizeof(rocshmem_ctx_t)));
  return ctx.ctx_opaque;
}

/**
 * Copies a device symbol from rocSHMEM to the user's HIP module
 * via device-to-device memcpy (graph-capture compatible).
 * Returns 0 on success, ROCSHMEM_ERROR on failure.
 */
template <typename Symbol>
static int copy_device_symbol_to_module(Symbol &builtin_symbol,
    const char *module_symbol_name, size_t expected_size, hipModule_t module,
    hipStream_t stream, const char *label) {
  void *source {nullptr};
  hipError_t err = hipGetSymbolAddress(&source, HIP_SYMBOL(builtin_symbol));
  if (err != hipSuccess) {
    LOG_ERROR("Failed to get address of built-in %s: %s",
              label, hipGetErrorString(err));
    return ROCSHMEM_ERROR;
  }
  if (source == nullptr) {
    LOG_ERROR("Built-in %s has null address", label);
    return ROCSHMEM_ERROR;
  }

  void *target {nullptr};
  size_t symbol_size {0};
  err = hipModuleGetGlobal(&target, &symbol_size, module, module_symbol_name);
  if (err != hipSuccess) {
    LOG_ERROR("Failed to get %s symbol from module: %s",
              label, hipGetErrorString(err));
    return ROCSHMEM_ERROR;
  }
  if (symbol_size != expected_size) {
    LOG_ERROR("Symbol size mismatch for %s. Expected %zu, got %zu",
              label, expected_size, symbol_size);
    return ROCSHMEM_ERROR;
  }

  err = hipMemcpyAsync(target, source, expected_size,
                       hipMemcpyDeviceToDevice, stream);
  if (err != hipSuccess) {
    LOG_ERROR("Failed to copy %s to device: %s",
              label, hipGetErrorString(err));
    return ROCSHMEM_ERROR;
  }
  return ROCSHMEM_SUCCESS;
}

__host__ int rocshmem_hipmodule_init(hipModule_t module, hipStream_t stream) {
  if (stream == nullptr) {
    stream = hipStreamPerThread;
  }

  if (copy_device_symbol_to_module(ROCSHMEM_CTX_DEFAULT, "ROCSHMEM_CTX_DEFAULT",
                                   sizeof(rocshmem_ctx_t), module, stream,
                                   "ROCSHMEM_CTX_DEFAULT") != ROCSHMEM_SUCCESS) {
    return ROCSHMEM_ERROR;
  }
  if (copy_device_symbol_to_module(device::ROCSHMEM_TEAM_WORLD,
                                   "ROCSHMEM_TEAM_WORLD",
                                   sizeof(rocshmem_team_t), module, stream,
                                   "ROCSHMEM_TEAM_WORLD") != ROCSHMEM_SUCCESS) {
    return ROCSHMEM_ERROR;
  }
  if (copy_device_symbol_to_module(device::ROCSHMEM_TEAM_SHARED,
                                   "ROCSHMEM_TEAM_SHARED",
                                   sizeof(rocshmem_team_t), module, stream,
                                   "ROCSHMEM_TEAM_SHARED") != ROCSHMEM_SUCCESS) {
    return ROCSHMEM_ERROR;
  }
  {
    void *probe{nullptr}; size_t probe_size{0};
    if (hipModuleGetGlobal(&probe, &probe_size, module,
                           "_ZN8rocshmem8constmemE") == hipSuccess) {
      copy_device_symbol_to_module(constmem, "_ZN8rocshmem8constmemE",
                                   sizeof(constmem_t), module, stream,
                                   "constmem");
    } else {
      LOG_WARN("constmem not in module — module does not use rocshmem "
               "device APIs that read constmem directly");
    }
  }
  return ROCSHMEM_SUCCESS;
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

__device__ void rocshmem_barrier() {
  rocshmem_ctx_barrier(ROCSHMEM_CTX_DEFAULT, device::ROCSHMEM_TEAM_WORLD);
}

__device__ void rocshmem_barrier_wave() {
  rocshmem_ctx_barrier_wave(ROCSHMEM_CTX_DEFAULT, device::ROCSHMEM_TEAM_WORLD);
}

__device__ void rocshmem_barrier_wg() {
  rocshmem_ctx_barrier_wg(ROCSHMEM_CTX_DEFAULT, device::ROCSHMEM_TEAM_WORLD);
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

__host__ void set_team_world_device(rocshmem_team_t team_world) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(device::ROCSHMEM_TEAM_WORLD), &team_world,
                              sizeof(rocshmem_team_t), 0,
                              hipMemcpyHostToDevice));
}

__host__ void set_team_shared_device(rocshmem_team_t team_shared) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(device::ROCSHMEM_TEAM_SHARED), &team_shared,
                              sizeof(rocshmem_team_t), 0,
                              hipMemcpyHostToDevice));
}

__device__ ContextTy *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<ContextTy *>(ctx.ctx_opaque);
}

namespace {

__device__ __forceinline__ Context *get_base_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

template <BackendType B>
struct DirectBackendContext;

#if defined(USE_GDA)
template <>
struct DirectBackendContext<BackendType::GDA_BACKEND> {
  using Type = GDAContext;
};
#endif

#if defined(USE_RO)
template <>
struct DirectBackendContext<BackendType::RO_BACKEND> {
  using Type = ROContext;
};
#endif

#if defined(USE_IPC)
template <>
struct DirectBackendContext<BackendType::IPC_BACKEND> {
  using Type = IPCContext;
};
#endif

template <BackendType B>
__device__ __forceinline__ typename DirectBackendContext<B>::Type *
get_backend_ctx(rocshmem_ctx_t ctx) {
  return static_cast<typename DirectBackendContext<B>::Type *>(
      get_base_internal_ctx(ctx));
}

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH(CTX, FUNC)                     \
  switch (constmem.backend_type) {                                      \
    case BackendType::GDA_BACKEND:                                      \
      get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC;             \
      break;                                                            \
    case BackendType::RO_BACKEND:                                       \
      get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC;              \
      break;                                                            \
    case BackendType::IPC_BACKEND:                                      \
    default:                                                            \
      get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC;             \
      break;                                                            \
  }
#elif defined(USE_GDA)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH(CTX, FUNC) \
  get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC
#elif defined(USE_RO)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH(CTX, FUNC) \
  get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC
#elif defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH(CTX, FUNC) \
  get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC
#endif

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(CTX, FUNC)                 \
  if (constmem.backend_type == BackendType::GDA_BACKEND) {              \
    auto ret1 = get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC;   \
    return ret1;                                                        \
  } else if (constmem.backend_type == BackendType::RO_BACKEND) {        \
    auto ret2 = get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC;    \
    return ret2;                                                        \
  } else {                                                              \
    auto ret3 = get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC;   \
    return ret3;                                                        \
  }
#elif defined(USE_GDA)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(CTX, FUNC)                 \
  auto ret_val = get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC;  \
  return ret_val;
#elif defined(USE_RO)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(CTX, FUNC)                 \
  auto ret_val = get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC;   \
  return ret_val;
#elif defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(CTX, FUNC)                 \
  auto ret_val = get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC;  \
  return ret_val;
#endif

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR(CTX, FUNC)             \
  void *ret_val{nullptr};                                               \
  switch (constmem.backend_type) {                                      \
    case BackendType::GDA_BACKEND:                                      \
      ret_val = get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC;   \
      break;                                                            \
    case BackendType::RO_BACKEND:                                       \
      ret_val = get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC;    \
      break;                                                            \
    case BackendType::IPC_BACKEND:                                      \
    default:                                                            \
      ret_val = get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC;   \
      break;                                                            \
  }                                                                     \
  return ret_val;
#elif defined(USE_GDA)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR(CTX, FUNC)             \
  void *ret_val{nullptr};                                               \
  ret_val = get_backend_ctx<BackendType::GDA_BACKEND>(CTX)->FUNC;       \
  return ret_val;
#elif defined(USE_RO)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR(CTX, FUNC)             \
  void *ret_val{nullptr};                                               \
  ret_val = get_backend_ctx<BackendType::RO_BACKEND>(CTX)->FUNC;        \
  return ret_val;
#elif defined(USE_IPC)
#define ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR(CTX, FUNC)             \
  void *ret_val{nullptr};                                               \
  ret_val = get_backend_ctx<BackendType::IPC_BACKEND>(CTX)->FUNC;       \
  return ret_val;
#endif

#define ROCSHMEM_DIRECT_CTX_MEM_HELPER(NAME, STAT, BACKEND_FUNC)        \
  __device__ __forceinline__ void direct_ctx_##NAME(                    \
      rocshmem_ctx_t ctx, void *dest, const void *source,               \
      size_t nelems, int pe) {                                          \
    if (nelems == 0) {                                                  \
      return;                                                           \
    }                                                                   \
                                                                        \
    get_base_internal_ctx(ctx)->ctxStats.incStat(STAT);                 \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH(                                   \
        ctx, BACKEND_FUNC(dest, source, nelems, pe));                   \
  }

ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem, NUM_PUT, putmem)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem, NUM_GET, getmem)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem_nbi, NUM_PUT_NBI, putmem_nbi)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem_nbi, NUM_GET_NBI, getmem_nbi)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem_wg, NUM_PUT_WG, putmem_wg)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem_wg, NUM_GET_WG, getmem_wg)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem_nbi_wg, NUM_PUT_NBI_WG, putmem_nbi_wg)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem_nbi_wg, NUM_GET_NBI_WG, getmem_nbi_wg)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem_wave, NUM_PUT_WAVE, putmem_wave)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem_wave, NUM_GET_WAVE, getmem_wave)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(putmem_nbi_wave, NUM_PUT_NBI_WAVE, putmem_nbi_wave)
ROCSHMEM_DIRECT_CTX_MEM_HELPER(getmem_nbi_wave, NUM_GET_NBI_WAVE, getmem_nbi_wave)

#undef ROCSHMEM_DIRECT_CTX_MEM_HELPER

/*
 * Non-template direct_ctx_* helpers mirroring Context::* bodies in
 * context_device.cpp exactly (guard, stat, DISPATCH semantics).
 */
__device__ __forceinline__ void direct_ctx_threadfence_system(
    rocshmem_ctx_t /*ctx*/) {
  __threadfence_system();
}

__device__ __forceinline__ void direct_ctx_fence(rocshmem_ctx_t ctx) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FENCE);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, fence());
}

__device__ __forceinline__ void direct_ctx_fence(rocshmem_ctx_t ctx, int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FENCE);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, fence(pe));
}

__device__ __forceinline__ void direct_ctx_quiet(rocshmem_ctx_t ctx) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_QUIET);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, quiet());
}

__device__ __forceinline__ void direct_ctx_pe_quiet(rocshmem_ctx_t ctx,
                                                     size_t pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_PE_QUIET);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, pe_quiet(pe));
}

__device__ __forceinline__ void *direct_ctx_shmem_ptr(rocshmem_ctx_t ctx,
                                                       const void *dest,
                                                       int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_SHMEM_PTR);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR(ctx, shmem_ptr(dest, pe));
}

/*
 * The barrier-family and sync-family helpers below are deliberately
 * __noinline__, unlike the other direct_ctx_* helpers in this file. Each one
 * expands (under all_backends) to a 3-way GDA/RO/IPC switch over a full
 * collective synchronization implementation (internal_sync/
 * internal_direct_barrier/internal_atomic_barrier, plus a quiet() call) --
 * inlining that into shared test kernels that call multiple of the
 * regular/wave/wg variants from one kernel body grows resource usage enough 
 * to drop occupancy by a full wave (5->4 waves/SIMD, confirmed via gfx950
 * all_backends resource-usage comparison). This happened even without
 * __forceinline__: because these helpers now live in the same translation
 * unit as their callers (unlike the Context::barrier* /sync* methods they
 * replace, which lived in a separate TU and only merged in at the more
 * conservative cross-TU LTO inlining stage), the ordinary per-TU inliner
 * still folded them in. Unlike putmem/getmem (hot-loop, per-call overhead
 * dominates), barrier/sync calls are infrequent and network-round-trip-
 * dominated, so the call overhead __noinline__ costs here is not worth an
 * occupancy regression.
 */

#define ROCSHMEM_DIRECT_CTX_SYNC_HELPER(NAME, STAT, BACKEND_FUNC)       \
  __device__ __noinline__ void direct_ctx_##NAME(                       \
      rocshmem_ctx_t ctx) {                                             \
    get_base_internal_ctx(ctx)->ctxStats.incStat(STAT);                 \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, BACKEND_FUNC());              \
  }

ROCSHMEM_DIRECT_CTX_SYNC_HELPER(barrier_all, NUM_BARRIER_ALL, barrier_all)
ROCSHMEM_DIRECT_CTX_SYNC_HELPER(barrier_all_wg, NUM_BARRIER_ALL_WG, barrier_all_wg)
ROCSHMEM_DIRECT_CTX_SYNC_HELPER(barrier_all_wave, NUM_BARRIER_ALL_WAVE, barrier_all_wave)
ROCSHMEM_DIRECT_CTX_SYNC_HELPER(sync_all, NUM_SYNC_ALL, sync_all)
ROCSHMEM_DIRECT_CTX_SYNC_HELPER(sync_all_wg, NUM_SYNC_ALL_WG, sync_all_wg)
ROCSHMEM_DIRECT_CTX_SYNC_HELPER(sync_all_wave, NUM_SYNC_ALL_WAVE, sync_all_wave)

#undef ROCSHMEM_DIRECT_CTX_SYNC_HELPER

__device__ __noinline__ void direct_ctx_barrier(rocshmem_ctx_t ctx, 
                                                rocshmem_team_t team) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_BARRIER);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, barrier(team));
}

__device__ __noinline__ void direct_ctx_barrier_wave(rocshmem_ctx_t ctx,
                                                     rocshmem_team_t team) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_BARRIER_WAVE);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, barrier_wave(team));
}

__device__ __noinline__ void direct_ctx_barrier_wg(rocshmem_ctx_t ctx,
                                                   rocshmem_team_t team) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_BARRIER_WG);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, barrier_wg(team));
}

/*
 * NOTE: rocshmem_ctx_sync/_wave/_wg all three call sync_wg (pre-existing
 * quirk in rocshmem_gpu.cpp being replicated exactly, not fixed).
 */
__device__ __noinline__ void direct_ctx_sync_wg(rocshmem_ctx_t ctx,
                                                rocshmem_team_t team) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_SYNC_WG);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, sync_wg(team));
}

__device__ __noinline__ void direct_ctx_broadcastmem_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems, int pe_root) {
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
      ctx, broadcastmem_wg(team, dest, source, nelems, pe_root));
}

__device__ __noinline__ int direct_ctx_broadcastmem_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems, int pe_root) {
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, broadcastmem_wave(team, dest, source, nelems, pe_root));
}

__device__ __noinline__ void direct_ctx_alltoallmem_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems) {
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ALLTOALL);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
    ctx, alltoallmem_wg(team, dest, source, nelems));
}

__device__ __noinline__ int direct_ctx_alltoallmem_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems) {
  if (nelems == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ALLTOALL);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, alltoallmem_wave(team, dest, source, nelems));
}

__device__ __noinline__ void direct_ctx_fcollectmem_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems) {
  if (nelems == 0) {
    return;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FCOLLECT);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
    ctx, fcollectmem_wg(team, dest, source, nelems));
}

__device__ __noinline__ int direct_ctx_fcollectmem_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, void *dest, const void *source,
    int nelems) {
  if (nelems == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FCOLLECT);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, fcollectmem_wave(team, dest, source, nelems));
}

#define ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(SUFFIX, STATS_SUFFIX)           \
  __device__ __forceinline__ void direct_ctx_putmem_signal##SUFFIX(           \
      rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,      \
      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {              \
    if (nelems == 0) {                                                        \
      return;                                                                 \
    }                                                                         \
    get_base_internal_ctx(ctx)->ctxStats.incStat(                             \
      NUM_PUT_SIGNAL##STATS_SUFFIX);                                          \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH(                                         \
        ctx, putmem_signal##SUFFIX(dest, source, nelems, sig_addr, signal,    \
                                   sig_op, pe));                              \
  }

ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(, )
ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(_wg, _WG)
ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(_wave, _WAVE)
ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(_nbi, _NBI)
ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(_nbi_wg, _NBI_WG)
ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF(_nbi_wave, _NBI_WAVE)

#undef ROCSHMEM_DIRECT_CTX_PUTMEM_SIGNAL_DEF

#define ROCSHMEM_DIRECT_SIGNAL_FETCH_DEF(SUFFIX)                              \
  __device__ __forceinline__ uint64_t direct_ctx_signal_fetch##SUFFIX(        \
      rocshmem_ctx_t ctx, const uint64_t *sig_addr) {                         \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(                                     \
      ctx, signal_fetch##SUFFIX(sig_addr));                                   \
  }

ROCSHMEM_DIRECT_SIGNAL_FETCH_DEF()
ROCSHMEM_DIRECT_SIGNAL_FETCH_DEF(_wg)
ROCSHMEM_DIRECT_SIGNAL_FETCH_DEF(_wave)

#undef ROCSHMEM_DIRECT_SIGNAL_FETCH_DEF

/*
 * Templated direct_ctx_* helpers mirroring Context::* template bodies in
 * context_tmpl_device.hpp exactly (guard, stat, DISPATCH semantics).
 */
template <typename T>
__device__ __forceinline__ void direct_ctx_p(rocshmem_ctx_t ctx, T *dest,
                                             T value, int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_P);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, p(dest, value, pe));
}

template <typename T>
__device__ __forceinline__ T direct_ctx_g(rocshmem_ctx_t ctx, const T *source,
                                          int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_G);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, g(source, pe));
}

#define ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(NAME, STAT, BACKEND_FUNC)     \
  template <typename T>                                                \
  __device__ __forceinline__ void direct_ctx_##NAME(                   \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems,     \
      int pe) {                                                        \
    if (nelems == 0) {                                                 \
      return;                                                          \
    }                                                                  \
    get_base_internal_ctx(ctx)->ctxStats.incStat(STAT);                \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH(                                  \
        ctx, BACKEND_FUNC(dest, source, nelems, pe));                  \
  }

ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put, NUM_PUT, put)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put_nbi, NUM_PUT_NBI, put_nbi)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get, NUM_GET, get)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get_nbi, NUM_GET_NBI, get_nbi)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put_wg, NUM_PUT_WG, put_wg)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put_nbi_wg, NUM_PUT_NBI_WG, put_nbi_wg)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get_wg, NUM_GET_WG, get_wg)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get_nbi_wg, NUM_GET_NBI_WG, get_nbi_wg)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put_wave, NUM_PUT_WAVE, put_wave)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(put_nbi_wave, NUM_PUT_NBI_WAVE, put_nbi_wave)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get_wave, NUM_GET_WAVE, get_wave)
ROCSHMEM_DIRECT_CTX_T_MEM_HELPER(get_nbi_wave, NUM_GET_NBI_WAVE, get_nbi_wave)

#undef ROCSHMEM_DIRECT_CTX_T_MEM_HELPER

template <typename T, ROCSHMEM_OP Op>
__device__ __forceinline__ int direct_ctx_reduce_wg(rocshmem_ctx_t ctx,
                                                    rocshmem_team_t team,
                                                    T *dest, const T *source,
                                                    int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_REDUCE);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, reduce_wg<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T, ROCSHMEM_OP Op>
__device__ __forceinline__ int direct_ctx_reduce_scatter_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,
    int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_REDUCE_SCATTER);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, reduce_scatter_wg<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T, ROCSHMEM_OP Op>
__device__ __forceinline__ int direct_ctx_reduce_wave(rocshmem_ctx_t ctx,
                                                      rocshmem_team_t team,
                                                      T *dest,
                                                      const T *source,
                                                      int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_REDUCE);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, reduce_wave<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T, ROCSHMEM_OP Op>
__device__ __forceinline__ int direct_ctx_reduce_scatter_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,
    int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_wave()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_REDUCE_SCATTER);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, reduce_scatter_wave<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T>
__device__ __forceinline__ void direct_ctx_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,
    int nelems, int pe_root) {
  if (nelems == 0) {
    return;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_BROADCAST);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
      ctx, broadcast_wg<T>(team, dest, source, nelems, pe_root));
}

template <typename T>
__device__ __forceinline__ int direct_ctx_broadcast_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,
    int nelems, int pe_root) {
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, broadcast_wave<T>(team, dest, source, nelems, pe_root));
}

template <typename T>
__device__ __forceinline__ void direct_ctx_alltoall_wg(rocshmem_ctx_t ctx,
                                                       rocshmem_team_t team,
                                                       T *dest,
                                                       const T *source,
                                                       int nelems) {
  if (nelems == 0) {
    return;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ALLTOALL);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
    ctx, alltoall_wg<T>(team, dest, source, nelems));
}

template <typename T>
__device__ __forceinline__ int direct_ctx_alltoall_wave(rocshmem_ctx_t ctx,
                                                        rocshmem_team_t team,
                                                        T *dest,
                                                        const T *source,
                                                        int nelems) {
  if (nelems == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ALLTOALL);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, alltoall_wave<T>(team, dest, source, nelems));
}

/* NOTE: no nelems guard, matches Context::alltoallv<T> exactly. */
template <typename T>
__device__ __forceinline__ void direct_ctx_alltoallv(
    rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[]) {
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ALLTOALLV);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
      ctx, alltoallv<T>(team, dest, dest_nelems, dest_displs, source,
                        source_nelems, source_displs));
}

template <typename T>
__device__ __forceinline__ void direct_ctx_fcollect_wg(rocshmem_ctx_t ctx,
                                                       rocshmem_team_t team,
                                                       T *dest,
                                                       const T *source,
                                                       int nelems) {
  if (nelems == 0) {
    return;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FCOLLECT);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(
    ctx, fcollect_wg<T>(team, dest, source, nelems));
}

template <typename T>
__device__ __forceinline__ int direct_ctx_fcollect_wave(rocshmem_ctx_t ctx,
                                                        rocshmem_team_t team,
                                                        T *dest,
                                                        const T *source,
                                                        int nelems) {
  if (nelems == 0) {
    return ROCSHMEM_SUCCESS;
  }
  if (is_thread_zero_in_block()) {
    get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_FCOLLECT);
  }
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
      ctx, fcollect_wave<T>(team, dest, source, nelems));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_fetch_add(rocshmem_ctx_t ctx,
                                                   void *dst, T value,
                                                   int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_FADD);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, amo_fetch_add(dst, value, pe));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_fetch_cas(rocshmem_ctx_t ctx,
                                                   void *dst, T value,
                                                   T cond, int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_FCSWAP);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(
    ctx, amo_fetch_cas(dst, value, cond, pe));
}

template <typename T>
__device__ __noinline__ void direct_ctx_amo_add(rocshmem_ctx_t ctx,
                                                void *dst, T value,
                                                int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_ADD);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, amo_add(dst, value, pe));
}

template <typename T>
__device__ __noinline__ void direct_ctx_amo_set(rocshmem_ctx_t ctx,
                                                void *dst, T value,
                                                int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_SET);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, amo_set(dst, value, pe));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_swap(rocshmem_ctx_t ctx,
                                              void *dst, T value, int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_SWAP);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, amo_swap(dst, value, pe));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_fetch_and(rocshmem_ctx_t ctx,
                                                   void *dst, T value,
                                                   int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_FETCH_AND);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, amo_fetch_and(dst, value, pe));
}

template <typename T>
__device__ __noinline__ void direct_ctx_amo_and(rocshmem_ctx_t ctx,
                                                void *dst, T value,
                                                int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_AND);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, amo_and(dst, value, pe));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_fetch_or(rocshmem_ctx_t ctx,
                                                  void *dst, T value,
                                                  int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_FETCH_OR);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, amo_fetch_or(dst, value, pe));
}

template <typename T>
__device__ __noinline__ void direct_ctx_amo_or(rocshmem_ctx_t ctx,
                                               void *dst, T value,
                                               int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_OR);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, amo_or(dst, value, pe));
}

template <typename T>
__device__ __noinline__ T direct_ctx_amo_fetch_xor(rocshmem_ctx_t ctx,
                                                   void *dst, T value,
                                                   int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_FETCH_XOR);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET(ctx, amo_fetch_xor(dst, value, pe));
}

template <typename T>
__device__ __noinline__ void direct_ctx_amo_xor(rocshmem_ctx_t ctx,
                                                void *dst, T value,
                                                int pe) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_ATOMIC_XOR);
  ROCSHMEM_DIRECT_BACKEND_DISPATCH(ctx, amo_xor(dst, value, pe));
}

#define ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(SUFFIX, STATS_SUFFIX)              \
  template <typename T>                                                       \
  __device__ __forceinline__ void direct_ctx_put_signal##SUFFIX(              \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems,            \
      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {              \
    if (nelems == 0) {                                                        \
      return;                                                                 \
    }                                                                         \
    get_base_internal_ctx(ctx)->ctxStats.incStat(                             \
      NUM_PUT_SIGNAL##STATS_SUFFIX);                                          \
    ROCSHMEM_DIRECT_BACKEND_DISPATCH(                                         \
        ctx, put_signal##SUFFIX(dest, source, nelems, sig_addr, signal,       \
                                sig_op, pe));                                 \
  }

ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(, )
ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(_wg, _WG)
ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(_wave, _WAVE)
ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(_nbi, _NBI)
ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(_nbi_wg, _NBI_WG)
ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF(_nbi_wave, _NBI_WAVE)

#undef ROCSHMEM_DIRECT_CTX_PUT_SIGNAL_DEF

/*
 * direct_ctx_test/direct_ctx_wait_until* helpers mirroring Context::test/
 * wait_until* bodies in context_tmpl_device.hpp exactly (stat increment,
 * forward to the base Context template method). These do not use
 * ROCSHMEM_DIRECT_BACKEND_DISPATCH because Context::test/wait_until* are
 * plain non-virtual templates on the base Context class, not per-backend
 * DISPATCH-switched methods -- test() itself is backend-agnostic
 * (uncached_load + comparison), so there is nothing to switch on.
 */
template <typename T>
__device__ __forceinline__ int direct_ctx_test(rocshmem_ctx_t ctx, T *ivars,
                                               int cmp, T val) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_TEST);
  return get_base_internal_ctx(ctx)->test(ivars, cmp, val);
}

template <typename T>
__device__ __forceinline__ void direct_ctx_wait_until(rocshmem_ctx_t ctx,
                                                      T *ivars, int cmp,
                                                      T val) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL);
  get_base_internal_ctx(ctx)->wait_until(ivars, cmp, val);
}

template <typename T>
__device__ __forceinline__ void direct_ctx_wait_until_all(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, const int *status, int cmp,
    T val) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_ALL);
  get_base_internal_ctx(ctx)->wait_until_all(ivars, nelems, status, cmp, val);
}

template <typename T>
__device__ __forceinline__ size_t direct_ctx_wait_until_any(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, const int *status, int cmp,
    T val) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_ANY);
  return get_base_internal_ctx(ctx)->wait_until_any(ivars, nelems, status,
                                                     cmp, val);
}

template <typename T>
__device__ __forceinline__ size_t direct_ctx_wait_until_some(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, size_t *indices,
    const int *status, int cmp, T val) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_SOME);
  return get_base_internal_ctx(ctx)->wait_until_some(ivars, nelems, indices,
                                                      status, cmp, val);
}

template <typename T>
__device__ __forceinline__ size_t direct_ctx_wait_until_any_vector(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, const int *status, int cmp,
    T *vals) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_ANY_VECTOR);
  return get_base_internal_ctx(ctx)->wait_until_any_vector(ivars, nelems,
                                                            status, cmp, vals);
}

template <typename T>
__device__ __forceinline__ void direct_ctx_wait_until_all_vector(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, const int *status, int cmp,
    T *vals) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_ALL_VECTOR);
  get_base_internal_ctx(ctx)->wait_until_all_vector(ivars, nelems, status,
                                                     cmp, vals);
}

template <typename T>
__device__ __forceinline__ size_t direct_ctx_wait_until_some_vector(
    rocshmem_ctx_t ctx, T *ivars, size_t nelems, size_t *indices,
    const int *status, int cmp, T *vals) {
  get_base_internal_ctx(ctx)->ctxStats.incStat(NUM_WAIT_UNTIL_SOME_VECTOR);
  return get_base_internal_ctx(ctx)->wait_until_some_vector(
      ivars, nelems, indices, status, cmp, vals);
}

#undef ROCSHMEM_DIRECT_BACKEND_DISPATCH
#undef ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET
#undef ROCSHMEM_DIRECT_BACKEND_DISPATCH_RET_PTR

/*
 * Part 2: direct dispatch for device_backend_proxy->create_ctx/destroy_ctx,
 * bypassing Backend::create_ctx/destroy_ctx's runtime switch(this->type)
 * (backend_bc_device.cpp) in favor of the already-resident
 * constmem.backend_type scalar.
 */
template <BackendType B>
struct DirectBackend;

#if defined(USE_GDA)
template <>
struct DirectBackend<BackendType::GDA_BACKEND> {
  using Type = GDABackend;
};
#endif

#if defined(USE_RO)
template <>
struct DirectBackend<BackendType::RO_BACKEND> {
  using Type = ROBackend;
};
#endif

#if defined(USE_IPC)
template <>
struct DirectBackend<BackendType::IPC_BACKEND> {
  using Type = IPCBackend;
};
#endif

template <BackendType B>
__device__ __forceinline__ typename DirectBackend<B>::Type *get_backend(
    Backend *b) {
  return static_cast<typename DirectBackend<B>::Type *>(b);
}

__device__ __forceinline__ bool direct_create_ctx(int64_t options,
                                                   rocshmem_ctx_t *ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch (constmem.backend_type) {
    case BackendType::GDA_BACKEND:
      return get_backend<BackendType::GDA_BACKEND>(device_backend_proxy)
          ->create_ctx(options, ctx);
    case BackendType::RO_BACKEND:
      return get_backend<BackendType::RO_BACKEND>(device_backend_proxy)
          ->create_ctx(options, ctx);
    case BackendType::IPC_BACKEND:
    default:
      return get_backend<BackendType::IPC_BACKEND>(device_backend_proxy)
          ->create_ctx(options, ctx);
  }
#elif defined(USE_GDA)
  return get_backend<BackendType::GDA_BACKEND>(device_backend_proxy)
      ->create_ctx(options, ctx);
#elif defined(USE_RO)
  return get_backend<BackendType::RO_BACKEND>(device_backend_proxy)
      ->create_ctx(options, ctx);
#elif defined(USE_IPC)
  return get_backend<BackendType::IPC_BACKEND>(device_backend_proxy)
      ->create_ctx(options, ctx);
#endif
}

__device__ __forceinline__ void direct_destroy_ctx(rocshmem_ctx_t *ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch (constmem.backend_type) {
    case BackendType::GDA_BACKEND:
      get_backend<BackendType::GDA_BACKEND>(device_backend_proxy)
          ->destroy_ctx(ctx);
      break;
    case BackendType::RO_BACKEND:
      get_backend<BackendType::RO_BACKEND>(device_backend_proxy)
          ->destroy_ctx(ctx);
      break;
    case BackendType::IPC_BACKEND:
    default:
      get_backend<BackendType::IPC_BACKEND>(device_backend_proxy)
          ->destroy_ctx(ctx);
      break;
  }
#elif defined(USE_GDA)
  get_backend<BackendType::GDA_BACKEND>(device_backend_proxy)
      ->destroy_ctx(ctx);
#elif defined(USE_RO)
  get_backend<BackendType::RO_BACKEND>(device_backend_proxy)->destroy_ctx(ctx);
#elif defined(USE_IPC)
  get_backend<BackendType::IPC_BACKEND>(device_backend_proxy)
      ->destroy_ctx(ctx);
#endif
}

}  // namespace

__device__ int rocshmem_wg_ctx_create(long options, rocshmem_ctx_t *ctx) {
  LOGD_API("device::wg_ctx_create (options=%ld)", options);
  bool result{true};
  if (get_flat_block_id() == 0) {
    ctx->team_opaque = reinterpret_cast<TeamInfo *>(ROCSHMEM_CTX_DEFAULT.team_opaque);
    result = direct_create_ctx(options, ctx);
    if (!result) {
      *ctx = ROCSHMEM_CTX_INVALID;
    }
  }
  __syncthreads();
  return result == true ? 0 : -1;
}

__device__ int rocshmem_wg_team_create_ctx(rocshmem_team_t team, long options,
                                           rocshmem_ctx_t *ctx) {
  LOGD_API("device::wg_team_create_ctx (team=%zd, options=%ld)",
    (intptr_t)team, options);
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  }

  bool result{true};
  if (get_flat_block_id() == 0) {
    Team *team_obj{get_internal_team(team)};
    TeamInfo *info_wrt_world = team_obj->tinfo_wrt_world;
    ctx->team_opaque = info_wrt_world;
    result = direct_create_ctx(options, ctx);
    if (!result) {
      *ctx = ROCSHMEM_CTX_INVALID;
    }
  }
  __syncthreads();

  return result == true ? 0 : -1;
}

__device__ void rocshmem_wg_ctx_destroy([[maybe_unused]] rocshmem_ctx_t *ctx) {
  LOGD_API("device::wg_ctx_destroy (ctx=%zd)",
    ctx->ctx_opaque);

  if (get_flat_block_id() == 0 && *ctx != ROCSHMEM_CTX_INVALID) {
    direct_destroy_ctx(ctx);
  }
}

__device__ void rocshmem_ctx_threadfence_system(rocshmem_ctx_t ctx) {
  LOGD_API("device::ctx_threadfence_system (ctx=%zd)", ctx.ctx_opaque);

  direct_ctx_threadfence_system(ctx);
}

__device__ __attribute__((used)) __forceinline__ void rocshmem_ctx_putmem(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::ctx_putmem (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_putmem(ctx, dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ __attribute__((used)) __forceinline__ void rocshmem_put(
    rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::put (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_put<T>(ctx, dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::p (ctx=%zd, dest=%p, value=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)value, pe, pe_in_world);

  direct_ctx_p<T>(ctx, dest, value, pe_in_world);
}

template <typename T>
__device__ T rocshmem_g(rocshmem_ctx_t ctx, const T *source, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::g (ctx=%zd, source=%p, pe=%d w%d)",
    ctx.ctx_opaque, source, pe, pe_in_world);

  return direct_ctx_g<T>(ctx, source, pe_in_world);
}

__device__ void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void *dest,
    const void *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::ctx_getmem (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_getmem(ctx, dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_get(rocshmem_ctx_t ctx, T *dest, const T *source,
    size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::get (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_get<T>(ctx, dest, source, nelems, pe_in_world);
}

__device__ __attribute__((used)) __forceinline__ void rocshmem_ctx_putmem_nbi(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::ctx_putmem_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_putmem_nbi(ctx, dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ __attribute__((used)) __forceinline__ void rocshmem_put_nbi(
    rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::put_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_put_nbi<T>(ctx, dest, source, nelems, pe_in_world);
}

__device__ void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::ctx_getmem_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_getmem_nbi(ctx, dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void rocshmem_get_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::get_nbi (ctx=%zd, dest=%p, source=%p, nelems=%zd, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, pe_in_world);

  direct_ctx_get_nbi<T>(ctx, dest, source, nelems, pe_in_world);
}

__device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx) {
  LOGD_API("device::ctx_fence (ctx=%zd)", ctx.ctx_opaque);

  direct_ctx_fence(ctx);
}

__device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe) {

  int pe_in_world = translate_pe(ctx, pe);
  LOGD_API("device::ctx_fence (ctx=%zd, pe=%d w%d))",
    ctx.ctx_opaque, pe, pe_in_world);

  direct_ctx_fence(ctx, pe_in_world);
}

__device__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  LOGD_API("device::ctx_quiet (ctx=%zd)",
    ctx.ctx_opaque);

  direct_ctx_quiet(ctx);
}

__device__ void rocshmem_ctx_pe_quiet(rocshmem_ctx_t ctx, const int *target_pes, size_t npes) {
  LOGD_API("device::ctx_pe_quiet (ctx=%zd)", ctx.ctx_opaque);

  for (size_t i = 0; i < npes;  i++) {
    direct_ctx_pe_quiet(ctx, translate_pe(ctx, target_pes[i]));
  }
}

__device__ void *rocshmem_ptr(const void *dest, int pe) {
  LOGD_API("device::ptr (dest=%p, pe=%d w%d",
    dest, pe, pe);

  return direct_ctx_shmem_ptr(ROCSHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T, ROCSHMEM_OP Op>
__device__ int rocshmem_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                   T *dest, const T *source, int nreduce) {
  LOGD_API("device::reduce_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nreduce=%d",
    ctx.ctx_opaque, team, dest, source, nreduce);

  return direct_ctx_reduce_wg<T, Op>(ctx, team, dest, source, nreduce);
}

template <typename T, ROCSHMEM_OP Op>
__device__ int rocshmem_reduce_scatter_wg(rocshmem_ctx_t ctx,
                                          rocshmem_team_t team, T *dest,
                                          const T *source, int nreduce) {
  LOGD_API("device::reduce_scatter_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nreduce=%d",
    ctx.ctx_opaque, team, dest, source, nreduce);

  return direct_ctx_reduce_scatter_wg<T, Op>(ctx, team, dest, source, nreduce);
}

template <typename T, ROCSHMEM_OP Op>
__device__ int rocshmem_reduce_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                    T *dest, const T *source, int nreduce) {
  LOGD_API("device::reduce_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nreduce=%d",
    ctx.ctx_opaque, team, dest, source, nreduce);

  return direct_ctx_reduce_wave<T, Op>(ctx, team, dest, source, nreduce);
}

template <typename T, ROCSHMEM_OP Op>
__device__ int rocshmem_reduce_scatter_wave(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team, T *dest,
                                            const T *source, int nreduce) {
  LOGD_API("device::reduce_scatter_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nreduce=%d",
    ctx.ctx_opaque, team, dest, source, nreduce);

  return direct_ctx_reduce_scatter_wave<T, Op>(ctx, team, dest, source, nreduce);
}

template <typename T>
__device__ void rocshmem_broadcast_wg(rocshmem_ctx_t ctx,
                                       rocshmem_team_t team, T *dest,
                                       const T *source, int nelems,
                                       int pe_root) {
  LOGD_API("device::broadcast_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d, root=%d)",
    ctx.ctx_opaque, team, dest, source, nelems, pe_root);

  direct_ctx_broadcast_wg<T>(ctx, team, dest, source, nelems, pe_root);
}

__device__ void rocshmem_ctx_broadcastmem_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
  void *dest, const void *source, int nelems, int pe_root) {
    LOGD_API("device::broadcastmem_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d, root=%d)",
      ctx.ctx_opaque, team, dest, source, nelems, pe_root);

    direct_ctx_broadcastmem_wg(ctx, team, dest, source, nelems, pe_root);
}

template <typename T>
__device__ int rocshmem_broadcast_wave(rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest,
                                       const T *source, int nelems, int pe_root) {
  LOGD_API("device::broadcast_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d, root=%d)",
    ctx.ctx_opaque, team, dest, source, nelems, pe_root);

  return direct_ctx_broadcast_wave<T>(ctx, team, dest, source, nelems, pe_root);
}

__device__ int rocshmem_ctx_broadcastmem_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
  void *dest, const void *source, int nelems, int pe_root) {
    LOGD_API("device::broadcastmem_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d, root=%d)",
      ctx.ctx_opaque, team, dest, source, nelems, pe_root);

    return direct_ctx_broadcastmem_wave(ctx, team, dest, source, nelems, pe_root);
}

template <typename T>
__device__ void rocshmem_ctx_alltoall_wg(rocshmem_ctx_t ctx,
                                         rocshmem_team_t team, T *dest,
                                         const T *source, int nelems) {
  LOGD_API("device::ctx_alltoall_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d)",
              ctx.ctx_opaque, team, dest, source, nelems);

  direct_ctx_alltoall_wg<T>(ctx, team, dest, source, nelems);
}

template <typename T>
__device__ void rocshmem_alltoall_wg(rocshmem_team_t team, T *dest,
                                     const T *source, int nelems) {
  LOGD_API("device::alltoall_wg (team=%zd, dest=%p, source=%p, nelems=%d)",
              team, dest, source, nelems);

  direct_ctx_alltoall_wg<T>(ROCSHMEM_CTX_DEFAULT, team, dest, source, nelems);
}

__device__ void rocshmem_ctx_alltoallmem_wg(rocshmem_ctx_t ctx,
          rocshmem_team_t team, void *dest, const void *source, int nelems){
  LOGD_API("device::ctx_alltoallmem_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d)",
              ctx.ctx_opaque, team, dest, source, nelems);

  direct_ctx_alltoallmem_wg(ctx, team, dest, source, nelems);
}

template <typename T>
__device__ void rocshmem_alltoallv_wg(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
  LOGD_API("device::alltoallv_wg(team=%zd, dest=%p, source=%p)",
              team, dest, source);

  direct_ctx_alltoallv<T>(ROCSHMEM_CTX_DEFAULT, team,
                         dest, dest_nelems, dest_displs,
                         source, source_nelems, source_displs);
}

template <typename T>
__device__ int rocshmem_ctx_alltoall_wave(rocshmem_ctx_t ctx,
                                           rocshmem_team_t team,
                                           T *dest, const T *source,
                                           int nelems) {
  LOGD_API("device::ctx_alltoall_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d)",
              ctx.ctx_opaque, team, dest, source, nelems);

  return direct_ctx_alltoall_wave<T>(ctx, team, dest, source, nelems);
}

__device__ int rocshmem_ctx_alltoallmem_wave(rocshmem_ctx_t ctx,
          rocshmem_team_t team, void *dest, const void *source, int nelems){
  LOGD_API("device::ctx_alltoallmem_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d)",
              ctx.ctx_opaque, team, dest, source, nelems);

  return direct_ctx_alltoallmem_wave(ctx, team, dest, source, nelems);
}

template <typename T>
__device__ void rocshmem_fcollect_wg(rocshmem_ctx_t ctx,
                                      rocshmem_team_t team, T *dest,
                                      const T *source, int nelems) {
  LOGD_API("device::fcollect_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d",
    ctx.ctx_opaque, team, dest, source, nelems);

  direct_ctx_fcollect_wg<T>(ctx, team, dest, source, nelems);
}

__device__ void rocshmem_ctx_fcollectmem_wg(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems) {
  LOGD_API("device::fcollectmem_wg (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d",
    ctx.ctx_opaque, team, dest, source, nelems);

  direct_ctx_fcollectmem_wg(ctx, team, dest, source, nelems);
}

template <typename T>
__device__ int rocshmem_fcollect_wave(rocshmem_ctx_t ctx,
    rocshmem_team_t team, T *dest, const T *source, int nelems) {
  LOGD_API("device::fcollect_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d",
    ctx.ctx_opaque, team, dest, source, nelems);

  return direct_ctx_fcollect_wave<T>(ctx, team, dest, source, nelems);
}

__device__ int rocshmem_ctx_fcollectmem_wave(rocshmem_ctx_t ctx,
    rocshmem_team_t team, void *dest, const void *source, int nelems) {
  LOGD_API("device::fcollectmem_wave (ctx=%zd, team=%zd, dest=%p, source=%p, nelems=%d",
    ctx.ctx_opaque, team, dest, source, nelems);

  return direct_ctx_fcollectmem_wave(ctx, team, dest, source, nelems);
}

template <typename T>
__device__ void rocshmem_wait_until(T *ivars, int cmp, T val) {
  LOGD_API("device::wait_until (ivars=%p, cmp=%d, val=%g)",
    ivars, cmp, (double)val);

  direct_ctx_wait_until<T>(ROCSHMEM_CTX_DEFAULT, ivars, cmp, val);
}

template <typename T>
__device__ void rocshmem_wait_until_all(T *ivars, size_t nelems, const int* status,
                                         int cmp, T val) {
  LOGD_API("device::wait_until_all (ivars=%p, nelems=%zd cmp=%d, val=%g)",
    ivars, nelems, cmp, (double)val);

  direct_ctx_wait_until_all<T>(ROCSHMEM_CTX_DEFAULT, ivars, nelems, status,
                               cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_any(T *ivars, size_t nelems, const int* status,
                                           int cmp, T val) {
  LOGD_API("device::wait_until_any (ivars=%p, nelems=%zd cmp=%d, val=%g)",
    ivars, nelems, cmp, (double)val);

  return direct_ctx_wait_until_any<T>(ROCSHMEM_CTX_DEFAULT, ivars, nelems,
                                      status, cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_some(T *ivars, size_t nelems, size_t* indices,
                                          const int* status, int cmp,
                                          T val) {
  LOGD_API("device::wait_until_some (ivars=%p, nelems=%zd cmp=%d, val=%g)",
    ivars, nelems, cmp, (double)val);

  return direct_ctx_wait_until_some<T>(ROCSHMEM_CTX_DEFAULT, ivars, nelems,
                                       indices, status, cmp, val);
}

template <typename T>
__device__ size_t rocshmem_wait_until_any_vector(T *ivars, size_t nelems, const int* status,
                                                  int cmp, T* vals) {
  LOGD_API("device::wait_until_any_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)",
    ivars, nelems, cmp, vals);

  return direct_ctx_wait_until_any_vector<T>(ROCSHMEM_CTX_DEFAULT, ivars,
                                             nelems, status, cmp, vals);
}

template <typename T>
__device__ void rocshmem_wait_until_all_vector(T *ivars, size_t nelems, const int* status,
                                                int cmp, T* vals) {
  LOGD_API("device::wait_until_all_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)",
    ivars, nelems, cmp, vals);

  direct_ctx_wait_until_all_vector<T>(ROCSHMEM_CTX_DEFAULT, ivars, nelems,
                                      status, cmp, vals);
}

template <typename T>
__device__ size_t rocshmem_wait_until_some_vector(T *ivars, size_t nelems,
                                                 size_t* indices,
                                                 const int* status,
                                                 int cmp, T* vals) {
  LOGD_API("device::wait_until_some_vector (ivars=%p, nelems=%zd cmp=%d, vals=%p)",
    ivars, nelems, cmp, vals);

  return direct_ctx_wait_until_some_vector<T>(ROCSHMEM_CTX_DEFAULT, ivars,
                                              nelems, indices, status, cmp,
                                              vals);
}

template <typename T>
__device__ int rocshmem_test(T *ivars, int cmp, T val) {
  LOGD_API("device::test (ivars=%p, cmp=%d, val=%g)",
    ivars, cmp, (double)val);

  return direct_ctx_test<T>(ROCSHMEM_CTX_DEFAULT, ivars, cmp, val);
}

__global__ ATTR_NO_INLINE void rocshmem_barrier_all_kernel(){
  rocshmem_barrier_all();
}

__global__ ATTR_NO_INLINE void rocshmem_barrier_kernel(rocshmem_team_t team){
  rocshmem_ctx_barrier(ROCSHMEM_CTX_DEFAULT, team);
}

__global__ ATTR_NO_INLINE void rocshmem_quiet_kernel(){
  rocshmem_quiet();
}

__global__ ATTR_NO_INLINE void rocshmem_sync_all_kernel(){
  rocshmem_sync_all();
}

__global__ ATTR_NO_INLINE void rocshmem_team_sync_kernel(rocshmem_team_t team){
  rocshmem_ctx_sync(ROCSHMEM_CTX_DEFAULT, team);
}

__global__ ATTR_NO_INLINE void rocshmem_alltoallmem_kernel(rocshmem_team_t team,
                                                           void *dest,
                                                           const void *source,
                                                           size_t size) {
  rocshmem_ctx_alltoall_wg<char>(ROCSHMEM_CTX_DEFAULT, team, (char *) dest,
                                 (const char *) source, (int) size);
}

template <typename T, ROCSHMEM_OP Op>
__global__ ATTR_NO_INLINE void rocshmem_reduce_on_stream_kernel(rocshmem_team_t team,
                                            T *dest,
                                            const T *source,
                                            int nreduce)
{
  rocshmem_reduce_wg<T, Op>(ROCSHMEM_CTX_DEFAULT, team, dest,
                            source, nreduce);
}

#define REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, Op, Op_API) \
    template \
    __global__ ATTR_NO_INLINE void rocshmem_reduce_on_stream_kernel<T, Op_API>(rocshmem_team_t team, \
                                                                T *dest, \
                                                                const T *source, \
                                                                int nreduce); \

#define REDUCTION_ON_STREAM_KERNEL_DEF_GEN_ARITH(T, TNAME) \
    REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM) \
    REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN) \
    REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX) \
    REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define REDUCTION_ON_STREAM_KERNEL_DEF_GEN_BITWISE(T, TNAME)  \
  REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)  \
  REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, and, ROCSHMEM_AND) \
  REDUCTION_ON_STREAM_KERNEL_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define INT_REDUCTION_ON_STREAM_KERNEL_GEN(T, TNAME) \
  REDUCTION_ON_STREAM_KERNEL_DEF_GEN_ARITH(T, TNAME) \
  REDUCTION_ON_STREAM_KERNEL_DEF_GEN_BITWISE(T, TNAME)

#define FLOAT_REDUCTION_ON_STREAM_KERNEL_GEN(T, TNAME) REDUCTION_ON_STREAM_KERNEL_DEF_GEN_ARITH(T, TNAME)

__global__ ATTR_NO_INLINE void rocshmem_broadcastmem_kernel(
    rocshmem_team_t team, void *dest, const void *source, size_t nelems,
    int pe_root) {
  rocshmem_broadcast_wg<char>(ROCSHMEM_CTX_DEFAULT, team, (char *) dest, (const char *) source,
                              (int) nelems, pe_root);
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
  LOGD_API("device::barrier_all (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_barrier_all(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_barrier_all_wave() {
  LOGD_API("device::barrier_all_wave (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_barrier_all_wave(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_barrier_all_wg() {
  LOGD_API("device::barrier_all_wg (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_barrier_all_wg(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_ctx_barrier(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  LOGD_API("device::ctx_barrier (ctx=%zd, team=%zd)",
    ctx.ctx_opaque, team);

  direct_ctx_barrier(ctx, team);
}

__device__ void rocshmem_ctx_barrier_wave(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  LOGD_API("device::ctx_barrier_wave (ctx=%zd, team=%zd)",
    ctx.ctx_opaque, team);

  direct_ctx_barrier_wave(ctx, team);
}

__device__ void rocshmem_ctx_barrier_wg(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  LOGD_API("device::ctx_barrier_wg (ctx=%zd, team=%zd)",
    ctx.ctx_opaque, team);

  direct_ctx_barrier_wg(ctx, team);
}

__device__ void rocshmem_sync_all() {
  LOGD_API("device::sync_all (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_sync_all(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_sync_all_wave() {
  LOGD_API("device::sync_all_wave (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_sync_all_wave(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_sync_all_wg() {
  LOGD_API("device::sync_all_wg (ctx=%zd)",
    ROCSHMEM_CTX_DEFAULT.ctx_opaque);

  direct_ctx_sync_all_wg(ROCSHMEM_CTX_DEFAULT);
}

__device__ void rocshmem_ctx_sync(rocshmem_ctx_t ctx,
                                  rocshmem_team_t team) {
  LOGD_API("device::ctx_sync (ctx=%zd, team=%zd)",
    ctx.ctx_opaque, team);

  direct_ctx_sync_wg(ctx, team);
}

__device__ void rocshmem_ctx_sync_wave(rocshmem_ctx_t ctx,
                                       rocshmem_team_t team) {
  LOGD_API("device::ctx_sync_wave (ctx=%zd, team=%zd)",
      ctx.ctx_opaque, team);

  direct_ctx_sync_wg(ctx, team);
}

__device__ void rocshmem_ctx_sync_wg(rocshmem_ctx_t ctx,
                                     rocshmem_team_t team) {
  LOGD_API("device::ctx_sync_wg (ctx=%zd, team=%zd)",
      ctx.ctx_opaque, team);

  direct_ctx_sync_wg(ctx, team);
}

__device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx) {
  LOGD_API("device::ctx_n_pes (ctx=%zd)",
    ctx.ctx_opaque);

  if (ctx.team_opaque) {
    TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
    return tinfo->size;
  }
  return constmem.num_pes;
}

__device__ int rocshmem_n_pes() {
  return constmem.num_pes;
}

__device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx) {
  LOGD_API("device::ctx_my_pe (ctx=%zd)",
    ctx.ctx_opaque);

  if (!ctx.team_opaque) {
    return constmem.my_pe;
  }

  TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
  int my_pe{get_base_internal_ctx(ctx)->my_pe};
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
  return constmem.my_pe;
}

__device__ int rocshmem_team_n_pes(rocshmem_team_t team) {
  LOGD_API("device::team_n_pes (team=%zd)", team);
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->num_pes;
  }
}

__device__ int rocshmem_team_my_pe(rocshmem_team_t team) {
  LOGD_API("device::team_my_pe (team=%zd)", team);
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->my_pe;
  }
}

template <typename T>
__device__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  LOGD_API("device::atomic_fetch_add (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_add<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                           T val, int pe) {
  LOGD_API("device::atomic_compare_swap (ctx=%zd, dest=%p, cond=%g, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, cond, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_cas<T>(ctx, dest, val, cond, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  LOGD_API("device::atomic_fetch_inc (ctx=%zd, dest=%p, pe=%d w%d)",
    ctx.ctx_opaque, dest, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_add<T>(ctx, dest, 1, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch(rocshmem_ctx_t ctx, T *source, int pe) {
  LOGD_API("device::atomic_fetch (ctx=%zd, source=%p, pe=%d w%d)",
    ctx.ctx_opaque, source, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_add<T>(ctx, source, 0, pe);
}

template <typename T>
__device__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  LOGD_API("device::atomic_add (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  direct_ctx_amo_add<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  LOGD_API("device::atomic_inc (ctx=%zd, dest=%p, pe=%d w%d)",
    ctx.ctx_opaque, dest, pe, translate_pe(ctx, pe));

  direct_ctx_amo_add<T>(ctx, dest, 1, pe);
}

template <typename T>
__device__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  LOGD_API("device::atomic_set (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  direct_ctx_amo_set<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  LOGD_API("device::atomic_swap (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_swap<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_and(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  LOGD_API("device::atomic_fetch_and (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_and<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_and(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  LOGD_API("device::atomic_and (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  direct_ctx_amo_and<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_or(rocshmem_ctx_t ctx, T *dest, T val,
                                       int pe) {
  LOGD_API("device::atomic_fetch_or (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_or<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_or(rocshmem_ctx_t ctx, T *dest, T val,
                                    int pe) {
  LOGD_API("device::atomic_or (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  direct_ctx_amo_or<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ T rocshmem_atomic_fetch_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  LOGD_API("device::atomic_fetch_xor (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  return direct_ctx_amo_fetch_xor<T>(ctx, dest, val, pe);
}

template <typename T>
__device__ void rocshmem_atomic_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  LOGD_API("device::atomic_xor (ctx=%zd, dest=%p, val=%g, pe=%d w%d)",
    ctx.ctx_opaque, dest, (double)val, pe, translate_pe(ctx, pe));

  direct_ctx_amo_xor<T>(ctx, dest, val, pe);
}

/**
 *      SHMEM X RMA API for WG and Wave level
 */
__device__ void rocshmem_ctx_putmem_wave(rocshmem_ctx_t ctx, void *dest,
                                          const void *source, size_t nelems,
                                          int pe) {
  LOGD_API("device::ctx_putmem_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_putmem_wave(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_wg(rocshmem_ctx_t ctx, void *dest,
                                        const void *source, size_t nelems,
                                        int pe) {
  LOGD_API("device::ctx_putmem_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_putmem_wg(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_nbi_wave(rocshmem_ctx_t ctx, void *dest,
                                              const void *source,
                                              size_t nelems, int pe) {
  LOGD_API("device::ctx_putmem_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_putmem_nbi_wave(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_putmem_nbi_wg(rocshmem_ctx_t ctx, void *dest,
                                            const void *source, size_t nelems,
                                            int pe) {
  LOGD_API("device::ctx_putmem_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_putmem_nbi_wg(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wave(rocshmem_ctx_t ctx, T *dest,
                                   const T *source, size_t nelems, int pe) {
  LOGD_API("device::put_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_put_wave<T>(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                 size_t nelems, int pe) {
  LOGD_API("device::put_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_put_wg<T>(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                       const T *source, size_t nelems,
                                       int pe) {
  LOGD_API("device::put_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_put_nbi_wave<T>(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                     const T *source, size_t nelems, int pe) {
  LOGD_API("device::put_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_put_nbi_wg<T>(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_wg(rocshmem_ctx_t ctx, void *dest,
                                        const void *source, size_t nelems,
                                        int pe) {
  LOGD_API("device::ctx_getmem_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_getmem_wg(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_wave(rocshmem_ctx_t ctx, void *dest,
                                          const void *source, size_t nelems,
                                          int pe) {
  LOGD_API("device::ctx_getmem_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_getmem_wave(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                 size_t nelems, int pe) {
  LOGD_API("device::get_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_get_wg<T>(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wave(rocshmem_ctx_t ctx, T *dest,
                                   const T *source, size_t nelems, int pe) {
  LOGD_API("device::get_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_get_wave<T>(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_nbi_wg(rocshmem_ctx_t ctx, void *dest,
                                            const void *source, size_t nelems,
                                            int pe) {
  LOGD_API("device::ctx_getmem_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_getmem_nbi_wg(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                     const T *source, size_t nelems, int pe) {
  LOGD_API("device::get_nbi_wg (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_get_nbi_wg<T>(ctx, dest, source, nelems, pe);
}

__device__ void rocshmem_ctx_getmem_nbi_wave(rocshmem_ctx_t ctx, void *dest,
                                              const void *source,
                                              size_t nelems, int pe) {
  LOGD_API("device::ctx_getmem_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_getmem_nbi_wave(ctx, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                       const T *source, size_t nelems,
                                       int pe) {
  LOGD_API("device::get_nbi_wave (ctx=%zd, dest=%p, source=%p, nelems=%d, pe=%d w%d)",
    ctx.ctx_opaque, dest, source, nelems, pe, translate_pe(ctx, pe));

  direct_ctx_get_nbi_wave<T>(ctx, dest, source, nelems, pe);
}

#define ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(SUFFIX)                                             \
  __device__ __attribute__((used)) __forceinline__                                         \
  void rocshmem_ctx_putmem_signal##SUFFIX(rocshmem_ctx_t ctx,                              \
                                                      void *dest, const void *source,      \
                                                      size_t nelems,                       \
                                                      uint64_t *sig_addr, uint64_t signal, \
                                                      int sig_op,                          \
                                                      int pe) {                            \
    LOGD_API("device::ctx_putmem_signal##SUFFIX (ctx=%zd, dest=%p, "                       \
      "source=%p, nelems=%d, sig_addr=%p, signal=%ld, sig_op=%d, pe=%d w%d)\n",            \
      ctx.ctx_opaque, dest, source, nelems,                                                \
      sig_addr, signal, sig_op, pe, translate_pe(ctx, pe));                                \
                                                                                           \
    direct_ctx_putmem_signal##SUFFIX(ctx, dest, source, nelems,                            \
                                     sig_addr, signal, sig_op, pe);                        \
  }                                                                                        \
                                                                                           \
  template <typename T>                                                                    \
  __device__ void rocshmem_ctx_put_signal##SUFFIX(rocshmem_ctx_t ctx,                      \
                                                   T *dest, const T *source,               \
                                                   size_t nelems,                          \
                                                   uint64_t *sig_addr, uint64_t signal,    \
                                                   int sig_op, int pe) {                   \
    LOGD_API("device::ctx_put_signal##SUFFIX (ctx=%zd, dest=%p, "                          \
      "source=%p, nelems=%d, sig_addr=%p, signal=%ld, sig_op=%d, pe=%d w%d)\n",            \
      ctx.ctx_opaque, dest, source, nelems,                                                \
      sig_addr, signal, sig_op, pe, translate_pe(ctx, pe));                                \
                                                                                           \
    direct_ctx_put_signal##SUFFIX<T>(ctx, dest, source, nelems,                            \
                                     sig_addr, signal, sig_op, pe);                        \
  }

ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF()
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_wg)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_wave)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi_wg)
ROCSHMEM_CTX_PUTMEM_SIGNAL_DEF(_nbi_wave)

#define ROCSHMEM_SIGNAL_FETCH_DEF(SUFFIX)                                          \
  __device__ uint64_t rocshmem_signal_fetch##SUFFIX(const uint64_t *sig_addr) {    \
    return direct_ctx_signal_fetch##SUFFIX(ROCSHMEM_CTX_DEFAULT, sig_addr);        \
  }

ROCSHMEM_SIGNAL_FETCH_DEF()
ROCSHMEM_SIGNAL_FETCH_DEF(_wg)
ROCSHMEM_SIGNAL_FETCH_DEF(_wave)

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__device__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe,
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
      int nreduce);                                                            \
  template __device__ int rocshmem_reduce_scatter_wg<T, Op>(                   \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nreduce);                                                            \
  template __device__ int rocshmem_reduce_wave<T, Op>(                         \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nreduce);                                                            \
  template __device__ int rocshmem_reduce_scatter_wave<T, Op>(                 \
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
      int nelems, int pe_root);                                                \
  template __device__ int rocshmem_broadcast_wave<T>(                          \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelems, int pe_root);                                                \
  template __device__ void rocshmem_ctx_alltoall_wg<T>(                        \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelems);                                                             \
  template __device__ void rocshmem_alltoall_wg<T>(                            \
      rocshmem_team_t team, T * dest, const T *source,                         \
      int nelems);                                                             \
  template __device__ int rocshmem_ctx_alltoall_wave<T>(rocshmem_ctx_t ctx,    \
      rocshmem_team_t team, T *dest, const T *source, int nelems);             \
  template __device__ void rocshmem_alltoallv_wg<T>(                           \
                                      rocshmem_team_t team,                    \
                                      T *dest, const size_t dest_nelems[],     \
                                      const size_t dest_displs[],              \
                                      T *source, const size_t source_nelems[], \
                                      const size_t source_displs[]);           \
  template __device__ void rocshmem_fcollect_wg<T>(                            \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelems);                                                             \
  template __device__ int rocshmem_fcollect_wave<T>(                           \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,     \
      int nelems);                                                             \
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

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                                   \
  __device__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce_wg(                     \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,         \
      int nreduce) {                                                              \
    return rocshmem_reduce_wg<T, Op>(ctx, team, dest, source, nreduce);           \
  }                                                                               \
  __device__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce_wave(                   \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,         \
      int nreduce) {                                                              \
    return rocshmem_reduce_wave<T, Op>(ctx, team, dest, source, nreduce);         \
  }    
    
#define REDUCE_SCATTER_DEF_GEN(T, TNAME, Op_API, Op)                              \
  __device__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce_scatter_wg(             \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,         \
      int nreduce) {                                                              \
    return rocshmem_reduce_scatter_wg<T, Op>(ctx, team, dest, source, nreduce);   \
  }

#define REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, Op_API, Op)                         \
  __device__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce_scatter_wave(           \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,         \
      int nreduce) {                                                              \
    return rocshmem_reduce_scatter_wave<T, Op>(ctx, team, dest, source, nreduce); \
  }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME)                        \
  REDUCTION_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)                 \
  REDUCTION_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)                 \
  REDUCTION_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)                 \
  REDUCTION_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)               \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)            \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)            \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)            \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)          \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)       \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)       \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)       \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME)                      \
  REDUCTION_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)                   \
  REDUCTION_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)                 \
  REDUCTION_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)                 \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)              \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)            \
  REDUCE_SCATTER_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)            \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)         \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)       \
  REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

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
      int nelems, int pe_root) {                                              \
    rocshmem_broadcast_wg<T>(ctx, team, dest, source, nelems, pe_root);       \
  }                                                                           \
  __device__ int rocshmem_ctx_##TNAME##_broadcast_wave(                       \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelems, int pe_root) {                                              \
    return rocshmem_broadcast_wave<T>(ctx, team, dest,                        \
                                       source, nelems, pe_root);              \
  }                                                                           \
  __device__ void rocshmem_ctx_##TNAME##_alltoall_wg(                         \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelems) {                                                           \
    rocshmem_ctx_alltoall_wg<T>(ctx, team, dest, source, nelems);             \
  }                                                                           \
  __device__ void rocshmem_##TNAME##_alltoall_wg(                             \
      rocshmem_team_t team, T *dest, const T *source,                         \
      int nelems) {                                                           \
    rocshmem_alltoall_wg<T>(team, dest, source, nelems);                      \
  }                                                                           \
  __device__ int rocshmem_ctx_##TNAME##_alltoall_wave(rocshmem_ctx_t ctx,     \
      rocshmem_team_t team, T *dest, const T *source, int nelems) {           \
    return rocshmem_ctx_alltoall_wave<T>(ctx, team, dest, source, nelems);    \
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
      int nelems) {                                                           \
    rocshmem_fcollect_wg<T>(ctx, team, dest, source, nelems);                 \
  }                                                                           \
   __device__ int rocshmem_ctx_##TNAME##_fcollect_wave(                       \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelems) {                                                           \
    return rocshmem_fcollect_wave<T>(ctx, team, dest, source, nelems);        \
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
  __device__ void rocshmem_ctx_put_signal##SUFFIX(rocshmem_ctx_t ctx,                    \
                                                    T *dest, const T *source,            \
                                                    size_t nelems,                       \
                                                    uint64_t *sig_addr, uint64_t signal, \
                                                    int sig_op, int pe);                 \
                                                                                         \
  template <typename T>                                                                  \
  __device__ void rocshmem_put_signal##SUFFIX(T *dest, const T *source, size_t nelems,   \
                                                uint64_t *sig_addr, uint64_t signal,     \
                                                int sig_op, int pe);                     \

#define RMA_SIGNAL_SUFFIX_DEF(T, TNAME, SUFFIX)                                                   \
  __device__ void rocshmem_ctx_##TNAME##_put_signal##SUFFIX(rocshmem_ctx_t ctx,                   \
                                                             T *dest, const T *source,            \
                                                             size_t nelems,                       \
                                                             uint64_t *sig_addr, uint64_t signal, \
                                                             int sig_op, int pe) {                \
    rocshmem_ctx_put_signal##SUFFIX<T>(ctx, dest, source, nelems, sig_addr, signal, sig_op, pe);  \
  }                                                                                               \
                                                                                                  \
  __device__ void rocshmem_##TNAME##_put_signal##SUFFIX(T *dest, const T *source, size_t nelems,  \
                                                         uint64_t *sig_addr, uint64_t signal,     \
                                                         int sig_op, int pe) {                    \
    rocshmem_put_signal##SUFFIX(dest, source, nelems, sig_addr, signal, sig_op, pe);              \
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

INT_REDUCTION_ON_STREAM_KERNEL_GEN(int, int)
INT_REDUCTION_ON_STREAM_KERNEL_GEN(long, long)
INT_REDUCTION_ON_STREAM_KERNEL_GEN(long long, longlong)
INT_REDUCTION_ON_STREAM_KERNEL_GEN(short, short)

FLOAT_REDUCTION_ON_STREAM_KERNEL_GEN(float, float)
FLOAT_REDUCTION_ON_STREAM_KERNEL_GEN(double, double)
// clang-format on

}  // namespace rocshmem
