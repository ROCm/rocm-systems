/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ROCMWRAP_H_
#define NCCL_ROCMWRAP_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>  // hsa_amd_portable_export_dmabuf (DMA-BUF export)
#include <unistd.h>
#include "checks.h"

#ifndef CU_STREAM_WRITE_VALUE_DEFAULT
#define CU_STREAM_WRITE_VALUE_DEFAULT 0
#endif

// === ROCm/HIP version milestones ==========================================
// Every "magic" version number lives here, once. HIP_VERSION (compile time)
// and cudaDriverGetVersion() (runtime) both encode as
// MAJOR*10000000 + MINOR*100000 + PATCH.
#define ROCM_VER_7_0_2_2 70051831  // 7.0.2.x backport range, lower bound
#define ROCM_VER_7_0_3_0 70060000  // 7.0.2.x backport range, exclusive upper bound
#define ROCM_VER_7_12_0 71200000  // hipMemcpyBatchAsync native min
#define ROCM_VER_7_12_60540 71260540  // upstream cuMem/VMM + DMA-BUF export

// === Generic version-range predicates =====================================
// Constant expressions, so they are valid in both #if directives (compile
// time, against HIP_VERSION) and regular C++ (runtime driver version).
#define NCCL_VER_GE(v, lo) ((v) >= (lo))
#define NCCL_VER_IN(v, lo, hi) ((v) >= (lo) && (v) < (hi))

// === Per-feature VERSION predicates ========================================
// Each capability names its own support window by composing milestones; do NOT
// reuse a predicate across features that have different windows.

// cuMem VMM + DMA-BUF driver export: native 7.12 OR the 7.0.2.x backport.
#define NCCL_CUMEM_VERSION_SUPPORTED(v) \
  (NCCL_VER_GE(v, ROCM_VER_7_12_60540) || NCCL_VER_IN(v, ROCM_VER_7_0_2_2, ROCM_VER_7_0_3_0))

// cuMem HOST allocations: native only. NOT part of the 7.0.2.x backport (relies
// on hipDeviceAttributeHostNumaId, which is absent there).
#define NCCL_CUMEM_HOST_VERSION_SUPPORTED(v) NCCL_VER_GE(v, ROCM_VER_7_12_60540)

// Back-compat alias for the few call sites that compare against the native
// minimum directly.
#define NCCL_CUMEM_NATIVE_MIN_VERSION ROCM_VER_7_12_60540

// hipMemcpyBatchAsync: native 7.12 OR the 7.0.2.x backport. No device-attribute
// probe exists for the batch API, so this version window is the only runtime guard.
#define NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(v) \
  (NCCL_VER_GE(v, ROCM_VER_7_12_0) || NCCL_VER_IN(v, ROCM_VER_7_0_2_2, ROCM_VER_7_0_3_0))

// === Capability gate: both the symbol probe and the version must agree =====
// CMake sets RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED when
// check_symbol_exists(hipMemGetHandleForAddressRange ...) succeeds. The symbol
// is declared in headers on backport builds outside the supported window too,
// so it cannot gate the feature on its own: the version predicate has to agree.
#if defined(RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED)
#define NCCL_CUMEM_DMABUF_EXPORT_PROBE 1
#else
#define NCCL_CUMEM_DMABUF_EXPORT_PROBE 0
#endif

// Both gate inputs are parameters so every (probe, version) combination can be
// asserted in unit tests; the gate itself is this build's instantiation.
#define NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(probe, v) ((probe) && NCCL_CUMEM_VERSION_SUPPORTED(v))

#define NCCL_CUMEM_DMABUF_EXPORT_GATE NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(NCCL_CUMEM_DMABUF_EXPORT_PROBE, HIP_VERSION)

// Linux kernel floor for cuMem auto-detect. Encoding matches uname major/minor
// packing used by ncclGetKernelVersionCode() in rocmwrap.cc.
#define NCCL_KERNEL_VERSION_CODE(major, minor) (((major) << 16) | ((minor) << 8))
#define NCCL_CUMEM_MIN_KERNEL_VERSION NCCL_KERNEL_VERSION_CODE(6, 8)

// Kernel gate used by ncclCuMemCapabilityCheck. Auto-detect (param <= 0, including
// the default NCCL_CUMEM_ENABLE=-2) requires Linux >= 6.8. An explicit
// NCCL_CUMEM_ENABLE>0 overrides the floor so validation hosts on older kernels
// can still force the VMM path. Both inputs are parameters so unit tests can
// cover every combination without depending on the host kernel.
#define NCCL_CUMEM_KERNEL_GATE_FOR(kernelVersionCode, cumemEnableParam) \
  ((kernelVersionCode) >= NCCL_CUMEM_MIN_KERNEL_VERSION || (cumemEnableParam) > 0)

// HIP: implemented in rma_proxy_launch.cc (hipStreamBatchMemOp + old-HIP fallback).
// CUDA: implemented in cudawrap.cc (cuStreamBatchMemOp).
ncclResult_t ncclCuStreamBatchMemOp(cudaStream_t stream, unsigned int numOps, CUstreamBatchMemOpParams* batchParams);

// Re-declare the DMA-BUF export entry as a weak reference. hsa_init,
// hsa_system_get_info and hsa_status_string are required and resolve as hard
// dependencies, but hsa_amd_portable_export_dmabuf is optional: older ROCr
// runtimes may not export it. A weak reference resolves to NULL at load time
// when the symbol is absent (instead of failing librccl's load with an
// undefined symbol), and pfn_hsa_amd_portable_export_dmabuf below stays the
// runtime feature gate. This declaration must precede every use of the symbol
// so all references (including the HSACHECK* macro call sites) are emitted weak.
extern "C" hsa_status_t hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf, uint64_t* offset)
  __attribute__((weak));

// hsa_init, hsa_system_get_info and hsa_status_string are called directly via the
// hsa-runtime64 library that librccl links against. Only the DMA-BUF export entry
// keeps a function-pointer indirection, because it doubles as the runtime feature
// gate: pfn_hsa_amd_portable_export_dmabuf stays NULL when the platform does not
// support DMA-BUF.
typedef hsa_status_t (*PFN_hsa_amd_portable_export_dmabuf)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);

#ifdef __HIP_PLATFORM_AMD__
#define CUPFN(symbol) symbol
#else
#define CUPFN(symbol) pfn_##symbol
#endif

// Call sites go through pfn_##cmd so the (optional, weakly-linked)
// hsa_amd_portable_export_dmabuf stays gated by its function pointer and is not
// referenced as a symbol in every consumer TU. hsa_status_string is a required
// HSA entry point and is called directly.
#define HSACHECK(cmd) \
  do { \
    hsa_status_t err = pfn_##cmd; \
    if (err != HSA_STATUS_SUCCESS) { \
      const char* errStr; \
      hsa_status_string(err, &errStr); \
      WARN("HSA failure '%s' at %s:%d", errStr, __FILE__, __LINE__); \
      return ncclUnhandledCudaError; \
    } \
  } while (false)

#define HSACHECKGOTO(cmd, res, label) \
  do { \
    hsa_status_t err = pfn_##cmd; \
    if (err != HSA_STATUS_SUCCESS) { \
      const char* errStr; \
      hsa_status_string(err, &errStr); \
      WARN("HSA failure '%s' at %s:%d", errStr, __FILE__, __LINE__); \
      res = ncclUnhandledCudaError; \
      goto label; \
    } \
  } while (false)

// Check CUDA PFN driver calls
#define CUCHECK(cmd) \
  do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__); \
      (void)hipGetLastError(); /* clear sticky HIP error state */ \
      return ncclUnhandledCudaError; \
    } \
  } while (false)

#define CUCHECKGOTO(cmd, res, label) \
  do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__); \
      (void)hipGetLastError(); /* clear sticky HIP error state */ \
      res = ncclUnhandledCudaError; \
      goto label; \
    } \
  } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) \
  do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
      INFO(NCCL_ALL, "%s:%d HIP failure '%s'", __FILE__, __LINE__, hipGetErrorString(err)); \
    } \
  } while (false)

#define CUCHECKTHREAD(cmd, args) \
  do { \
    hsa_status_t err = pfn_##cmd; \
    if (err != HSA_STATUS_SUCCESS) { \
      INFO(NCCL_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError; \
      return args; \
    } \
  } while (0)

#define DECLARE_ROCM_PFN_EXTERN(symbol) extern PFN_##symbol pfn_##symbol

DECLARE_ROCM_PFN_EXTERN(hsa_amd_portable_export_dmabuf); // DMA-BUF feature gate

// Export `buffer` through the HSA portable DMA-BUF exporter and register the resulting fd
// with a net plugin's regMrDmaBuf (identical signature for ncclNet_t and ncclCollNet_t).
// The fd is closed on every path, so it can neither leak nor be clobbered by a later
// export into a caller-owned variable. Returns false when either step fails, so the caller
// can fall back to a plain regMr; callers must still gate on
// pfn_hsa_amd_portable_export_dmabuf being non-NULL.
static inline bool ncclHsaRegMrDmaBuf(ncclResult_t (*regMrDmaBuf)(void*, void*, size_t, int, uint64_t, int, void**),
                                      void* comm, void* buffer, size_t size, int type, void** mhandle) {
  int fd = -1;
  uint64_t offset = 0;
  hsa_status_t status = pfn_hsa_amd_portable_export_dmabuf(buffer, size, &fd, &offset);
  if (status != HSA_STATUS_SUCCESS) {
    const char* errStr;
    hsa_status_string(status, &errStr);
    WARN("hsa_amd_portable_export_dmabuf failed for %p (size %zu): '%s'", buffer, size, errStr);
    return false;
  }
  ncclResult_t res = regMrDmaBuf(comm, buffer, size, type, offset, fd, mhandle);
  (void)close(fd);
  if (res != ncclSuccess) {
    INFO(NCCL_NET, "DMA-BUF registration of %p (size %zu) failed with %d", buffer, size, res);
    return false;
  }
  return true;
}

extern int ncclCuMemEnable();
extern int ncclIsCuMemSupported();
extern int ncclCuMemRuntimeSupported();
extern int ncclCuMemHostEnable();
extern int64_t rcclParamForceEnableDMABUF();
extern int64_t ncclParamDmaBufEnable();

// Handle type used for cuMemCreate()
extern CUmemAllocationHandleType ncclCuMemHandleType;

ncclResult_t rocmLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

// [RCCL] cudawrap.h (now hipified) also defines ncclCudaDriverVersion. When a
// translation unit pulls both rocmwrap.h and cudawrap.h in (e.g. via alloc.h)
// we'd otherwise get a redefinition error. Pick a single canonical location.
#ifndef NCCL_CUDA_DRIVER_VERSION_DEFINED
#define NCCL_CUDA_DRIVER_VERSION_DEFINED
inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = __atomic_load_n(&ncclCudaDriverVersionCache, __ATOMIC_RELAXED);
  if (version == -1) {
    CUDACHECK(cudaDriverGetVersion(&version));
    __atomic_store_n(&ncclCudaDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}
#endif

// [RCCL] Upstream NCCL added this helper in cudawrap.h; on HIP we provide the
// equivalent that resolves the "legacy NULL" stream alias. Guard against
// redefinition when both rocmwrap.h and cudawrap.h land in the same TU.
#ifndef NCCL_CUDA_STREAM_IS_LEGACY_NULL_DEFINED
#define NCCL_CUDA_STREAM_IS_LEGACY_NULL_DEFINED
static inline ncclResult_t ncclCudaStreamIsLegacyNull(cudaStream_t stream, bool* isLegacy) {
#if CUDART_VERSION >= 11030
  unsigned long long streamId = ~0ULL;
  CUDACHECK(cudaStreamGetId(stream, &streamId));
  *isLegacy = (streamId == 0);
#else
  *isLegacy = (stream == NULL) || (stream == cudaStreamLegacy);
#endif
  return ncclSuccess;
}
#endif

#endif
