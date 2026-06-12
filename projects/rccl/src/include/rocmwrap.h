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
#include "checks.h"

// Re-declare the DMA-BUF export entry as a weak reference. hsa_init,
// hsa_system_get_info and hsa_status_string are required and resolve as hard
// dependencies, but hsa_amd_portable_export_dmabuf is optional: older ROCr
// runtimes may not export it. A weak reference resolves to NULL at load time
// when the symbol is absent (instead of failing librccl's load with an
// undefined symbol), and pfn_hsa_amd_portable_export_dmabuf below stays the
// runtime feature gate. This declaration must precede every use of the symbol
// so all references (including the HSACHECK* macro call sites) are emitted weak.
extern "C" hsa_status_t hsa_amd_portable_export_dmabuf(
    const void* ptr, size_t size, int* dmabuf, uint64_t* offset) __attribute__((weak));

// hsa_init, hsa_system_get_info and hsa_status_string are called directly via the
// hsa-runtime64 library that librccl links against. Only the DMA-BUF export entry
// keeps a function-pointer indirection, because it doubles as the runtime feature
// gate: pfn_hsa_amd_portable_export_dmabuf stays NULL when the platform does not
// support DMA-BUF.

#ifndef CU_STREAM_WRITE_VALUE_DEFAULT
#define CU_STREAM_WRITE_VALUE_DEFAULT 0
#endif

// HIP: implemented in rma_proxy_launch.cc (hipStreamBatchMemOp + old-HIP fallback).
// CUDA: implemented in cudawrap.cc (cuStreamBatchMemOp).
ncclResult_t ncclCuStreamBatchMemOp(cudaStream_t stream, unsigned int numOps, CUstreamBatchMemOpParams* batchParams);
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
#define HSACHECK(cmd) do {				      \
    hsa_status_t err = pfn_##cmd;				      \
    if( err != HSA_STATUS_SUCCESS ) {				      \
      const char *errStr;				      \
      hsa_status_string(err, &errStr);	      \
      WARN("HSA failure '%s' at %s:%d", errStr, __FILE__, __LINE__); \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define HSACHECKGOTO(cmd, res, label) do {		      \
    hsa_status_t err = pfn_##cmd;				      \
    if( err != HSA_STATUS_SUCCESS ) {				      \
      const char *errStr;				      \
      hsa_status_string(err, &errStr);	      \
      WARN("HSA failure '%s' at %s:%d", errStr, __FILE__, __LINE__); \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    hipError_t err = cmd;				      \
    if( err != hipSuccess ) {				      \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__);		      \
      (void)hipGetLastError(); /* clear sticky HIP error state */   \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    hipError_t err = cmd;				      \
    if( err != hipSuccess ) {				      \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__);		      \
      (void)hipGetLastError(); /* clear sticky HIP error state */   \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    hipError_t err = cmd;						\
    if( err != hipSuccess ) {						\
      INFO(NCCL_ALL,"%s:%d HIP failure '%s'", __FILE__, __LINE__, hipGetErrorString(err));	\
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    hsa_status_t err = pfn_##cmd;						\
    if (err != HSA_STATUS_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_ROCM_PFN_EXTERN(symbol) extern PFN_##symbol pfn_##symbol

DECLARE_ROCM_PFN_EXTERN(hsa_amd_portable_export_dmabuf); // DMA-BUF feature gate

extern int ncclCuMemEnable();
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
