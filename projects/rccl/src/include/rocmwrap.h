/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ROCMWRAP_H_
#define NCCL_ROCMWRAP_H_

#include <hsa/hsa.h>
#include "checks.h"

typedef hsa_status_t (*PFN_hsa_init)();
typedef hsa_status_t (*PFN_hsa_system_get_info)(hsa_system_info_t attribute, void* value);
typedef hsa_status_t (*PFN_hsa_status_string)(hsa_status_t status, const char ** status_string);
typedef hsa_status_t (*PFN_hsa_amd_portable_export_dmabuf)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);

#ifdef __HIP_PLATFORM_AMD__
#define CUPFN(symbol) symbol
#else
#define CUPFN(symbol) pfn_##symbol
#endif

#define HSACHECK(cmd) do {				      \
    hsa_status_t err = pfn_##cmd;				      \
    if( err != HSA_STATUS_SUCCESS ) {				      \
      const char *errStr;				      \
      pfn_hsa_status_string(err, &errStr);	      \
      WARN("HIP failure '%s'", errStr);		      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    hipError_t err = cmd;				      \
    if( err != hipSuccess ) {				      \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__);		      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    hipError_t err = cmd;				      \
    if( err != hipSuccess ) {				      \
      WARN("HIP failure '%s' at %s:%d", hipGetErrorString(err), __FILE__, __LINE__);		      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    hsa_status_t err = pfn_##cmd;						\
    if( err != HSA_STATUS_SUCCESS ) {						\
      const char *errStr;						\
      pfn_hsa_status_string(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d HIP failure '%s'", __FILE__, __LINE__, errStr);	\
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

DECLARE_ROCM_PFN_EXTERN(hsa_amd_portable_export_dmabuf); // DMA-BUF support

/* ROCr Driver functions loaded with dlsym() */
DECLARE_ROCM_PFN_EXTERN(hsa_init);
DECLARE_ROCM_PFN_EXTERN(hsa_system_get_info);
DECLARE_ROCM_PFN_EXTERN(hsa_status_string);

extern int ncclCuMemEnable();
extern int ncclCuMemHostEnable();

// Handle type used for cuMemCreate()
extern CUmemAllocationHandleType ncclCuMemHandleType;

ncclResult_t rocmLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

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
