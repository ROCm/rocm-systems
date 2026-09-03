/*************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef INSPECTOR_GPU_COMPAT_H_
#define INSPECTOR_GPU_COMPAT_H_

// The inspector body is written against the CUDA spellings. On ROCm the names
// below are aliased onto their HIP equivalents so the rest of the plugin
// compiles for both platforms from a single set of sources.

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// The plugin is host-only, so the host API header is enough; hip_runtime.h
// would pull in device intrinsics that need the compiler in HIP language mode.
#include <hip/hip_runtime_api.h>

typedef hipError_t  cudaError_t;
typedef hipError_t  CUresult;
typedef hipDevice_t CUdevice;
typedef hipUUID     CUuuid;

// Both spellings are needed: the sources compare runtime-API results against
// cudaSuccess and driver-API results against CUDA_SUCCESS, which are distinct
// enumerators of distinct types under CUDA. HIP returns hipError_t from both.
constexpr hipError_t cudaSuccess  = hipSuccess;
constexpr hipError_t CUDA_SUCCESS = hipSuccess;

static inline const char* cudaGetErrorString(hipError_t err) {
  return hipGetErrorString(err);
}

static inline hipError_t cudaGetDevice(int* device) {
  return hipGetDevice(device);
}

// inspector_cudawrap.cc resolves the driver entry points by name at runtime.
// ROCm exports the driver API from libamdhip64 rather than a separate driver
// library, and spells the error-string helper hipDrvGetErrorString to keep it
// distinct from the runtime's hipGetErrorString.
#define INSPECTOR_GPU_DRIVER_LIB           "libamdhip64.so"
#define INSPECTOR_GPU_DRIVER_LIB_ALT       "libamdhip64.so.6"
#define INSPECTOR_GPU_SYM_GET_ERROR_STRING "hipDrvGetErrorString"
#define INSPECTOR_GPU_SYM_DEVICE_GET       "hipDeviceGet"
#define INSPECTOR_GPU_SYM_DEVICE_GET_UUID  "hipDeviceGetUuid"

#else

#include <cuda.h>
#include <cuda_runtime.h>

#define INSPECTOR_GPU_DRIVER_LIB           "libcuda.so"
#define INSPECTOR_GPU_DRIVER_LIB_ALT       "libcuda.so.1"
#define INSPECTOR_GPU_SYM_GET_ERROR_STRING "cuGetErrorString"
#define INSPECTOR_GPU_SYM_DEVICE_GET       "cuDeviceGet"
#define INSPECTOR_GPU_SYM_DEVICE_GET_UUID  "cuDeviceGetUuid"

#endif

#endif // INSPECTOR_GPU_COMPAT_H_
