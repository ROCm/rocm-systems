// Minimal HIP stub for CPU-only RCCL unit tests.
// Provides type definitions needed to compile RCCL headers without HIP runtime.
#pragma once

#include <cstdint>
#include <cstddef>

#define __device__
#define __host__
#define __global__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};

struct uint4 { unsigned int x, y, z, w; };
struct int4  { int x, y, z, w; };

// Stream/event/device types
using hipStream_t = void*;
using hipError_t = int;
using hipEvent_t = void*;
using hipDeviceProp_t = struct { char name[256]; char gcnArchName[256]; int major; int minor; };
using hipDevice_t = int;
using hipIpcMemHandle_t = struct { char reserved[64]; };
using hipIpcEventHandle_t = struct { char reserved[64]; };
using hipPointerAttribute_t = struct { int type; int device; void* devicePointer; void* hostPointer; };

// Memory types
using hipMemGenericAllocationHandle_t = uint64_t;
using hipMemAllocationHandleType = int;

// Error codes
enum {
    hipSuccess = 0,
    hipErrorNotReady = 600,
    hipErrorStreamCaptureInvalidated = 901,
};

// Memory copy kinds
enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
};

// Stream flags
static constexpr hipStream_t hipStreamLegacy = nullptr;

// Batch mem op params (opaque struct used in function pointer typedefs)
struct hipStreamBatchMemOpParams { uint64_t padding[8]; };

// API stubs (never called, just need to compile)
inline const char* hipGetErrorString(hipError_t) { return "stub"; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipDriverGetVersion(int* v) { *v = 60400; return hipSuccess; }
inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipGetDevice(int* d) { *d = 0; return hipSuccess; }
inline hipError_t hipGetDeviceCount(int* c) { *c = 0; return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipMalloc(void**, size_t) { return hipSuccess; }
inline hipError_t hipFree(void*) { return hipSuccess; }
inline hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return hipSuccess; }
inline hipError_t hipMemset(void*, int, size_t) { return hipSuccess; }
inline hipError_t hipHostMalloc(void** p, size_t s, unsigned) { *p = nullptr; return hipSuccess; }
inline hipError_t hipHostFree(void*) { return hipSuccess; }
inline hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned) { return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
inline hipError_t hipEventCreate(hipEvent_t*) { return hipSuccess; }
inline hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipSuccess; }

// HIP compatibility aliases (CUDA-style names used by RCCL via hipify)
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
static constexpr int cudaSuccess = hipSuccess;
static constexpr int cudaErrorStreamCaptureInvalidated = hipErrorStreamCaptureInvalidated;
inline const char* cudaGetErrorString(cudaError_t e) { return hipGetErrorString(e); }
