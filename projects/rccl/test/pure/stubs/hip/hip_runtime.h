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

// GPU vector types
struct ulong2 { unsigned long x, y; };
struct ushort2 { unsigned short x, y; };
struct ulonglong2 { unsigned long long x, y; };

// GPU device intrinsics (never called on CPU, just need declarations to compile)
inline int __clz(unsigned int x) { return x ? __builtin_clz(x) : 32; }
inline int __popc(unsigned int x) { return __builtin_popcount(x); }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __ffs(unsigned int x) { return __builtin_ffs(x); }
inline int __fns(unsigned long long, int, int) { return 0; }
inline int __shfl(int, int, int w=64) { return 0; }
inline unsigned int __shfl(unsigned int, int, int w=64) { return 0; }
inline unsigned long long __shfl(unsigned long long, int, int w=64) { return 0; }
inline float __shfl(float, int, int w=64) { return 0.0f; }
inline unsigned long long __ballot(int) { return 0; }
inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_system() {}
inline int __lane_id() { return 0; }
inline unsigned int __umulhi(unsigned int, unsigned int) { return 0; }
inline unsigned long long __umul64hi(unsigned long long, unsigned long long) { return 0; }
inline long long clock64() { return 0; }
template<typename T> inline T __ldg(const T* p) { return *p; }

// AMD GCN builtins
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 0
#define __HIP_MEMORY_SCOPE_WORKGROUP 1
#define __HIP_MEMORY_SCOPE_AGENT 2
#define __HIP_MEMORY_SCOPE_SYSTEM 3
template<typename T> inline void __hip_atomic_store(T*, T, int, int) {}
template<typename T> inline T __hip_atomic_load(const T* p, int, int) { return *p; }
template<typename T> inline T __hip_atomic_exchange(T* p, T v, int, int) { T old = *p; *p = v; return old; }
template<typename T> inline T __hip_atomic_compare_exchange_strong(T* p, T e, T d, int, int, int) { T old = *p; if(old==e) *p=d; return old; }
template<typename T> inline T __hip_atomic_fetch_add(T* p, T v, int, int) { T old = *p; *p += v; return old; }
inline void __builtin_amdgcn_fence(int, const char*) {}
inline void __builtin_amdgcn_s_sleep(int) {}
template<typename T> inline void __builtin_nontemporal_store(T, T*) {}
template<typename T> inline T __builtin_nontemporal_load(const T*) { return T{}; }

// Thread/block indices (device-side, never used in host code)
struct { int x, y, z; } static constexpr threadIdx = {0,0,0};
struct { int x, y, z; } static constexpr blockDim = {1,1,1};
struct { int x, y, z; } static constexpr blockIdx = {0,0,0};
struct { int x, y, z; } static constexpr gridDim = {1,1,1};

// __shfl_sync variants defined by nccl_device/hip_compat.h — not duplicated here

// Additional CUDA aliases needed by RCCL headers
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventSynchronize(e) hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaHostAllocDefault 0

// Resource buffer pointers (device-side stubs)
template<typename T> inline T* ncclGetResourceBufferMultimemPointer(void*, int) { return nullptr; }
template<typename T> inline T* ncclGetResourceBufferLocalPointer(void*, int) { return nullptr; }
template<typename T> inline T* ncclGetResourceBufferPeerPointer(void*, int, int) { return nullptr; }

// Device property struct
struct cudaDeviceProp_stub {
  char name[256]; char gcnArchName[256]; int major; int minor;
  size_t totalGlobalMem; size_t sharedMemPerBlock; int warpSize;
  int multiProcessorCount; int maxThreadsPerBlock;
};
#define cudaDeviceProp cudaDeviceProp_stub
#define cudaGetDeviceProperties(prop, dev) hipGetDeviceProperties((hipDeviceProp_t*)(prop), dev)

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
#define cudaStreamLegacy hipStreamLegacy

// Batch mem op params (opaque struct used in function pointer typedefs)
struct hipStreamBatchMemOpParams { uint64_t padding[8]; };
using CUstreamBatchMemOpParams = hipStreamBatchMemOpParams;
using CUmemAllocationHandleType = hipMemAllocationHandleType;

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
inline hipError_t hipHostMalloc(void** p, size_t s, unsigned f=0) { *p = nullptr; return hipSuccess; }
inline hipError_t hipHostFree(void*) { return hipSuccess; }
inline hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned) { return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
inline hipError_t hipEventCreate(hipEvent_t*) { return hipSuccess; }
inline hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipSuccess; }

// HIP device API stubs
struct hipDeviceArch_t { unsigned data; };
inline hipError_t hipDeviceGetAttribute(int* v, int, int) { *v = 0; return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int) { return hipSuccess; }
inline hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipSuccess; }
inline hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t) { return hipSuccess; }
inline hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return hipSuccess; }
inline hipError_t hipEventQuery(hipEvent_t) { return hipSuccess; }
inline hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned) { return hipSuccess; }

// Device memory allocation flags
enum { hipDeviceMallocDefault = 0, hipDeviceMallocFinegrained = 1 };
enum { hipHostMallocMapped = 0, hipHostMallocNonCoherent = 0 };
enum { hipDeviceAttributeDirectManagedMemAccessFromHost = 0 };
enum { hipStreamNonBlocking = 1 };
using hipStreamCaptureMode = int;
inline hipError_t hipExtMallocWithFlags(void** p, size_t, unsigned) { *p = nullptr; return hipSuccess; }
inline hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode*) { return hipSuccess; }

// HIP compatibility aliases (CUDA-style names used by RCCL via hipify)
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
static constexpr int cudaSuccess = hipSuccess;
static constexpr int cudaErrorStreamCaptureInvalidated = hipErrorStreamCaptureInvalidated;
inline const char* cudaGetErrorString(cudaError_t e) { return hipGetErrorString(e); }
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaFree hipFree
inline hipError_t cudaFreeHost(void*) { return hipSuccess; }
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFreeHost hipHostFree
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaHostAlloc hipHostMalloc
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrDirectManagedMemAccessFromHost hipDeviceAttributeDirectManagedMemAccessFromHost
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaStreamCaptureModeRelaxed 0
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaHostGetDevicePointer(a,b,c) hipSuccess
#define cudaIpcGetMemHandle(a,b) hipSuccess
#define cudaIpcOpenMemHandle(a,b,c) hipSuccess
#define cudaIpcCloseMemHandle(a) hipSuccess
#define CUDART_VERSION 0

// Additional types needed by real RCCL headers
using cudaMemPool_t = void*;
using cudaHostFn_t = void(*)(void*);
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
struct cudaIpcMemHandle_t { char reserved[64]; };
inline hipError_t cudaStreamGetId(hipStream_t, unsigned long long* id) { *id = 0; return hipSuccess; }

#ifndef hipMemAllocationTypeUncached
#define hipMemAllocationTypeUncached 0
#endif
#ifndef hipDeviceMallocUncached
#define hipDeviceMallocUncached 0
#endif
#ifndef hipHostMallocUncached
#define hipHostMallocUncached 0
#endif
#ifndef HIP_UNCACHED_MEMORY
// don't define — let the non-uncached path be taken
#endif
