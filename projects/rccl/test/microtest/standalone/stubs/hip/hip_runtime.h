// Declaration-based HIP stub for rccl-UnitTestsMicro (CPU-only / g++).
// Types + enums + device macros here; function BODIES live in the branch
// fakes/hip_fakes.cc (and micro_link_stubs.cc for leftovers). This header
// only DECLARES the fakes functions, so it never collides with those defs.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>   // std::min / std::max

// ---- device-code macros / intrinsics (host build: neutralize) ----
#define __device__
#define __host__
#define __global__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)
struct ulong2 { unsigned long x, y; };
struct ushort2 { unsigned short x, y; };
struct ulonglong2 { unsigned long long x, y; };
struct uint4 { unsigned int x, y, z, w; };
struct int4  { int x, y, z, w; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int _x=1,unsigned int _y=1,unsigned int _z=1):x(_x),y(_y),z(_z){} };
inline int __clz(unsigned int x){return x?__builtin_clz(x):32;}
inline int __popc(unsigned int x){return __builtin_popcount(x);}
inline int __popcll(unsigned long long x){return __builtin_popcountll(x);}
inline int __ffs(unsigned int x){return __builtin_ffs(x);}
inline void __syncthreads(){}
inline void __threadfence(){}
inline void __threadfence_system(){}
template<typename T> inline T __ldg(const T* p){return *p;}
#define __HIP_MEMORY_SCOPE_SYSTEM 3
#define __HIP_MEMORY_SCOPE_AGENT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 1
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 0
inline void __builtin_amdgcn_fence(int,const char*){}
inline void __builtin_amdgcn_s_sleep(int){}
#ifndef __clang__
// clang provides these as real builtins; only g++ needs the stubs.
template<typename T> inline void __builtin_nontemporal_store(T,T*){}
template<typename T> inline T __builtin_nontemporal_load(const T*){return T{};}
#endif
// thread/block indices (device-side; host build sees zeros)
struct { int x, y, z; } static constexpr threadIdx = {0,0,0};
struct { int x, y, z; } static constexpr blockDim  = {1,1,1};
struct { int x, y, z; } static constexpr blockIdx  = {0,0,0};
struct { int x, y, z; } static constexpr gridDim   = {1,1,1};
inline int __lane_id(){return 0;}
inline unsigned long long __ballot(int){return 0;}
inline int __activemask(){return 0;}
inline unsigned int __umulhi(unsigned int,unsigned int){return 0;}
inline unsigned long long __umul64hi(unsigned long long,unsigned long long){return 0;}
inline long long clock64(){return 0;}
inline int __shfl(int v,int,int=64){return v;}
inline unsigned int __shfl(unsigned int v,int,int=64){return v;}
inline unsigned long long __shfl(unsigned long long v,int,int=64){return v;}
inline float __shfl(float v,int,int=64){return v;}
inline int __shfl_sync(unsigned long long,int v,int,int=64){return v;}
#ifndef __clang__
// clang provides the HIP atomic builtins natively; only g++ needs the stubs.
template<typename T> inline void __hip_atomic_store(T* p,T v,int,int){*p=v;}
template<typename T> inline T __hip_atomic_load(const T* p,int,int){return *p;}
template<typename T> inline T __hip_atomic_exchange(T* p,T v,int,int){T o=*p;*p=v;return o;}
template<typename T> inline T __hip_atomic_compare_exchange_strong(T* p,T e,T d,int,int,int){T o=*p;if(o==e)*p=d;return o;}
template<typename T> inline T __hip_atomic_fetch_add(T* p,T v,int,int){T o=*p;*p+=v;return o;}
#endif
// unqualified min/max used inside RCCL device headers (host build)
template<class T> inline T min(T a, T b){ return a < b ? a : b; }
template<class T> inline T max(T a, T b){ return a > b ? a : b; }
// resource-buffer pointer helpers referenced by device headers
template<typename T> inline T* ncclGetResourceBufferMultimemPointer(void*, int){ return nullptr; }
template<typename T> inline T* ncclGetResourceBufferLocalPointer(void*, int){ return nullptr; }
template<typename T> inline T* ncclGetResourceBufferPeerPointer(void*, int, int){ return nullptr; }
template<typename T> inline T* ncclGetResourceBuffer(void*, int){ return nullptr; }

// ---- error enum ----
typedef enum {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorNotReady = 600,
  hipErrorStreamCaptureInvalidated = 901,
  hipErrorNotSupported = 801,
  hipErrorInvalidDevicePointer = 17,
  hipErrorPeerAccessAlreadyEnabled = 704,
} hipError_t;

// ---- opaque handle types ----
typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef int   hipDevice_t;
typedef void* hipDeviceptr_t;
typedef struct ihipMemGenericAllocationHandle* hipMemGenericAllocationHandle_t;
typedef int   hipMemAllocationHandleType;
typedef int   hipStreamCaptureMode;
typedef enum hipMemAllocationGranularity_flags {
  hipMemAllocationGranularityMinimum = 0,
  hipMemAllocationGranularityRecommended = 1,
} hipMemAllocationGranularity_flags;
typedef struct { char reserved[64]; } hipIpcMemHandle_t;
typedef struct { char reserved[64]; } hipIpcEventHandle_t;
typedef struct { char name[256]; char gcnArchName[256]; int major; int minor; } hipDeviceProp_t;
typedef struct { int type; int device; void* devicePointer; void* hostPointer; } hipPointerAttribute_t;
struct hipDeviceArch_t { unsigned data; };
typedef void* hipMemPool_t;
typedef int hipMemLocationType;
typedef void* hipCtx_t;
typedef void* hipGraph_t;
typedef void* hipGraphExec_t;
typedef void (*hipHostFn_t)(void*);
struct hipStreamBatchMemOpParams { uint64_t padding[8]; };
static hipStream_t const hipStreamLegacy = (hipStream_t)2;
hipError_t hipEventSynchronize(hipEvent_t);

// ---- enums used by p2p.cc / headers ----
typedef enum {
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 30,
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
} hipPointer_attribute;
typedef enum {
  hipDeviceAttributeHdpMemFlushCntl = 100,
  hipDeviceAttributeComputeCapabilityMajor = 23,
  hipDeviceAttributeComputeCapabilityMinor = 61,
  hipDeviceAttributeDirectManagedMemAccessFromHost = 200,
  hipDeviceAttributeVirtualMemoryManagementSupported = 201,
} hipDeviceAttribute_t;
typedef enum {
  hipMemcpyHostToHost=0, hipMemcpyHostToDevice=1, hipMemcpyDeviceToHost=2,
  hipMemcpyDeviceToDevice=3, hipMemcpyDefault=4,
} hipMemcpyKind;
enum { hipMemHandleTypeNone = 0, hipMemHandleTypePosixFileDescriptor = 1, hipMemHandleTypeWin32 = 2, hipMemHandleTypeWin32Kmt = 4 };
enum { hipMemAllocationTypePinned = 1 };
enum { hipMemLocationTypeDevice = 1 };
enum { hipMemAccessFlagsProtNone = 0, hipMemAccessFlagsProtRead = 1, hipMemAccessFlagsProtReadWrite = 3 };
enum { hipStreamNonBlocking = 1 };
enum { hipStreamCaptureModeGlobal = 0, hipStreamCaptureModeThreadLocal = 1, hipStreamCaptureModeRelaxed = 2 };
enum { hipDeviceMallocDefault = 0, hipDeviceMallocFinegrained = 1, hipDeviceMallocUncached = 3 };
enum { hipIpcMemLazyEnablePeerAccess = 1 };
enum { hipMemAllocationTypeUncached = 0 };
enum { hipHostMallocDefault = 0, hipHostMallocMapped = 0, hipHostMallocNonCoherent = 0 };

struct hipMemLocation { int type; int id; };
struct hipMemAllocationProp {
  int type; int requestedHandleTypes; hipMemLocation location;
  void* win32HandleMetaData; struct { unsigned char compressionType; unsigned char gpuDirectRDMACapable; unsigned short usage; } allocFlags;
};
struct hipMemAccessDesc { hipMemLocation location; int flags; };

// ---- functions DEFINED in fakes/hip_fakes.cc (declare only here) ----
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int, int);
hipError_t hipDeviceEnablePeerAccess(int, unsigned int);
hipError_t hipDeviceGet(hipDevice_t* device, int);
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t, int);
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int);
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipExtMallocWithFlags(void** ptr, size_t, unsigned int);
hipError_t hipFree(void*);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
const char* hipGetErrorString(hipError_t);
hipError_t hipGetLastError(void);
hipError_t hipHostFree(void*);
hipError_t hipHostMalloc(void** ptr, size_t, unsigned int flags = 0);
hipError_t hipIpcCloseMemHandle(void*);
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t, unsigned int);
hipError_t hipMemAddressFree(void*, size_t);
hipError_t hipMemAddressReserve(void** ptr, size_t, size_t, void*, unsigned long long);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t, const hipMemAllocationProp*, unsigned long long);
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp*, hipMemAllocationGranularity_flags);
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp*, hipMemGenericAllocationHandle_t);
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType);
hipError_t hipMemMap(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long);
hipError_t hipMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t);
hipError_t hipMemUnmap(void*, size_t);
hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t);
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int);
hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode*);
hipError_t hipMemExportToShareableHandle(void* shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags);
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);

// ---- functions NOT in hip_fakes.cc but referenced by RCCL headers: declare;
//      defined in micro_link_stubs.cc ----
hipError_t hipSetDevice(int);
hipError_t hipMalloc(void**, size_t);
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t hipMemset(void*, int, size_t);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipDriverGetVersion(int*);
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int);
hipError_t hipStreamCreate(hipStream_t*);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*);
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int);
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t*, hipEvent_t);

// ---- typed-pointer template forwarders ----
// Real HIP declares these allocation entry points as templates over T** so
// RCCL can pass typed pointers (e.g. ncclRecvMem**). g++ tolerated the
// implicit T**->void** with -fpermissive; clang (correctly) does not. These
// forwarders route typed calls to the void** overloads defined in the fakes.
template<class T> inline hipError_t hipHostMalloc(T** ptr, size_t s, unsigned int flags = 0) {
  return hipHostMalloc(reinterpret_cast<void**>(ptr), s, flags);
}
template<class T> inline hipError_t hipMalloc(T** ptr, size_t s) {
  return hipMalloc(reinterpret_cast<void**>(ptr), s);
}
template<class T> inline hipError_t hipExtMallocWithFlags(T** ptr, size_t s, unsigned int flags) {
  return hipExtMallocWithFlags(reinterpret_cast<void**>(ptr), s, flags);
}

// ---- CUDA-name compatibility (a few RCCL spots still use them) ----
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
static const hipError_t cudaSuccess = hipSuccess;
