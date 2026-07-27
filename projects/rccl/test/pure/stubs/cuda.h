// Stub cuda.h — CUDA driver API types for CPU-only RCCL unit tests.
#pragma once
#include <hip/hip_runtime.h>

// Basic driver types
using CUresult = int;

// CUdeviceptr must be comparable with pointers (alloc.h compares CUdeviceptr with void*)
struct CUdeviceptr {
  uintptr_t val;
  CUdeviceptr() : val(0) {}
  CUdeviceptr(uintptr_t v) : val(v) {}
  operator uintptr_t() const { return val; }
  template<typename T> bool operator==(T* p) const { return val == (uintptr_t)p; }
  template<typename T> bool operator!=(T* p) const { return val != (uintptr_t)p; }
  friend bool operator==(void* p, const CUdeviceptr& d) { return (uintptr_t)p == d.val; }
  friend bool operator!=(void* p, const CUdeviceptr& d) { return (uintptr_t)p != d.val; }
};
using CUcontext = void*;
using CUdevice = int;
using CUmemGenericAllocationHandle = uint64_t;
// CUstreamBatchMemOpParams and CUmemAllocationHandleType defined in hip/hip_runtime.h
using CUfunction = void*;
using CUmodule = void*;
#define CUDA_SUCCESS 0

// Memory allocation structs
struct CUmemAllocationProp {
  int type;
  struct { int type; int id; } location;
  int requestedHandleTypes;
  struct { int gpuDirectRDMACapable; } allocFlags;
};
struct CUmemAccessDesc {
  struct { int type; int id; } location;
  int flags;
};
struct CUmulticastObjectProp { int dummy; };

// Enums
enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 300,
  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 301,
  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 302,
};
using CUdevice_attribute = int;

typedef int CUmemLocationType;
enum {
  CU_MEM_LOCATION_TYPE_DEVICE = 1,
  CU_MEM_LOCATION_TYPE_HOST = 2,
  CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
  CU_MEM_ALLOCATION_TYPE_PINNED = 1,
  CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
  CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
};

// Driver API function stubs (never called at runtime)
inline CUresult cuDeviceGet(CUdevice*, int) { return 0; }
inline CUresult cuDeviceGetAttribute(int* v, int, int) { if(v) *v = 0; return 0; }
inline CUresult cuMemGetAllocationGranularity(size_t* g, const CUmemAllocationProp*, int) { if(g) *g = 4096; return 0; }
inline CUresult cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) { return 0; }
inline CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) { return 0; }
inline CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) { return 0; }
inline CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) { return 0; }
inline CUresult cuMemUnmap(CUdeviceptr, size_t) { return 0; }
inline CUresult cuMemRelease(CUmemGenericAllocationHandle) { return 0; }
inline CUresult cuMemAddressFree(CUdeviceptr, size_t) { return 0; }
inline CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle*, void*) { return 0; }
inline CUresult cuMemGetAddressRange(CUdeviceptr*, size_t*, CUdeviceptr) { return 0; }
inline CUresult cuMemGetAddressRange(CUdeviceptr* b, size_t* s, void* p) { return 0; }
inline CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp*, CUmemGenericAllocationHandle) { return 0; }

// Compatibility aliases
#define cudaGetLastError hipGetLastError
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaStreamLegacy hipStreamLegacy
