/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "bitops.h"
#include "utils.h"
#include "p2p.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include "rccl_vars.h"
#include <atomic>
#include <mutex>

#if CUDART_VERSION >= 11030
#include <cuda.h>
#include "cudawrap.h"
#endif

// Global flag to detect process shutdown. Set by atexit handler before
// HIP runtime static destructors run. This prevents use-after-free crashes
// when RCCL proxy threads try to free GPU memory during process exit.
inline std::atomic<bool>& rcclShutdownFlag() {
  static std::atomic<bool> flag{false};
  return flag;
}

inline void rcclShutdownHandler() {
  rcclShutdownFlag().store(true, std::memory_order_release);
}

inline void rcclRegisterShutdownHandler() {
  static std::once_flag once;
  std::call_once(once, []() {
    atexit(rcclShutdownHandler);
  });
}

uint64_t clockNano(); // from utils.h with which we have a circular dependency

template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

struct ncclSideStream {
  cudaStream_t stream;
  uint64_t refCount;
};

inline std::unordered_map<int64_t, ncclSideStream> sideStream;
inline pthread_mutex_t sideStreamLock = PTHREAD_MUTEX_INITIALIZER;
extern ncclResult_t getBusId(int cudaDev, int64_t *busId);

static inline ncclResult_t ncclCreateSideStream(int cudaDev) {
  ncclResult_t res = ncclSuccess;
  int64_t busId;
  NCCLCHECK(getBusId(cudaDev, &busId));
  pthread_mutex_lock(&sideStreamLock);
  if (auto it = sideStream.find(busId); it != sideStream.end()) {
    it->second.refCount++;
    INFO(NCCL_ALLOC, "Side stream %p of dev %d busid %lx inc count to %ld",
      it->second.stream, cudaDev, busId, it->second.refCount);
  } else {
    cudaStream_t stream;
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), res, fail);
    sideStream.emplace(busId, ncclSideStream{stream, 1});
    INFO(NCCL_ALLOC, "Created side stream %p of dev %d busid %lx",
      stream, cudaDev, busId);
  }
fail:
  pthread_mutex_unlock(&sideStreamLock);
  return res;
};

static inline ncclResult_t ncclDestroySideStream(int cudaDev) {
  ncclResult_t res = ncclSuccess;
  int64_t busId;
  NCCLCHECK(getBusId(cudaDev, &busId));
  pthread_mutex_lock(&sideStreamLock);
  if (auto it = sideStream.find(busId); it != sideStream.end()) {
    it->second.refCount--;
    if (it->second.refCount== 0) {
      INFO(NCCL_ALLOC, "Destroyed side stream %p of dev %d busid %lx",
        it->second.stream, cudaDev, busId);
      CUDACHECKGOTO(cudaStreamDestroy(it->second.stream), res, fail);
      sideStream.erase(it);
    } else {
      INFO(NCCL_ALLOC, "Side stream %p of dev %d busid %lx dec count to %ld",
        it->second.stream, cudaDev, busId, it->second.refCount);
    }
  } else {
    WARN("Side stream of dev %d busid %lx was not found for destroy", cudaDev, busId);
  }
fail:
  pthread_mutex_unlock(&sideStreamLock);
  return res;
};

static inline ncclResult_t getSideStream(cudaStream_t *stream) {
  int cudaDev;
  int64_t busId;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(getBusId(cudaDev, &busId));
  pthread_mutex_lock(&sideStreamLock);
  if (auto it = sideStream.find(busId); it != sideStream.end()) {
    *stream = it->second.stream;
    INFO(NCCL_ALLOC, "Found side stream %p of dev %d busid %lx count %ld",
      it->second.stream, cudaDev, busId, it->second.refCount);
  } else {
    *stream = 0;
    WARN("Side stream of dev %d busid %lx was not found", cudaDev, busId);
  }
  pthread_mutex_unlock(&sideStreamLock);
  return ncclSuccess;
}

#if CUDART_VERSION >= 12020

static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int cpuNumaNodeId = -1;
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  prop.location.id = cpuNumaNodeId;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  if (handlep) *handlep = handle;
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else /* CUDART_VERSION >= 12020 */

static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, void* handlep, size_t size) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

#endif  /* CUDART_VERSION >= 12020 */

template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  int managed = 0;
  CUDACHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
  if (nelem > 0) {
    if (managed) {
#if defined(HIP_UNCACHED_MEMORY)
      CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), hipDeviceMallocUncached), result, finish);
#else
      CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), hipDeviceMallocFinegrained), result, finish);
#endif
    } else
#if defined(HIP_HOST_UNCACHED_MEMORY)
      CUDACHECKGOTO(hipHostMalloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped | hipHostMallocUncached), result, finish);
#else
      CUDACHECKGOTO(hipHostMalloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
#endif
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess;
  // Check if process is shutting down to avoid use-after-free in HIP runtime
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCudaHostFree: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  if (nelem > 0) {
    T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
      return ncclSystemError;
    }
    //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), p);
    memset(p, 0, nelem*ncclSizeOfT<T>());
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  T* oldp = *ptr;
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
    return ncclSystemError;
  }
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * ncclSizeOfT<T>());
  if (oldp) free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*ncclSizeOfT<T>());
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*ncclSizeOfT<T>(), nelem*ncclSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

struct __attribute__ ((aligned(64))) allocationTracker {
  union {
    struct {
      uint64_t totalAlloc;
      uint64_t totalAllocSize;
    };
    char align[64];
  };
};
static_assert(sizeof(struct allocationTracker) == 64, "allocationTracker must be size of 64 bytes");
#define MAX_ALLOC_TRACK_NGPU 128
extern struct allocationTracker allocTracker[];

#if ROCM_VERSION >= 70000

#include "rocmwrap.h"

// ncclCuMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, *handleIn, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  if (cudaDev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[cudaDev].totalAlloc, 1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[cudaDev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemAllocAddr: Memory used = %ld on device = %d", allocTracker[cudaDev].totalAllocSize, cudaDev);
  return result;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  // Check if process is shutting down to avoid use-after-free in HIP runtime
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCuMemFreeAddr: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }
  ncclResult_t result = ncclSuccess;
  size_t size = 0;
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));

  int dev;
  size *= -1;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[dev].totalAlloc, -1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[dev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemFreeAddr: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  return result;
}

static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;
  // Query device to see if RDMA support is available
  // CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  if (handlep) *handlep = handle;
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
  
  if (cudaDev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[cudaDev].totalAlloc, 1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[cudaDev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemAlloc: Memory used = %ld on device = %d", allocTracker[cudaDev].totalAllocSize, cudaDev);

  return result;
}

static inline ncclResult_t ncclCuMemFree(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  // Check if process is shutting down to avoid use-after-free in HIP runtime
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCuMemFree: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));

  int dev;
  size *= -1;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[dev].totalAlloc, -1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[dev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemFree: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  return result;
}

#else

extern int ncclCuMemEnable();

static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, int type, size_t size) {
  WARN("CUMEM not supported prior to ROCm 7.0");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  WARN("CUMEM not supported prior to ROCm 7.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  WARN("CUMEM not supported prior to ROCm 7.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  WARN("CUMEM not supported prior to ROCm 7.0");
  return ncclInternalError;
}
#endif

template <typename T>
ncclResult_t ncclCudaMallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) 
    CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  else {
     int dev;
     CUDACHECK(hipGetDevice(&dev));
     if (dev < MAX_ALLOC_TRACK_NGPU) {
        __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*ncclSizeOfT<T>(), __ATOMIC_RELAXED);
     }
     INFO(NCCL_ALLOC, "ncclCudaMallocDebug: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  }
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p flags %d", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr, flags);
  return result;
}
#define ncclCudaMalloc(...) ncclCudaMallocDebug( __FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  int dev;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream, sidestream;
  NCCLCHECK(getSideStream(&sidestream));
  stream = sidestream;
  if (sidestream == nullptr)
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
  CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  if (sidestream == nullptr)
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc %ld bytes", nelem*ncclSizeOfT<T>());
  else {
      CUDACHECK(hipGetDevice(&dev));
      if (dev < MAX_ALLOC_TRACK_NGPU) {
    	 __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_RELAXED);
    	 __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*ncclSizeOfT<T>(), __ATOMIC_RELAXED);
      }
      INFO(NCCL_ALLOC, "ncclCudaCallocDebug: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  }
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p flags %d", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr, flags);
  return result;
}
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(const char *filefunc, int line, T** ptr, size_t nelem, hipStream_t stream, unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  int dev;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish); 
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc async %ld bytes", nelem*ncclSizeOfT<T>());
  else {
     CUDACHECK(hipGetDevice(&dev));
     if (dev < MAX_ALLOC_TRACK_NGPU) {
       __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_RELAXED);
       __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*ncclSizeOfT<T>(), __ATOMIC_RELAXED);
     }
     INFO(NCCL_ALLOC, "ncclCudaCallocDebug: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  }
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p flags %d", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr, flags);
  return result;
}
#define ncclCudaCallocAsync(...) ncclCudaCallocAsyncDebug(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream, sidestream;
  NCCLCHECK(getSideStream(&sidestream));
  stream = sidestream;
  if (sidestream == nullptr)
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  if (sidestream == nullptr)
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem*ncclSizeOfT<T>(), cudaMemcpyDefault, stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  if (ptr == NULL) return ncclSuccess;

  // Check if process is shutting down. The atexit handler sets this flag
  // BEFORE HIP runtime static destructors run, so we can safely skip the free.
  // The OS will reclaim all memory when the process exits anyway.
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCudaFree: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }

  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);

  // get the size of the allocation for tracking
  {
     CUdeviceptr baseAddress;
     size_t retrievedSize;

     CUDACHECK(cuMemGetAddressRange(&baseAddress, &retrievedSize, ptr));
     retrievedSize *= -1;

     if (ptr == baseAddress) {
        int dev;
        CUDACHECK(hipGetDevice(&dev));
        if (dev < MAX_ALLOC_TRACK_NGPU) {
           __atomic_fetch_add(&allocTracker[dev].totalAlloc, -1, __ATOMIC_RELAXED);
           __atomic_fetch_add(&allocTracker[dev].totalAllocSize, retrievedSize, __ATOMIC_RELAXED);
        }
        INFO(NCCL_ALLOC, "ncclCudaFree: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
     }
  }

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  if (size > 0) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) return ncclSystemError;
    void* p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ncclSystemError;
    memset(p, 0, size);
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
