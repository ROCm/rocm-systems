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
#include "mem_manager.h"
#include <sys/mman.h>
struct ncclComm;
#include "os.h"
#include <memory>
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

#if ROCM_VERSION >= 71200
#include <hip/hip_runtime.h>
#include "rocmwrap.h"
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

// C++14-compatible wrapper that captures function pointers through template parameters.
template <typename FunctionPtr, FunctionPtr Function>
struct ncclDeleterWrapper {
  template <typename... Args>
  constexpr auto operator()(Args &&...args) const { return Function(std::forward<Args>(args)...); }
}; // struct ncclDeleterWrapper

using ncclDeleterFree = ncclDeleterWrapper<decltype(&std::free), std::free>;
template <typename T>
using ncclUniquePtr = std::unique_ptr<T, ncclDeleterFree>;
template <typename T>
using ncclUniqueArrayPtr = std::unique_ptr<T[], ncclDeleterFree>;

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

#if CUDART_VERSION >= 12020 || ROCM_VERSION >= 71200

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
  bool handleCreated = false;
  bool addressReserved = false;
  bool mapped = false;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
#if defined(__HIP_PLATFORM_AMD__)
  // CLR rejects HostNuma; only Device or Host are accepted.
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  // HIP/CLR requires host id to be 0. cpuNumaNodeId can exceed GPU count and fail.
  prop.location.id = 0;             // ignored on the Host path
#else
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  prop.location.id = cpuNumaNodeId;
#endif
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECKGOTO(cuMemCreate(&handle, size, &prop, 0), result, fail);
  handleCreated = true;
  /* Reserve a virtual address range */
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0), result, fail);
  addressReserved = true;
  /* Map the virtual address range to the physical allocation */
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0), result, fail);
  mapped = true;
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1), result, fail);

  /* Now allow RW access to the newly mapped memory from the CPU */
#if defined(__HIP_PLATFORM_AMD__)
  // CLR rejects HostNuma here too; mirror the Host fallback used at allocation.
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST;
  accessDesc.location.id = 0;
#else
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
#endif
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1), result, fail);

  if (handlep) *handlep = handle;
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %p numa %d dev %d granularity %ld", size, *ptr, (void*)(uintptr_t)handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
fail:
  WARN("ncclCuMemHostAlloc failed (size %zu, dev %d): cleaning up partial allocation", size, cudaDev);
  if (mapped) (void)cuMemUnmap((CUdeviceptr)*ptr, size);
  if (addressReserved) (void)cuMemAddressFree((CUdeviceptr)*ptr, size);
  if (handleCreated) (void)cuMemRelease(handle);
  *ptr = nullptr;
  return result;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  // ROCM-2696: Proper initialization of base and size is required for cuMemGetAddressRange
  // base is dereferenced in cuMemGetAddressRange without checking for nullptr
  CUdeviceptr base = nullptr;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(&base, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle %p", size, ptr, (void*)(uintptr_t)handle);
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
    memset((void*)p, 0, nelem*ncclSizeOfT<T>());
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclCallocDebug(ncclUniquePtr<T>& ptr, size_t nelem, const char *filefunc, int line) {
  typename ncclUniquePtr<T>::pointer p = nullptr;
  ncclResult_t result = ncclCallocDebug(&p, nelem, filefunc, line);
  ptr.reset(p);
  return result;
}

template <typename T>
ncclResult_t ncclCallocDebug(ncclUniqueArrayPtr<T>& ptr, size_t nelem, const char *filefunc, int line) {
  typename ncclUniqueArrayPtr<T>::pointer p = nullptr;
  ncclResult_t result = ncclCallocDebug(&p, nelem, filefunc, line);
  ptr.reset(p);
  return result;
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

// [RCCL] Helper introduced upstream in NCCL 2.29.7 -- maps a virtual address
// range to a physical allocation and grants RW access on the given device.
// Used by mem_manager.cc and the per-allocator helpers below.
static inline ncclResult_t ncclCuMemMapAndSetAccess(void *ptr, size_t size,
  CUmemGenericAllocationHandle handle,
  int cudaDev) {
  ncclResult_t result = ncclSuccess;
  CUCHECK(cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1));
  return result;
}

// ncclCuMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  int cudaDev;
  bool addressReserved = false;
  bool mapped = false;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Reserve a virtual address range */
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0), result, fail);
  addressReserved = true;
  /* Map the virtual address range to the physical allocation */
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ptr, size, 0, *handleIn, 0), result, fail);
  mapped = true;
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1), result, fail);
  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %p", size, *ptr, (void*)(uintptr_t)*handleIn);
  if (cudaDev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[cudaDev].totalAlloc, 1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[cudaDev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemAllocAddr: Memory used = %ld on device = %d", allocTracker[cudaDev].totalAllocSize, cudaDev);
  return result;
fail:
  WARN("ncclCuMemAllocAddr failed (size %zu, dev %d): cleaning up partial allocation", size, cudaDev);
  if (mapped) (void)cuMemUnmap((CUdeviceptr)*ptr, size);
  if (addressReserved) (void)cuMemAddressFree((CUdeviceptr)*ptr, size);
  *ptr = nullptr;
  return result;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr, struct ncclMemManager* manager, int numSegments = 1) {
  if (ptr == NULL) return ncclSuccess;
  // Check if process is shutting down to avoid use-after-free in HIP runtime
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCuMemFreeAddr: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }

  // RCCL: Skip if Suspend already unmapped this VA. The reservation is kept by Suspend and
  // will be released by ncclMemManagerDestroy walking the entry list.
  if (ncclMemEntryAlreadyReleased(manager, ptr)) {
    INFO(NCCL_ALLOC, "ncclCuMemFreeAddr: %p already released by Suspend", ptr);
    return ncclSuccess;
  }

  ncclResult_t result = ncclSuccess;
  size_t totalSize = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    size_t segmentSize = 0;
    // ROCM-2696: Proper initialization of base and size is required for cuMemGetAddressRange
    // base is dereferenced in cuMemGetAddressRange without checking for nullptr
    CUdeviceptr base = nullptr;
    // RCCL: cast through char* before pointer arithmetic
    CUCHECK(cuMemGetAddressRange(&base, &segmentSize, (CUdeviceptr)((char*)ptr + totalSize)));
    CUCHECK(cuMemUnmap((CUdeviceptr)((char*)ptr + totalSize), segmentSize));
    totalSize += segmentSize;
  }

  // Untrack from memory manager
  if (manager != nullptr) {
    NCCLCHECK(ncclMemUntrack(manager, ptr, totalSize));
  }

  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, totalSize));

  int dev;
  size_t trackSize = totalSize;
  trackSize *= -1;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[dev].totalAlloc, -1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[dev].totalAllocSize, trackSize, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemFreeAddr: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  return result;
}

static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep,
                                          CUmemAllocationHandleType type, size_t size,
                                          struct ncclMemManager* manager = nullptr,
                                          ncclMemType_t memType = ncclMemPersist) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  bool handleCreated = false;
  bool addressReserved = false;
  bool mapped = false;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
#if defined(HIP_VMM_UNCACHED_MEMORY)
  prop.type = hipMemAllocationTypeUncached;
#else
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
#endif
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;
#if defined(__HIP_PLATFORM_AMD__)
  // ROCM-2550: Use cuDeviceGetAttribute to check if RDMA support is available
  // TODO: Remove once ROCM-2550 is fixed and uncomment the commented code below.
  // Always enable gpuDirectRDMACapable: the non-RDMA VMM code path in
  // HIP crashes (SIGSEGV in hipMemMap) after many allocations.
  flag = 1;
  prop.allocFlags.gpuDirectRDMACapable = flag;
  // // Query device to see if RDMA support is available
  // CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  // if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
#endif
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECKGOTO(cuMemCreate(&handle, size, &prop, 0), result, fail);
  handleCreated = true;
  /* Reserve a virtual address range */
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0), result, fail);
  addressReserved = true;
  /* Map the virtual address range to the physical allocation */
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0), result, fail);
  mapped = true;
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1), result, fail);
  if (handlep) *handlep = handle;
  // ROCM-20370 Workaround: hsa_amd_vmem_map writes internal bookkeeping metadata
  // into the user-visible buffer after the kernel driver's SDMA clear,
  // leaving non-zero residue at specific offsets. Zero it now so that
  // structures like ncclSendMem/ncclRecvMem (head, tail, ptrExchange,
  // redOpArgExchange) start at zero. Use a non-blocking stream with
  // relaxed capture mode so this is safe when called from the proxy
  // thread during graph capture on the main thread.
  {
    cudaStreamCaptureMode capMode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&capMode), result, fail);
    cudaStream_t zeroStream;
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking), result, restoreCapMode);
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, size, zeroStream), result, destroyStream);
    CUDACHECKGOTO(cudaStreamSynchronize(zeroStream), result, destroyStream);
destroyStream:
    CUDACHECK(cudaStreamDestroy(zeroStream));
restoreCapMode:
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&capMode));
    if (result != ncclSuccess) goto fail;
  }
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %p", size, *ptr, (void*)(uintptr_t)handle);

  /* Track allocation in memory manager */
  if (manager != nullptr) {
    NCCLCHECKGOTO(ncclMemTrack(manager, *ptr, size, handle, type, memType), result, fail);
  }

  if (cudaDev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[cudaDev].totalAlloc, 1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[cudaDev].totalAllocSize, size, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemAlloc: Memory used = %ld on device = %d", allocTracker[cudaDev].totalAllocSize, cudaDev);
  return result;
fail:
  WARN("ncclCuMemAlloc failed (size %zu, dev %d): cleaning up partial allocation", size, cudaDev);
  if (mapped) (void)cuMemUnmap((CUdeviceptr)*ptr, size);
  if (addressReserved) (void)cuMemAddressFree((CUdeviceptr)*ptr, size);
  if (handleCreated) (void)cuMemRelease(handle);
  *ptr = nullptr;
  return result;
}

static inline ncclResult_t ncclCuMemFree(void *ptr, struct ncclMemManager* manager, int numSegments = 1) {
  if (ptr == NULL) return ncclSuccess;
  // Check if process is shutting down to avoid use-after-free in HIP runtime
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCuMemFree: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }

  // RCCL: skip only tracked entries already torn down by Suspend; persistent
  // and other untracked pointers must still be freed here.
  if (ncclMemEntryAlreadyReleased(manager, ptr)) {
    INFO(NCCL_ALLOC, "ncclCuMemFree: %p already released by Suspend", ptr);
    return ncclSuccess;
  }

  ncclResult_t result = ncclSuccess;
  size_t totalSize = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    CUmemGenericAllocationHandle handle;
    size_t segmentSize = 0;
    CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)((char*)ptr + totalSize)));
    CUCHECK(cuMemRelease(handle));
    // ROCM-2696: Proper initialization of base and size is required for cuMemGetAddressRange
    // base is dereferenced in cuMemGetAddressRange without checking for nullptr
    CUdeviceptr base = nullptr;
    // RCCL: cast through char* before pointer arithmetic 
    CUCHECK(cuMemGetAddressRange(&base, &segmentSize, (CUdeviceptr)((char*)ptr + totalSize)));
    TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle %p segment %d numSegments %d",
          segmentSize, ptr, (void*)(uintptr_t)handle, segment, numSegments);
    CUCHECK(cuMemUnmap((CUdeviceptr)((char*)ptr + totalSize), segmentSize));
    CUCHECK(cuMemRelease(handle));
    totalSize += segmentSize;
  }

  // Update tracking with total size after processing all segments
  if (manager != nullptr) {
    NCCLCHECK(ncclMemUntrack(manager, ptr, totalSize));
  }

  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, totalSize));

  int dev;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
     __atomic_fetch_add(&allocTracker[dev].totalAlloc, -1, __ATOMIC_RELAXED);
     __atomic_fetch_add(&allocTracker[dev].totalAllocSize, -(int64_t)totalSize, __ATOMIC_RELAXED);
  }
  INFO(NCCL_ALLOC, "ncclCuMemFree: Memory used = %ld on device = %d", allocTracker[dev].totalAllocSize, dev);
  return result;
}

// Get the base and size of all segments that span a given user buffer
static inline ncclResult_t ncclCuMemGetAddressRange(CUdeviceptr userBuff, size_t userBuffSize, CUdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize, int* numSegments) {
  *totalMappedBufferSize = 0;
  *mappedPtrBase = 0;
  if (numSegments) *numSegments = 0;
  CUdeviceptr userBuffStart = userBuff;
  CUdeviceptr userBuffEnd = (CUdeviceptr)((char*)userBuffStart + userBuffSize);
  CUdeviceptr mappedPtrEnd = userBuffStart;
  CUdeviceptr baseSend;
  size_t baseSendSize;

  while ((char*)mappedPtrEnd < (char*)userBuffEnd) {
    CUCHECK(cuMemGetAddressRange(&baseSend, &baseSendSize, mappedPtrEnd));

    if (*totalMappedBufferSize == 0) {
      *mappedPtrBase = baseSend;
    }
    *totalMappedBufferSize += baseSendSize;
    mappedPtrEnd = (CUdeviceptr)((char*)baseSend + baseSendSize);

    if (numSegments) *numSegments = *numSegments + 1;
  }
  return ncclSuccess;
}

#else

extern int ncclCuMemEnable();

static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, int type, size_t size,
                                          struct ncclMemManager* manager,
                                          ncclMemType_t memType = ncclMemPersist) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void *ptr, struct ncclMemManager* manager, int numSegments = 1) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr, struct ncclMemManager* manager, int numSegments = 1) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemGetAddressRange(CUdeviceptr userBuff, size_t userBuffSize, CUdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize, int* numSegments) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemMapAndSetAccess(void *ptr, size_t size,
  CUmemGenericAllocationHandle handle, int cudaDev) {
  WARN("CUMEM requires ROCM_VERSION >= 7.0.0");
  return ncclInternalError;
}
#endif

template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line,
                                 struct ncclMemManager* manager,
                                 ncclMemType_t memType = ncclMemPersist,
                                 unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>(), manager, memType), result, finish);
    } else {
      CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
    }
  }
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
#define ncclCudaMalloc(ptr, nelem, ...) ncclCudaMallocDebug(ptr, nelem, __FILE__, __LINE__, ##__VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line,
                                 struct ncclMemManager* manager,
                                 ncclMemType_t memType = ncclMemPersist,
                                 unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  int dev;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    // Need a side stream so as not to interfere with graph capture.
    cudaStream_t stream, sidestream;
    NCCLCHECK(getSideStream(&sidestream));
    stream = sidestream;
    if (sidestream == nullptr)
      CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>(), manager, memType), result, finish);
    } else {
      CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
    }
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    if (sidestream == nullptr)
      CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }
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
#define ncclCudaCalloc(ptr, nelem, ...) ncclCudaCallocDebug(ptr, nelem, __FILE__, __LINE__, ##__VA_ARGS__)

// [RCCL] Upstream NCCL 2.29 added a `struct ncclMemManager*` parameter to
// the *Debug helpers (and a defaulted ncclMemType_t). RCCL's variants here
// keep the original `flags` overload (used heavily across the codebase) and
// add a manager/memType overload that simply ignores both values: the
// manager-driven tracking lives in mem_manager.cc and isn't wired through
// the HIP allocator path yet. This way new upstream call sites compile
// without forcing every old AMD call site to change.
template <typename T>
ncclResult_t ncclCudaMallocDebug(const char *filefunc, int line, T** ptr, size_t nelem,
                                 struct ncclMemManager* /*manager*/,
                                 ncclMemType_t /*memType*/ = ncclMemPersist) {
  return ncclCudaMallocDebug(filefunc, line, ptr, nelem);
}

template <typename T>
ncclResult_t ncclCudaCallocDebug(const char *filefunc, int line, T** ptr, size_t nelem,
                                 struct ncclMemManager* /*manager*/,
                                 ncclMemType_t /*memType*/ = ncclMemPersist) {
  return ncclCudaCallocDebug(filefunc, line, ptr, nelem);
}

template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(T** ptr, size_t nelem, hipStream_t stream, const char *filefunc, int line,
                                      struct ncclMemManager* manager,
                                      ncclMemType_t memType = ncclMemPersist,
                                      unsigned int flags = hipDeviceMallocDefault) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  int dev;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>(), manager, memType), result, finish);
    } else {
      CUDACHECKGOTO(hipExtMallocWithFlags((void**)ptr, nelem*ncclSizeOfT<T>(), flags), result, finish);
    }
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
#define ncclCudaCallocAsync(ptr, nelem, stream, ...) ncclCudaCallocAsyncDebug(ptr, nelem, stream, __FILE__, __LINE__, ##__VA_ARGS__)

// [RCCL] Manager/memType overload for ncclCudaCallocAsyncDebug; see the note
// above ncclCudaMallocDebug for rationale.
template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(const char *filefunc, int line, T** ptr, size_t nelem,
                                      hipStream_t stream,
                                      struct ncclMemManager* /*manager*/,
                                      ncclMemType_t /*memType*/ = ncclMemPersist) {
  return ncclCudaCallocAsyncDebug(filefunc, line, ptr, nelem, stream);
}

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
ncclResult_t ncclCudaMemset(T* dst, int value, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  CUDACHECKGOTO(cudaMemsetAsync((void*)dst, value, nelem * ncclSizeOfT<T>(), stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
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
ncclResult_t ncclCudaFree(T* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  if (ptr == NULL) return ncclSuccess;

  // Check if process is shutting down. The atexit handler sets this flag
  // BEFORE HIP runtime static destructors run, so we can safely skip the free.
  // The OS will reclaim all memory when the process exits anyway.
  if (rcclShutdownFlag().load(std::memory_order_acquire)) {
    INFO(NCCL_ALLOC, "ncclCudaFree: Skipping free (process shutdown) pointer %p", ptr);
    return ncclSuccess;
  }

  // RCCL: skip only tracked entries already torn down by Suspend; persistent
  // and other untracked pointers must still be freed here.
  if (ncclMemEntryAlreadyReleased(manager, (void*)ptr)) {
    INFO(NCCL_ALLOC, "ncclCudaFree: %p already released by Suspend", (void*)ptr);
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
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr, manager, numSegments), result, finish);
  } else {
    if (numSegments > 1) {
      result = ncclUnhandledCudaError;
      goto finish;
    } else {
      CUDACHECKGOTO(cudaFree(ptr), result, finish);
    }
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
    long page_size = ncclOsGetPageSize();
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
