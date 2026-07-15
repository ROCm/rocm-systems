/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <set>
#include "hip_internal.hpp"
#include "hip_vm.hpp"
namespace hip {

static_assert(static_cast<uint32_t>(hipMemAccessFlagsProtNone) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kNone),
              "Mem Access Flag None mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAccessFlagsProtRead) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kReadOnly),
              "Mem Access Flag Read mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAccessFlagsProtReadWrite) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kReadWrite),
              "Mem Access Flag Read Write mismatch with ROCclr!");

static_assert(static_cast<uint32_t>(hipMemLocationTypeDevice) ==
                  static_cast<uint32_t>(amd::Device::VmmLocationType::kDevice),
              "Mem Location Type Device mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemLocationTypeHost) ==
                  static_cast<uint32_t>(amd::Device::VmmLocationType::kHost),
              "Mem Location Type Host mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemLocationTypeHostNuma) ==
                  static_cast<uint32_t>(amd::Device::VmmLocationType::kHostNuma),
              "Mem Location Type HostNuma mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemLocationTypeHostNumaCurrent) ==
                  static_cast<uint32_t>(amd::Device::VmmLocationType::kHostNumaCurrent),
              "Mem Location Type HostNumaCurrent mismatch with ROCclr!");

int VmmOwnerDeviceIndex(const hipMemAllocationProp& prop) {
  // For host-backed allocations location.id is not a device index (it is a NUMA
  // node id for HostNuma, ignored otherwise). The VA-level map/unmap operations
  // are anchored on the current device in that case.
  if (prop.location.type == hipMemLocationTypeDevice) {
    return prop.location.id;
  }
  return static_cast<int>(hip::getCurrentDevice()->deviceId());
}

hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  HIP_INIT_API(hipMemAddressFree, devPtr, size);
  hipError_t status = hipSuccess;
  if (devPtr == nullptr || size == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Memory* memObj = amd::MemObjMap::FindVirtualMemObj(devPtr);
  if (memObj == nullptr) {
    LogPrintfError("Cannot find the Virtual MemObj entry for this addr %p", devPtr);
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Single call frees address range for all devices.
  if (!(g_devices[0]->devices()[0]->virtualFree(devPtr))) {
    status = hipErrorUnknown;
  }
  memObj->release();
  HIP_RETURN(status);
}

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags) {
  HIP_INIT_API(hipMemAddressReserve, ptr, size, alignment, addr, flags);

  if (ptr == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const auto& dev_info = g_devices[0]->devices()[0]->info();
  if (size == 0 || ((size % dev_info.virtualMemAllocGranularityMinimum_) != 0) ||
      ((alignment & (alignment - 1)) != 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Initialize the ptr, single virtual alloc call would reserve va range for all devices.
  *ptr = nullptr;
  *ptr = g_devices[0]->devices()[0]->virtualAlloc(addr, size, alignment);
  if (*ptr == nullptr) {
    HIP_RETURN(hipErrorOutOfMemory);
  }

  // If requested address was not allocated, printf error message.
  if (addr != nullptr && addr == *ptr) {
    LogPrintfError("Requested address was not allocated. Allocated address : %p ", *ptr);
  }

  HIP_RETURN(hipSuccess, ReturnPtrValue(ptr));
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags) {
  HIP_INIT_API(hipMemCreate, handle, size, prop, flags);

  //  Currently we do not support Pinned memory
  if (handle == nullptr || size == 0 || flags != 0 || prop == nullptr ||
      (prop->type != hipMemAllocationTypePinned && prop->type != hipMemAllocationTypeUncached) ||
      (prop->location.type != hipMemLocationTypeDevice &&
          prop->location.type != hipMemLocationTypeHost &&
          prop->location.type != hipMemLocationTypeHostNuma &&
          prop->location.type != hipMemLocationTypeHostNumaCurrent)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // location.id is interpreted per location type: a device index for Device, a
  // NUMA node id for HostNuma, and ignored for Host / HostNumaCurrent.
  if (prop->location.type == hipMemLocationTypeDevice) {
    if (prop->location.id < 0 || prop->location.id >= g_devices.size()) {
      HIP_RETURN(hipErrorInvalidDevice);
    }
  } else if (prop->location.type == hipMemLocationTypeHostNuma) {
    uint32_t numNumaNodes = hip::getCurrentDevice()->devices()[0]->numHostNumaNodes();
    if (prop->location.id < 0 || static_cast<uint32_t>(prop->location.id) >= numNumaNodes) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  if (prop->requestedHandleTypes != hipMemHandleTypeNone &&
      prop->requestedHandleTypes != hipMemHandleTypePosixFileDescriptor &&
      prop->requestedHandleType  != hipMemHandleTypeFabric) {
    HIP_RETURN(hipErrorNotSupported);
  }

  // When ROCCLR_MEM_PHYMEM is set, ROCr impl gets and stores unique hsa handle. Flag no-op on PAL.
  cl_svm_mem_flags ihipFlags = ROCCLR_MEM_PHYMEM;
  if (prop->type == hipMemAllocationTypeUncached) {
    ihipFlags |= CL_MEM_SVM_ATOMICS | ROCCLR_MEM_HSA_UNCACHED;
  }

  const bool isHostNuma = (prop->location.type == hipMemLocationTypeHostNuma ||
                           prop->location.type == hipMemLocationTypeHostNumaCurrent);
  bool useHostDevice = (prop->location.type == hipMemLocationTypeHost) || isHostNuma;
  // NUMA node selector handed to ROCclr: explicit node for HostNuma; -1 (default/
  // resolve-current) for HostNumaCurrent and non-NUMA allocations.
  const int numaNode =
      (prop->location.type == hipMemLocationTypeHostNuma) ? prop->location.id : -1;
  if (isHostNuma) {
    // Pack the host-NUMA marker + node into the upper bits so roc::Memory::create
    // routes to the CPU-pool allocator. Stored node value is (node + 1); 0 means
    // "resolve current node" (HostNumaCurrent, numaNode == -1).
    ihipFlags |= ROCCLR_MEM_HOST_NUMA;
    ihipFlags |= (static_cast<cl_svm_mem_flags>(numaNode + 1) << ROCCLR_MEM_HOST_NUMA_NODE_SHIFT) &
                 ROCCLR_MEM_HOST_NUMA_NODE_MASK;
  }
  hip::Device* dev = hip::getCurrentDevice();
  amd::Context* curDevContext = dev->asContext();
  amd::Context* amdContext = useHostDevice ? hip::host_context : curDevContext;

  if (amdContext == nullptr) {
    HIP_RETURN(hipErrorOutOfMemory);
  }

  const auto& dev_info = amdContext->devices()[0]->info();

  if (dev_info.maxPhysicalMemAllocSize_ < size) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  if (size % dev_info.memBaseAddrAlign_ != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  void* ptr = amd::SvmBuffer::malloc(*amdContext, ihipFlags, size, dev_info.memBaseAddrAlign_,
                                     useHostDevice ? curDevContext->svmDevices()[0] : nullptr);

  // Handle out of memory cases,
  if (ptr == nullptr) {
    size_t free = 0, total = 0;
    hipError_t hip_error = ihipMemGetInfo(&free, &total);
    if (hip_error == hipSuccess) {
      LogPrintfError(
          "Allocation failed : Device memory : required :%zu | free :%zu"
          "| total :%zu",
          size, free, total);
    }
    HIP_RETURN(hipErrorOutOfMemory);
  }

  // Add this to amd::Memory object, so this ptr is accesible for other hipmemory operations.
  size_t offset = 0;  // this is ignored
  amd::Memory* phys_mem_obj = getMemoryObject(dev, ptr, offset);
  // saves the current device id so that it can be accessed later. For host-NUMA
  // allocations location.id is a NUMA node id (not a device index), so record the
  // current device for the device-indexed bookkeeping and keep the node separately.
  phys_mem_obj->getUserData().deviceId =
      (prop->location.type == hipMemLocationTypeDevice) ? prop->location.id : dev->deviceId();
  phys_mem_obj->getUserData().locationType = prop->location.type;
  phys_mem_obj->getUserData().numaNode = numaNode;
  phys_mem_obj->getUserData().data = new hip::GenericAllocation(*phys_mem_obj, size, *prop);
  *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(phys_mem_obj->getUserData().data);

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags) {
  HIP_INIT_API(hipMemExportToShareableHandle, shareableHandle, handle, handleType, flags);

  if (flags != 0 || handle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (shareableHandle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GenericAllocation* ga = reinterpret_cast<hip::GenericAllocation*>(handle);
  if (ga == nullptr) {
    LogError("Generic Allocation is nullptr");
    HIP_RETURN(hipErrorNotInitialized);
  }

  if (ga->GetProperties().requestedHandleTypes != handleType) {
    LogPrintfError("HandleType mismatch memoryHandleType: %d, requestedHandleTypes: %d",
                   ga->GetProperties().requestedHandleTypes, handleType);
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Memory::HandleType htype = static_cast<amd::Memory::HandleType>(handleType);

  if (!ga->asAmdMemory().getContext().devices()[0]->ExportShareableVMMHandle(
          ga->asAmdMemory(), flags, shareableHandle, htype)) {
    LogPrintfError("Exporting Handle failed with flags: %d", flags);
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
  HIP_INIT_API(hipMemGetAccess, flags, location, ptr);

  if (flags == nullptr || location == nullptr || ptr == nullptr ||
      location->type != hipMemLocationTypeDevice || location->id >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue)
  }

  // Convert the access flags to amd::Device access flag
  auto& dev = g_devices[location->id];
  amd::Device::VmmAccess access_flags = static_cast<amd::Device::VmmAccess>(0);

  if (!dev->devices()[0]->GetMemAccess(ptr, &access_flags)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *flags = static_cast<unsigned long long>(access_flags);

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option) {

  HIP_INIT_API(hipMemGetAllocationGranularity, granularity, prop, option);

  if (granularity == nullptr || prop == nullptr || (prop->type != hipMemAllocationTypePinned &&
      prop->type != hipMemAllocationTypeUncached) ||
      (prop->location.type != hipMemLocationTypeDevice &&
       prop->location.type != hipMemLocationTypeHost &&
       prop->location.type != hipMemLocationTypeHostNuma &&
       prop->location.type != hipMemLocationTypeHostNumaCurrent) ||
      (prop->location.type == hipMemLocationTypeDevice &&
       prop->location.id >= g_devices.size()) ||
      (option != hipMemAllocationGranularityMinimum &&
       option != hipMemAllocationGranularityRecommended)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // All host location types (Host / HostNuma / HostNumaCurrent) use host_context.
  bool useHostDevice = (prop->location.type != hipMemLocationTypeDevice);
  amd::Context* curDevContext = hip::getCurrentDevice()->asContext();
  amd::Context* amdContext = useHostDevice ? hip::host_context : curDevContext;
  const auto& dev_info = amdContext->devices()[0]->info();

  if (option == hipMemAllocationGranularityMinimum) {
    *granularity = dev_info.virtualMemAllocGranularityMinimum_;
  } else {
    *granularity = dev_info.virtualMemAllocGranularityRecommended_;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle) {
  HIP_INIT_API(hipMemGetAllocationPropertiesFromHandle, prop, handle);

  if (handle == nullptr || prop == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *prop = reinterpret_cast<hip::GenericAllocation*>(handle)->GetProperties();

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType) {
  HIP_INIT_API(hipMemImportFromShareableHandle, handle, osHandle, shHandleType);

  if (handle == nullptr || osHandle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  amd::Memory::HandleType htype = static_cast<amd::Memory::HandleType>(shHandleType);
  amd::Memory* phys_mem_obj = device->ImportShareableVMMHandle(osHandle, htype);

  if (phys_mem_obj == nullptr) {
    LogError("failed to new a va range curr_mem_obj object!");
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = hip::getCurrentDevice()->deviceId();
  prop.requestedHandleTypes = shHandleType;

  phys_mem_obj->getUserData().deviceId = hip::getCurrentDevice()->deviceId();
  phys_mem_obj->getUserData().data = new hip::GenericAllocation(*phys_mem_obj, 0, prop);
  *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(phys_mem_obj->getUserData().data);

  if (!amd::MemObjMap::FindMemObj(phys_mem_obj->getSvmPtr())) {
    amd::MemObjMap::AddMemObj(phys_mem_obj->getSvmPtr(), phys_mem_obj);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) {
  HIP_INIT_API(hipMemMap, ptr, size, offset, handle, flags);

  if (ptr == nullptr || handle == nullptr || size == 0 || offset != 0 || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Re-interpret the ga handle.
  hip::GenericAllocation* ga = reinterpret_cast<hip::GenericAllocation*>(handle);

  // Pick the owning device. The backend self-classifies the remaining failure
  // causes (missing/invalid VA reservation, out-of-bounds range) via cl_int
  // return codes, so the HIP layer keeps only the checks the backend cannot
  // perform: the device-id index guard and size-vs-granularity alignment.

  // Owner device id must index g_devices[] safely before any backend call.
  // For host-backed allocations location.id is not a device index, so resolve via
  // VmmOwnerDeviceIndex (anchors on the current device).
  int owner_dev_id = VmmOwnerDeviceIndex(ga->GetProperties());
  if (owner_dev_id < 0 || static_cast<size_t>(owner_dev_id) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Device* dev = g_devices[owner_dev_id]->devices()[0];

  // The backend does not validate size-vs-granularity alignment; a misaligned
  // size would otherwise fail in the HW map and surface as OOM instead of the
  // correct invalid-value.
  size_t granularity = dev->info().virtualMemAllocGranularityMinimum_;
  if (granularity != 0 && (size % granularity) != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  ga->retain();

  // Direct synchronous path, do not wait on streams or other work
  cl_int cl_err = dev->virtualMap(ptr, size, &ga->asAmdMemory());
  if (cl_err != CL_SUCCESS) {
    ga->release();
    HIP_RETURN(ConvertCLErrorIntoHIPError(cl_err));
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream) {
  HIP_INIT_API(hipMemMapArrayAsync, mapInfoList, count, stream);

  if (mapInfoList == nullptr || count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  HIP_INIT_API(hipMemRelease, handle);

  if (handle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Re-interpret the ga handle and make sure it is not already released.
  hip::GenericAllocation* ga = reinterpret_cast<hip::GenericAllocation*>(handle);
  ga->release();

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
  HIP_INIT_API(hipMemRetainAllocationHandle, handle, addr);

  if (handle == nullptr || addr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Memory* mem = amd::MemObjMap::FindMemObj(addr);
  if (mem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // hipMalloc and other non-VMM allocations do not have phys_mem_obj
  amd::Memory* phys_mem_obj = mem->getUserData().phys_mem_obj;
  if (phys_mem_obj == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto ga = reinterpret_cast<hip::GenericAllocation*>(
      phys_mem_obj->getUserData().data);
  if (ga == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  ga->retain();
  *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(ga);

  HIP_RETURN(hipSuccess);
}

static inline address NextSubBufferPtr(const amd::Memory* mem) {
  return reinterpret_cast<address>(mem->getSvmPtr()) + mem->getSize();
}

static hipError_t ValidateSubBufferCoverage(amd::Memory* vaddr_sub_buffer_obj, size_t range_size) {
  // Validate that the requested range size is within the parent sub-buffer bounds.
  if (vaddr_sub_buffer_obj == nullptr || (vaddr_sub_buffer_obj->parent() != nullptr &&
                                          range_size > (vaddr_sub_buffer_obj->parent()->getSize() -
                                                        vaddr_sub_buffer_obj->getOrigin()))) {
    return hipErrorInvalidValue;
  }

  address range_end_address =
      reinterpret_cast<address>(vaddr_sub_buffer_obj->getSvmPtr()) + range_size;
  size_t covered_size = 0;
  amd::Memory* current_sub_buffer_obj = vaddr_sub_buffer_obj;
  // Validate that the size matches the sum of sub-buffer sizes
  while (current_sub_buffer_obj && NextSubBufferPtr(current_sub_buffer_obj) <= range_end_address) {
    if (range_size > covered_size &&
        range_size < covered_size + current_sub_buffer_obj->getSize()) {
      return hipErrorInvalidValue;
    }
    covered_size += current_sub_buffer_obj->getSize();
    current_sub_buffer_obj = amd::MemObjMap::FindMemObj(NextSubBufferPtr(current_sub_buffer_obj));
  }
  if (covered_size != range_size) {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  HIP_INIT_API(hipMemSetAccess, ptr, size, desc, count);

  if (ptr == nullptr || size == 0 || desc == nullptr || count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Ensure that the specified size parameter matches the sum of a complete set of
  // sub-buffers in the range, disallowing partial sub-buffer coverage.
  amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(ptr);
  hipError_t status = ValidateSubBufferCoverage(vaddr_sub_obj, size);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  for (size_t desc_idx = 0; desc_idx < count; ++desc_idx) {
    hipMemLocationType accessLocationType = desc[desc_idx].location.type;
    if (accessLocationType != hipMemLocationTypeDevice &&
        accessLocationType != hipMemLocationTypeHost &&
        accessLocationType != hipMemLocationTypeHostNuma &&
        accessLocationType != hipMemLocationTypeHostNumaCurrent) {
      HIP_RETURN(hipErrorInvalidValue);
    }

    // location.id is a device index for Device access and a NUMA node id for
    // HostNuma (ignored for Host / HostNumaCurrent). Validate accordingly and pick
    // a device to dispatch through (the CPU agent is resolved inside SetMemAccess).
    int numaNode = -1;
    amd::Device* accessDev = nullptr;
    if (accessLocationType == hipMemLocationTypeDevice) {
      if (desc[desc_idx].location.id < 0 || desc[desc_idx].location.id >= g_devices.size()) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      accessDev = g_devices[desc[desc_idx].location.id]->devices()[0];
    } else {
      accessDev = hip::getCurrentDevice()->devices()[0];
      if (accessLocationType == hipMemLocationTypeHostNuma) {
        if (desc[desc_idx].location.id < 0 ||
            static_cast<uint32_t>(desc[desc_idx].location.id) >= accessDev->numHostNumaNodes()) {
          HIP_RETURN(hipErrorInvalidValue);
        }
        numaNode = desc[desc_idx].location.id;
      }
    }

    amd::Device::VmmAccess access_flags = static_cast<amd::Device::VmmAccess>(desc[desc_idx].flags);
    if (access_flags != amd::Device::VmmAccess::kNone &&
        access_flags != amd::Device::VmmAccess::kReadOnly &&
        access_flags != amd::Device::VmmAccess::kReadWrite) {
      HIP_RETURN(hipErrorInvalidValue);
    }

    if (!accessDev->SetMemAccess(ptr, size, access_flags,
                                 static_cast<amd::Device::VmmLocationType>(accessLocationType),
                                 numaNode)) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemUnmap(void* ptr, size_t size) {
  HIP_INIT_API(hipMemUnmap, ptr, size);

  if (ptr == nullptr || size == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(ptr);
  hipError_t status = ValidateSubBufferCoverage(vaddr_sub_obj, size);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  // Direct synchronous path; synchronize all devices with access to memory.
  //
  // Pass 1 (validate, no mutation): walk every sub-buffer in [ptr, ptr+size)
  // and verify it has the bookkeeping the unmap loop will rely on. Bailing
  // here avoids leaving the range in an inconsistent half-unmapped state
  // (some sub-buffers already torn down, others still mapped). Simultaneously
  // collect the union of owner devices and access devices so we only walk
  // the chain once.
  std::set<int> sync_device_ids;
  {
    amd::Memory* it = vaddr_sub_obj;
    address end_addr = reinterpret_cast<address>(vaddr_sub_obj->getSvmPtr()) + size;
    while (it && NextSubBufferPtr(it) <= end_addr) {
      amd::Memory* phys_mem_obj = it->getUserData().phys_mem_obj;
      if (phys_mem_obj == nullptr) {
        LogPrintfError("hipMemUnmap: sub_obj at %p missing phys_mem_obj", it->getSvmPtr());
        HIP_RETURN(hipErrorInvalidValue);
      }
      auto* ga = reinterpret_cast<hip::GenericAllocation*>(phys_mem_obj->getUserData().data);
      if (ga == nullptr) {
        LogPrintfError("hipMemUnmap: sub_obj at %p has null ga", it->getSvmPtr());
        HIP_RETURN(hipErrorInvalidValue);
      }
      int owner_id = VmmOwnerDeviceIndex(ga->GetProperties());
      if (owner_id < 0 || static_cast<size_t>(owner_id) >= g_devices.size()) {
        LogPrintfError("hipMemUnmap: sub_obj at %p has out-of-range owner id %d",
                       it->getSvmPtr(), owner_id);
        HIP_RETURN(hipErrorInvalidValue);
      }
      sync_device_ids.insert(owner_id);

      // Probe each device's access at THIS sub-buffer's VA. The previous
      // implementation only probed `ptr`, which missed any device whose
      // access window covers a strictly-interior sub-range.
      void* sub_va = it->getSvmPtr();
      for (size_t dev_idx = 0; dev_idx < g_devices.size(); ++dev_idx) {
        amd::Device::VmmAccess access_flags = amd::Device::VmmAccess::kNone;
        if (g_devices[dev_idx]->devices()[0]->GetMemAccess(sub_va, &access_flags) &&
            access_flags != amd::Device::VmmAccess::kNone) {
          sync_device_ids.insert(static_cast<int>(dev_idx));
        }
      }

      it = amd::MemObjMap::FindMemObj(NextSubBufferPtr(it));
    }
  }

  // Pass 2: SyncAllStreams once per device in the union BEFORE the sub-buffer
  // loop, so the unmap doesn't race in-flight access-device work.
  for (int dev_id : sync_device_ids) {
    g_devices[dev_id]->SyncAllStreams();
  }

  // Pass 3: Sub-buffer unmap loop. Pass 1 validated every entry, so the only
  // remaining failure mode here is a HW virtualUnmap failure.
  cl_int first_cl_err = CL_SUCCESS;
  address end_address = reinterpret_cast<address>(vaddr_sub_obj->getSvmPtr()) + size;
  while (vaddr_sub_obj && NextSubBufferPtr(vaddr_sub_obj) <= end_address) {
    amd::Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;
    hip::GenericAllocation* ga =
        reinterpret_cast<hip::GenericAllocation*>(phys_mem_obj->getUserData().data);
    void* sub_va = vaddr_sub_obj->getSvmPtr();
    size_t sub_size = vaddr_sub_obj->getSize();
    address next_ptr = NextSubBufferPtr(vaddr_sub_obj);

    // Each sub-buffer is unmapped by the device that owns its physical backing.
    amd::Device* sub_dev = g_devices[VmmOwnerDeviceIndex(ga->GetProperties())]->devices()[0];
    cl_int cl_err = sub_dev->virtualUnmap(sub_va, sub_size);
    if (cl_err != CL_SUCCESS) {
      LogPrintfError("hipMemUnmap: virtualUnmap failed for va: %p", sub_va);
      if (first_cl_err == CL_SUCCESS) {
        first_cl_err = cl_err;
      }
      vaddr_sub_obj = amd::MemObjMap::FindMemObj(next_ptr);
      continue;
    }

    // Release the ga ref only on successful HW unmap.
    ga->release();

    // sub_obj already released inside UnmapMemObjBookkeeping (called from
    // virtualUnmap on success).
    vaddr_sub_obj = amd::MemObjMap::FindMemObj(next_ptr);
  }

  if (first_cl_err != CL_SUCCESS) {
    HIP_RETURN(ConvertCLErrorIntoHIPError(first_cl_err));
  }

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
