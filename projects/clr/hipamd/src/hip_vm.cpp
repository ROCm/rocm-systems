/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
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

hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  HIP_INIT_API(hipMemAddressFree, devPtr, size);
  hipError_t status = hipSuccess;
  if (devPtr == nullptr || size == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Memory* memObj = amd::MemObjMap::FindVirtualMemObj(devPtr);
  if (memObj == nullptr) {
    LogPrintfError("Cannot find the Virtual MemObj entry for this addr 0x%x", devPtr);
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
  if (size == 0 || ((size % dev_info.virtualMemAllocGranularity_) != 0) ||
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
    LogPrintfError("Requested address was not allocated. Allocated address : 0x%x ", *ptr);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags) {
  HIP_INIT_API(hipMemCreate, handle, size, prop, flags);

  //  Currently we do not support Pinned memory
  if (handle == nullptr || size == 0 || flags != 0 || prop == nullptr ||
      (prop->type != hipMemAllocationTypePinned && prop->type != hipMemAllocationTypeUncached) ||
      (prop->location.type != hipMemLocationTypeDevice &&
          prop->location.type != hipMemLocationTypeHost)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (prop->location.id < 0 || prop->location.id >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (prop->requestedHandleTypes != hipMemHandleTypeNone &&
      prop->requestedHandleTypes != hipMemHandleTypePosixFileDescriptor) {
    HIP_RETURN(hipErrorNotSupported);
  }

  // When ROCCLR_MEM_PHYMEM is set, ROCr impl gets and stores unique hsa handle. Flag no-op on PAL.
  unsigned int ihipFlags = ROCCLR_MEM_PHYMEM;
  if (prop->type == hipMemAllocationTypeUncached) {
    ihipFlags |= CL_MEM_SVM_ATOMICS | ROCCLR_MEM_HSA_UNCACHED;
  }

  bool useHostDevice = (prop->location.type == hipMemLocationTypeHost);
  amd::Context* curDevContext = hip::getCurrentDevice()->asContext();
  amd::Context* amdContext = useHostDevice ? hip::host_context : curDevContext;

  if (amdContext == nullptr) {
    return hipErrorOutOfMemory;
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
    hipError_t hip_error = hipMemGetInfo(&free, &total);
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
  amd::Memory* phys_mem_obj = getMemoryObject(ptr, offset);
  // saves the current device id so that it can be accessed later
  phys_mem_obj->getUserData().deviceId = prop->location.id;
  phys_mem_obj->getUserData().locationType = prop->location.type;
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

  if (!ga->asAmdMemory().getContext().devices()[0]->ExportShareableVMMHandle(
          ga->asAmdMemory(), flags, shareableHandle)) {
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
       prop->location.type != hipMemLocationTypeHost) ||
      prop->location.id >= g_devices.size() ||
      (option != hipMemAllocationGranularityMinimum &&
       option != hipMemAllocationGranularityRecommended)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  bool useHostDevice = (prop->location.type == hipMemLocationTypeHost);
  amd::Context* curDevContext = hip::getCurrentDevice()->asContext();
  amd::Context* amdContext = useHostDevice ? hip::host_context : curDevContext;
  const auto& dev_info = amdContext->devices()[0]->info();

  *granularity = dev_info.virtualMemAllocGranularity_;

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
  amd::Memory* phys_mem_obj = device->ImportShareableVMMHandle(osHandle);

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

  // Re-interpret the ga handle and set the mapped flag
  hip::GenericAllocation* ga = reinterpret_cast<hip::GenericAllocation*>(handle);
  ga->retain();

  auto& queue = *g_devices[ga->GetProperties().location.id]->NullStream();
  // Map the physical address to virtual address
  amd::Command* cmd = new amd::VirtualMapCommand(queue, amd::Command::EventWaitList{}, ptr, size,
                                                 &ga->asAmdMemory());
  cmd->enqueue();
  cmd->awaitCompletion();
  cmd->release();

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

  *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(
      mem->getUserData().phys_mem_obj->getUserData().data);

  if (*handle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  HIP_INIT_API(hipMemSetAccess, ptr, size, desc, count);

  if (ptr == nullptr || size == 0 || desc == nullptr || count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Ensure that the specified size parameter matches the total size of a complete set of
  // sub-buffers, disallowing partial sub-buffer coverage
  auto mem_object = amd::MemObjMap::FindMemObj(ptr);
  hipMemLocationType memLocationType = hipMemLocationTypeNone;

  if (mem_object) {
    memLocationType = static_cast<hipMemLocationType>(mem_object->getUserData().locationType);
    if (mem_object->parent()) {
      size_t accumulated_buffer_size = 0;
      for (auto sub_buffer : mem_object->parent()->subBuffers()) {
        accumulated_buffer_size += sub_buffer->getSize();
        if (accumulated_buffer_size > size) {
          HIP_RETURN(hipErrorInvalidValue);
        } else if (accumulated_buffer_size == size) {
          break;
        }
      }

      if (accumulated_buffer_size != size) {
        HIP_RETURN(hipErrorInvalidValue);
      }
    }
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (size_t desc_idx = 0; desc_idx < count; ++desc_idx) {
    hipMemLocationType accessLocationType = desc[desc_idx].location.type;
    if (accessLocationType != hipMemLocationTypeDevice && accessLocationType != hipMemLocationTypeHost) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    if (accessLocationType == hipMemLocationTypeHost &&
        memLocationType != hipMemLocationTypeHost) {
      HIP_RETURN(hipErrorInvalidValue)
    }

    if (desc[desc_idx].location.id >= g_devices.size()) {
      HIP_RETURN(hipErrorInvalidValue)
    }

    auto& dev = g_devices[desc[desc_idx].location.id];
    amd::Device::VmmAccess access_flags = static_cast<amd::Device::VmmAccess>(desc[desc_idx].flags);
    if (access_flags != amd::Device::VmmAccess::kNone &&
        access_flags != amd::Device::VmmAccess::kReadOnly &&
        access_flags != amd::Device::VmmAccess::kReadWrite) {
      HIP_RETURN(hipErrorInvalidValue);
    }

    if (!dev->devices()[0]->SetMemAccess(ptr, size, access_flags,
                                         static_cast<amd::Device::VmmLocationType>(accessLocationType))) {
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

  // Helper lambda to get the next sub-buffer pointer
  auto next_subbuffer_ptr = [](const amd::Memory* mem) -> address {
    return reinterpret_cast<address>(mem->getSvmPtr()) + mem->getSize();
  };

  amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(ptr);
  // Validate that the size is within range
  if (vaddr_sub_obj == nullptr ||
      (vaddr_sub_obj->parent() != nullptr &&
       size > (vaddr_sub_obj->parent()->getSize() - vaddr_sub_obj->getOrigin()))) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  address end_address = reinterpret_cast<address>(vaddr_sub_obj->getSvmPtr()) + size;
  size_t total_processed_size = 0;
  amd::Memory* check_obj = vaddr_sub_obj;
  // Validate that the size matches the sum of sub-buffer sizes
  while (check_obj && next_subbuffer_ptr(check_obj) <= end_address) {
    if (size > total_processed_size && size < total_processed_size + check_obj->getSize()) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    total_processed_size += check_obj->getSize();
    check_obj = amd::MemObjMap::FindMemObj(next_subbuffer_ptr(check_obj));
  }
  if (total_processed_size != size) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Unmap all sub-buffers in the range
  while (vaddr_sub_obj && next_subbuffer_ptr(vaddr_sub_obj) <= end_address) {
    amd::Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;
    if (phys_mem_obj == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }

    amd::Command* cmd = new amd::VirtualMapCommand(
        *hip::getCurrentDevice()->NullStream(), amd::Command::EventWaitList{},
        vaddr_sub_obj->getSvmPtr(), vaddr_sub_obj->getSize(), nullptr);
    cmd->enqueue();
    cmd->awaitCompletion();
    cmd->release();
    // restore the original pa of the generic allocation
    hip::GenericAllocation* ga =
        reinterpret_cast<hip::GenericAllocation*>(phys_mem_obj->getUserData().data);
    ga->release();

    address next_ptr = next_subbuffer_ptr(vaddr_sub_obj);
    vaddr_sub_obj->release();
    vaddr_sub_obj = amd::MemObjMap::FindMemObj(next_ptr);
  }

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
