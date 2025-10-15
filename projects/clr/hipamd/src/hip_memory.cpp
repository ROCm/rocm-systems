/* Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc.

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
#include "device.hpp"
#include "hip/driver_types.h"
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_conversions.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "platform/external_memory.hpp"
namespace hip {

// Guards global hipArray set
amd::Monitor hipArraySetLock{};
std::unordered_set<hipArray*> hipArraySet;

template <typename T> T ReturnPtrValue(T* ptr) { return (ptr != nullptr) ? *ptr : nullptr; }

// ================================================================================================
amd::Memory* getMemoryObject(const void* ptr, size_t& offset, size_t size) {
  auto memObj = amd::MemObjMap::FindMemObj(ptr, &offset);
  if (memObj == nullptr) {
    // If memObj not found, use arena_mem_obj. arena_mem_obj is null, if HMM is disabled.
    memObj =
        (hip::getCurrentDevice()->asContext()->svmDevices()[0])->GetArenaMemObj(ptr, offset, size);
  }

  // On Windows, when using hipHostRegister, the map may contain a single memory object for
  // multiple devices. This is because device addresses can overlap.
  // The offset needs to be calculated relative to the memory of the current device.
  if (IS_WINDOWS && (memObj != nullptr) && (memObj->getMemFlags() & CL_MEM_USE_HOST_PTR)) {
    device::Memory* currentDevMem =
        memObj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0], false);
    if (currentDevMem != nullptr) {
      size_t currentDevOffset = reinterpret_cast<uint64_t>(ptr) - currentDevMem->virtualAddress();
      if (currentDevOffset < memObj->getSize()) {
        offset = currentDevOffset;
      }
    }
  }
  return memObj;
}

hipMemoryType getMemoryType(const amd::Memory* memory) {
  if (memory == nullptr) {
    return hipMemoryTypeHost;
  }

  return ((CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_USE_HOST_PTR) & memory->getMemFlags())
             ? hipMemoryTypeHost
             : hipMemoryTypeDevice;
}

// ================================================================================================
amd::Memory* getMemoryObjectWithOffset(const void* ptr, const size_t size) {
  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(ptr, offset);

  if (memObj != nullptr) {
    if (size > (memObj->getSize() - offset)) {
      return nullptr;
    }
    memObj = new (memObj->getContext()) amd::Buffer(*memObj, memObj->getMemFlags(), offset, size);
    if (memObj == nullptr) {
      ;
      return nullptr;
    }

    if (!memObj->create(nullptr)) {
      memObj->release();
      return nullptr;
    }
  }

  return memObj;
}

// ================================================================================================
hipError_t ihipFree(void* ptr) {
  if (ptr == nullptr) {
    return hipSuccess;
  }

  size_t offset = 0;
  amd::Memory* memory_object = getMemoryObject(ptr, offset);
  if (memory_object != nullptr) {
    auto device_id = memory_object->getUserData().deviceId;

    // Find out if memory belongs to any memory pool
    if (!g_devices[device_id]->FreeMemory(memory_object, nullptr)) {
      // Wait on the device, associated with the current memory object during allocation
      g_devices[device_id]->SyncAllStreams();

      // External mem is not svm.
      if (memory_object->isInterop()) {
        amd::MemObjMap::RemoveMemObj(ptr);
        memory_object->release();
      } else {
        amd::SvmBuffer::free(memory_object->getContext(), ptr);
      }
    }
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

// ================================================================================================
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                                   const hipExternalMemoryHandleDesc* memHandleDesc) {
  HIP_INIT_API(hipImportExternalMemory, extMem_out, memHandleDesc);
  if (extMem_out == nullptr || memHandleDesc == nullptr ||
      (memHandleDesc->flags != 0 && memHandleDesc->flags != hipExternalMemoryDedicated) ||
      memHandleDesc->size == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if ((memHandleDesc->type < hipExternalMemoryHandleTypeOpaqueFd) ||
      (memHandleDesc->type > hipExternalMemoryHandleTypeD3D11ResourceKmt)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Context& amdContext = *hip::getCurrentDevice()->asContext();
#ifdef _WIN32
  auto ext_buffer = new (amdContext)
      amd::ExternalBuffer(amdContext, memHandleDesc->size, memHandleDesc->handle.win32.handle,
                          static_cast<amd::ExternalMemory::HandleType>(memHandleDesc->type),
                          memHandleDesc->handle.win32.name);
#else
  auto ext_buffer = new (amdContext)
      amd::ExternalBuffer(amdContext, memHandleDesc->size, memHandleDesc->handle.fd,
                          static_cast<amd::ExternalMemory::HandleType>(memHandleDesc->type));
#endif

  if (!ext_buffer) {
    HIP_RETURN(hipErrorOutOfMemory);
  }

  if (!ext_buffer->create()) {
    ext_buffer->release();
    HIP_RETURN(hipErrorOutOfMemory);
  }
  *extMem_out = ext_buffer;

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem,
                                            const hipExternalMemoryBufferDesc* bufferDesc) {
  HIP_INIT_API(hipExternalMemoryGetMappedBuffer, devPtr, extMem, bufferDesc);

  if (devPtr == nullptr || extMem == nullptr || bufferDesc == nullptr || bufferDesc->flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto buf = reinterpret_cast<amd::ExternalBuffer*>(extMem);
  const device::Memory* devMem = buf->getDeviceMemory(*hip::getCurrentDevice()->devices()[0]);

  if (devMem == nullptr || ((bufferDesc->offset + bufferDesc->size) > devMem->size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *devPtr = reinterpret_cast<void*>(devMem->virtualAddress() + bufferDesc->offset);
  amd::MemObjMap::AddMemObj(*devPtr, buf);
  buf->retain();
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
  HIP_INIT_API(hipDestroyExternalMemory, extMem);

  if (extMem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<amd::ExternalBuffer*>(extMem)->release();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  HIP_INIT_API(hipImportExternalSemaphore, extSem_out, semHandleDesc);
  if (extSem_out == nullptr || semHandleDesc == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if ((semHandleDesc->type < hipExternalSemaphoreHandleTypeOpaqueFd) ||
      (semHandleDesc->type > hipExternalSemaphoreHandleTypeD3D12Fence)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Device* device = hip::getCurrentDevice()->devices()[0];

#ifdef _WIN32
  if (semHandleDesc->handle.win32.handle == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (device->importExtSemaphore(
          extSem_out, semHandleDesc->handle.win32.handle,
          static_cast<amd::ExternalSemaphoreHandleType>(semHandleDesc->type))) {
#else
  if (semHandleDesc->handle.fd == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (device->importExtSemaphore(
          extSem_out, semHandleDesc->handle.fd,
          static_cast<amd::ExternalSemaphoreHandleType>(semHandleDesc->type))) {
#endif
    HIP_RETURN(hipSuccess);
  }
  HIP_RETURN(hipErrorNotSupported);
}


hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream) {
  HIP_INIT_API(hipSignalExternalSemaphoresAsync, extSemArray, paramsArray, numExtSems, stream);
  if (extSemArray == nullptr || paramsArray == nullptr || !hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (unsigned int i = 0; i < numExtSems; i++) {
    if (extSemArray[i] != nullptr) {
      if (paramsArray[i].flags != 0) {
        // flags should be 0 for any type other than hipExternalSemaphoreHandleTypeNvSciSync.
        // But hipExternalSemaphoreHandleTypeNvSciSync is not supported on AMD device.
        HIP_RETURN(hipErrorInvalidValue);
      }
      amd::ExternalSemaphoreCmd* command = new amd::ExternalSemaphoreCmd(
          *hip_stream, extSemArray[i], paramsArray[i].params.fence.value,
          amd::ExternalSemaphoreCmd::COMMAND_SIGNAL_EXTSEMAPHORE);
      if (command == nullptr) {
        return hipErrorOutOfMemory;
      }
      command->enqueue();
      command->release();
    } else {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                          const hipExternalSemaphoreWaitParams* paramsArray,
                                          unsigned int numExtSems, hipStream_t stream) {
  HIP_INIT_API(hipWaitExternalSemaphoresAsync, extSemArray, paramsArray, numExtSems, stream);
  if (extSemArray == nullptr || paramsArray == nullptr || !hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (unsigned int i = 0; i < numExtSems; i++) {
    if (extSemArray[i] != nullptr) {
      if (paramsArray[i].flags != 0) {
        // flags should be 0 for any type other than hipExternalSemaphoreHandleTypeNvSciSync.
        // But hipExternalSemaphoreHandleTypeNvSciSync is not supported on AMD device.
        HIP_RETURN(hipErrorInvalidValue);
      }
      amd::ExternalSemaphoreCmd* command = new amd::ExternalSemaphoreCmd(
          *hip_stream, extSemArray[i], paramsArray[i].params.fence.value,
          amd::ExternalSemaphoreCmd::COMMAND_WAIT_EXTSEMAPHORE);
      if (command == nullptr) {
        return hipErrorOutOfMemory;
      }
      command->enqueue();
      command->release();
    } else {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
  HIP_INIT_API(hipDestroyExternalSemaphore, extSem);
  if (extSem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  device->DestroyExtSemaphore(extSem);
  HIP_RETURN(hipSuccess);
}


// ================================================================================================
hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }
  if (sizeBytes == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }

  bool useHostDevice = (flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) != 0;
  amd::Context* curDevContext = hip::getCurrentDevice()->asContext();
  amd::Context* amdContext = useHostDevice ? hip::host_context : curDevContext;

  if (amdContext == nullptr) {
    return hipErrorOutOfMemory;
  }

  const auto& dev_info = amdContext->devices()[0]->info();
  hip::getCurrentDevice()->SetActiveStatus();

  size_t max_device_size = IS_LINUX
                               ? dev_info.maxMemAllocSize_
                               : (dev_info.maxMemAllocSize_ + dev_info.maxPhysicalMemAllocSize_);

  if ((useHostDevice && dev_info.maxPhysicalMemAllocSize_ < sizeBytes) ||
      (!useHostDevice && max_device_size < sizeBytes)) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*amdContext, flags, sizeBytes, dev_info.memBaseAddrAlign_,
                                useHostDevice ? curDevContext->svmDevices()[0] : nullptr);

  if (*ptr == nullptr) {
    if (!useHostDevice) {
      size_t free = 0, total = 0;
      hipError_t err = hipMemGetInfo(&free, &total);
      if (err == hipSuccess) {
        LogPrintfError("Allocation failed : Device memory : required :%zu | free :%zu | total :%zu",
                       sizeBytes, free, total);
      }
    } else {
      LogPrintfError("Allocation failed : Pinned Memory, size :%zu", sizeBytes);
    }
    return hipErrorOutOfMemory;
  }
  size_t offset = 0;  // this is ignored
  amd::Memory* memObj = getMemoryObject(*ptr, offset);
  // saves the current device id so that it can be accessed later
  memObj->getUserData().deviceId = hip::getCurrentDevice()->deviceId();
  return hipSuccess;
}

// ================================================================================================
hipError_t ihipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }
  if (sizeBytes == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }

  *ptr = nullptr;
  const unsigned int coherentFlags = hipHostMallocCoherent | hipHostMallocNonCoherent;

  // can't have both Coherent and NonCoherent flags set at the same time
  if ((flags & coherentFlags) == coherentFlags) {
    LogPrintfError(
        "Cannot have both coherent and non-coherent flags "
        "at the same time, flags: %u coherent flags: %u",
        flags, coherentFlags);
    return hipErrorInvalidValue;
  }

  unsigned int ihipFlags = CL_MEM_SVM_FINE_GRAIN_BUFFER;
  if (flags & hipHostMallocUncached) {
    if (IS_WINDOWS) {
      return hipErrorInvalidValue;
    }
    if (flags & (hipHostMallocNonCoherent | hipHostMallocCoherent)) {
      return hipErrorInvalidValue;
    }
    ihipFlags |= ROCCLR_MEM_HSA_UNCACHED;
  }

  if (flags == 0 ||
      flags & (hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocNumaUser |
               hipHostMallocUncached) ||
      (!(flags & hipHostMallocNonCoherent) && HIP_HOST_COHERENT)) {
    ihipFlags |= CL_MEM_SVM_ATOMICS;
  }

  if (flags & hipHostMallocNumaUser) {
    ihipFlags |= CL_MEM_FOLLOW_USER_NUMA_POLICY;
  }

  if (flags & hipHostMallocNonCoherent) {
    ihipFlags &= ~CL_MEM_SVM_ATOMICS;
  }

  hipError_t status = ihipMalloc(ptr, sizeBytes, ihipFlags);

  if ((status == hipSuccess) && ((*ptr) != nullptr)) {
    size_t offset = 0;  // This is ignored
    amd::Memory* svmMem = getMemoryObject(*ptr, offset);
    // Save the HIP memory flags so that they can be accessed later
    svmMem->getUserData().flags = flags;
  }

  return status;
}

// ================================================================================================
bool IsHtoHMemcpyValid(void* dst, const void* src, hipMemcpyKind kind) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
  if (src && dst && srcMemory == nullptr && dstMemory == nullptr) {
    if (!g_devices[0]->devices()[0]->info().hmmCpuMemoryAccessible_ &&
        kind != hipMemcpyHostToHost && kind != hipMemcpyDefault) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  if (dst == nullptr || src == nullptr) {
    return hipErrorInvalidValue;
  }
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if (srcMemory != nullptr) {
    // Validate Mem Access in case of VMM Memory
    if (!srcMemory->ValidateMemAccess(*hip::getCurrentDevice()->devices()[0], false)) {
      return hipErrorUnknown;
    }

    // If the mem object is a VMM sub buffer (subbuffer has parent set),
    // then use parent's size for validation.
    if (srcMemory->parent() && (srcMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
      srcMemory = srcMemory->parent();
    }

    // Size validation
    if (sizeBytes > (srcMemory->getSize() - sOffset)) {
      return hipErrorInvalidValue;
    }
  }

  if (dstMemory != nullptr) {
    // Validate Mem Access in case of VMM Memory
    if (!dstMemory->ValidateMemAccess(*hip::getCurrentDevice()->devices()[0], true)) {
      return hipErrorUnknown;
    }

    // If the mem object is a VMM sub buffer (subbuffer has parent set),
    // then use parent's size for validation.
    if (dstMemory->parent() && (dstMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
      dstMemory = dstMemory->parent();
    }

    // Size validation
    if (sizeBytes > (dstMemory->getSize() - dOffset)) {
      return hipErrorInvalidValue;
    }
  }

  // If src and dst ptr are null then kind must be either h2h or def.
  if (!IsHtoHMemcpyValid(dst, src, kind)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hip::MemcpyType ihipGetMemcpyType(const void* src, void* dst, hipMemcpyKind kind) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
  hip::MemcpyType type;
  if (srcMemory == nullptr && dstMemory == nullptr) {
    type = hipHostToHost;
  } else if ((srcMemory == nullptr) && (dstMemory != nullptr)) {
    type = hipWriteBuffer;
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {
    type = hipReadBuffer;
  } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
    // Check if the queue device doesn't match the device on any memory object.
    // And any of them are not host allocation.
    // Hence it's a P2P transfer, because the app has requested access to another GPU
    if ((srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) &&
        ((srcMemory->getContext().devices().size() == 1) &&
         (dstMemory->getContext().devices().size() == 1))) {
      type = hipCopyBufferP2P;
    } else if (kind == hipMemcpyDeviceToDeviceNoCU) {
      type = hipCopyBufferSDMA;
    } else {
      type = hipCopyBuffer;
    }
  }
  return type;
}

// ================================================================================================
hipError_t ihipMemcpyCommand(amd::Command*& command, void* dst, const void* src, size_t sizeBytes,
                             hipMemcpyKind kind, hip::Stream& stream, bool isAsync) {
  amd::Command::EventWaitList waitList;
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
  amd::Device* queueDevice = &stream.device();
  amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
  hip::MemcpyType type = ihipGetMemcpyType(src, dst, kind);
  hip::Stream* pStream = &stream;
  switch (type) {
    case hipWriteBuffer:
      if (queueDevice != dstMemory->GetDeviceById() &&
          !(dstMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
        pStream = hip::getNullStream(dstMemory->GetDeviceById()->context());
        amd::Command* cmd = stream.getLastQueuedCommand(true);
        if (cmd != nullptr) {
          waitList.push_back(cmd);
        }
      }
      command = new amd::WriteMemoryCommand(*pStream, CL_COMMAND_WRITE_BUFFER, waitList,
                                            *dstMemory->asBuffer(), dOffset, sizeBytes, src, 0, 0,
                                            copyMetadata);
      break;
    case hipReadBuffer:
      if (queueDevice != srcMemory->GetDeviceById() &&
          !(srcMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
        pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
        amd::Command* cmd = stream.getLastQueuedCommand(true);
        if (cmd != nullptr) {
          waitList.push_back(cmd);
        }
      }
      command = new amd::ReadMemoryCommand(*pStream, CL_COMMAND_READ_BUFFER, waitList,
                                           *srcMemory->asBuffer(), sOffset, sizeBytes, dst, 0, 0,
                                           copyMetadata);
      break;
    case hipCopyBufferP2P:
      command = new amd::CopyMemoryP2PCommand(stream, CL_COMMAND_COPY_BUFFER, waitList,
                                              *srcMemory->asBuffer(), *dstMemory->asBuffer(),
                                              sOffset, dOffset, sizeBytes);
      if (command == nullptr) {
        return hipErrorOutOfMemory;
      }
      // Make sure runtime has valid memory for the command execution. P2P access
      // requires page table mapping on the current device to another GPU memory
      if (!static_cast<amd::CopyMemoryP2PCommand*>(command)->validateMemory()) {
        delete command;
        return hipErrorInvalidValue;
      }
      break;
    case hipCopyBufferSDMA:
      copyMetadata.copyEnginePreference_ = amd::CopyMetadata::CopyEnginePreference::SDMA;
    case hipCopyBuffer:
      if ((srcMemory->GetDeviceById() == dstMemory->GetDeviceById()) &&
          queueDevice != srcMemory->GetDeviceById()) {
        pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
        amd::Command* cmd = stream.getLastQueuedCommand(true);
        if (cmd != nullptr) {
          waitList.push_back(cmd);
        }
      } else if (srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) {
        // Scenarios such as DtoH where dst is pinned memory
        if ((queueDevice != srcMemory->GetDeviceById()) &&
            (dstMemory->getContext().devices().size() != 1) &&
            !(srcMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
          pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
          amd::Command* cmd = stream.getLastQueuedCommand(true);
          if (cmd != nullptr) {
            waitList.push_back(cmd);
          }
          // Scenarios such as HtoD where src is pinned memory
        } else if ((queueDevice != dstMemory->GetDeviceById()) &&
                   (srcMemory->getContext().devices().size() != 1) &&
                   !(dstMemory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
          pStream = hip::getNullStream(dstMemory->GetDeviceById()->context());
          amd::Command* cmd = stream.getLastQueuedCommand(true);
          if (cmd != nullptr) {
            waitList.push_back(cmd);
          }
        }
      }
      command = new amd::CopyMemoryCommand(*pStream, CL_COMMAND_COPY_BUFFER, waitList,
                                           *srcMemory->asBuffer(), *dstMemory->asBuffer(), sOffset,
                                           dOffset, sizeBytes, copyMetadata);
      break;
  }
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  if (waitList.size() > 0) {
    waitList[0]->release();
  }
  return hipSuccess;
}

bool IsHtoHMemcpy(void* dst, const void* src) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
  if (srcMemory == nullptr && dstMemory == nullptr) {
    return true;
  }
  return false;
}

// ================================================================================================
void ihipHtoHMemcpy(void* dst, const void* src, size_t sizeBytes, hip::Stream& stream) {
  stream.finish();
  memcpy(dst, src, sizeBytes);
}

// ================================================================================================
hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                      hip::Stream& stream, bool isHostAsync, bool isGPUAsync) {
  hipError_t status;
  if (sizeBytes == 0) {
    // Skip if nothing needs writing.
    return hipSuccess;
  }
  status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    return status;
  }
  if (src == dst && kind == hipMemcpyDefault) {
    return hipSuccess;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  hipMemoryType srcMemoryType = getMemoryType(srcMemory);
  hipMemoryType dstMemoryType = getMemoryType(dstMemory);

  if (srcMemory == nullptr && dstMemory == nullptr) {
    ihipHtoHMemcpy(dst, src, sizeBytes, stream);
    return hipSuccess;
  } else if (((srcMemory == nullptr) && (dstMemory != nullptr)) ||
             ((srcMemory != nullptr) && (dstMemory == nullptr))) {
    // Unpinned copy wait behavior is enforced in the lower copy layers so skip
    // wait at top level except for MT path
    isHostAsync &= AMD_DIRECT_DISPATCH ? true : false;
  } else if (srcMemory->GetDeviceById() == dstMemory->GetDeviceById()) {
    // Device to Device copies do not need to host side synchronization.
    if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice) &&
        (!srcMemory->getUserData().sync_mem_ops_ || !dstMemory->getUserData().sync_mem_ops_)) {
      isHostAsync = true;
    }
    // Any Host to any Host need host side synchronization.
    if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeHost)) {
      isHostAsync = false;
    }
  }

  amd::Command* command = nullptr;
  status = ihipMemcpyCommand(command, dst, src, sizeBytes, kind, stream, isHostAsync);
  if (status != hipSuccess) {
    return status;
  }
  command->enqueue();
  if (!isHostAsync) {
    command->queue()->finishCommand(command);
  } else if (!isGPUAsync) {
    hip::Stream* pStream = hip::getNullStream(dstMemory->GetDeviceById()->context());
    amd::Command::EventWaitList waitList;
    waitList.push_back(command);
    amd::Command* depdentMarker = new amd::Marker(*pStream, false, waitList);
    if (depdentMarker != nullptr) {
      depdentMarker->enqueue();
      depdentMarker->release();
    }
  } else {
    amd::HostQueue* newQueue = command->queue();
    if (newQueue != &stream) {
      amd::Command::EventWaitList waitList;
      amd::Command* cmd = newQueue->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
        amd::Command* depdentMarker = new amd::Marker(stream, true, waitList);
        if (depdentMarker != nullptr) {
          depdentMarker->enqueue();
          depdentMarker->release();
        }
        cmd->release();
      }
    }
  }
  command->release();
  return hipSuccess;
}

// ================================================================================================
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipExtMallocWithFlags, ptr, sizeBytes, flags);

  unsigned int ihipFlags = 0;
  if (flags == hipDeviceMallocDefault) {
    ihipFlags = 0;
  } else if (flags == hipDeviceMallocFinegrained) {
    ihipFlags = CL_MEM_SVM_ATOMICS;
  } else if (flags == hipDeviceMallocUncached) {
    ihipFlags = CL_MEM_SVM_ATOMICS | ROCCLR_MEM_HSA_UNCACHED;
  } else if (flags == hipDeviceMallocContiguous) {
    ihipFlags = ROCCLR_MEM_HSA_CONTIGUOUS | ROCCLR_MEM_HSA_UNCACHED;
  } else if (flags == hipMallocSignalMemory) {
    ihipFlags = CL_MEM_SVM_ATOMICS | CL_MEM_SVM_FINE_GRAIN_BUFFER | ROCCLR_MEM_HSA_SIGNAL_MEMORY;
    if (sizeBytes != 8) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status = ihipMalloc(ptr, sizeBytes, ihipFlags);

  if ((status == hipSuccess) && ((*ptr) != nullptr)) {
    size_t offset = 0;  // This is ignored
    amd::Memory* svmMem = getMemoryObject(*ptr, offset);
    // Save the HIP memory flags so that they can be accessed later
    svmMem->getUserData().flags = flags;
  }
  HIP_RETURN(status, ReturnPtrValue(ptr));
}

hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(hipMalloc, ptr, sizeBytes);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN_DURATION(ihipMalloc(ptr, sizeBytes, 0), ReturnPtrValue(ptr));
}

hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipHostMalloc, ptr, sizeBytes, flags);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  if (ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status = ihipHostMalloc(ptr, sizeBytes, flags);
  HIP_RETURN_DURATION(status, ReturnPtrValue(ptr));
}

hipError_t hipFree(void* ptr) {
  HIP_INIT_API(hipFree, ptr);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipFree(ptr));
}

hipError_t hipMemcpy_common(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                            hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();
  hip::Stream* hip_stream = nullptr;

  if (stream != nullptr && stream != hipStreamLegacy) {
    hip_stream = hip::getStream(stream);
  } else {
    hip_stream = hip::getNullStream();
  }

  if (hip_stream == nullptr) {
    return hipErrorInvalidValue;
  }
  return ihipMemcpy(dst, src, sizeBytes, kind, *hip_stream);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy, dst, src, sizeBytes, kind);
  HIP_RETURN_DURATION(hipMemcpy_common(dst, src, sizeBytes, kind));
}

hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy, dst, src, sizeBytes, kind);
  HIP_RETURN_DURATION(hipMemcpy_common(dst, src, sizeBytes, kind, getPerThreadDefaultStream()));
}

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream) {
  HIP_INIT_API(hipMemcpyWithStream, dst, src, sizeBytes, kind, stream);
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  // Since this API does a MemcpyAsync + sync, and sync during a graph capture
  // is not allowed, we raise an error here if user tries to use this during a
  // stream capture.
  if (hip::Stream::StreamCaptureOngoing(stream) == true) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN_DURATION(ihipMemcpy(dst, src, sizeBytes, kind, *hip_stream, false));
}

hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  HIP_INIT_API(hipMemPtrGetInfo, ptr, size);

  if (ptr == nullptr) {
    *size = 0;
    HIP_RETURN(hipSuccess);
  }

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);

  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *size = svmMem->getSize();

  HIP_RETURN(hipSuccess);
}

hipError_t hipHostFree(void* ptr) {
  HIP_INIT_API(hipHostFree, ptr);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  size_t offset = 0;
  amd::Memory* memory_object = getMemoryObject(ptr, offset);
  if (memory_object != nullptr) {
    if (memory_object->getSvmPtr() == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(ihipFree(ptr));
}

hipError_t ihipArrayDestroy(hipArray_t array) {
  if (array == nullptr) {
    return hipErrorInvalidValue;
  }
  {
    amd::ScopedLock lock(hipArraySetLock);
    if (hip::hipArraySet.find(array) == hip::hipArraySet.end()) {
      return hipErrorContextIsDestroyed;
    } else {
      hip::hipArraySet.erase(array);
    }
  }
  cl_mem memObj = reinterpret_cast<cl_mem>(array->data);
  if (is_valid(memObj) == false) {
    return hipErrorInvalidValue;
  }

  auto image = as_amd(memObj);
  // Wait on the device, associated with the current memory object during allocation
  g_devices[image->getUserData().deviceId]->SyncAllStreams();
  image->release();

  delete array;
  return hipSuccess;
}

hipError_t hipFreeArray(hipArray_t array) {
  HIP_INIT_API(hipFreeArray, array);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipArrayDestroy(array));
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  HIP_INIT_API(hipMemGetAddressRange, pbase, psize, dptr);

  // Since we are using SVM buffer DevicePtr and HostPtr is the same
  void* ptr = dptr;
  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);
  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorNotFound);
  }

  *pbase = svmMem->getSvmPtr();
  *psize = svmMem->getSize();

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  HIP_INIT_API(hipMemGetInfo, free, total);

  if (free == nullptr && total == nullptr) {
    HIP_RETURN(hipSuccess);
  }

  size_t freeMemory[2];
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  if (device == nullptr) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (!device->globalFreeMemory(freeMemory)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (free != nullptr) {
    *free = freeMemory[0] * Ki;
  }

  if (total != nullptr) {
    *total = device->info().globalMemSize_;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t ihipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth) {
  amd::Device* device = hip::getCurrentDevice()->devices()[0];

  if ((ptr == nullptr) || (pitch == nullptr)) {
    return hipErrorInvalidValue;
  }

  if ((width == 0) || (height == 0) || (depth == 0)) {
    *ptr = nullptr;
    return hipSuccess;
  }

  // avoid size_t overflow for pitch calculation
  if (width > (std::numeric_limits<size_t>::max() - device->info().imagePitchAlignment_)) {
    return hipErrorInvalidValue;
  }

  *pitch = amd::alignUp(width, device->info().imagePitchAlignment_);

  size_t sizeBytes = *pitch * height * depth;

  if (device->info().maxMemAllocSize_ < sizeBytes) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*hip::getCurrentDevice()->asContext(), 0, sizeBytes,
                                device->info().memBaseAddrAlign_);

  if (*ptr == nullptr) {
    return hipErrorOutOfMemory;
  }
  size_t offset = 0;  // this is ignored
  amd::Memory* memObj = getMemoryObject(*ptr, offset);
  memObj->getUserData().pitch_ = *pitch;
  memObj->getUserData().width_ = width;
  memObj->getUserData().height_ = height;
  memObj->getUserData().depth_ = depth;
  // saves the current device id so that it can be accessed later
  memObj->getUserData().deviceId = hip::getCurrentDevice()->deviceId();

  return hipSuccess;
}


hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  HIP_INIT_API(hipMallocPitch, ptr, pitch, width, height);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipMallocPitch(ptr, pitch, width, height, 1), ReturnPtrValue(ptr));
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  HIP_INIT_API(hipMalloc3D, pitchedDevPtr, extent);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  size_t pitch = 0;

  if (pitchedDevPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status =
      ihipMallocPitch(&pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);

  if (status == hipSuccess) {
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
  }

  HIP_RETURN(status, *pitchedDevPtr);
}

amd::Image* ihipImageCreate(const cl_channel_order channelOrder, const cl_channel_type channelType,
                            const cl_mem_object_type imageType, const size_t imageWidth,
                            const size_t imageHeight, const size_t imageDepth,
                            const size_t imageArraySize, const size_t imageRowPitch,
                            const size_t imageSlicePitch, const uint32_t numMipLevels,
                            const size_t offset, amd::Memory* buffer, hipError_t& status) {
  status = hipSuccess;
  const amd::Image::Format imageFormat({channelOrder, channelType});
  if (!imageFormat.isValid()) {
    LogPrintfError("Invalid Image format for channel Order:%u Type:%u", channelOrder, channelType);
    status = hipErrorInvalidValue;
    return nullptr;
  }

  amd::Context& context = *hip::getCurrentDevice()->asContext();
  if (!imageFormat.isSupported(context, imageType)) {
    LogPrintfError("Image type: %u not supported", imageType);
    status = hipErrorInvalidValue;
    return nullptr;
  }

  const std::vector<amd::Device*>& devices = context.devices();
  if (!devices[0]->info().imageSupport_) {
    LogPrintfError("Device: 0x%x does not support image", devices[0]);
    status = hipErrorNotSupported;
    return nullptr;
  }

  if (!amd::Image::validateDimensions(devices, imageType, imageWidth, imageHeight, imageDepth,
                                      imageArraySize)) {
    DevLogError("Image does not have valid dimensions");
    status = hipErrorInvalidValue;
    return nullptr;
  }

  if (numMipLevels > 0) {
    size_t max_dim = std::max(std::max(imageWidth, imageHeight), imageDepth);
    size_t mip_levels = 0;
    for (mip_levels = 0; max_dim > 0; max_dim >>= 1, mip_levels++);
    // empty for loop

    if (mip_levels < numMipLevels) {
      LogPrintfError("Invalid Mip Levels: %d", numMipLevels);
      status = hipErrorInvalidValue;
      return nullptr;
    }
  }

  // TODO validate the image descriptor.
  bool isExternalBuffer = buffer != nullptr ? buffer->isInterop() : false;
  amd::Image* image = nullptr;
  if (isExternalBuffer) {
    // We will create mipmap image view on top of external buffer which is from Vulkan or D3D
    // image. The buffer contains subchunks of tiled image, or it's just a linear image. Lower
    // level PAL will infer memory layout (offset, pitch, slice pitch, size) for each level
    // array in terms of tiling mode(linear or optimal), extent, format and mipmap levels.
    // Note that RocR will support mipmap in the future.
    switch (imageType) {
      case CL_MEM_OBJECT_IMAGE1D:
      case CL_MEM_OBJECT_IMAGE2D:
      case CL_MEM_OBJECT_IMAGE3D:
        image = new (buffer->getContext())
            amd::Image(*buffer->asBuffer(), imageType, CL_MEM_READ_WRITE, imageFormat, imageWidth,
                       (imageHeight == 0) ? 1 : imageHeight, (imageDepth == 0) ? 1 : imageDepth,
                       imageRowPitch, imageSlicePitch, numMipLevels, offset);
        break;
      default:
        LogPrintfError("Cannot create image of imageType: 0x%x for external buffer", imageType);
    }
  } else if (buffer != nullptr) {
    switch (imageType) {
      case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      case CL_MEM_OBJECT_IMAGE2D:
        image = new (buffer->getContext())
            amd::Image(*buffer->asBuffer(), imageType, CL_MEM_READ_WRITE, imageFormat,
                       (imageWidth == 0) ? 1 : imageWidth, (imageHeight == 0) ? 1 : imageHeight,
                       (imageDepth == 0) ? 1 : imageDepth, imageRowPitch, imageSlicePitch,
                       numMipLevels, offset);
        break;
      default:
        LogPrintfError("Cannot create image of imageType: 0x%x", imageType);
    }
  } else {
    switch (imageType) {
      case CL_MEM_OBJECT_IMAGE1D:
      case CL_MEM_OBJECT_IMAGE2D:
      case CL_MEM_OBJECT_IMAGE3D:
        image = new (context)
            amd::Image(context, imageType, CL_MEM_READ_WRITE, imageFormat, imageWidth,
                       (imageHeight == 0) ? 1 : imageHeight, (imageDepth == 0) ? 1 : imageDepth,
                       imageWidth * imageFormat.getElementSize(),               /* row pitch */
                       imageWidth * imageHeight * imageFormat.getElementSize(), /* slice pitch */
                       numMipLevels);
        break;
      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        image = new (context)
            amd::Image(context, imageType, CL_MEM_READ_WRITE, imageFormat, imageWidth,
                       imageArraySize, 1, /* image depth */
                       imageWidth * imageFormat.getElementSize(),
                       imageWidth * imageHeight * imageFormat.getElementSize(), numMipLevels);
        break;
      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        image = new (context)
            amd::Image(context, imageType, CL_MEM_READ_WRITE, imageFormat, imageWidth, imageHeight,
                       imageArraySize, imageWidth * imageFormat.getElementSize(),
                       imageWidth * imageHeight * imageFormat.getElementSize(), numMipLevels);
        break;
      default:
        LogPrintfError("Cannot create image of imageType: 0x%x", imageType);
    }
  }

  if (image == nullptr) {
    status = hipErrorOutOfMemory;
    return nullptr;
  }

  if (!image->create(nullptr)) {
    LogPrintfError("Cannot create image: 0x%x", image);
    status = hipErrorOutOfMemory;
    delete image;
    return nullptr;
  }
  // Save device ID image was creted on
  image->getUserData().deviceId = hip::getCurrentDevice()->deviceId();
  return image;
}

hipError_t ihipArrayCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray,
                           unsigned int numMipmapLevels) {
  if (!array || !pAllocateArray) {
    return hipErrorInvalidValue;
  }
  // NumChannels specifies the number of packed components per HIP array element; it may be 1, 2, or
  // 4;
  if ((pAllocateArray->NumChannels != 1) && (pAllocateArray->NumChannels != 2) &&
      (pAllocateArray->NumChannels != 4)) {
    return hipErrorInvalidValue;
  }
  unsigned int flags = hipArrayDefault | hipArrayLayered | hipArraySurfaceLoadStore |
                       hipArrayTextureGather;  // hipArrayCubemap isn't supported
  if (pAllocateArray->Flags & (~flags)) {
    return hipErrorInvalidValue;
  }
  if (pAllocateArray->Flags & hipArrayTextureGather) {
    // hipArrayTextureGather only works for 2D
    if (pAllocateArray->Width == 0 || pAllocateArray->Height == 0 || pAllocateArray->Depth > 0)
      return hipErrorInvalidValue;
  }

  const cl_channel_order channelOrder = hip::getCLChannelOrder(pAllocateArray->NumChannels, 0);
  const cl_channel_type channelType =
      hip::getCLChannelType(pAllocateArray->Format, hipReadModeElementType);
  const cl_mem_object_type imageType = hip::getCLMemObjectType(
      pAllocateArray->Width, pAllocateArray->Height, pAllocateArray->Depth, pAllocateArray->Flags);
  hipError_t status = hipSuccess;
  amd::Image* image = ihipImageCreate(channelOrder, channelType, imageType, pAllocateArray->Width,
                                      pAllocateArray->Height, pAllocateArray->Depth,
                                      // The number of layers is determined by the depth extent.
                                      pAllocateArray->Depth,       /* array size */
                                      0,                           /* row pitch */
                                      0,                           /* slice pitch */
                                      numMipmapLevels, 0, nullptr, /* buffer */
                                      status);

  if (image == nullptr) {
    return status;
  }

  cl_mem memObj = as_cl<amd::Memory>(image);
  *array = new hipArray{reinterpret_cast<void*>(memObj)};

  // It is UB to call hipGet*() on an array created via hipArrayCreate()/hipArray3DCreate().
  // This is due to hip not differentiating between runtime and driver types.
  // TODO change the hipArray struct in driver_types.h.
  (*array)->desc = hip::getChannelFormatDesc(pAllocateArray->NumChannels, pAllocateArray->Format);
  (*array)->width = pAllocateArray->Width;
  (*array)->height = pAllocateArray->Height;
  (*array)->depth = pAllocateArray->Depth;
  (*array)->Format = pAllocateArray->Format;
  (*array)->NumChannels = pAllocateArray->NumChannels;
  (*array)->flags = pAllocateArray->Flags;
  {
    amd::ScopedLock lock(hipArraySetLock);
    hip::hipArraySet.insert(*array);
  }
  return hipSuccess;
}

hipError_t hipArrayCreate(hipArray_t* array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(hipArrayCreate, array, pAllocateArray);
  if (pAllocateArray == nullptr) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_ARRAY3D_DESCRIPTOR desc = {
      pAllocateArray->Width,  pAllocateArray->Height,      0, /* Depth */
      pAllocateArray->Format, pAllocateArray->NumChannels, hipArrayDefault /* Flags */};

  HIP_RETURN(ihipArrayCreate(array, &desc, 0));
}


hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags) {
  HIP_INIT_API(hipMallocArray, array, desc, width, height, flags);
  if (array == nullptr || desc == nullptr) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_ARRAY3D_DESCRIPTOR allocateArray = {width,
                                          height,
                                          0, /* Depth */
                                          hip::getArrayFormat(*desc),
                                          hip::getNumChannels(*desc),
                                          flags};
  if (!hip::CheckArrayFormat(*desc)) {
    return hipErrorInvalidValue;
  }
  HIP_RETURN(ihipArrayCreate(array, &allocateArray, 0 /* numMipLevels */));
}

hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(hipArray3DCreate, array, pAllocateArray);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  if (pAllocateArray == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(ihipArrayCreate(array, pAllocateArray, 0 /* numMipLevels */));
}

hipError_t hipMalloc3DArray(hipArray_t* array, const hipChannelFormatDesc* desc, hipExtent extent,
                            unsigned int flags) {
  HIP_INIT_API(hipMalloc3DArray, array, desc, extent, flags);
  if (array == nullptr || desc == nullptr) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_ARRAY3D_DESCRIPTOR allocateArray = {extent.width,
                                          extent.height,
                                          extent.depth,
                                          hip::getArrayFormat(*desc),
                                          hip::getNumChannels(*desc),
                                          flags};
  if (!hip::CheckArrayFormat(*desc)) {
    return hipErrorInvalidValue;
  }

  HIP_RETURN(ihipArrayCreate(array, &allocateArray, 0));
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  HIP_INIT_API(hipHostGetFlags, flagsPtr, hostPtr);

  if (flagsPtr == nullptr || hostPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(hostPtr, offset);

  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // To match with Nvidia behaviour validate that hostPtr passed
  // was allocated using hipHostAlloc(), and not hipMalloc()
  if (!(svmMem->getMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Retrieve HIP memory flags
  *flagsPtr = svmMem->getUserData().flags;

  HIP_RETURN(hipSuccess);
}

hipError_t ihipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  if (hostPtr == nullptr || sizeBytes == 0 ||
      flags & ~(hipHostRegisterPortable | hipHostRegisterMapped | hipExtHostRegisterCoarseGrained |
                hipExtHostRegisterUncached)) {
    return hipErrorInvalidValue;
  } else {
    unsigned int memFlags = CL_MEM_USE_HOST_PTR | CL_MEM_SVM_ATOMICS;
    if (flags & hipExtHostRegisterUncached) {
      if (IS_WINDOWS) {
        return hipErrorInvalidValue;
      }
      memFlags |= ROCCLR_MEM_HSA_UNCACHED;
    }

    amd::Memory* mem =
        new (*hip::host_context) amd::Buffer(*hip::host_context, memFlags, sizeBytes);

    constexpr bool sysMemAlloc = false;
    constexpr bool skipAlloc = false;
    constexpr bool forceAlloc = true;
    if (!mem->create(hostPtr, sysMemAlloc, skipAlloc, forceAlloc)) {
      mem->release();
      LogPrintfError("Cannot create memory for size: %u with flags: %d", sizeBytes, flags);
      return hipErrorInvalidValue;
    }

    amd::MemObjMap::AddMemObj(hostPtr, mem);
    for (const auto& device : g_devices) {
      // Since the amd::Memory object is shared between all devices
      // it's fine to have multiple addresses mapped to it
      const device::Memory* devMem = mem->getDeviceMemory(*device->devices()[0]);
      void* vAddr = reinterpret_cast<void*>(devMem->virtualAddress());
      if ((hostPtr != vAddr) && (amd::MemObjMap::FindMemObj(vAddr) == nullptr)) {
        amd::MemObjMap::AddMemObj(vAddr, mem);
      }
    }


    if (mem != nullptr) {
      mem->getUserData().deviceId = hip::getCurrentDevice()->deviceId();
      // Save the HIP memory flags so that they can be accessed later
      mem->getUserData().flags = flags;
    }
    return hipSuccess;
  }
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipHostRegister, hostPtr, sizeBytes, flags);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipHostRegister(hostPtr, sizeBytes, flags));
}

hipError_t ihipHostUnregister(void* hostPtr) {
  if (hostPtr == nullptr) {
    return hipErrorInvalidValue;
  }
  size_t offset = 0;
  amd::Memory* mem = getMemoryObject(hostPtr, offset);

  if (mem != nullptr) {
    // Wait on the device, associated with the current memory object during allocation
    g_devices[mem->getUserData().deviceId]->SyncAllStreams();

    amd::MemObjMap::RemoveMemObj(hostPtr);
    for (const auto& device : g_devices) {
      const device::Memory* devMem = mem->getDeviceMemory(*device->devices()[0]);
      if (devMem != nullptr) {
        void* vAddr = reinterpret_cast<void*>(devMem->virtualAddress());
        if ((vAddr != hostPtr) && amd::MemObjMap::FindMemObj(vAddr)) {
          amd::MemObjMap::RemoveMemObj(vAddr);
        }
      }
    }
    mem->release();
    return hipSuccess;
  }

  LogPrintfError("Cannot unregister host_ptr: 0x%x", hostPtr);
  return hipErrorHostMemoryNotRegistered;
}


hipError_t hipHostUnregister(void* hostPtr) {
  HIP_INIT_API(hipHostUnregister, hostPtr);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipHostUnregister(hostPtr));
}

hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipHostAlloc, ptr, sizeBytes, flags);
  CHECK_STREAM_CAPTURE_SUPPORTED();

  if (ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (flags & ~(hipHostAllocPortable | hipHostAllocMapped | hipHostAllocWriteCombined |
                hipHostAllocUncached)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status = ihipHostMalloc(ptr, sizeBytes, flags);
  HIP_RETURN_DURATION(status, ReturnPtrValue(ptr));
}

hipError_t hipMemcpyAsync_common(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                                 hipStream_t stream) {
  STREAM_CAPTURE(hipMemcpyAsync, stream, dst, src, sizeBytes, kind);

  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  getStreamPerThread(stream);
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    return hipErrorInvalidValue;
  }
  return ihipMemcpy(dst, src, sizeBytes, kind, *hip_stream, true);
}

inline hipError_t ihipMemcpySymbol_validate(const void* symbol, size_t sizeBytes, size_t offset,
                                            size_t& sym_size, hipDeviceptr_t& device_ptr) {
  HIP_RETURN_ONFAIL(
      PlatformState::instance().getStatGlobalVar(symbol, ihipGetDevice(), &device_ptr, &sym_size));

  /* Size Check to make sure offset is correct */
  if ((offset + sizeBytes) > sym_size) {
    LogPrintfError("Trying to access out of bounds, offset: %u sizeBytes: %u sym_size: %u", offset,
                   sizeBytes, sym_size);
    HIP_RETURN(hipErrorInvalidValue);
  }

  device_ptr = reinterpret_cast<address>(device_ptr) + offset;
  return hipSuccess;
}

hipError_t hipMemcpyToSymbol_common(const void* symbol, const void* src, size_t sizeBytes,
                                    size_t offset, hipMemcpyKind kind,
                                    hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();

  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    HIP_RETURN(hipErrorInvalidMemcpyDirection);
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }

  /* Copy memory from source to destination address */
  return hipMemcpy_common(device_ptr, src, sizeBytes, kind, stream);
}

hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyToSymbol, symbol, src, sizeBytes, offset, kind);
  HIP_RETURN_DURATION(hipMemcpyToSymbol_common(symbol, src, sizeBytes, offset, kind));
}

hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyToSymbol, symbol, src, sizeBytes, offset, kind);
  HIP_RETURN_DURATION(
      hipMemcpyToSymbol_common(symbol, src, sizeBytes, offset, kind, getPerThreadDefaultStream()));
}

hipError_t hipMemcpyFromSymbol_common(void* dst, const void* symbol, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind,
                                      hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();

  if (kind != hipMemcpyDeviceToHost && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    HIP_RETURN(hipErrorInvalidMemcpyDirection);
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }

  /* Copy memory from source to destination address */
  return hipMemcpy_common(dst, device_ptr, sizeBytes, kind, stream);
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromSymbol, symbol, dst, sizeBytes, offset, kind);
  HIP_RETURN_DURATION(hipMemcpyFromSymbol_common(dst, symbol, sizeBytes, offset, kind));
}

hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromSymbol, symbol, dst, sizeBytes, offset, kind);
  HIP_RETURN_DURATION(hipMemcpyFromSymbol_common(dst, symbol, sizeBytes, offset, kind,
                                                 getPerThreadDefaultStream()));
}

hipError_t hipMemcpyToSymbolAsync_common(const void* symbol, const void* src, size_t sizeBytes,
                                         size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  STREAM_CAPTURE(hipMemcpyToSymbolAsync, stream, symbol, src, sizeBytes, offset, kind);

  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }

  return hipMemcpyAsync_common(device_ptr, src, sizeBytes, kind, stream);
}

hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyToSymbolAsync, symbol, src, sizeBytes, offset, kind, stream);
  HIP_RETURN_DURATION(hipMemcpyToSymbolAsync_common(symbol, src, sizeBytes, offset, kind, stream));
}

hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyToSymbolAsync, symbol, src, sizeBytes, offset, kind, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipMemcpyToSymbolAsync_common(symbol, src, sizeBytes, offset, kind, stream));
}

hipError_t hipMemcpyFromSymbolAsync_common(void* dst, const void* symbol, size_t sizeBytes,
                                           size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  STREAM_CAPTURE(hipMemcpyFromSymbolAsync, stream, dst, symbol, sizeBytes, offset, kind);

  if (kind != hipMemcpyDeviceToHost && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }

  /* Copy memory from source to destination address */
  return hipMemcpyAsync(dst, device_ptr, sizeBytes, kind, stream);
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyFromSymbolAsync, dst, symbol, sizeBytes, offset, kind, stream);
  HIP_RETURN_DURATION(
      hipMemcpyFromSymbolAsync_common(dst, symbol, sizeBytes, offset, kind, stream));
}

hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyFromSymbolAsync, dst, symbol, sizeBytes, offset, kind, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(
      hipMemcpyFromSymbolAsync_common(dst, symbol, sizeBytes, offset, kind, stream));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount) {
  HIP_INIT_API(hipMemcpyHtoD, dstDevice, srcHost, ByteCount);
  CHECK_STREAM_CAPTURING();
  hip::Stream* stream = hip::getStream(nullptr);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipMemcpy(dstDevice, srcHost, ByteCount, hipMemcpyHostToDevice, *stream));
}

hipError_t hipMemcpyDtoH(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoH, dstHost, srcDevice, ByteCount);
  CHECK_STREAM_CAPTURING();
  hip::Stream* stream = hip::getStream(nullptr);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipMemcpy(dstHost, srcDevice, ByteCount, hipMemcpyDeviceToHost, *stream));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoD, dstDevice, srcDevice, ByteCount);
  CHECK_STREAM_CAPTURING();
  hip::Stream* stream = hip::getStream(nullptr);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(
      ihipMemcpy(dstDevice, srcDevice, ByteCount, hipMemcpyDeviceToDevice, *stream));
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
  HIP_INIT_API(hipMemcpyAsync, dst, src, sizeBytes, kind, stream);
  HIP_RETURN_DURATION(hipMemcpyAsync_common(dst, src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyAsync, dst, src, sizeBytes, kind, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipMemcpyAsync_common(dst, src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyHtoDAsync, dstDevice, srcHost, ByteCount, stream);
  hipMemcpyKind kind = hipMemcpyHostToDevice;
  STREAM_CAPTURE(hipMemcpyHtoDAsync, stream, dstDevice, srcHost, ByteCount, kind);
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipMemcpy(dstDevice, srcHost, ByteCount, kind, *hip_stream, true));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount, stream);
  hipMemcpyKind kind = hipMemcpyDeviceToDevice;
  STREAM_CAPTURE(hipMemcpyDtoDAsync, stream, dstDevice, srcDevice, ByteCount, kind);
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipMemcpy(dstDevice, srcDevice, ByteCount, kind, *hip_stream, true));
}

hipError_t hipMemcpyDtoHAsync(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyDtoHAsync, dstHost, srcDevice, ByteCount, stream);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;
  STREAM_CAPTURE(hipMemcpyDtoHAsync, stream, dstHost, srcDevice, ByteCount, kind);
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipMemcpy(dstHost, srcDevice, ByteCount, kind, *hip_stream, true));
}

hipError_t ihipMemcpyAtoDCommand(amd::Command*& command, void* dstDevice, amd::Coord3D dstOrigin,
                                 amd::Image* srcImage, amd::Coord3D srcOrigin,
                                 amd::Coord3D copyRegion, amd::BufferRect srcRect,
                                 amd::BufferRect dstRect, hip::Stream* stream) {
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstDevice, dOffset);

  amd::CopyMemoryCommand* cpyMemCmd = new amd::CopyMemoryCommand(
      *stream, CL_COMMAND_COPY_IMAGE_TO_BUFFER, amd::Command::EventWaitList{}, *srcImage,
      *dstMemory, srcOrigin, dstOrigin, copyRegion, srcRect, dstRect);

  if (cpyMemCmd == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!cpyMemCmd->validatePeerMemory()) {
    delete cpyMemCmd;
    return hipErrorInvalidValue;
  }
  command = cpyMemCmd;
  return hipSuccess;
}

hipError_t ihipMemcpyDtoACommand(amd::Command*& command, amd::Image* dstImage,
                                 amd::Coord3D dstOrigin, void* srcDevice, amd::Coord3D srcOrigin,
                                 amd::Coord3D copyRegion, amd::BufferRect srcRect,
                                 amd::BufferRect dstRect, hip::Stream* stream) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcDevice, sOffset);

  amd::CopyMemoryCommand* cpyMemCmd = new amd::CopyMemoryCommand(
      *stream, CL_COMMAND_COPY_BUFFER_TO_IMAGE, amd::Command::EventWaitList{}, *srcMemory,
      *dstImage, srcOrigin, dstOrigin, copyRegion, srcRect, dstRect);

  if (cpyMemCmd == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!cpyMemCmd->validatePeerMemory()) {
    delete cpyMemCmd;
    return hipErrorInvalidValue;
  }
  command = cpyMemCmd;
  return hipSuccess;
}

hipError_t ihipMemcpyDtoDCommand(amd::Command*& command, void* dstDevice, void* srcDevice,
                                 amd::Coord3D copyRegion, amd::BufferRect srcRect,
                                 amd::BufferRect dstRect, hip::Stream* stream) {
  size_t srcOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcDevice, srcOffset);
  size_t dstOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstDevice, dstOffset);

  amd::Command::EventWaitList waitList;
  amd::CopyMemoryCommand* copyCommand;
  amd::Device* queueDevice = &stream->device();
  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);

  bool p2pcopy = false;
  // Check if the queue device doesn't match the device on any memory object.
  // And any of them are not host allocation.
  // Hence it's a P2P transfer, because the app has requested access to another GPU
  if ((srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) &&
      ((srcMemory->getContext().devices().size() == 1) &&
       (dstMemory->getContext().devices().size() == 1))) {
    copyCommand =
        new amd::CopyMemoryP2PCommand(*stream, CL_COMMAND_COPY_BUFFER_RECT, waitList, *srcMemory,
                                      *dstMemory, srcStart, dstStart, copyRegion, srcRect, dstRect);
    p2pcopy = true;
  } else {
    hip::Stream* pStream = stream;
    if ((srcMemory->GetDeviceById() == dstMemory->GetDeviceById()) &&
        (queueDevice != srcMemory->GetDeviceById())) {
      pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
      amd::Command* cmd = stream->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
      }
    } else if (srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) {
      // Scenarios such as DtoH where dst is pinned memory
      if ((queueDevice != srcMemory->GetDeviceById()) &&
          (dstMemory->getContext().devices().size() != 1)) {
        pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
        amd::Command* cmd = stream->getLastQueuedCommand(true);
        if (cmd != nullptr) {
          waitList.push_back(cmd);
        }
        // Scenarios such as HtoD where src is pinned memory
      } else if ((queueDevice != dstMemory->GetDeviceById()) &&
                 (srcMemory->getContext().devices().size() != 1)) {
        pStream = hip::getNullStream(dstMemory->GetDeviceById()->context());
        amd::Command* cmd = stream->getLastQueuedCommand(true);
        if (cmd != nullptr) {
          waitList.push_back(cmd);
        }
      }
    }
    copyCommand =
        new amd::CopyMemoryCommand(*pStream, CL_COMMAND_COPY_BUFFER_RECT, waitList, *srcMemory,
                                   *dstMemory, srcStart, dstStart, copyRegion, srcRect, dstRect);
  }

  if (copyCommand == nullptr) {
    return hipErrorOutOfMemory;
  }
  // Make sure runtime has valid memory for the command execution. P2P access
  // requires page table mapping on the current device to another GPU memory
  if ((p2pcopy && !static_cast<amd::CopyMemoryP2PCommand*>(copyCommand)->validateMemory()) ||
      (!p2pcopy && !copyCommand->validatePeerMemory())) {
    delete copyCommand;
    return hipErrorInvalidValue;
  }
  if (waitList.size() > 0) {
    waitList[0]->release();
  }
  command = copyCommand;
  return hipSuccess;
}

hipError_t ihipMemcpyDtoHCommand(amd::Command*& command, void* dstHost, amd::Coord3D dstOrigin,
                                 void* srcDevice, amd::Coord3D srcOrigin, amd::Coord3D copyRegion,
                                 amd::BufferRect srcRect, amd::BufferRect dstRect,
                                 hip::Stream* stream, bool isAsync = false) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcDevice, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstHost, dOffset);

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
  if (dstMemory) {
    amd::CopyMemoryCommand* copyCommand = new amd::CopyMemoryCommand(
        *stream, CL_COMMAND_COPY_BUFFER_RECT, amd::Command::EventWaitList{}, *srcMemory, *dstMemory,
        srcOrigin, dstOrigin, copyRegion, srcRect, dstRect, copyMetadata);
    if (copyCommand == nullptr) {
      return hipErrorOutOfMemory;
    }
    command = copyCommand;
  } else {
    amd::Command::EventWaitList waitList;
    auto* pStream = hip::getNullStream(srcMemory->GetDeviceById()->context());
    if (stream->DeviceId() != srcMemory->getUserData().deviceId) {
      amd::Command* cmd = pStream->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
      }
    }

    amd::ReadMemoryCommand* readCommand =
        new amd::ReadMemoryCommand(*stream, CL_COMMAND_READ_BUFFER_RECT, waitList, *srcMemory,
                                   srcStart, copyRegion, dstHost, srcRect, dstRect, copyMetadata);
    if (readCommand == nullptr) {
      return hipErrorOutOfMemory;
    }

    if (!readCommand->validatePeerMemory()) {
      delete readCommand;
      return hipErrorInvalidValue;
    }
    command = readCommand;

    if (!waitList.empty()) {
      waitList[0]->release();
    }
  }

  return hipSuccess;
}

hipError_t ihipMemcpyHtoDCommand(amd::Command*& command, void* dstDevice, amd::Coord3D dstOrigin,
                                 const void* srcHost, amd::Coord3D srcOrigin,
                                 amd::Coord3D copyRegion, amd::BufferRect srcRect,
                                 amd::BufferRect dstRect, hip::Stream* stream,
                                 bool isAsync = false) {
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstDevice, dOffset);
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcHost, sOffset);

  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
  if (srcMemory) {
    amd::CopyMemoryCommand* copyCommand = new amd::CopyMemoryCommand(
        *stream, CL_COMMAND_COPY_BUFFER_RECT, amd::Command::EventWaitList{}, *srcMemory, *dstMemory,
        srcOrigin, dstOrigin, copyRegion, srcRect, dstRect, copyMetadata);
    if (copyCommand == nullptr) {
      return hipErrorOutOfMemory;
    }
    command = copyCommand;
  } else {
    amd::WriteMemoryCommand* writeCommand = new amd::WriteMemoryCommand(
        *stream, CL_COMMAND_WRITE_BUFFER_RECT, amd::Command::EventWaitList{}, *dstMemory, dstStart,
        copyRegion, srcHost, dstRect, srcRect, copyMetadata);
    if (writeCommand == nullptr) {
      return hipErrorOutOfMemory;
    }

    if (!writeCommand->validatePeerMemory()) {
      delete writeCommand;
      return hipErrorInvalidValue;
    }
    command = writeCommand;
  }

  return hipSuccess;
}

hipError_t ihipMemcpyHtoH(void* dstHost, const void* srcHost, amd::Coord3D copyRegion,
                          amd::BufferRect srcRect, amd::BufferRect dstRect, hip::Stream* stream) {
  if (stream) {
    stream->finish();
  }

  for (size_t slice = 0; slice < copyRegion[2]; slice++) {
    for (size_t row = 0; row < copyRegion[1]; row++) {
      const void* srcRow = static_cast<const char*>(srcHost) + srcRect.start_ +
                           row * srcRect.rowPitch_ + slice * srcRect.slicePitch_;
      void* dstRow = static_cast<char*>(dstHost) + dstRect.start_ + row * dstRect.rowPitch_ +
                     slice * dstRect.slicePitch_;
      std::memcpy(dstRow, srcRow, copyRegion[0]);
    }
  }

  return hipSuccess;
}

hipError_t ihipMemcpyAtoACommand(amd::Command*& command, amd::Image* dstImage,
                                 amd::Coord3D dstOrigin, amd::Image* srcImage,
                                 amd::Coord3D srcOrigin, amd::Coord3D copyRegion,
                                 hip::Stream* stream) {
  amd::CopyMemoryCommand* cpyMemCmd =
      new amd::CopyMemoryCommand(*stream, CL_COMMAND_COPY_IMAGE, amd::Command::EventWaitList{},
                                 *srcImage, *dstImage, srcOrigin, dstOrigin, copyRegion);

  if (cpyMemCmd == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!cpyMemCmd->validatePeerMemory()) {
    delete cpyMemCmd;
    return hipErrorInvalidValue;
  }
  command = cpyMemCmd;
  return hipSuccess;
}

size_t ihipGetbufferStart(const size_t* bufferOrigin, const size_t* region, size_t bufferRowPitch,
                          size_t bufferSlicePitch) {
  size_t rowPitch_ = (bufferRowPitch != 0) ? bufferRowPitch : region[0];
  // Find the buffer's slice pitch
  size_t slicePitch_ = (bufferSlicePitch != 0) ? bufferSlicePitch : rowPitch_ * region[1];
  // Find the region start offset
  // For the layered 1D image, bufferOrigin[2] is always 0; for 1D image, bufferOrigin[1] is
  // always 0. So the logic still holds true for all cases.
  return bufferOrigin[2] * slicePitch_ + bufferOrigin[1] * rowPitch_ + bufferOrigin[0];
}

hipError_t ihipMemcpyHtoACommand(amd::Command*& command, amd::Image* dstImage,
                                 amd::Coord3D dstOrigin, const void* srcHost,
                                 amd::Coord3D srcOrigin, amd::Coord3D copyRegion,
                                 size_t srcRowPitch, size_t srcSlicePitch, hip::Stream* stream,
                                 bool isAsync = false) {
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcHost, sOffset);
  size_t start = ihipGetbufferStart(static_cast<size_t*>(srcOrigin),
                                    static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch);

  amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
  if (srcMemory) {
    amd::CopyMemoryCommand* copyCommand = new amd::CopyMemoryCommand(
        *stream, CL_COMMAND_COPY_BUFFER_TO_IMAGE, amd::Command::EventWaitList{}, *srcMemory,
        *dstImage, srcOrigin, dstOrigin, copyRegion, copyMetadata);
    if (copyCommand == nullptr) {
      return hipErrorOutOfMemory;
    }
    command = copyCommand;
  } else {
    hip::Stream* pStream = stream;
    amd::Device* queueDevice = &stream->device();
    amd::Command::EventWaitList waitList;
    if (queueDevice != dstImage->GetDeviceById()) {
      pStream = hip::getNullStream(dstImage->GetDeviceById()->context());
      amd::Command* cmd = stream->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
      }
    }

    amd::WriteMemoryCommand* writeMemCmd = new amd::WriteMemoryCommand(
        *pStream, CL_COMMAND_WRITE_IMAGE, waitList, *dstImage, dstOrigin, copyRegion,
        static_cast<const char*>(srcHost) + start, srcRowPitch, srcSlicePitch, copyMetadata);
    if (writeMemCmd == nullptr) {
      return hipErrorOutOfMemory;
    }

    if (!writeMemCmd->validatePeerMemory()) {
      delete writeMemCmd;
      return hipErrorInvalidValue;
    }
    command = writeMemCmd;

    if (!waitList.empty()) {
      waitList[0]->release();
    }
  }

  return hipSuccess;
}

hipError_t ihipMemcpyAtoHCommand(amd::Command*& command, void* dstHost, amd::Coord3D dstOrigin,
                                 amd::Image* srcImage, amd::Coord3D srcOrigin,
                                 amd::Coord3D copyRegion, size_t dstRowPitch, size_t dstSlicePitch,
                                 hip::Stream* stream, bool isAsync = false) {
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstHost, dOffset);
  size_t start = ihipGetbufferStart(static_cast<size_t*>(dstOrigin),
                                    static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch);

  amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
  if (dstMemory) {
    amd::CopyMemoryCommand* copyCommand = new amd::CopyMemoryCommand(
        *stream, CL_COMMAND_COPY_IMAGE_TO_BUFFER, amd::Command::EventWaitList{}, *srcImage,
        *dstMemory, srcOrigin, dstOrigin, copyRegion, copyMetadata);
    if (copyCommand == nullptr) {
      return hipErrorOutOfMemory;
    }
    command = copyCommand;
  } else {
    hip::Stream* pStream = stream;
    amd::Device* queueDevice = &stream->device();
    amd::Command::EventWaitList waitList;
    if (queueDevice != srcImage->GetDeviceById()) {
      pStream = hip::getNullStream(srcImage->GetDeviceById()->context());
      amd::Command* cmd = stream->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
      }
    }

    amd::ReadMemoryCommand* readMemCmd = new amd::ReadMemoryCommand(
        *pStream, CL_COMMAND_READ_IMAGE, waitList, *srcImage, srcOrigin, copyRegion,
        static_cast<char*>(dstHost) + start, dstRowPitch, dstSlicePitch, copyMetadata);

    if (readMemCmd == nullptr) {
      return hipErrorOutOfMemory;
    }

    if (!readMemCmd->validatePeerMemory()) {
      delete readMemCmd;
      return hipErrorInvalidValue;
    }
    command = readMemCmd;

    if (!waitList.empty()) {
      waitList[0]->release();
    }
  }

  return hipSuccess;
}

void ihipCopyMemParamSet(const HIP_MEMCPY3D* pCopy, hipMemoryType& srcMemType,
                         hipMemoryType& dstMemType) {
  size_t offset = 0;
  // If {src/dst}MemoryType is hipMemoryTypeUnified, {src/dst}Device and {src/dst}Pitch
  // specify the (unified virtual address space)
  // base address of the source data and the bytes per row to apply. {src/dst}Array is ignored.
  hipMemoryType srcMemoryType = pCopy->srcMemoryType;
  if (srcMemoryType == hipMemoryTypeUnified) {
    amd::Memory* memObj = getMemoryObject(pCopy->srcDevice, offset);
    srcMemoryType = getMemoryType(memObj);
    if (memObj == nullptr) {
      const_cast<HIP_MEMCPY3D*>(pCopy)->srcXInBytes += offset;
    }

    if (srcMemoryType == hipMemoryTypeHost) {
      // {src/dst}Host may be unitialized. Copy over {src/dst}Device into it if we
      //  detect system memory.
      const_cast<HIP_MEMCPY3D*>(pCopy)->srcHost = pCopy->srcDevice;
    }
    // We don't need detect memory type again for hipMemoryTypeUnified
    const_cast<HIP_MEMCPY3D*>(pCopy)->srcMemoryType = srcMemoryType;
  }
  offset = 0;
  hipMemoryType dstMemoryType = pCopy->dstMemoryType;
  if (dstMemoryType == hipMemoryTypeUnified) {
    amd::Memory* memObj = getMemoryObject(pCopy->dstDevice, offset);
    dstMemoryType = getMemoryType(memObj);
    if (memObj == nullptr) {
      const_cast<HIP_MEMCPY3D*>(pCopy)->dstXInBytes += offset;
    }

    if (dstMemoryType == hipMemoryTypeHost) {
      const_cast<HIP_MEMCPY3D*>(pCopy)->dstHost = pCopy->dstDevice;
    }
    // We don't need detect memory type again for hipMemoryTypeUnified
    const_cast<HIP_MEMCPY3D*>(pCopy)->dstMemoryType = dstMemoryType;
  }
  offset = 0;
  // If {src/dst}MemoryType is hipMemoryTypeHost, check if the memory was prepinned.
  // In that case upgrade the copy type to hipMemoryTypeDevice to avoid extra pinning.
  if (srcMemoryType == hipMemoryTypeHost) {
    srcMemoryType =
        getMemoryObject(pCopy->srcHost, offset) ? hipMemoryTypeDevice : hipMemoryTypeHost;

    if (srcMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(pCopy)->srcDevice = const_cast<void*>(pCopy->srcHost);
    }
  }
  offset = 0;
  if (dstMemoryType == hipMemoryTypeHost) {
    dstMemoryType =
        getMemoryObject(pCopy->dstHost, offset) ? hipMemoryTypeDevice : hipMemoryTypeHost;

    if (dstMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(pCopy)->dstDevice = const_cast<void*>(pCopy->dstHost);
    }
  }
  srcMemType = srcMemoryType;
  dstMemType = dstMemoryType;
}

hipError_t validateImageObject(hipArray_t array, amd::Coord3D& origin, amd::Coord3D& copyRegion,
                               amd::BufferRect* rect, amd::Image*& image) {
  if (array == nullptr) {
    return hipErrorInvalidValue;
  }

  cl_mem memObj = reinterpret_cast<cl_mem>(array->data);
  if (!is_valid(memObj)) {
    return hipErrorInvalidValue;
  }

  image = as_amd(memObj)->asImage();
  if (!image->validateRegion(origin, copyRegion)) {
    return hipErrorInvalidValue;
  }

  if (!rect->create(static_cast<size_t*>(origin), static_cast<size_t*>(copyRegion),
                    image->getRowPitch(), image->getSlicePitch())) {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

hipError_t validateMemoryObject(const void* ptr, bool isDeviceMemory, amd::Coord3D& origin,
                                amd::Coord3D& copyRegion, size_t rowPitch, size_t slicePitch,
                                amd::BufferRect& rect) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!rect.create(static_cast<size_t*>(origin), static_cast<size_t*>(copyRegion), rowPitch,
                   slicePitch)) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::Memory* memory = nullptr;
  memory = getMemoryObject(ptr, offset);
  if (memory == nullptr && isDeviceMemory) {
    return hipErrorInvalidValue;
  }

  amd::Coord3D start(rect.start_, 0, 0);
  amd::Coord3D size(rect.end_, 1, 1);
  if (memory && !memory->validateRegion(start, size)) {
    return hipErrorInvalidValue;
  }

  rect.start_ += offset;

  origin.c[0] = rect.offset(0, 0, 0);  // Get the physical offset of the logic origin
  origin.c[1] = origin.c[2] = 0;

  return hipSuccess;
}

hipError_t ihipDrvMemcpy3D_validate(const HIP_MEMCPY3D* pCopy, amd::Coord3D& srcOrigin,
                                    amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                    amd::BufferRect* outSrcRect, amd::BufferRect* outDstRect,
                                    amd::Image** outSrcImage, amd::Image** outDstImage) {
  hipError_t status = hipSuccess;

  if (copyRegion.c[0] * copyRegion.c[1] * copyRegion.c[2] <= 0) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  amd::Image* srcImage = nullptr;
  amd::Image* dstImage = nullptr;
  amd::Coord3D tempCopyRegion = copyRegion;

  if (pCopy->srcMemoryType == hipMemoryTypeArray || pCopy->dstMemoryType == hipMemoryTypeArray) {
    void* arr = pCopy->srcMemoryType == hipMemoryTypeArray ? pCopy->srcArray : pCopy->dstArray;
    if (arr == nullptr) {
      return hipErrorInvalidValue;
    }
    auto element_size = hip::getElementSize((hipArray_const_t)arr);
    copyRegion.c[0] /= element_size;
    tempCopyRegion.c[0] /= element_size;

    if (pCopy->srcMemoryType == hipMemoryTypeArray) {
      static_cast<size_t*>(srcOrigin)[0] /= element_size;
      if (hip::isLayered1D(pCopy->srcArray)) {
        tempCopyRegion.c[1] = 1;
      }
    }

    if (pCopy->dstMemoryType == hipMemoryTypeArray) {
      static_cast<size_t*>(dstOrigin)[0] /= element_size;
      if (hip::isLayered1D(pCopy->dstArray)) {
        tempCopyRegion.c[1] = 1;
      }
    }
  }

  if (pCopy->srcMemoryType == hipMemoryTypeArray) {
    status = validateImageObject(pCopy->srcArray, srcOrigin, tempCopyRegion, &srcRect, srcImage);
  } else if (pCopy->srcMemoryType == hipMemoryTypeDevice) {
    status = validateMemoryObject(pCopy->srcDevice, true, srcOrigin, tempCopyRegion,
                                  pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, srcRect);
  } else if (pCopy->srcMemoryType == hipMemoryTypeHost) {
    status = validateMemoryObject(pCopy->srcHost, false, srcOrigin, tempCopyRegion, pCopy->srcPitch,
                                  pCopy->srcPitch * pCopy->srcHeight, srcRect);
  }

  if (status != hipSuccess) {
    return status;
  }

  if (pCopy->dstMemoryType == hipMemoryTypeArray) {
    status = validateImageObject(pCopy->dstArray, dstOrigin, tempCopyRegion, &dstRect, dstImage);
  } else if (pCopy->dstMemoryType == hipMemoryTypeDevice) {
    status = validateMemoryObject(pCopy->dstDevice, true, dstOrigin, tempCopyRegion,
                                  pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, dstRect);
  } else if (pCopy->dstMemoryType == hipMemoryTypeHost) {
    status = validateMemoryObject(pCopy->dstHost, false, dstOrigin, tempCopyRegion, pCopy->dstPitch,
                                  pCopy->dstPitch * pCopy->dstHeight, dstRect);
  }

  if (status != hipSuccess) {
    return status;
  }

  *outSrcRect = srcRect;
  *outDstRect = dstRect;
  *outSrcImage = srcImage;
  *outDstImage = dstImage;

  return hipSuccess;
}

hipError_t ihipDrvMemcpy3D_validate(const HIP_MEMCPY3D* pCopy) {
  amd::Coord3D srcOrigin = {pCopy->srcXInBytes, pCopy->srcY, pCopy->srcZ};
  amd::Coord3D dstOrigin = {pCopy->dstXInBytes, pCopy->dstY, pCopy->dstZ};
  amd::Coord3D copyRegion = {pCopy->WidthInBytes, pCopy->Height, pCopy->Depth};
  amd::BufferRect srcRect, dstRect;
  amd::Image *srcImage, *dstImage;
  return ihipDrvMemcpy3D_validate(pCopy, srcOrigin, dstOrigin, copyRegion, &srcRect, &dstRect,
                                  &srcImage, &dstImage);
}

hipError_t ihipGetMemcpyParam3DCommand(amd::Command*& command, const HIP_MEMCPY3D* pCopy,
                                       hip::Stream* stream) {
  if (pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) {
    return hipSuccess;
  }

  hipMemoryType srcMemoryType;
  hipMemoryType dstMemoryType;
  ihipCopyMemParamSet(pCopy, srcMemoryType, dstMemoryType);

  amd::Coord3D srcOrigin = {pCopy->srcXInBytes, pCopy->srcY, pCopy->srcZ};
  amd::Coord3D dstOrigin = {pCopy->dstXInBytes, pCopy->dstY, pCopy->dstZ};
  amd::Coord3D copyRegion = {pCopy->WidthInBytes, pCopy->Height, pCopy->Depth};
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  amd::Image* srcImage = nullptr;
  amd::Image* dstImage = nullptr;

  auto status = ihipDrvMemcpy3D_validate(pCopy, srcOrigin, dstOrigin, copyRegion, &srcRect,
                                         &dstRect, &srcImage, &dstImage);

  if (status != hipSuccess) {
    return status;
  }

  if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Host to Device.
    return ihipMemcpyHtoDCommand(command, pCopy->dstDevice, dstOrigin, pCopy->srcHost, srcOrigin,
                                 copyRegion, srcRect, dstRect, stream);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeHost)) {
    // Device to Host.
    return ihipMemcpyDtoHCommand(command, pCopy->dstHost, dstOrigin, pCopy->srcDevice, srcOrigin,
                                 copyRegion, srcRect, dstRect, stream);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Device to Device.
    return ihipMemcpyDtoDCommand(command, pCopy->dstDevice, pCopy->srcDevice, copyRegion, srcRect,
                                 dstRect, stream);
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeArray)) {
    // Host to Image.
    return ihipMemcpyHtoACommand(command, dstImage, dstOrigin, pCopy->srcHost, srcOrigin,
                                 copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight,
                                 stream);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeHost)) {
    // Image to Host.
    return ihipMemcpyAtoHCommand(command, pCopy->dstHost, dstOrigin, srcImage, srcOrigin,
                                 copyRegion, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight,
                                 stream);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeArray)) {
    // Device to Image.
    return ihipMemcpyDtoACommand(command, dstImage, dstOrigin, pCopy->srcDevice,
                                 {srcRect.start_, 0, 0}, copyRegion, srcRect, dstRect, stream);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Image to Device.
    return ihipMemcpyAtoDCommand(command, pCopy->dstDevice, {dstRect.start_, 0, 0}, srcImage,
                                 srcOrigin, copyRegion, srcRect, dstRect, stream);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeArray)) {
    // Image to Image.
    return ihipMemcpyAtoACommand(command, dstImage, dstOrigin, srcImage, srcOrigin, copyRegion,
                                 stream);
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeHost)) {
    // Host to Host
    return ihipMemcpyHtoH(pCopy->dstHost, pCopy->srcHost, copyRegion, srcRect, dstRect, stream);
  } else {
    ShouldNotReachHere();
  }

  return hipSuccess;
}

inline hipError_t ihipMemcpyCmdEnqueue(amd::Command* command, bool isAsync = false,
                                       hip::Stream* stream = nullptr) {
  hipError_t status = hipSuccess;
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  if (!isAsync) {
    command->queue()->finishCommand(command);
  } else if (stream != nullptr) {
    auto* newQueue = command->queue();
    if (newQueue != stream) {
      amd::Command::EventWaitList waitList;
      amd::Command* cmd = newQueue->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
        amd::Command* depdentMarker = new amd::Marker(*stream, true, waitList);
        if (depdentMarker != nullptr) {
          depdentMarker->enqueue();
          depdentMarker->release();
        }
        cmd->release();
      }
    }
  }
  command->release();
  return status;
}

hipError_t ihipMemcpyParam3D(const HIP_MEMCPY3D* pCopy, hipStream_t stream, bool isAsync = false) {
  hipError_t status;
  if (pCopy == nullptr) {
    return hipErrorInvalidValue;
  }
  getStreamPerThread(stream);
  hipMemoryType srcMemoryType;
  hipMemoryType dstMemoryType;
  ihipCopyMemParamSet(pCopy, srcMemoryType, dstMemoryType);

  hip::Stream* hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    return hipErrorInvalidValue;
  }

  amd::Command* command = nullptr;
  status = ihipGetMemcpyParam3DCommand(command, pCopy, hip_stream);
  if (command != nullptr) {
    if (status != hipSuccess) {
      return status;
    }
    // Transfers from device memory to pageable host memory and transfers from any
    // host memory to any host memory are synchronous with respect to the host.
    // Device to Device copies do not need to host side synchronization.
    if (dstMemoryType == hipMemoryTypeHost || ((pCopy->srcMemoryType == hipMemoryTypeHost) &&
                                               (pCopy->dstMemoryType == hipMemoryTypeHost))) {
      isAsync = false;
    } else if ((pCopy->srcMemoryType == hipMemoryTypeDevice) &&
               (pCopy->dstMemoryType == hipMemoryTypeDevice)) {
      // Device to Device copies dont need to wait for host synchronization
      isAsync = true;
    }
    return ihipMemcpyCmdEnqueue(command, isAsync, hip_stream);
  }

  return status;
}

hipError_t ihipMemcpyParam2D(const hip_Memcpy2D* pCopy, hipStream_t stream, bool isAsync = false) {
  HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*pCopy);

  return ihipMemcpyParam3D(&desc, stream, isAsync);
}

hipError_t ihipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind, hipStream_t stream,
                        bool isAsync = false) {
  hip_Memcpy2D desc = {};
  if ((width == 0) || (height == 0)) {
    return hipSuccess;
  }
  if ((width > dpitch) || (width > spitch)) {
    return hipErrorInvalidPitchValue;
  }

  desc.srcXInBytes = 0;
  desc.srcY = 0;
  desc.srcMemoryType = std::get<0>(hip::getMemoryType(kind));
  desc.srcHost = src;
  desc.srcDevice = const_cast<void*>(src);
  desc.srcArray = nullptr;  // Ignored.
  desc.srcPitch = spitch;

  desc.dstXInBytes = 0;
  desc.dstY = 0;
  desc.dstMemoryType = std::get<1>(hip::getMemoryType(kind));
  desc.dstHost = dst;
  desc.dstDevice = dst;
  desc.dstArray = nullptr;  // Ignored.
  desc.dstPitch = dpitch;

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  HIP_INIT_API(hipMemcpyParam2D, pCopy);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpyParam2D(pCopy, nullptr));
}

hipError_t hipMemcpy2DValidateParams(hipMemcpyKind kind, hipStream_t stream = nullptr) {
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }

  getStreamPerThread(stream);

  return hipSuccess;
}

hipError_t hipMemcpy2DValidateBuffer(const void* buf, size_t pitch, size_t width) {
  if (buf == nullptr) {
    return hipErrorInvalidValue;
  }

  if (pitch == 0 || pitch < width) {
    return hipErrorInvalidPitchValue;
  }

  return hipSuccess;
}

hipError_t hipMemcpy2DValidateArray(hipArray_const_t arr, size_t wOffset, size_t hOffset,
                                    size_t width, size_t height) {
  if (arr == nullptr) {
    return hipErrorInvalidHandle;
  }

  int FormatSize = hip::getElementSize(arr);
  if ((width + wOffset) > (arr->width * FormatSize)) {
    return hipErrorInvalidValue;
  }
  if (arr->height == 0) {  // 1D hipArray
    if (height + hOffset > 1) {
      return hipErrorInvalidValue;
    }
  } else if ((height + hOffset) > (arr->height)) {  // 2D hipArray
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

hipError_t hipMemcpy2D_common(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, hipMemcpyKind kind,
                              hipStream_t stream = nullptr, bool isAsync = false) {
  hipError_t validateParams = hipSuccess, validateSrc = hipSuccess, validateDst = hipSuccess;
  if ((validateParams = hipMemcpy2DValidateParams(kind, stream)) != hipSuccess) {
    return validateParams;
  }
  return ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind, stream, isAsync);
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2D, dst, dpitch, src, spitch, width, height, kind);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(hipMemcpy2D_common(dst, dpitch, src, spitch, width, height, kind));
}

hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2D, dst, dpitch, src, spitch, width, height, kind);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(hipMemcpy2D_common(dst, dpitch, src, spitch, width, height, kind,
                                         getPerThreadDefaultStream()));
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DAsync, dst, dpitch, src, spitch, width, height, kind, stream);
  STREAM_CAPTURE(hipMemcpy2DAsync, stream, dst, dpitch, src, spitch, width, height, kind);
  HIP_RETURN_DURATION(
      hipMemcpy2D_common(dst, dpitch, src, spitch, width, height, kind, stream, true));
}

hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DAsync, dst, dpitch, src, spitch, width, height, kind, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  STREAM_CAPTURE(hipMemcpy2DAsync, stream, dst, dpitch, src, spitch, width, height, kind);
  HIP_RETURN_DURATION(
      hipMemcpy2D_common(dst, dpitch, src, spitch, width, height, kind, stream, true));
}

hipError_t ihipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                               size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                               hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = 0;
  desc.srcY = 0;
  desc.srcMemoryType = std::get<0>(hip::getMemoryType(kind));
  desc.srcHost = const_cast<void*>(src);
  desc.srcDevice = const_cast<void*>(src);
  desc.srcArray = nullptr;
  desc.srcPitch = spitch;

  desc.dstXInBytes = wOffset;
  desc.dstY = hOffset;
  desc.dstMemoryType = hipMemoryTypeArray;
  desc.dstHost = nullptr;
  desc.dstDevice = nullptr;
  desc.dstArray = dst;
  desc.dstPitch = 0;  // Ignored.

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpy2DToArray_common(hipArray_t dst, size_t wOffset, size_t hOffset,
                                     const void* src, size_t spitch, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream = nullptr,
                                     bool isAsync = false) {
  hipError_t validateParams = hipSuccess, validateSrc = hipSuccess, validateDst = hipSuccess;
  if ((validateParams = hipMemcpy2DValidateParams(kind, stream)) != hipSuccess) {
    return validateParams;
  }
  if ((validateSrc = hipMemcpy2DValidateBuffer(src, spitch, width)) != hipSuccess) {
    return validateSrc;
  }
  if ((validateDst = hipMemcpy2DValidateArray(dst, wOffset, hOffset, width, height)) !=
      hipSuccess) {
    return validateDst;
  }
  return ihipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind, stream,
                             isAsync);
}

hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DToArray, dst, wOffset, hOffset, src, spitch, width, height, kind);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(
      hipMemcpy2DToArray_common(dst, wOffset, hOffset, src, spitch, width, height, kind));
}

hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DToArray, dst, wOffset, hOffset, src, spitch, width, height, kind);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(hipMemcpy2DToArray_common(dst, wOffset, hOffset, src, spitch, width, height,
                                                kind, getPerThreadDefaultStream()));
}

hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyToArray, dst, wOffset, hOffset, src, count, kind);
  CHECK_STREAM_CAPTURING();
  if (dst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const size_t arrayHeight = (dst->height != 0) ? dst->height : 1;
  const size_t witdthInBytes = count / arrayHeight;

  const size_t height = (count / dst->width) / hip::getElementSize(dst);

  HIP_RETURN_DURATION(ihipMemcpy2DToArray(dst, wOffset, hOffset, src, 0 /* spitch */, witdthInBytes,
                                          height, kind, nullptr));
}

hipError_t ihipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc,
                                 size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind,
                                 hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = wOffsetSrc;
  desc.srcY = hOffsetSrc;
  desc.srcMemoryType = hipMemoryTypeArray;
  desc.srcHost = nullptr;
  desc.srcDevice = nullptr;
  desc.srcArray = const_cast<hipArray_t>(src);
  desc.srcPitch = 0;  // Ignored.

  desc.dstXInBytes = 0;
  desc.dstY = 0;
  desc.dstMemoryType = std::get<1>(hip::getMemoryType(kind));
  desc.dstHost = dst;
  desc.dstDevice = dst;
  desc.dstArray = nullptr;
  desc.dstPitch = dpitch;

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpyFromArray_common(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                     size_t hOffset, size_t count, hipMemcpyKind kind,
                                     hipStream_t stream) {
  CHECK_STREAM_CAPTURING();
  if (src == nullptr) {
    return hipErrorInvalidValue;
  }

  const size_t arrayHeight = (src->height != 0) ? src->height : 1;
  const size_t witdthInBytes = count / arrayHeight;

  const size_t height = (count / src->width) / hip::getElementSize(src);

  return ihipMemcpy2DFromArray(dst, 0 /* dpitch */, src, wOffsetSrc, hOffset, witdthInBytes, height,
                               kind, stream);
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromArray, dst, src, wOffsetSrc, hOffset, count, kind);
  HIP_RETURN_DURATION(
      hipMemcpyFromArray_common(dst, src, wOffsetSrc, hOffset, count, kind, nullptr));
}

hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromArray, dst, src, wOffsetSrc, hOffset, count, kind);
  HIP_RETURN_DURATION(hipMemcpyFromArray_common(dst, src, wOffsetSrc, hOffset, count, kind,
                                                getPerThreadDefaultStream()));
}

hipError_t ihipMemcpy3D_validate(const hipMemcpy3DParms* p) {
  // Passing more than one non-zero source or destination will cause hipMemcpy3D() to
  // return an error.
  if (p == nullptr || ((p->srcArray != nullptr) && (p->srcPtr.ptr != nullptr)) ||
      ((p->dstArray != nullptr) && (p->dstPtr.ptr != nullptr))) {
    return hipErrorInvalidValue;
  }
  // The struct passed to hipMemcpy3D() must specify one of srcArray or srcPtr and one of dstArray
  // or dstPtr.
  if (((p->srcArray == nullptr) && (p->srcPtr.ptr == nullptr)) ||
      ((p->dstArray == nullptr) && (p->dstPtr.ptr == nullptr))) {
    return hipErrorInvalidValue;
  }

  // If the source and destination are both arrays, hipMemcpy3D() will return an error if they do
  // not have the same element size.
  if (((p->srcArray != nullptr) && (p->dstArray != nullptr)) &&
      (hip::getElementSize(p->srcArray) != hip::getElementSize(p->dstArray))) {
    return hipErrorInvalidValue;
  }

  // Pitch should not be less than width for both src and dst.
  if (p->srcPtr.pitch < p->srcPtr.xsize || p->dstPtr.pitch < p->dstPtr.xsize) {
    return hipErrorInvalidPitchValue;
  }
  // dst/src pitch must be less than max pitch
  auto* deviceHandle = g_devices[hip::getCurrentDevice()->deviceId()]->devices()[0];
  const auto& info = deviceHandle->info();
  constexpr auto int32_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  auto maxPitch = std::min(info.maxMemAllocSize_, int32_max);

  // negative pitch cases
  if (p->dstPtr.pitch >= maxPitch || p->srcPtr.pitch >= maxPitch) {
    return hipErrorInvalidValue;
  }

  if (p->dstArray == nullptr && p->srcArray == nullptr) {
    if ((p->dstPtr.pitch != 0 && (p->extent.width + p->dstPos.x > p->dstPtr.pitch)) ||
        (p->srcPtr.pitch != 0 && (p->extent.width + p->srcPos.x > p->srcPtr.pitch))) {
      return hipErrorInvalidValue;
    }
    auto totalExtentBytes = p->extent.width * p->extent.height * p->extent.depth;
    // get memory obj of the PitchPtr
    size_t offset = 0;
    amd::Memory* srcPtrMemObj = getMemoryObject(p->srcPtr.ptr, offset);
    amd::Memory* dstPtrMemObj = getMemoryObject(p->dstPtr.ptr, offset);

    if (dstPtrMemObj != nullptr && (p->dstPtr.xsize != 0 && p->dstPtr.ysize != 0)) {
      // Use the memoryObj to get 3d data
      const auto& dstUsrData = dstPtrMemObj->getUserData();
      //  dst ptr out of bound cases for linear memory
      if (dstUsrData.pitch_ == 0 || dstUsrData.height_ == 0 || dstUsrData.depth_ == 0) {
        auto dstDepth = dstPtrMemObj->getSize() / (p->dstPtr.xsize * p->dstPtr.ysize);
        if ((p->dstPtr.xsize * (p->dstPtr.ysize - p->dstPos.y) * (dstDepth - p->dstPos.z)) <
            totalExtentBytes) {
          return hipErrorInvalidValue;
        }
        // out of bound dst ptr for 3d memory
      } else if ((dstUsrData.pitch_ * (dstUsrData.height_ - p->dstPos.y) *
                  (dstUsrData.depth_ - p->dstPos.z)) < totalExtentBytes) {
        return hipErrorInvalidValue;
      }
    }

    if (srcPtrMemObj != nullptr && (p->srcPtr.xsize != 0 && p->srcPtr.ysize != 0)) {
      const auto& srcUsrData = srcPtrMemObj->getUserData();
      //  src ptr out of bound cases for linear memory
      if (srcUsrData.pitch_ == 0 || srcUsrData.height_ == 0 || srcUsrData.depth_ == 0) {
        auto srcDepth = srcPtrMemObj->getSize() / (p->srcPtr.xsize * p->srcPtr.ysize);
        if ((p->srcPtr.xsize * (p->srcPtr.ysize - p->srcPos.y) * (srcDepth - p->srcPos.z)) <
            totalExtentBytes) {
          return hipErrorInvalidValue;
        }
        // out of bound src ptr for 3d memory
      } else if ((srcUsrData.pitch_ * (srcUsrData.height_ - p->srcPos.y) *
                  (srcUsrData.depth_ - p->srcPos.z)) < totalExtentBytes) {
        return hipErrorInvalidValue;
      }
    }
  }

  if (static_cast<uint32_t>(p->kind) > hipMemcpyDefault && p->kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  // If src and dst ptr are null then kind must be either h2h or def.
  if (!IsHtoHMemcpyValid(p->dstPtr.ptr, p->srcPtr.ptr, p->kind)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t ihipMemcpy3DCommand(amd::Command*& command, const hipMemcpy3DParms* p,
                               hip::Stream* stream) {
  const HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*p);
  return ihipGetMemcpyParam3DCommand(command, &desc, stream);
}

hipError_t ihipMemcpy3D(const hipMemcpy3DParms* p, hipStream_t stream, bool isAsync) {
  hipError_t status = ihipMemcpy3D_validate(p);
  if (status != hipSuccess) {
    return status;
  }
  const HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*p);

  return ihipMemcpyParam3D(&desc, stream, isAsync);
}

hipError_t hipMemcpy3D_common(const hipMemcpy3DParms* p, hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();
  return ihipMemcpy3D(p, stream);
}

hipError_t hipMemcpy3D(const hipMemcpy3DParms* p) {
  HIP_INIT_API(hipMemcpy3D, p);
  HIP_RETURN_DURATION(hipMemcpy3D_common(p));
}

hipError_t hipMemcpy3D_spt(const hipMemcpy3DParms* p) {
  HIP_INIT_API(hipMemcpy3D, p);
  HIP_RETURN_DURATION(hipMemcpy3D_common(p, getPerThreadDefaultStream()));
}

hipError_t hipMemcpy3DAsync_common(const hipMemcpy3DParms* p, hipStream_t stream) {
  STREAM_CAPTURE(hipMemcpy3DAsync, stream, p);
  return ihipMemcpy3D(p, stream, true);
}

hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy3DAsync, p, stream);
  HIP_RETURN_DURATION(hipMemcpy3DAsync_common(p, stream));
}

hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy3DAsync, p, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipMemcpy3DAsync_common(p, stream));
}

hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  HIP_INIT_API(hipDrvMemcpy3D, pCopy);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpyParam3D(pCopy, nullptr));
}

hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  HIP_INIT_API(hipDrvMemcpy3DAsync, pCopy, stream);
  HIP_RETURN_DURATION(ihipMemcpyParam3D(pCopy, stream, true));
}

hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyBatchAsync, dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx,
               stream);
  // validate stream
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  // validate inputs
  if (dsts == nullptr || srcs == nullptr || sizes == nullptr || failIdx == nullptr || count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // no support for memcpy attributes
  if (numAttrs > 0) {
    HIP_RETURN(hipErrorNotSupported);
  }

  hipError_t status = hipSuccess;

  *failIdx = SIZE_MAX;
  for (int i = 0; i < count; ++i) {
    if (sizes[i] == 0) {
      *failIdx = i;
      status = hipErrorInvalidValue;
      break;
    }
    status = ihipMemcpy(dsts[i], srcs[i], sizes[i], hipMemcpyDefault, *hip::getStream(stream), true,
                        true);
    if (status != hipSuccess) {
      *failIdx = i;
      break;
    }
  }
  HIP_RETURN(status);
}

hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t* failIdx,
                                 unsigned long long flags, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy3DBatchAsync, numOps, opList, failIdx, flags, stream);
  if (flags != 0 || opList == nullptr || numOps == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hipError_t status = hipSuccess;

  *failIdx = SIZE_MAX;
  for (int i = 0; i < numOps; ++i) {
    hipMemcpy3DParms parms = getMemcpy3DParms(opList[i]);
    status = ihipMemcpy3D(&parms, stream, true);
    if (status != hipSuccess) {
      *failIdx = i;
      break;
    }
  }
  HIP_RETURN(status);
}

hipError_t packFillMemoryCommand(amd::Command*& command, amd::Memory* memory, size_t offset,
                                 int64_t value, size_t valueSize, size_t sizeBytes,
                                 hip::Stream* stream) {
  if ((memory == nullptr) || (stream == nullptr)) {
    return hipErrorInvalidValue;
  }

  amd::Command::EventWaitList waitList;
  amd::Coord3D fillOffset(offset, 0, 0);
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  // surface=[pitch, width, height]
  amd::Coord3D surface(sizeBytes, sizeBytes, 1);
  amd::FillMemoryCommand* fillMemCommand =
      new amd::FillMemoryCommand(*stream, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, valueSize, fillOffset, fillSize, surface);
  if (fillMemCommand == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!fillMemCommand->validatePeerMemory()) {
    delete fillMemCommand;
    return hipErrorInvalidValue;
  }
  command = fillMemCommand;
  return hipSuccess;
}

hipError_t ihipMemset_validate(void* dst, int64_t value, size_t valueSize, size_t sizeBytes) {
  if (sizeBytes == 0) {
    // Skip if nothing needs filling.
    return hipSuccess;
  }

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(dst, offset);
  if (memory == nullptr) {
    // dst ptr is host ptr hence error
    return hipErrorInvalidValue;
  }

  // Validate Mem Access in case of VMM Memory
  if (!memory->ValidateMemAccess(*hip::getCurrentDevice()->devices()[0], true)) {
    return hipErrorUnknown;
  }

  // In case of vmm sub object, validate using parents vaddr mem object.
  if (memory->parent() && (memory->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
    memory = memory->parent();
  }

  // Return error if sizeBytes passed to memcpy is more than the actual size allocated
  if (sizeBytes > (memory->getSize() - offset)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t ihipGraphMemsetParams_validate(const hipMemsetParams* pNodeParams) {
  if (pNodeParams == nullptr) {
    return hipErrorInvalidValue;
  }

  if (pNodeParams->width == 0) {
    return hipErrorInvalidValue;
  }

  if (pNodeParams->elementSize != 1 && pNodeParams->elementSize != 2 &&
      pNodeParams->elementSize != 4) {
    return hipErrorInvalidValue;
  }

  if (pNodeParams->height <= 0) {
    return hipErrorInvalidValue;
  }

  size_t discardOffset = 0;
  amd::Memory* memObj = getMemoryObject(pNodeParams->dst, discardOffset);
  if (memObj != nullptr) {
    // Validate Mem Access in case of VMM Memory
    if (!memObj->ValidateMemAccess(*hip::getCurrentDevice()->devices()[0], true)) {
      return hipErrorUnknown;
    }

    if ((pNodeParams->pitch * pNodeParams->height) > memObj->getSize()) {
      return hipErrorInvalidValue;
    }
  }

  return hipSuccess;
}

hipError_t ihipMemsetCommand(std::vector<amd::Command*>& commands, void* dst, int64_t value,
                             size_t valueSize, size_t sizeBytes, hip::Stream* stream) {
  hipError_t hip_error = hipSuccess;
  auto aligned_dst = amd::alignUp(reinterpret_cast<address>(dst), sizeof(uint64_t));
  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(dst, offset);
  size_t n_head_bytes = 0;
  size_t n_tail_bytes = 0;
  amd::Command* command;

  hip_error = packFillMemoryCommand(command, memory, offset, value, valueSize, sizeBytes, stream);
  commands.push_back(command);

  return hip_error;
}

hipError_t ihipMemset(void* dst, int64_t value, size_t valueSize, size_t sizeBytes,
                      hipStream_t stream, bool isAsync = false) {
  hipError_t hip_error = hipSuccess;
  do {
    // Nothing to do, fill size is 0. Returns hipSuccess.
    if (sizeBytes == 0) {
      break;
    }

    // In case of validation failure stop processing. Returns hip_error.
    hip_error = ihipMemset_validate(dst, value, valueSize, sizeBytes);
    if (hip_error != hipSuccess) {
      break;
    }
    // This is required to comply with the spec
    // spec says hipMemset will be asynchronous when destination memory is device memory
    // and pointer is non-offseted
    if (isAsync == false) {
      size_t offset = 0;
      amd::Memory* memObj = getMemoryObject(dst, offset);
      auto flags = memObj->getMemFlags();
      if ((memObj->getUserData().sync_mem_ops_) ||
          (offset == 0 &&
           !(flags & (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS | CL_MEM_USE_HOST_PTR)))) {
        isAsync = true;
      }
    }
    std::vector<amd::Command*> commands;
    hip::Stream* hip_stream = hip::getStream(stream);
    if (hip_stream == nullptr) {
      return hipErrorOutOfMemory;
    }
    hip_error = ihipMemsetCommand(commands, dst, value, valueSize, sizeBytes, hip_stream);
    if (hip_error != hipSuccess) {
      break;
    }

    for (auto command : commands) {
      command->enqueue();
      if (!isAsync) {
        hip_stream->finish();
      }
      command->release();
    }
  } while (0);
  return hip_error;
}

hipError_t hipMemset_common(void* dst, int value, size_t sizeBytes, hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();
  return ihipMemset(dst, value, sizeof(int8_t), sizeBytes, stream);
}

hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(hipMemset, dst, value, sizeBytes);
  HIP_RETURN(hipMemset_common(dst, value, sizeBytes, getPerThreadDefaultStream()));
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(hipMemset, dst, value, sizeBytes);
  HIP_RETURN(hipMemset_common(dst, value, sizeBytes));
}

hipError_t hipMemsetAsync_common(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  size_t valueSize = sizeof(int8_t);
  STREAM_CAPTURE(hipMemsetAsync, stream, dst, value, valueSize, sizeBytes);
  return ihipMemset(dst, value, sizeof(int8_t), sizeBytes, stream, true);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(hipMemsetAsync, dst, value, sizeBytes, stream);
  HIP_RETURN(hipMemsetAsync_common(dst, value, sizeBytes, stream));
}

hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(hipMemsetAsync, dst, value, sizeBytes, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipMemsetAsync_common(dst, value, sizeBytes, stream));
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t count) {
  HIP_INIT_API(hipMemsetD8, dst, value, count);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN(ihipMemset(dst, value, sizeof(int8_t), count * sizeof(int8_t), nullptr));
}

hipError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char value, size_t count,
                            hipStream_t stream) {
  HIP_INIT_API(hipMemsetD8Async, dst, value, count, stream);
  int iValue = value;
  size_t valueSize = sizeof(int8_t);
  size_t sizeBytes = count * sizeof(int8_t);
  STREAM_CAPTURE(hipMemsetAsync, stream, dst, iValue, valueSize, sizeBytes);
  HIP_RETURN(ihipMemset(dst, value, valueSize, sizeBytes, stream, true));
}

hipError_t hipMemsetD16(hipDeviceptr_t dst, unsigned short value, size_t count) {
  HIP_INIT_API(hipMemsetD16, dst, value, count);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN(ihipMemset(dst, value, sizeof(int16_t), count * sizeof(int16_t), nullptr));
}

hipError_t hipMemsetD16Async(hipDeviceptr_t dst, unsigned short value, size_t count,
                             hipStream_t stream) {
  HIP_INIT_API(hipMemsetD16Async, dst, value, count, stream);
  int iValue = value;
  size_t valueSize = sizeof(int16_t);
  size_t sizeBytes = count * sizeof(int16_t);
  STREAM_CAPTURE(hipMemsetAsync, stream, dst, iValue, valueSize, sizeBytes);
  HIP_RETURN(ihipMemset(dst, value, valueSize, sizeBytes, stream, true));
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
  HIP_INIT_API(hipMemsetD32, dst, value, count);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN(ihipMemset(dst, value, sizeof(int32_t), count * sizeof(int32_t), nullptr));
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
  HIP_INIT_API(hipMemsetD32Async, dst, value, count, stream);
  int iValue = value;
  size_t valueSize = sizeof(int32_t);
  size_t sizeBytes = count * sizeof(int32_t);
  STREAM_CAPTURE(hipMemsetAsync, stream, dst, iValue, valueSize, sizeBytes);
  HIP_RETURN(ihipMemset(dst, value, valueSize, sizeBytes, stream, true));
}

hipError_t ihipMemset3D_validate(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                 size_t sizeBytes) {
  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(pitchedDevPtr.ptr, offset, sizeBytes);

  if (memory == nullptr) {
    return hipErrorInvalidValue;
  }
  // Return error if sizeBytes passed to memcpy is more than the actual size allocated
  if (sizeBytes > (memory->getSize() - offset)) {
    return hipErrorInvalidValue;
  }
  if (pitchedDevPtr.pitch == memory->getUserData().pitch_) {
    if (extent.height > memory->getUserData().height_) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t ihipMemset3DCommand(std::vector<amd::Command*>& commands, hipPitchedPtr pitchedDevPtr,
                               int value, hipExtent extent, hip::Stream* stream,
                               size_t elementSize = 1) {
  size_t offset = 0;
  auto sizeBytes = extent.width * extent.height * extent.depth;
  amd::Memory* memory = getMemoryObject(pitchedDevPtr.ptr, offset);
  if (pitchedDevPtr.pitch == extent.width) {
    return ihipMemsetCommand(commands, pitchedDevPtr.ptr, value, elementSize,
                             static_cast<size_t>(sizeBytes), stream);
  }
  // Workaround for cases when pitch > row until fill kernel will be updated to support pitch.
  // Fall back to filling one row at a time.
  amd::Coord3D origin(offset);
  amd::Coord3D region(extent.width, extent.height, extent.depth);
  amd::Coord3D surface(pitchedDevPtr.pitch, pitchedDevPtr.xsize, pitchedDevPtr.ysize);
  amd::BufferRect rect;
  if (pitchedDevPtr.pitch == 0 ||
      !rect.create(static_cast<size_t*>(origin),
                   static_cast<size_t*>(
                       amd::Coord3D{pitchedDevPtr.xsize, pitchedDevPtr.ysize, extent.depth}),
                   pitchedDevPtr.pitch, 0)) {
    return hipErrorInvalidValue;
  }
  amd::FillMemoryCommand* command;
  command =
      new amd::FillMemoryCommand(*stream, CL_COMMAND_FILL_BUFFER, amd::Command::EventWaitList{},
                                 *memory->asBuffer(), &value, elementSize, origin, region, surface);
  commands.push_back(command);
  return hipSuccess;
}


hipError_t ihipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                        hipStream_t stream, bool isAsync = false, size_t elementSize = 1) {
  auto sizeBytes = extent.width * extent.height * extent.depth;

  if (sizeBytes == 0) {
    // sizeBytes is zero hence returning early as nothing to be set
    return hipSuccess;
  }
  hipError_t status = ihipMemset3D_validate(pitchedDevPtr, value, extent, sizeBytes);
  if (status != hipSuccess) {
    return status;
  }
  // This is required to comply with the spec
  // spec says hipMemset will be asynchronous when destination memory is device memory
  // and pointer is non-offseted
  if (isAsync == false) {
    size_t offset = 0;
    amd::Memory* memObj = getMemoryObject(pitchedDevPtr.ptr, offset);
    auto flags = memObj->getMemFlags();
    if (offset == 0 &&
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_SVM_ATOMICS | CL_MEM_SVM_FINE_GRAIN_BUFFER))) {
      isAsync = true;
    }
  }
  hip::Stream* hip_stream = hip::getStream(stream);
  std::vector<amd::Command*> commands;
  status = ihipMemset3DCommand(commands, pitchedDevPtr, value, extent, hip_stream, elementSize);
  if (status != hipSuccess) {
    return status;
  }
  for (auto& command : commands) {
    command->enqueue();
    if (!isAsync) {
      hip_stream->finish();
    }
    command->release();
  }
  return hipSuccess;
}

hipError_t hipMemset2D_common(void* dst, size_t pitch, int value, size_t width, size_t height,
                              hipStream_t stream = nullptr, size_t elementSize = 1) {
  CHECK_STREAM_CAPTURING();
  return ihipMemset3D({dst, pitch, width, height}, value, {width, height, 1}, stream, false,
                      elementSize);
}

hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(hipMemset2D, dst, pitch, value, width, height);
  hipStream_t stream = getPerThreadDefaultStream();
  HIP_RETURN(hipMemset2D_common(dst, pitch, value, width, height, stream));
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(hipMemset2D, dst, pitch, value, width, height);
  HIP_RETURN(hipMemset2D_common(dst, pitch, value, width, height));
}

hipError_t hipMemset2DAsync_common(void* dst, size_t pitch, int value, size_t width, size_t height,
                                   hipStream_t stream, size_t elementSize = 1) {
  STREAM_CAPTURE(hipMemset2DAsync, stream, dst, pitch, value, width, height);
  return ihipMemset3D({dst, pitch, width, height}, value, {width, height, 1}, stream, true,
                      elementSize);
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream) {
  HIP_INIT_API(hipMemset2DAsync, dst, pitch, value, width, height, stream);
  HIP_RETURN(hipMemset2DAsync_common(dst, pitch, value, width, height, stream));
}

hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream) {
  HIP_INIT_API(hipMemset2DAsync, dst, pitch, value, width, height, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipMemset2DAsync_common(dst, pitch, value, width, height, stream));
}

hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width,
                         size_t height) {
  HIP_INIT_API(hipMemsetD2D8, dst, dstPitch, value, width, height);
  HIP_RETURN(
      hipMemset2D_common(dst, dstPitch, value, width, height, nullptr, sizeof(unsigned char)));
}

hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                              size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(hipMemsetD2D8Async, dst, dstPitch, value, width, height, stream);
  HIP_RETURN(
      hipMemset2DAsync_common(dst, dstPitch, value, width, height, stream, sizeof(unsigned char)));
}

hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width,
                          size_t height) {
  HIP_INIT_API(hipMemsetD2D16, dst, dstPitch, value, width, height);
  HIP_RETURN(
      hipMemset2D_common(dst, dstPitch, value, width, height, nullptr, sizeof(unsigned short)));
}

hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                               size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(hipMemsetD2D16Async, dst, dstPitch, value, width, height, stream);
  HIP_RETURN(
      hipMemset2DAsync_common(dst, dstPitch, value, width, height, stream, sizeof(unsigned short)));
}

hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height) {
  HIP_INIT_API(hipMemsetD2D32, dst, dstPitch, value, width, height);
  HIP_RETURN(
      hipMemset2D_common(dst, dstPitch, value, width, height, nullptr, sizeof(unsigned int)));
}

hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                               size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(hipMemsetD2D32Async, dst, dstPitch, value, width, height, stream);
  HIP_RETURN(
      hipMemset2DAsync_common(dst, dstPitch, value, width, height, stream, sizeof(unsigned int)));
}

// ================================================================================================
hipError_t hipMemset3D_common(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                              hipStream_t stream = nullptr) {
  CHECK_STREAM_CAPTURING();
  return ihipMemset3D(pitchedDevPtr, value, extent, stream);
}

// ================================================================================================
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  HIP_INIT_API(hipMemset3D, pitchedDevPtr, value, extent);
  HIP_RETURN(hipMemset3D_common(pitchedDevPtr, value, extent));
}

// ================================================================================================
hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  HIP_INIT_API(hipMemset3D, pitchedDevPtr, value, extent);
  hipStream_t stream = getPerThreadDefaultStream();
  HIP_RETURN(hipMemset3D_common(pitchedDevPtr, value, extent, stream));
}

// ================================================================================================
hipError_t hipMemset3DAsync_common(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                   hipStream_t stream) {
  STREAM_CAPTURE(hipMemset3DAsync, stream, pitchedDevPtr, value, extent);
  return ihipMemset3D(pitchedDevPtr, value, extent, stream, true);
}

// ================================================================================================
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream) {
  HIP_INIT_API(hipMemset3DAsync, pitchedDevPtr, value, extent, stream);
  HIP_RETURN(hipMemset3DAsync_common(pitchedDevPtr, value, extent, stream));
}

// ================================================================================================
hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream) {
  HIP_INIT_API(hipMemset3DAsync, pitchedDevPtr, value, extent, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipMemset3DAsync_common(pitchedDevPtr, value, extent, stream));
}

hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  HIP_INIT_API(hipMemAllocPitch, dptr, pitch, widthInBytes, height, elementSizeBytes);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  if (widthInBytes == 0 || height == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (elementSizeBytes != 4 && elementSizeBytes != 8 && elementSizeBytes != 16) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipMallocPitch(dptr, pitch, widthInBytes, height));
}

hipError_t hipMemAllocHost(void** ptr, size_t size) {
  HIP_INIT_API(hipMemAllocHost, ptr, size);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN_DURATION(hipHostMalloc(ptr, size, 0), ReturnPtrValue(ptr));
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* dev_ptr) {
  HIP_INIT_API(hipIpcGetMemHandle, handle, dev_ptr);

  amd::Device* device = nullptr;
  amd::MemObjMap::IpcMemHandle* ihandle = nullptr;

  if ((handle == nullptr) || (dev_ptr == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  device = hip::getCurrentDevice()->devices()[0];
  ihandle = reinterpret_cast<amd::MemObjMap::IpcMemHandle*>(handle);
  ihandle->owners_device_id = hip::getCurrentDevice()->deviceId();

  if (!device->IpcCreate(dev_ptr, &(ihandle->psize), ihandle->ipc_handle, &(ihandle->poffset))) {
    LogPrintfError("IPC memory creation failed for memory: 0x%x", dev_ptr);
    HIP_RETURN(hipErrorInvalidValue);
  }
  ihandle->owners_process_id = amd::Os::getProcessId();

  HIP_RETURN(hipSuccess);
}

hipError_t hipIpcOpenMemHandle(void** dev_ptr, hipIpcMemHandle_t handle, unsigned int flags) {
  HIP_INIT_API(hipIpcOpenMemHandle, dev_ptr, &handle, flags);
  amd::Memory* amd_mem_obj = nullptr;
  amd::Device* device = nullptr;
  amd::MemObjMap::IpcMemHandle* ihandle = nullptr;
  size_t offset = 0;

  if (dev_ptr == nullptr || flags != hipIpcMemLazyEnablePeerAccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  /* Call the IPC Attach from Device class */
  device = hip::getCurrentDevice()->devices()[0];
  ihandle = reinterpret_cast<amd::MemObjMap::IpcMemHandle*>(&handle);

  if (ihandle->psize == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (ihandle->owners_process_id == amd::Os::getProcessId()) {
    HIP_RETURN(hipErrorInvalidContext);
  }

  if (ihandle->owners_device_id >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Device* peer_device = g_devices[ihandle->owners_device_id]->asContext()->devices()[0];
  device->enableP2P(peer_device);

  amd_mem_obj = amd::MemObjMap::FindIpcHandleMemObj(*ihandle);
  if (amd_mem_obj == nullptr) {
    if (!device->IpcAttach(ihandle->ipc_handle, ihandle->psize, ihandle->poffset, flags, dev_ptr)) {
      LogPrintfError(
          "Cannot attach ipc_handle: with ipc_size: %u"
          "ipc_offset: %u flags: %u",
          ihandle->psize, ihandle->poffset, flags);
      HIP_RETURN(hipErrorInvalidDevicePointer);
    }
    amd_mem_obj = getMemoryObject(*dev_ptr, offset);
    amd::MemObjMap::AddIpcHandleMemObj(*ihandle, amd_mem_obj);
  } else {
    // Handle was already opened by the same process
    *dev_ptr = amd_mem_obj->getSvmPtr();
    amd_mem_obj->retain();
  }

  amd_mem_obj->getUserData().deviceId = hip::getCurrentDevice()->deviceId();

  HIP_RETURN(hipSuccess, ReturnPtrValue(dev_ptr));
}

hipError_t hipIpcCloseMemHandle(void* dev_ptr) {
  HIP_INIT_API(hipIpcCloseMemHandle, dev_ptr);

  amd::Device* device = nullptr;
  amd::Memory* amd_mem_obj = nullptr;

  hip::getNullStream()->finish();

  if (dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  /* Call IPC Detach from Device class */
  device = hip::getCurrentDevice()->devices()[0];
  if (device == nullptr) {
    HIP_RETURN(hipErrorNoDevice);
  }

  amd_mem_obj = amd::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Memory object for the ptr: 0x%x cannot be null \n", dev_ptr);
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!amd_mem_obj->ipcShared()) {
    DevLogPrintfError("Memory object for the ptr: 0x%x is not ipcShared \n", dev_ptr);
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto device_id = amd_mem_obj->getUserData().deviceId;
  g_devices[device_id]->SyncAllStreams();

  amd_mem_obj->release();

  HIP_RETURN(hipSuccess);
}


hipError_t hipHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned flags) {
  HIP_INIT_API(hipHostGetDevicePointer, devicePointer, hostPointer, flags);

  if (devicePointer == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t offset = 0;

  amd::Memory* memObj = getMemoryObject(hostPointer, offset);
  if (!memObj) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *devicePointer = reinterpret_cast<void*>(
      memObj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0])->virtualAddress() + offset);

  HIP_RETURN(hipSuccess, ReturnPtrValue(devicePointer));
}

// ================================================================================================
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  HIP_INIT_API(hipPointerGetAttributes, attributes, ptr);

  if (attributes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  size_t offset = 0;
  amd::Memory* memObj = (ptr != nullptr) ? getMemoryObject(ptr, offset) : nullptr;
  int device = 0;
  device::Memory* devMem = nullptr;
  memset(attributes, 0, sizeof(hipPointerAttribute_t));

  if (memObj != nullptr) {
    attributes->type = getMemoryType(memObj);
    if (attributes->type == hipMemoryTypeHost) {
      if (memObj->getHostMem() != nullptr) {
        attributes->hostPointer = static_cast<char*>(memObj->getHostMem()) + offset;
      } else {
        attributes->hostPointer = static_cast<char*>(memObj->getSvmPtr()) + offset;
      }
    }
    // the pointer that attribute is retrieved for might not be on the current device
    for (const auto& device : g_devices) {
      if (device->deviceId() == memObj->getUserData().deviceId) {
        devMem = memObj->getDeviceMemory(*device->devices()[0]);
        break;
      }
    }
    // getDeviceMemory can fail, hence validate the sanity of the mem obtained
    if (nullptr == devMem) {
      DevLogPrintfError("getDeviceMemory for ptr failed : %p", ptr);
      HIP_RETURN(hipErrorMemoryAllocation);
    }

    attributes->devicePointer = reinterpret_cast<char*>(devMem->virtualAddress() + offset);
    constexpr uint32_t kManagedAlloc = (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR);
    attributes->isManaged =
        ((memObj->getMemFlags() & kManagedAlloc) == kManagedAlloc) ? true : false;
    attributes->allocationFlags = memObj->getUserData().flags;
    attributes->device = memObj->getUserData().deviceId;
    if (attributes->isManaged) {
      attributes->type = hipMemoryTypeManaged;
    }
    HIP_RETURN(hipSuccess);
  } else {
    attributes->type = hipMemoryTypeUnregistered;
    attributes->devicePointer = nullptr;
    attributes->hostPointer = nullptr;
    attributes->isManaged = false;
    attributes->allocationFlags = 0;
    attributes->device = hipInvalidDeviceId;
    if (ptr != nullptr) {
      LogPrintfError("Cannot get amd_mem_obj for ptr: %p", ptr);
    }
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t ihipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                   hipDeviceptr_t ptr) {
  if (attribute != HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS) {
    return hipErrorInvalidValue;
  }
  const unsigned int syncMemops = *reinterpret_cast<const unsigned int*>(value);
  if (syncMemops != 0 && syncMemops != 1) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(ptr, offset);
  if (memObj == nullptr) {
    return hipErrorInvalidDevicePointer;
  }

  memObj->getUserData().sync_mem_ops_ = static_cast<const bool>(syncMemops);

  return hipSuccess;
}

// ================================================================================================
hipError_t ihipPointerGetAttributes(void* data, hipPointer_attribute attribute,
                                    hipDeviceptr_t ptr) {
  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(ptr, offset);
  constexpr uint32_t kManagedAlloc = (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR);

  hipError_t status = hipSuccess;

  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_CONTEXT: {
      if (memObj) {
        amd::Context& context = memObj->getContext();
        int devId = getDeviceID(context);
        if (devId >= 0) {
          *reinterpret_cast<hipCtx_t*>(data) = reinterpret_cast<hipCtx_t>(g_devices[devId]);
        } else {
          return hipErrorInvalidValue;
        }
      } else {
        return hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      if (memObj) {  // checks for host type or device type
        *reinterpret_cast<uint32_t*>(data) = getMemoryType(memObj);
      } else {  // checks for array type
        // ptr must be a host allocation using malloc since memObj is null and is
        // not found in hipArraySet.
        if (hip::hipArraySet.find(static_cast<hipArray*>(ptr)) == hip::hipArraySet.end()) {
          *reinterpret_cast<uint32_t*>(data) = 0;
          return hipErrorInvalidValue;
        }
        cl_mem dstMemObj = reinterpret_cast<cl_mem>((static_cast<hipArray*>(ptr))->data);
        if (!is_valid(dstMemObj)) {
          *reinterpret_cast<uint32_t*>(data) = 0;
          return hipErrorInvalidValue;
        }
        amd::Image* dstImage = as_amd(dstMemObj)->asImage();
        if (dstImage) {
          *reinterpret_cast<uint32_t*>(data) = hipMemoryTypeArray;
        } else {
          *reinterpret_cast<uint32_t*>(data) = 0;
          return hipErrorInvalidValue;
        }
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_POINTER: {
      if (memObj) {
        device::Memory* devMem = memObj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0]);

        // getDeviceMemory can fail, hence validate the sanity of the mem obtained
        if (nullptr == devMem) {
          DevLogPrintfError("getDeviceMemory for ptr failed : %p", ptr);
          return hipErrorMemoryAllocation;
        }
        *reinterpret_cast<hipDeviceptr_t*>(data) =
            reinterpret_cast<hipDeviceptr_t*>(devMem->virtualAddress() + offset);
      } else {
        *reinterpret_cast<hipDeviceptr_t*>(data) = nullptr;
        return hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_HOST_POINTER: {
      if (memObj) {
        if (getMemoryType(memObj) == hipMemoryTypeHost) {
          if (memObj->getHostMem() != nullptr) {
            // Registered memory
            *reinterpret_cast<char**>(data) = static_cast<char*>(memObj->getHostMem()) + offset;
          } else {
            // Prepinned memory
            *reinterpret_cast<char**>(data) = static_cast<char*>(memObj->getSvmPtr()) + offset;
          }
        } else {
          *reinterpret_cast<char**>(data) = nullptr;
          status = hipErrorInvalidValue;
        }
      } else {  // Host Memory
        *reinterpret_cast<char**>(data) = nullptr;
        status = hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_P2P_TOKENS: {
      // Currently not supported, deprecated in cuda as well
      status = hipErrorNotSupported;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      if (memObj) {
        *reinterpret_cast<bool*>(data) = memObj->getUserData().sync_mem_ops_;
      } else {
        *reinterpret_cast<bool*>(data) = false;
        return hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID: {
      if (memObj) {
        *reinterpret_cast<uint32_t*>(data) = memObj->getUniqueId();
      } else {  // ptr passed must be allocated using HIP memory allocation API
        *reinterpret_cast<uint32_t*>(data) = 0;
        return hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_IS_MANAGED: {
      if (memObj) {
        *reinterpret_cast<bool*>(data) =
            ((memObj->getMemFlags() & kManagedAlloc) == kManagedAlloc) ? true : false;
      } else {
        *reinterpret_cast<bool*>(data) = false;
        return hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
      if (memObj) {
        *reinterpret_cast<int*>(data) = memObj->getUserData().deviceId;
      } else {
        // for host memory, -2 is returned by default similar to cuda
        *reinterpret_cast<int*>(data) = -2;
        status = hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE: {
      // TODO: Unclear what to be done for this attribute
      status = hipErrorNotSupported;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR: {
      if (memObj) {
        if (memObj->getHostMem() != nullptr) {
          *reinterpret_cast<hipDeviceptr_t*>(data) = static_cast<char*>(memObj->getHostMem());
        } else {
          device::Memory* devMem = memObj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0]);

          // getDeviceMemory can fail, hence validate the sanity of the mem obtained
          if (nullptr == devMem) {
            DevLogPrintfError("getDeviceMemory for ptr failed : %p", ptr);
            return hipErrorMemoryAllocation;
          }
          *reinterpret_cast<hipDeviceptr_t*>(data) =
              reinterpret_cast<char*>(devMem->virtualAddress());
        }
      } else {
        // Input is host memory pointer, invalid for device.
        *reinterpret_cast<hipDeviceptr_t*>(data) = nullptr;
        status = hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE: {
      if (memObj) {
        *reinterpret_cast<uint32_t*>(data) = memObj->getSize();
      } else {
        *reinterpret_cast<uint32_t*>(data) = 0;
        status = hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MAPPED: {
      if (memObj) {
        *reinterpret_cast<bool*>(data) = true;
      } else {
        *reinterpret_cast<bool*>(data) = false;
        status = hipErrorInvalidValue;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: {
      // hipMemAllocationHandleType is not yet supported
      LogPrintfWarning("attribute %d is not supported.", attribute);
      status = hipErrorNotSupported;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: {
      // GPUDirect RDMA API is not yet supported
      LogPrintfWarning("attribute %d is not supported.", attribute);
      status = hipErrorNotSupported;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: {
      if (memObj) {
        *reinterpret_cast<uint32_t*>(data) = memObj->getUserData().flags;
      } else {
        *reinterpret_cast<uint32_t*>(data) = 0;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: {
      // allocations from mempool are not yet supported
      LogPrintfWarning("attribute %d is not supported.", attribute);
      status = hipErrorNotSupported;
      break;
    }
    default: {
      LogPrintfError("Invalid attribute: %d ", attribute);
      status = hipErrorInvalidValue;
      break;
    }
  }
  return status;
}

// ================================================================================================
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr) {
  HIP_INIT_API(hipPointerSetAttribute, value, attribute, ptr);

  if (ptr == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipPointerSetAttribute(value, attribute, ptr));
}

// ================================================================================================

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  HIP_INIT_API(hipPointerGetAttribute, data, attribute, ptr);

  if (ptr == nullptr || data == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipPointerGetAttributes(data, attribute, ptr));
}

// ================================================================================================
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr) {
  HIP_INIT_API(hipDrvPointerGetAttributes, numAttributes, attributes, data, ptr);

  if (numAttributes == 0 || attributes == nullptr || data == nullptr || ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Ignore the status, hipDrvPointerGetAttributes always returns success
  // If the ptr is invalid, the queried attributes will be assigned default values
  for (int i = 0; i < numAttributes; ++i) {
    hipError_t status = ihipPointerGetAttributes(data[i], attributes[i], ptr);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipArrayDestroy(hipArray_t array) {
  HIP_INIT_API(hipArrayDestroy, array);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipArrayDestroy(array));
}

hipError_t ihipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* desc, hipArray_t array) {
  {
    amd::ScopedLock lock(hipArraySetLock);
    if (hip::hipArraySet.find(array) == hip::hipArraySet.end()) {
      return hipErrorInvalidHandle;
    }
  }

  desc->Width = array->width;
  desc->Height = array->height;
  desc->Depth = array->depth;
  desc->Format = array->Format;
  desc->NumChannels = array->NumChannels;
  desc->Flags = array->flags;

  return hipSuccess;
}

hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array) {
  HIP_INIT_API(hipArrayGetInfo, desc, extent, flags, array);
  CHECK_STREAM_CAPTURE_SUPPORTED();

  if (array == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  // If all output parameters are nullptr, then no need to proceed further
  if ((desc == nullptr) && (extent == nullptr) && (flags == nullptr)) {
    HIP_RETURN(hipSuccess);
  }

  HIP_ARRAY3D_DESCRIPTOR array3DDescriptor;
  hipError_t status = ihipArray3DGetDescriptor(&array3DDescriptor, array);

  // Fill each output parameter
  if (status == hipSuccess) {
    if (desc != nullptr) {
      *desc = hip::getChannelFormatDesc(array3DDescriptor.NumChannels, array3DDescriptor.Format);
    }

    if (extent != nullptr) {
      extent->width = array3DDescriptor.Width;
      extent->height = array3DDescriptor.Height;
      extent->depth = array3DDescriptor.Depth;
    }

    if (flags != nullptr) {
      *flags = array3DDescriptor.Flags;
    }
  }

  HIP_RETURN(status);
}

hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  HIP_INIT_API(hipArrayGetDescriptor, pArrayDescriptor, array);
  CHECK_STREAM_CAPTURE_SUPPORTED();

  if (array == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (pArrayDescriptor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_ARRAY3D_DESCRIPTOR array3DDescriptor;
  hipError_t status = ihipArray3DGetDescriptor(&array3DDescriptor, array);

  // Fill each output parameter
  if (status == hipSuccess) {
    pArrayDescriptor->Width = array3DDescriptor.Width;
    pArrayDescriptor->Height = array3DDescriptor.Height;
    pArrayDescriptor->Format = array3DDescriptor.Format;
    pArrayDescriptor->NumChannels = array3DDescriptor.NumChannels;
  }

  HIP_RETURN(status);
}

hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  HIP_INIT_API(hipArray3DGetDescriptor, pArrayDescriptor, array);
  CHECK_STREAM_CAPTURE_SUPPORTED();

  if (array == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (pArrayDescriptor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipArray3DGetDescriptor(pArrayDescriptor, array));
}

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyParam2DAsync, pCopy);
  STREAM_CAPTURE(hipMemcpyParam2DAsync, stream, pCopy);
  HIP_RETURN(ihipMemcpyParam2D(pCopy, stream, true));
}

// ================================================================================================
hipError_t ihipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                    hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                    size_t width, size_t height, hipStream_t stream,
                                    bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = wOffsetSrc;
  desc.srcY = hOffsetSrc;
  desc.srcMemoryType = hipMemoryTypeArray;
  desc.srcHost = nullptr;
  desc.srcDevice = nullptr;
  desc.srcArray = const_cast<hipArray_t>(src);
  desc.srcPitch = 0;  // Ignored.

  desc.dstXInBytes = wOffsetDst;
  desc.dstY = hOffsetDst;
  desc.dstMemoryType = hipMemoryTypeArray;
  desc.dstHost = nullptr;
  desc.dstDevice = nullptr;
  desc.dstArray = dst;
  desc.dstPitch = 0;  // Ignored.

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

// ================================================================================================
hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DArrayToArray, dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc,
               width, height, kind);
  CHECK_STREAM_CAPTURING();
  hipError_t validateParam = hipSuccess, validateSrc = hipSuccess, validateDst = hipSuccess;
  if ((validateParam = hipMemcpy2DValidateParams(kind)) != hipSuccess) {
    HIP_RETURN(validateParam);
  }
  if ((validateSrc = hipMemcpy2DValidateArray(src, wOffsetSrc, hOffsetSrc, width, height)) !=
      hipSuccess) {
    HIP_RETURN(validateSrc);
  }
  if ((validateDst = hipMemcpy2DValidateArray(dst, wOffsetDst, hOffsetDst, width, height)) !=
      hipSuccess) {
    HIP_RETURN(validateDst);
  }
  HIP_RETURN_DURATION(ihipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc,
                                               hOffsetSrc, width, height, nullptr));
}

// ================================================================================================

hipError_t hipMemcpy2DFromArray_common(void* dst, size_t dpitch, hipArray_const_t src,
                                       size_t wOffsetSrc, size_t hOffset, size_t width,
                                       size_t height, hipMemcpyKind kind,
                                       hipStream_t stream = nullptr, bool isAsync = false) {
  hipError_t validateParam = hipSuccess, validateSrc = hipSuccess, validateDst = hipSuccess;
  if ((validateParam = hipMemcpy2DValidateParams(kind, stream)) != hipSuccess) {
    return validateParam;
  }
  if ((validateSrc = hipMemcpy2DValidateArray(src, wOffsetSrc, hOffset, width, height)) !=
      hipSuccess) {
    return validateSrc;
  }
  if ((validateDst = hipMemcpy2DValidateBuffer(dst, dpitch, width)) != hipSuccess) {
    return validateDst;
  }
  return ihipMemcpy2DFromArray(dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind, stream,
                               isAsync);
}

// ================================================================================================
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DFromArray, dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(
      hipMemcpy2DFromArray_common(dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind));
}

// ================================================================================================
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                    size_t wOffsetSrc, size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DFromArray, dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind);
  hipStream_t stream = getPerThreadDefaultStream();
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(hipMemcpy2DFromArray_common(dst, dpitch, src, wOffsetSrc, hOffset, width,
                                                  height, kind, stream));
}

// ================================================================================================
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src,
                                     size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                     size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DFromArrayAsync, dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height,
               kind, stream);
  STREAM_CAPTURE(hipMemcpy2DFromArrayAsync, stream, dst, dpitch, src, wOffsetSrc, hOffsetSrc, width,
                 height, kind);
  HIP_RETURN_DURATION(hipMemcpy2DFromArray_common(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width,
                                                  height, kind, stream, true));
}

// ================================================================================================
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DFromArrayAsync, dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height,
               kind, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  STREAM_CAPTURE(hipMemcpy2DFromArrayAsync, stream, dst, dpitch, src, wOffsetSrc, hOffsetSrc, width,
                 height, kind);
  HIP_RETURN_DURATION(hipMemcpy2DFromArray_common(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width,
                                                  height, kind, stream, true));
}

// ================================================================================================
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DToArrayAsync, dst, wOffset, hOffset, src, spitch, width, height, kind,
               stream);
  STREAM_CAPTURE(hipMemcpy2DToArrayAsync, stream, dst, wOffset, hOffset, src, spitch, width, height,
                 kind);
  HIP_RETURN_DURATION(hipMemcpy2DToArray_common(dst, wOffset, hOffset, src, spitch, width, height,
                                                kind, stream, true));
}

// ================================================================================================
hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DToArrayAsync, dst, wOffset, hOffset, src, spitch, width, height, kind,
               stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  STREAM_CAPTURE(hipMemcpy2DToArrayAsync, stream, dst, wOffset, hOffset, src, spitch, width, height,
                 kind);
  HIP_RETURN_DURATION(hipMemcpy2DToArray_common(dst, wOffset, hOffset, src, spitch, width, height,
                                                kind, stream, true));
}

// ================================================================================================
hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoA, dstArray, dstOffset, srcArray, srcOffset, ByteCount);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpy2DArrayToArray(dstArray, dstOffset, 0, srcArray, srcOffset, 0,
                                               ByteCount, 1, nullptr));
}

// ================================================================================================
hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoD, dstDevice, srcArray, srcOffset, ByteCount);
  HIP_RETURN_DURATION(ihipMemcpy2DFromArray(dstDevice, 0, srcArray, srcOffset, 0, ByteCount, 1,
                                            hipMemcpyDeviceToDevice, nullptr));
}

// ================================================================================================
hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                              size_t ByteCount, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyAtoHAsync, dstHost, srcArray, srcOffset, ByteCount, stream);
  STREAM_CAPTURE(hipMemcpyAtoHAsync, stream, dstHost, srcArray, srcOffset, ByteCount);
  HIP_RETURN_DURATION(ihipMemcpy2DFromArray(dstHost, 0, srcArray, srcOffset, 0, ByteCount, 1,
                                            hipMemcpyDeviceToHost, stream, true));
}

// ================================================================================================
hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoA, dstArray, dstOffset, srcDevice, ByteCount);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpy2DToArray(dstArray, dstOffset, 0, srcDevice, 0, ByteCount, 1,
                                          hipMemcpyDeviceToDevice, nullptr));
}

// ================================================================================================
hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                              size_t ByteCount, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyHtoAAsync, dstArray, dstOffset, srcHost, ByteCount, stream);
  STREAM_CAPTURE(hipMemcpyHtoAAsync, stream, dstArray, dstOffset, srcHost, ByteCount);
  HIP_RETURN_DURATION(ihipMemcpy2DToArray(dstArray, dstOffset, 0, srcHost, 0, ByteCount, 1,
                                          hipMemcpyHostToDevice, stream, true));
}

hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyHtoA, dstArray, dstOffset, srcHost, ByteCount);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpy2DToArray(dstArray, dstOffset, 0, srcHost, 0, ByteCount, 1,
                                          hipMemcpyHostToDevice, nullptr));
}

hipError_t hipMemcpyAtoH(void* dstHost, hipArray_t srcArray, size_t srcOffset, size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoH, dstHost, srcArray, srcOffset, ByteCount);
  CHECK_STREAM_CAPTURING();
  HIP_RETURN_DURATION(ihipMemcpy2DFromArray(dstHost, 0, srcArray, srcOffset, 0, ByteCount, 1,
                                            hipMemcpyDeviceToHost, nullptr));
}

// ================================================================================================
hipError_t hipMallocHost(void** ptr, size_t size) {
  HIP_INIT_API(hipMallocHost, ptr, size);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN_DURATION(ihipMalloc(ptr, size, CL_MEM_SVM_FINE_GRAIN_BUFFER), ReturnPtrValue(ptr));
}

hipError_t hipFreeHost(void* ptr) {
  HIP_INIT_API(hipFreeHost, ptr);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  size_t offset = 0;
  amd::Memory* memory_object = getMemoryObject(ptr, offset);
  if (memory_object != nullptr) {
    if (memory_object->getSvmPtr() == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(ihipFree(ptr));
}

hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  HIP_INIT_API(hipDrvMemcpy2DUnaligned, pCopy);

  HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*pCopy);

  HIP_RETURN(ihipMemcpyParam3D(&desc, nullptr));
}

hipError_t ihipMipmapArrayCreate(hipMipmappedArray_t* mipmapped_array_pptr,
                                 HIP_ARRAY3D_DESCRIPTOR* mipmapped_array_desc_ptr,
                                 unsigned int num_mipmap_levels, size_t offset = 0,
                                 amd::Memory* buffer = nullptr) {
  bool mipMapSupport = true;
  amd::Context& context = *hip::getCurrentDevice()->asContext();
  const std::vector<amd::Device*>& devices = context.devices();
  for (auto& dev : devices) {
    if (!dev->settings().checkExtension(ClKhrMipMapImage)) {
      mipMapSupport = false;
    }
  }
  if (mipMapSupport == false) {
    LogPrintfError("Mipmap not supported on one of the devices, Mip Level: %d", num_mipmap_levels);
    return hipErrorNotSupported;
  }
  if (!mipmapped_array_pptr || !mipmapped_array_desc_ptr) {
    return hipErrorInvalidValue;
  }
  unsigned int flags = hipArrayDefault | hipArrayLayered | hipArraySurfaceLoadStore |
                       hipArrayTextureGather;  // hipArrayCubemap isn't supported
  if (mipmapped_array_desc_ptr->Flags & (~flags)) {
    return hipErrorInvalidValue;
  }
  if (mipmapped_array_desc_ptr->Flags & hipArrayTextureGather) {
    // hipArrayTextureGather only works for 2D
    if (mipmapped_array_desc_ptr->Width == 0 || mipmapped_array_desc_ptr->Height == 0 ||
        mipmapped_array_desc_ptr->Depth > 0)
      return hipErrorInvalidValue;
  }

  const cl_channel_order channel_order =
      hip::getCLChannelOrder(mipmapped_array_desc_ptr->NumChannels, 0);
  const cl_channel_type channel_type =
      hip::getCLChannelType(mipmapped_array_desc_ptr->Format, hipReadModeElementType);
  const cl_mem_object_type image_type =
      hip::getCLMemObjectType(mipmapped_array_desc_ptr->Width, mipmapped_array_desc_ptr->Height,
                              mipmapped_array_desc_ptr->Depth, mipmapped_array_desc_ptr->Flags);
  hipError_t status = hipSuccess;
  // Create a new amd::Image with mipmap
  amd::Image* image =
      ihipImageCreate(channel_order, channel_type, image_type, mipmapped_array_desc_ptr->Width,
                      mipmapped_array_desc_ptr->Height, mipmapped_array_desc_ptr->Depth,
                      mipmapped_array_desc_ptr->Depth, 0 /* row pitch */, 0 /* slice pitch */,
                      num_mipmap_levels, offset, buffer, status);

  if (image == nullptr) {
    return status;
  }

  cl_mem cl_mem_obj = as_cl<amd::Memory>(image);
  *mipmapped_array_pptr = new hipMipmappedArray();
  (*mipmapped_array_pptr)->data = reinterpret_cast<void*>(cl_mem_obj);

  (*mipmapped_array_pptr)->desc = hip::getChannelFormatDesc(mipmapped_array_desc_ptr->NumChannels,
                                                            mipmapped_array_desc_ptr->Format);
  (*mipmapped_array_pptr)->type = image_type;
  (*mipmapped_array_pptr)->width = mipmapped_array_desc_ptr->Width;
  (*mipmapped_array_pptr)->height = mipmapped_array_desc_ptr->Height;
  (*mipmapped_array_pptr)->depth = mipmapped_array_desc_ptr->Depth;
  (*mipmapped_array_pptr)->min_mipmap_level = 0;
  (*mipmapped_array_pptr)->max_mipmap_level = num_mipmap_levels - 1;
  (*mipmapped_array_pptr)->flags = mipmapped_array_desc_ptr->Flags;
  (*mipmapped_array_pptr)->format = mipmapped_array_desc_ptr->Format;
  (*mipmapped_array_pptr)->num_channels = mipmapped_array_desc_ptr->NumChannels;

  return hipSuccess;
}

hipError_t ihipMipmappedArrayDestroy(hipMipmappedArray_t mipmapped_array_ptr) {
  if (mipmapped_array_ptr == nullptr) {
    return hipErrorInvalidValue;
  }

  cl_mem mem_obj = reinterpret_cast<cl_mem>(mipmapped_array_ptr->data);
  if (is_valid(mem_obj) == false) {
    return hipErrorInvalidValue;
  }

  auto image = as_amd(mem_obj);
  // Wait on the device, associated with the current memory object during allocation
  g_devices[image->getUserData().deviceId]->SyncAllStreams();
  image->release();

  delete mipmapped_array_ptr;
  return hipSuccess;
}

hipError_t ihipMipmappedArrayGetLevel(hipArray_t* level_array_pptr,
                                      hipMipmappedArray_t mipmapped_array_ptr,
                                      unsigned int mip_level) {
  if (level_array_pptr == nullptr) {
    return hipErrorInvalidValue;
  }
  if (mipmapped_array_ptr == nullptr) {
    return hipErrorInvalidHandle;
  }
  if (mip_level < mipmapped_array_ptr->min_mipmap_level ||
      mipmapped_array_ptr->max_mipmap_level < mip_level) {
    return hipErrorInvalidValue;
  }
  // Convert the raw data to amd::Image
  cl_mem cl_mem_obj = reinterpret_cast<cl_mem>(mipmapped_array_ptr->data);
  if (is_valid(cl_mem_obj) == false) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = as_amd(cl_mem_obj)->asImage();
  if (image == nullptr) {
    return hipErrorInvalidValue;
  }

  // Create new hip Array parameter and create an image view with new mip level.
  (*level_array_pptr) = new hipArray();
  (*level_array_pptr)->data = as_cl<amd::Memory>(
      image->createView(image->getContext(), image->getImageFormat(), NULL, mip_level, 0));

  // Copy the new width, height & depth details of the flag to hipArray.
  cl_mem cl_mip_mem_obj = reinterpret_cast<cl_mem>((*level_array_pptr)->data);
  if (is_valid(cl_mem_obj) == false) {
    return hipErrorInvalidValue;
  }

  // Fill the hip_array info from newly created amd::Image's view
  amd::Image* mipmap_image = as_amd(cl_mip_mem_obj)->asImage();
  (*level_array_pptr)->width = mipmap_image->getWidth();
  (*level_array_pptr)->height = mipmap_image->getHeight();
  (*level_array_pptr)->depth = mipmap_image->getDepth();

  const cl_mem_object_type image_type =
      hip::getCLMemObjectType((*level_array_pptr)->width, (*level_array_pptr)->height,
                              (*level_array_pptr)->depth, mipmapped_array_ptr->flags);
  (*level_array_pptr)->type = image_type;
  (*level_array_pptr)->Format = mipmapped_array_ptr->format;
  (*level_array_pptr)->desc = mipmapped_array_ptr->desc;
  (*level_array_pptr)->NumChannels = hip::getNumChannels((*level_array_pptr)->desc);
  (*level_array_pptr)->isDrv = 0;
  (*level_array_pptr)->textureType = 0;
  (*level_array_pptr)->flags = mipmapped_array_ptr->flags;

  amd::ScopedLock lock(hipArraySetLock);
  hip::hipArraySet.insert(*level_array_pptr);

  return hipSuccess;
}

hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* mipmapped_array_pptr,
                                   HIP_ARRAY3D_DESCRIPTOR* mipmapped_array_desc_ptr,
                                   unsigned int num_mipmap_levels) {
  HIP_INIT_API(hipMipmappedArrayCreate, mipmapped_array_pptr, mipmapped_array_desc_ptr,
               num_mipmap_levels);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(
      ihipMipmapArrayCreate(mipmapped_array_pptr, mipmapped_array_desc_ptr, num_mipmap_levels));
}

hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t mipmapped_array_ptr) {
  HIP_INIT_API(hipMipmappedArrayDestroy, mipmapped_array_ptr);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipMipmappedArrayDestroy(mipmapped_array_ptr));
}

hipError_t hipMipmappedArrayGetLevel(hipArray_t* level_array_pptr,
                                     hipMipmappedArray_t mipmapped_array_ptr,
                                     unsigned int mip_level) {
  HIP_INIT_API(hipMipmappedArrayGetLevel, level_array_pptr, mipmapped_array_ptr, mip_level);

  HIP_RETURN(ihipMipmappedArrayGetLevel(level_array_pptr, mipmapped_array_ptr, mip_level));
}

hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                   const hipChannelFormatDesc* desc, hipExtent extent,
                                   unsigned int numLevels, unsigned int flags) {
  HIP_INIT_API(hipMallocMipmappedArray, mipmappedArray, desc, extent, numLevels, flags);
  if (mipmappedArray == nullptr || desc == nullptr) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_ARRAY3D_DESCRIPTOR allocateArray = {extent.width,
                                          extent.height,
                                          extent.depth,
                                          hip::getArrayFormat(*desc),
                                          hip::getNumChannels(*desc),
                                          flags};
  if (!hip::CheckArrayFormat(*desc)) {
    return hipErrorInvalidValue;
  }
  HIP_RETURN(ihipMipmapArrayCreate(mipmappedArray, &allocateArray, numLevels));
}

hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  HIP_INIT_API(hipFreeMipmappedArray, mipmappedArray);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipMipmappedArrayDestroy(mipmappedArray));
}

hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,
                                     hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
  HIP_INIT_API(hipGetMipmappedArrayLevel, levelArray, mipmappedArray, level);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipMipmappedArrayGetLevel(levelArray, const_cast<hipMipmappedArray_t>(mipmappedArray),
                                        level));
}
// ================================================================================================
hipError_t hipExternalMemoryGetMappedMipmappedArray(
    hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc) {
  HIP_INIT_API(hipExternalMemoryGetMappedMipmappedArray, mipmap, extMem, mipmapDesc);
  if (mipmap == nullptr || extMem == nullptr || mipmapDesc == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();

  auto buf = reinterpret_cast<amd::ExternalBuffer*>(extMem);
  const device::Memory* devMem = buf->getDeviceMemory(*hip::getCurrentDevice()->devices()[0]);

  HIP_ARRAY3D_DESCRIPTOR allocateArray = {mipmapDesc->extent.width,
                                          mipmapDesc->extent.height,
                                          mipmapDesc->extent.depth,
                                          hip::getArrayFormat(mipmapDesc->formatDesc),
                                          hip::getNumChannels(mipmapDesc->formatDesc),
                                          mipmapDesc->flags};
  if (!hip::CheckArrayFormat(mipmapDesc->formatDesc)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipMipmapArrayCreate(mipmap, &allocateArray, mipmapDesc->numLevels,
                                   (size_t)mipmapDesc->offset, buf));
}

hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t dptr, size_t size,
                                          hipMemRangeHandleType handleType,
                                          unsigned long long flags) {
  HIP_INIT_API(hipMemGetHandleForAddressRange, handle, dptr, size, handleType, flags);

  // We do not support any flags at this time.
  if (dptr == nullptr || size == 0 || handleType != hipMemRangeHandleTypeDmaBufFd || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue;)
  }

  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  if (!device->GetHandleForAddressRange(dptr, size, handle)) {
    HIP_RETURN(hipErrorInvalidValue;)
  }

  HIP_RETURN(hipSuccess);
}

}  // namespace hip
