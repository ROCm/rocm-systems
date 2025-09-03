/* Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc.

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
#include "hip_conversions.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"

namespace hip {

// Forward declaraiton of a function
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);
hipError_t ihipMemPrefetchAsync(const void* dev_ptr, size_t count, hipMemLocation location,
                                hipStream_t stream);
hipError_t ihipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                         hipMemLocation location);

// Make sure HIP defines match ROCclr to avoid double conversion
static_assert(hipCpuDeviceId == amd::CpuDeviceId, "CPU device ID mismatch with ROCclr!");
static_assert(hipInvalidDeviceId == amd::InvalidDeviceId,
              "Invalid device ID mismatch with ROCclr!");

static_assert(static_cast<uint32_t>(hipMemAdviseSetReadMostly) == amd::MemoryAdvice::SetReadMostly,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetReadMostly) ==
                  amd::MemoryAdvice::UnsetReadMostly,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetPreferredLocation) ==
                  amd::MemoryAdvice::SetPreferredLocation,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetPreferredLocation) ==
                  amd::MemoryAdvice::UnsetPreferredLocation,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetAccessedBy) == amd::MemoryAdvice::SetAccessedBy,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetAccessedBy) ==
                  amd::MemoryAdvice::UnsetAccessedBy,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetCoarseGrain) ==
                  amd::MemoryAdvice::SetCoarseGrain,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetCoarseGrain) ==
                  amd::MemoryAdvice::UnsetCoarseGrain,
              "Enum mismatch with ROCclr!");

static_assert(static_cast<uint32_t>(hipMemRangeAttributeReadMostly) ==
                  amd::MemRangeAttribute::ReadMostly,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributePreferredLocation) ==
                  amd::MemRangeAttribute::PreferredLocation,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributeAccessedBy) ==
                  amd::MemRangeAttribute::AccessedBy,
              "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributeLastPrefetchLocation) ==
                  amd::MemRangeAttribute::LastPrefetchLocation,
              "Enum mismatch with ROCclr!");

// ================================================================================================
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  HIP_INIT_API(hipMallocManaged, dev_ptr, size, flags);

  CHECK_STREAM_CAPTURE_SUPPORTED();

  if ((dev_ptr == nullptr) || (size == 0) ||
      ((flags != hipMemAttachGlobal) && (flags != hipMemAttachHost))) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  HIP_RETURN(ihipMallocManaged(dev_ptr, size, 0, 0), *dev_ptr);
}

// ================================================================================================
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
  HIP_INIT_API(hipMemPrefetchAsync, dev_ptr, count, device, stream);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  hipMemLocation location;
  if (device == hipCpuDeviceId) {
    location.type = hipMemLocationTypeHost;
    location.id = hipCpuDeviceId;
  } else {
    location.type = hipMemLocationTypeDevice;
    location.id = device;
  }
  HIP_RETURN(ihipMemPrefetchAsync(dev_ptr, count, location, stream));
}

// ================================================================================================
hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location,
                                  unsigned int flags, hipStream_t stream) {
  HIP_INIT_API(hipMemPrefetchAsync_v2, dev_ptr, count, location, flags, stream);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  if (flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(ihipMemPrefetchAsync(dev_ptr, count, location, stream));
}

// ================================================================================================
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  HIP_INIT_API(hipMemAdvise, dev_ptr, count, advice, device);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  hipMemLocation location;
  if (device == hipCpuDeviceId) {
    location.type = hipMemLocationTypeHost;
    location.id = hipCpuDeviceId;
  } else {
    location.type = hipMemLocationTypeDevice;
    location.id = device;
  }

  HIP_RETURN(ihipMemAdvise(dev_ptr, count, advice, location));
}

// ================================================================================================
hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                           hipMemLocation location) {
  HIP_INIT_API(hipMemAdvise_v2, dev_ptr, count, advice, location);
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(ihipMemAdvise(dev_ptr, count, advice, location));
}

// ================================================================================================
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
  HIP_INIT_API(hipMemRangeGetAttribute, data, data_size, attribute, dev_ptr, count);

  if ((data == nullptr) || (data_size == 0) || (dev_ptr == nullptr) || (count == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Shouldn't matter for which device the interface is called
  amd::Device* dev = g_devices[0]->devices()[0];

  // Get the allocation attribute from AMD HMM
  if (!dev->GetSvmAttributes(&data, &data_size, reinterpret_cast<int*>(&attribute), 1, dev_ptr,
                             count)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count) {
  HIP_INIT_API(hipMemRangeGetAttributes, data, data_sizes, attributes, num_attributes, dev_ptr,
               count);

  if ((data == nullptr) || (data_sizes == nullptr) || (attributes == nullptr) ||
      (num_attributes == 0) || (dev_ptr == nullptr) || (count == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (*data_sizes > 0) {
    for (int i = 0; i < *data_sizes; i++) {
      if (!data[i]) {
        HIP_RETURN(hipErrorInvalidValue);
      }
    }
  }

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(dev_ptr, offset);
  if (memObj) {
    if (!(memObj->getMemFlags() & (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR))) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Shouldn't matter for which device the interface is called
  amd::Device* dev = g_devices[0]->devices()[0];
  // Get the allocation attributes from AMD HMM
  if (!dev->GetSvmAttributes(data, data_sizes, reinterpret_cast<int*>(attributes), num_attributes,
                             dev_ptr, count)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags) {
  HIP_INIT_API(hipStreamAttachMemAsync, stream, dev_ptr, length, flags);
  // stream can be null, length can be 0.
  if (dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  getStreamPerThread(stream);

  if (flags != hipMemAttachGlobal && flags != hipMemAttachHost && flags != hipMemAttachSingle) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (flags == hipMemAttachSingle && !stream) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // host-accessible region of system-allocated pageable memory.
  // This type of memory may only be specified if the device associated with the
  // stream reports a non-zero value for the device attribute hipDevAttrPageableMemoryAccess.
  hip::Stream* hip_stream = (stream == nullptr || stream == hipStreamLegacy)
                                ? hip::getCurrentDevice()->NullStream()
                                : hip::getStream(stream);
  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(dev_ptr, offset);
  if (memObj == nullptr) {
    if (hip_stream->GetDevice()->devices()[0]->info().hmmCpuMemoryAccessible_ == 0) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    if (length == 0) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  } else {
    if (memObj->getMemFlags() & (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR)) {
      if (length != 0 && memObj->getSize() != length) {
        HIP_RETURN(hipErrorInvalidValue);
      }
    }
  }

  // Unclear what should be done for this interface in AMD HMM, since it's generic SVM alloc
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align, bool use_host_ptr) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  } else if (size == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }

  assert((hip::host_context != nullptr) && "Current host context must be valid");
  amd::Context& ctx = *hip::host_context;

  const amd::Device& dev = *ctx.devices()[0];

  // Allocate SVM fine grain buffer with the forced host pointer, avoiding explicit memory
  // allocation in the device driver
  if (use_host_ptr) {
    // If the host pointer is already allocated, map it to svm fine grain buffer
    *ptr =
        amd::SvmBuffer::malloc(ctx, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_USE_HOST_PTR, size,
                               (align == 0) ? dev.info().memBaseAddrAlign_ : align, nullptr, *ptr);
  } else {
    *ptr = amd::SvmBuffer::malloc(ctx, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR, size,
                                  (align == 0) ? dev.info().memBaseAddrAlign_ : align);
  }
  if (*ptr == nullptr) {
    return hipErrorMemoryAllocation;
  }
  size_t offset = 0;  // this is ignored
  amd::Memory* memObj = getMemoryObject(*ptr, offset);
  if (memObj == nullptr) {
    return hipErrorMemoryAllocation;
  }
  // saves the current device id so that it can be accessed later
  memObj->getUserData().deviceId = hip::getCurrentDevice()->deviceId();

  ClPrint(amd::LOG_INFO, amd::LOG_API, "ihipMallocManaged ptr=0x%zx", *ptr);
  return hipSuccess;
}
// ================================================================================================
hipError_t ihipMemPrefetchAsync(const void* dev_ptr, size_t count, hipMemLocation location,
                                hipStream_t stream) {
  if ((dev_ptr == nullptr) || (count == 0)) {
    return hipErrorInvalidValue;
  }

  getStreamPerThread(stream);

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(dev_ptr, offset);
  if ((memObj != nullptr) && (count > (memObj->getSize() - offset))) {
    return hipErrorInvalidValue;
  }
  // Compute the type of prefetch
  const bool isHost = (location.type == hipMemLocationTypeHost);
  const bool isHostNuma = (location.type == hipMemLocationTypeHostNuma);
  const bool isHostCurrent = (location.type == hipMemLocationTypeHostNumaCurrent);
  const bool cpuAccess = isHost || isHostNuma || isHostCurrent;

  // Determine the target device index:
  //  - for host-prefetch and host-current, always use device 0
  //  - for host-NUMA or device-prefetch, use the provided id
  int targetDevice = (isHost || isHostCurrent) ? hipCpuDeviceId : location.id;

  amd::Device* dev = nullptr;
  if (cpuAccess == false) {
    if (static_cast<size_t>(targetDevice) >= g_devices.size()) {
      return hipErrorInvalidDevice;
    }
    dev = g_devices[targetDevice]->devices()[0];
    if (memObj == nullptr && !dev->info().hmmCpuMemoryAccessible_) {
      return hipErrorNotSupported;
    }
  }
  hip::Stream* hip_stream = nullptr;
  // Pick the specified stream or Null one from the provided target device
  if (cpuAccess == true) {
    hip_stream = (stream == nullptr || stream == hipStreamLegacy)
                     ? hip::getCurrentDevice()->NullStream()
                     : hip::getStream(stream);
  } else {
    dev = g_devices[targetDevice]->devices()[0];
    hip_stream = (stream == nullptr || stream == hipStreamLegacy)
                     ? g_devices[targetDevice]->NullStream()
                     : hip::getStream(stream);
  }

  if (hip_stream == nullptr) {
    return hipErrorInvalidValue;
  }

  amd::Command::EventWaitList waitList;
  amd::SvmPrefetchAsyncCommand* command = new amd::SvmPrefetchAsyncCommand(
      *hip_stream, waitList, dev_ptr, count, dev, cpuAccess, targetDevice);
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();
  return hipSuccess;
}
// ================================================================================================
hipError_t ihipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                         hipMemLocation location) {
  if ((dev_ptr == nullptr) || (count == 0)) {
    return hipErrorInvalidValue;
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    return hipErrorStreamCaptureUnsupported;
  }

  // Determine device and CPU access from location
  int targetDevice = hipCpuDeviceId;
  bool use_cpu = true;
  bool isAdviseReadMostly =
      (advice == hipMemAdviseSetReadMostly) || (advice == hipMemAdviseUnsetReadMostly);

  switch (location.type) {
    case hipMemLocationTypeDevice:
      targetDevice = location.id;
      use_cpu = false;
      break;
    case hipMemLocationTypeHostNuma:
      targetDevice = location.id;  // NUMA node ID
      use_cpu = true;
      break;
    case hipMemLocationTypeHost:
    case hipMemLocationTypeHostNumaCurrent:
      targetDevice = hipCpuDeviceId;
      use_cpu = true;
      break;
    default:
      return hipErrorInvalidValue;
  }

  if (!isAdviseReadMostly && !use_cpu && (static_cast<size_t>(targetDevice) >= g_devices.size())) {
    return hipErrorInvalidDevice;
  }

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(dev_ptr, offset);
  if (memObj && count > (memObj->getSize() - offset)) {
    return hipErrorInvalidValue;
  }

  amd::Device* dev = (use_cpu || isAdviseReadMostly) ? g_devices[0]->devices()[0]
                                                     : g_devices[targetDevice]->devices()[0];

  // Set the allocation attributes in AMD HMM
  if (!dev->SetSvmAttributes(dev_ptr, count, static_cast<amd::MemoryAdvice>(advice), use_cpu,
                             targetDevice)) {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}
}  // namespace hip
