/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include "hip_mempool_impl.hpp"

namespace hip {
/**
 * API interfaces
 */
extern hipError_t ihipFree(void* ptr);

// Static declarations
namespace {
// ================================================================================================
inline bool IsMemPoolValid(MemoryPool* mem_pool) {
  bool result = false;
  for (auto it : g_devices) {
    if (result = it->IsMemoryPoolValid(mem_pool) == true) {
      break;
    }
  }
  return result;
}
}  // namespace

// ================================================================================================
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  HIP_INIT_API(hipDeviceGetDefaultMemPool, mem_pool, device);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (device < 0 || device >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(g_devices[device]->GetDefaultMemoryPool());
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  HIP_INIT_API(hipDeviceSetMemPool, device, mem_pool);
  if ((mem_pool == nullptr) || (device >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }
  auto poolDevice = reinterpret_cast<hip::MemoryPool*>(mem_pool)->Device();
  if (poolDevice->deviceId() != device) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  g_devices[device]->SetCurrentMemoryPool(reinterpret_cast<hip::MemoryPool*>(mem_pool));
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  HIP_INIT_API(hipDeviceGetMemPool, mem_pool, device);
  if ((mem_pool == nullptr) || (device >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(g_devices[device]->GetCurrentMemoryPool());
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  HIP_INIT_API(hipMallocAsync, dev_ptr, size, stream);
  if (dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  getStreamPerThread(stream);
  if (size == 0) {
    *dev_ptr = nullptr;
    HIP_RETURN(hipSuccess);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  auto hip_stream =
      (stream == nullptr || stream == hipStreamLegacy) ? hip::getCurrentDevice()->NullStream() : s;
  auto device = hip_stream->GetDevice();
  auto mem_pool = device->GetCurrentMemoryPool();

  // Return error if any stream other than the current stream is in capture mode
  if (device->StreamCaptureBlocking()) {
    if (s->GetCaptureStatus() != hipStreamCaptureStatusActive) {
      return hipErrorStreamCaptureUnsupported;
    }
  }

  STREAM_CAPTURE(hipMallocAsync, stream, reinterpret_cast<hipMemPool_t>(mem_pool), size, dev_ptr);

  *dev_ptr = mem_pool->AllocateMemory(size, hip_stream);
  if (*dev_ptr == nullptr) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
// @note: Runtime needs the new command for MT path, since the app can execute hipFreeAsync()
// before the graph execution is done. Hence there could be a race condition between
// memory allocatiom in graph, which occurs in a worker thread, and host execution of hipFreeAsync
class FreeAsyncCommand : public amd::Command {
 private:
  void* ptr_;          //!< Virtual address for asynchronious free
  hip::Event* event_;  //!< HIP event, associated with this memory release

 public:
  FreeAsyncCommand(amd::HostQueue& queue, void* ptr, hip::Event* event)
      : amd::Command(queue, 1, amd::Event::nullWaitList), ptr_(ptr), event_(event) {}

  virtual void submit(device::VirtualDevice& device) final {
    size_t offset = 0;
    auto memory = getMemoryObject(ptr_, offset);
    if (memory != nullptr) {
      auto id = memory->getUserData().deviceId;
      if (!g_devices[id]->FreeMemory(memory, static_cast<hip::Stream*>(queue()), event_)) {
        // @note It's not the most optimal logic.
        // The current implementation has unconditional waits
        if (ihipFree(ptr_) != hipSuccess) {
          setStatus(CL_INVALID_OPERATION);
        }
      }
    }
  }
};

// ================================================================================================
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  HIP_INIT_API(hipFreeAsync, dev_ptr, stream);

  getStreamPerThread(stream);

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  auto hip_stream =
      (stream == nullptr || stream == hipStreamLegacy) ? hip::getCurrentDevice()->NullStream() : s;

  auto device = hip_stream->GetDevice();
  // Return error if any stream other than the current stream is in capture mode
  if (device->StreamCaptureBlocking()) {
    if (s->GetCaptureStatus() != hipStreamCaptureStatusActive) {
      return hipErrorStreamCaptureUnsupported;
    }
  }

  if (dev_ptr == nullptr) {
    HIP_RETURN(hipSuccess);
  }

  STREAM_CAPTURE(hipFreeAsync, stream, dev_ptr);

  hip::Event* event = nullptr;
  bool graph_in_use = false;

  if (!AMD_DIRECT_DISPATCH) {
    // Note: This logic is required for multithreading execution only and
    // can reduce mempool reserved memory efficiency
    for (auto dev : g_devices) {
      graph_in_use |= dev->GetGraphMemoryPool()->GraphInUse();
    }
  }

  if (!graph_in_use) {
    size_t offset = 0;
    auto memory = getMemoryObject(dev_ptr, offset);
    if (memory != nullptr) {
      auto id = memory->getUserData().deviceId;
      if (!g_devices[id]->FreeMemory(memory, hip_stream, event)) {
        // @note It's not the most optimal logic.
        // The current implementation has unconditional waits
        return ihipFree(dev_ptr);
      }
    } else {
      HIP_RETURN(hipErrorInvalidValue);
    }
  } else {
    if (!AMD_DIRECT_DISPATCH) {
      // Add a marker to the stream to trace availability of this memory
      // Note: MT path requires the marker command to be created in the host thread,
      // so the queue thread could process it, because creating a command from the queue thread
      // may block the execution
      event = new hip::Event(0);
      if (event != nullptr) {
        if (hipSuccess != event->addMarker(hip_stream, nullptr)) {
          delete event;
          event = nullptr;
        } else {
          // Make sure runtime sends a notification to the worker thread
          auto result = event->ready();
        }
      }
    }

    auto cmd = new FreeAsyncCommand(*hip_stream, dev_ptr, event);
    if (cmd == nullptr) {
      HIP_RETURN(hipErrorUnknown);
    }
    cmd->enqueue();
    cmd->release();
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  HIP_INIT_API(hipMemPoolTrimTo, mem_pool, min_bytes_to_hold);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::MemoryPool* hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  hip_mem_pool->TrimTo(min_bytes_to_hold);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  HIP_INIT_API(hipMemPoolSetAttribute, mem_pool, attr, value);
  if (mem_pool == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  HIP_RETURN(hip_mem_pool->SetAttribute(attr, value));
}

// ================================================================================================
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  HIP_INIT_API(hipMemPoolGetAttribute, mem_pool, attr, value);
  if (mem_pool == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  HIP_RETURN(hip_mem_pool->GetAttribute(attr, value));
}

// ================================================================================================
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count) {
  HIP_INIT_API(hipMemPoolSetAccess, mem_pool, desc_list, count);
  if ((mem_pool == nullptr) || (desc_list == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (count > g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  for (int i = 0; i < count; ++i) {
    if (desc_list[i].location.type == hipMemLocationTypeDevice) {
      if (desc_list[i].location.id >= g_devices.size()) {
        HIP_RETURN(hipErrorInvalidDevice);
      }
      if (desc_list[i].flags == hipMemAccessFlagsProtNone) {
        HIP_RETURN(hipErrorInvalidDevice);
      }
      if (desc_list[i].flags > hipMemAccessFlagsProtReadWrite) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      auto device = g_devices[desc_list[i].location.id];
      hip_mem_pool->SetAccess(device, desc_list[i].flags);
    } else {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location) {
  HIP_INIT_API(hipMemPoolGetAccess, flags, mem_pool, location);
  if ((mem_pool == nullptr) || (location == nullptr) || (flags == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  if (location->type == hipMemLocationTypeDevice) {
    if (location->id >= g_devices.size()) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    auto device = g_devices[location->id];
    hip_mem_pool->GetAccess(device, flags);
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  HIP_INIT_API(hipMemPoolCreate, mem_pool, pool_props);
  if ((mem_pool == nullptr) || (pool_props == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // validate hipMemAllocationType value
  if (pool_props->allocType != hipMemAllocationTypePinned) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Make sure the pool creation occurs on a valid device
  if ((pool_props->location.type != hipMemLocationTypeDevice) ||
      (pool_props->location.id >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (IS_WINDOWS && pool_props->handleTypes == hipMemHandleTypePosixFileDescriptor) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }
  auto device = g_devices[pool_props->location.id];
  auto pool = new hip::MemoryPool(device, pool_props);
  if (pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(pool);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  HIP_INIT_API(hipMemPoolDestroy, mem_pool);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }
  hip::MemoryPool* hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);

  if (!IsMemPoolValid(hip_mem_pool)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto device = hip_mem_pool->Device();
  if (hip_mem_pool == device->GetDefaultMemoryPool()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip_mem_pool->ReleaseFreedMemory();

  // Force default pool if the current one is destroyed
  if (hip_mem_pool == device->GetCurrentMemoryPool()) {
    device->SetCurrentMemoryPool(device->GetDefaultMemoryPool());
  }

  hip_mem_pool->release();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream) {
  HIP_INIT_API(hipMallocFromPoolAsync, dev_ptr, size, mem_pool, stream);
  if ((dev_ptr == nullptr) || (mem_pool == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  getStreamPerThread(stream);
  if (size == 0) {
    *dev_ptr = nullptr;
    HIP_RETURN(hipSuccess);
  }
  STREAM_CAPTURE(hipMallocAsync, stream, mem_pool, size, dev_ptr);

  auto mpool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  auto hip_stream = (stream == nullptr || stream == hipStreamLegacy)
                        ? hip::getCurrentDevice()->NullStream()
                        : reinterpret_cast<hip::Stream*>(stream);
  *dev_ptr = mpool->AllocateMemory(size, hip_stream);
  if (*dev_ptr == nullptr) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags) {
  HIP_INIT_API(hipMemPoolExportToShareableHandle, shared_handle, mem_pool, handle_type, flags);
  if (mem_pool == nullptr || shared_handle == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto mpool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  if ((handle_type != mpool->Properties().handleTypes) || (handle_type == hipMemHandleTypeNone)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto handle = mpool->Export();
  if (!handle) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *reinterpret_cast<amd::Os::FileDesc*>(shared_handle) = handle;

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags) {
  HIP_INIT_API(hipMemPoolImportFromShareableHandle, mem_pool, shared_handle, handle_type, flags);
  if (mem_pool == nullptr || shared_handle == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (handle_type == hipMemHandleTypeNone) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto device = g_devices[0];
  auto pool = new hip::MemoryPool(device, nullptr, true);
  if (pool == nullptr) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  // Note: The interface casts the integer value of file handle under Linux into void*,
  // but compiler may not allow to cast it back. Hence, make a cast with a union...
  union {
    amd::Os::FileDesc desc;
    void* ptr;
  } handle;
  handle.ptr = shared_handle;
  if (!pool->Import(handle.desc)) {
    pool->release();
    HIP_RETURN(hipErrorOutOfMemory);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(pool);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* ptr) {
  HIP_INIT_API(hipMemPoolExportPointer, export_data, ptr);
  if (export_data == nullptr || ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t offset = 0;
  auto memory = getMemoryObject(ptr, offset);
  if (memory != nullptr) {
    auto id = memory->getUserData().deviceId;
    // Note: export_data must point to 64 bytes of shared memory
    auto shared = reinterpret_cast<hip::SharedMemPointer*>(export_data);

    if (!g_devices[id]->devices()[0]->IpcCreate(ptr, &shared->size_, &shared->handle_[0],
                                                &shared->offset_)) {
      HIP_RETURN(hipErrorOutOfMemory);
    }
  } else {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolImportPointer(void** ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data) {
  HIP_INIT_API(hipMemPoolImportPointer, ptr, mem_pool, export_data);
  if (mem_pool == nullptr || export_data == nullptr || ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto mpool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  auto shared = reinterpret_cast<hip::SharedMemPointer*>(export_data);
  if (!mpool->Device()->devices()[0]->IpcAttach(&shared->handle_[0], shared->size_, shared->offset_,
                                                0, ptr)) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  size_t offset = 0;
  auto memory = getMemoryObject(*ptr, offset);
  mpool->AddBusyMemory(memory);
  mpool->retain();
  HIP_RETURN(hipSuccess);
}
}  // namespace hip
