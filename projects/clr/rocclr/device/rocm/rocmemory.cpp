/* Copyright (c) 2008 - 2025 Advanced Micro Devices, Inc.

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

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "CL/cl_ext.h"

#include "utils/util.hpp"
#include "device/device.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocglinterop.hpp"
#include "thread/monitor.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "platform/interop_gl.hpp"
#include "platform/external_memory.hpp"

namespace amd::roc {

// ======================================= roc::Memory ============================================
Memory::Memory(const roc::Device& dev, amd::Memory& owner)
    : device::Memory(owner),
      dev_(dev),
      deviceMemory_(nullptr),
      kind_(MEMORY_KIND_NORMAL),
      amdImageDesc_(nullptr),
      persistent_host_ptr_(nullptr),
      pinnedMemory_(nullptr) {}

Memory::Memory(const roc::Device& dev, size_t size)
    : device::Memory(size),
      dev_(dev),
      deviceMemory_(nullptr),
      kind_(MEMORY_KIND_NORMAL),
      amdImageDesc_(nullptr),
      persistent_host_ptr_(nullptr),
      pinnedMemory_(nullptr) {}

Memory::~Memory() {
  // Destory pinned memory
  if (flags_ & PinnedMemoryAlloced) {
    pinnedMemory_->release();
  }

  dev().removeVACache(this);
  if (nullptr != mapMemory_) {
    mapMemory_->release();
  }
}

bool Memory::allocateMapMemory(size_t allocationSize) {
  assert(mapMemory_ == nullptr);

  void* mapData = nullptr;

  amd::Memory* mapMemory = dev().findMapTarget(owner()->getSize());
  if (mapMemory == nullptr) {
    // Create buffer object to contain the map target.
    mapMemory = new (dev().context())
        amd::Buffer(dev().context(), CL_MEM_ALLOC_HOST_PTR, owner()->getSize());

    if ((mapMemory == nullptr) || (!mapMemory->create())) {
      LogError("[OCL] Fail to allocate map target object");
      if (mapMemory) {
        mapMemory->release();
      }
      return false;
    }

    roc::Memory* hsaMapMemory = reinterpret_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev_));
    if (hsaMapMemory == nullptr) {
      mapMemory->release();
      return false;
    }
  }

  mapMemory_ = mapMemory;

  return true;
}

void* Memory::allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region, uint mapFlags,
                             size_t* rowPitch, size_t* slicePitch) {
  // Map/Unmap must be serialized.
  amd::ScopedLock lock(owner()->lockMemoryOps());

  incIndMapCount();
  // If the device backing storage is direct accessible, use it.
  if (isHostMemDirectAccess()) {
    if (owner()->getHostMem() != nullptr) {
      return (static_cast<char*>(owner()->getHostMem()) + origin[0]);
    }

    return (static_cast<char*>(deviceMemory_) + origin[0]);
  }
  if (IsPersistentDirectMap()) {
    return (static_cast<char*>(persistent_host_ptr_) + origin[0]);
  }

  // Allocate one if needed.
  if (indirectMapCount_ == 1) {
    if (!allocateMapMemory(owner()->getSize())) {
      decIndMapCount();
      DevLogPrintfError("Cannot allocate Map memory for size: %u", owner()->getSize());
      return nullptr;
    }
  } else {
    // Did the map resource allocation fail?
    if (mapMemory_ == nullptr) {
      LogError("Could not map target resource");
      return nullptr;
    }
  }

  void* mappedMemory = nullptr;
  void* hostMem = owner()->getHostMem();

  if (owner()->getSvmPtr() != nullptr) {
    owner()->commitSvmMemory();
    mappedMemory = owner()->getSvmPtr();
  } else if (hostMem != nullptr) {  // Otherwise, check for host memory.
    return (reinterpret_cast<address>(hostMem) + origin[0]);
  } else {
    mappedMemory = reinterpret_cast<address>(mapMemory_->getHostMem()) + origin[0];
  }

  return mappedMemory;
}

void Memory::decIndMapCount() {
  // Map/Unmap must be serialized.
  amd::ScopedLock lock(owner()->lockMemoryOps());

  if (indirectMapCount_ == 0) {
    LogError("decIndMapCount() called when indirectMapCount_ already zero");
    return;
  }

  // Decrement the counter and release indirect map if it's the last op
  if (--indirectMapCount_ == 0 && mapMemory_ != nullptr) {
    if (!dev().addMapTarget(mapMemory_)) {
      // Release the buffer object containing the map data.
      mapMemory_->release();
    }
    mapMemory_ = nullptr;
  }
}

void* Memory::cpuMap(device::VirtualDevice& vDev, uint flags, uint startLayer, uint numLayers,
                     size_t* rowPitch, size_t* slicePitch) {
  // Create the map target.
  void* mapTarget = allocMapTarget(amd::Coord3D(0), amd::Coord3D(0), 0, rowPitch, slicePitch);

  assert(mapTarget != nullptr);

  // CPU access requires a stall of the current queue
  static_cast<roc::VirtualGPU&>(vDev).releaseGpuMemoryFence();

  if (!isHostMemDirectAccess() && !IsPersistentDirectMap()) {
    if (!vDev.blitMgr().readBuffer(*this, mapTarget, amd::Coord3D(0), amd::Coord3D(size()), true)) {
      decIndMapCount();
      DevLogError("Cannot read buffer");
      return nullptr;
    }
  }

  return mapTarget;
}

void Memory::cpuUnmap(device::VirtualDevice& vDev) {
  if (!isHostMemDirectAccess() && !IsPersistentDirectMap()) {
    if (!vDev.blitMgr().writeBuffer(mapMemory_->getHostMem(), *this, amd::Coord3D(0),
                                    amd::Coord3D(size()), true)) {
      LogError("Fail sync the device memory on cpuUnmap");
    }
    // Wait on CPU for the transfer
    static_cast<roc::VirtualGPU&>(vDev).releaseGpuMemoryFence();
  }
  decIndMapCount();
}

// ================================================================================================
hsa_status_t Memory::interopMapBuffer(amd::Os::FileDesc fdn) {
  hsa_agent_t agent = dev().getBackendDevice();
  size_t size;
  size_t metadata_size = 0;
  void* metadata;
#if IS_WINDOWS
  int fd = 0;
  assert(!"Unimplemented");
#else
  auto fd = fdn;
#endif
  hsa_status_t status = Hsa::interop_map_buffer(1, &agent, fd, 0, &size, &interop_deviceMemory_,
                                                &metadata_size, (const void**)&metadata);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Map Interop memory %p, size 0x%zx", interop_deviceMemory_,
          size);
  deviceMemory_ = static_cast<char*>(interop_deviceMemory_);  // + out.buf_offset;
  if (status != HSA_STATUS_SUCCESS) return status;
  // if map_buffer wrote a legitimate SRD, copy it to amdImageDesc_
  // Note: Check if amdImageDesc_ is valid, because VA library maps linear planes of YUV image
  // as buffers for processing in HIP later
  if ((amdImageDesc_ != nullptr) && (metadata_size != 0) &&
      (reinterpret_cast<hsa_amd_image_descriptor_t*>(metadata)->deviceID ==
       amdImageDesc_->deviceID)) {
    memcpy(amdImageDesc_, metadata, metadata_size);
  }
  kind_ = MEMORY_KIND_INTEROP;
  assert(deviceMemory_ != nullptr && "Interop map failed to produce a pointer!");
  return status;
}

// Setup an interop buffer (dmabuf handle) as an OpenCL buffer
// ================================================================================================
bool Memory::createInteropBuffer(GLenum targetType, int miplevel) {
#if IS_WINDOWS
  return false;
#else
  assert(owner()->isInterop() && "Object is not an interop object.");

  mesa_glinterop_export_in in = {0};
  mesa_glinterop_export_out out = {0};

  in.version = MESA_GLINTEROP_EXPORT_IN_VERSION;
  out.version = MESA_GLINTEROP_EXPORT_OUT_VERSION;

  if (owner()->getMemFlags() & CL_MEM_READ_ONLY)
    in.access = MESA_GLINTEROP_ACCESS_READ_ONLY;
  else if (owner()->getMemFlags() & CL_MEM_WRITE_ONLY)
    in.access = MESA_GLINTEROP_ACCESS_WRITE_ONLY;
  else
    in.access = MESA_GLINTEROP_ACCESS_READ_WRITE;

  hsa_agent_t agent = dev().getBackendDevice();
  uint32_t id;
  Hsa::agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_CHIP_ID), &id);

  static constexpr int MaxMetadataSizeDwords = 64;
  static constexpr int MaxMetadataSizeBytes = MaxMetadataSizeDwords * sizeof(int);
  amdImageDesc_ = reinterpret_cast<hsa_amd_image_descriptor_t*>(new int[MaxMetadataSizeDwords + 2]);
  if (amdImageDesc_ == nullptr) {
    return false;
  }
  amdImageDesc_->version = 1;
  amdImageDesc_->deviceID = AmdVendor << 16 | id;

  in.target = targetType;
  in.obj = owner()->getInteropObj()->asGLObject()->getGLName();
  in.miplevel = miplevel;
  in.out_driver_data_size = MaxMetadataSizeBytes;
  in.out_driver_data = &amdImageDesc_->data[0];

  const auto& glenv = owner()->getContext().glenv();
  if (glenv->isEGL()) {
    if (!MesaInterop::Export(in, out, MesaInterop::MESA_INTEROP_EGL, glenv->getEglDpy(),
                             glenv->getEglOrigCtx()))
      return false;
  } else {
    if (!MesaInterop::Export(in, out, MesaInterop::MESA_INTEROP_GLX, glenv->getDpy(),
                             glenv->getOrigCtx()))
      return false;
  }

  if (interopMapBuffer(out.dmabuf_fd) != HSA_STATUS_SUCCESS) return false;

  close(out.dmabuf_fd);
  deviceMemory_ = static_cast<char*>(interop_deviceMemory_) + out.buf_offset;

  return true;
#endif
}

void Memory::destroyInteropBuffer() {
  assert(kind_ == MEMORY_KIND_INTEROP && "Memory must be interop type.");
  Hsa::interop_unmap_buffer(interop_deviceMemory_);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Unmap GL memory %p", deviceMemory_);
  deviceMemory_ = nullptr;
}

bool Memory::pinSystemMemory(void* hostPtr, size_t size) {
  size_t pinAllocSize;
  const static bool SysMem = true;
  amd::Memory* amdMemory = nullptr;
  amd::Memory* amdParent = owner()->parent();

  // If memory has a direct access already, then skip the host memory pinning
  if (isHostMemDirectAccess()) {
    return true;
  }

  // Memory was pinned already
  if (flags_ & PinnedMemoryAlloced) {
    return true;
  }

  // Check if runtime allocates a parent object
  if (amdParent != nullptr) {
    Memory* parent = dev().getRocMemory(amdParent);
    amd::Memory* amdPinned = parent->pinnedMemory_;
    if (amdPinned != nullptr) {
      // Create view on the parent's pinned memory
      amdMemory = new (amdPinned->getContext())
          amd::Buffer(*amdPinned, 0, owner()->getOrigin(), owner()->getSize());
      if ((amdMemory != nullptr) && !amdMemory->create()) {
        amdMemory->release();
        amdMemory = nullptr;
      }
    }
  }

  if (amdMemory == nullptr) {
    amdMemory = new (dev().context()) amd::Buffer(dev().context(), CL_MEM_USE_HOST_PTR, size);
    if ((amdMemory != nullptr) && !amdMemory->create(hostPtr, SysMem)) {
      amdMemory->release();
      return false;
    }
  }

  // Get device memory for this virtual device
  // @note: This will force real memory pinning
  Memory* srcMemory = dev().getRocMemory(amdMemory);

  if (srcMemory == nullptr) {
    // Release memory
    amdMemory->release();
    return false;
  } else {
    pinnedMemory_ = amdMemory;
    flags_ |= PinnedMemoryAlloced;
  }

  return true;
}

void Memory::syncCacheFromHost(VirtualGPU& gpu, device::Memory::SyncFlags syncFlags) {
  amd::ScopedLock lock(owner()->lockMemoryOps());
  // If the last writer was another GPU, then make a writeback
  if (!isHostMemDirectAccess() && (owner()->getLastWriter() != nullptr) &&
      (&dev() != owner()->getLastWriter())) {
    // Make sure GPU finished operation before synchronization with the backing store
    gpu.releaseGpuMemoryFence();
    mgpuCacheWriteBack(gpu);
  }

  // If host memory doesn't have direct access, then we have to synchronize
  if (!isHostMemDirectAccess() && (nullptr != owner()->getHostMem())) {
    bool hasUpdates = true;
    amd::Memory* amdParent = owner()->parent();

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (amdParent != nullptr)) {
      Memory* gpuMemory = dev().getRocMemory(amdParent);

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      gpuMemory->syncCacheFromHost(gpu, syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((version_ == owner()->getVersion()) || (&dev() == owner()->getLastWriter())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          Memory* gpuSub = reinterpret_cast<Memory*>(devSub);
          gpuSub->syncCacheFromHost(gpu, syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because this GPU device was the last writer
    if (&dev() != owner()->getLastWriter()) {
      // Update the latest version
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);

    // If host memory was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      Memory& pinned = *dev().getRocMemory(pinnedMemory_);
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().copyBuffer(pinned, *this, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result =
            gpu.blitMgr().copyBufferToImage(pinned, *this, origin, origin, image.getRegion(),
                                            Entire, image.getRowPitch(), image.getSlicePitch());
      }
    }

    if (!result) {
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().writeBuffer(owner()->getHostMem(), *this, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = gpu.blitMgr().writeImage(owner()->getHostMem(), *this, origin, image.getRegion(),
                                          image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    gpu.releaseGpuMemoryFence();

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}

void Memory::syncHostFromCache(device::VirtualDevice* vDev, device::Memory::SyncFlags syncFlags) {
  VirtualGPU* gpu = (vDev != nullptr) ? reinterpret_cast<VirtualGPU*>(vDev) : dev().xferQueue();
  // Sanity checks
  assert(owner() != nullptr);

  // If host memory doesn't have direct access, then we have to synchronize
  if (!isHostMemDirectAccess()) {
    bool hasUpdates = true;
    amd::Memory* amdParent = owner()->parent();

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (amdParent != nullptr)) {
      device::Memory* m = dev().getRocMemory(amdParent);

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      m->syncHostFromCache(gpu, syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((nullptr == owner()->getLastWriter()) || (version_ == owner()->getVersion())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      amd::ScopedLock lock(owner()->lockMemoryOps());
      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          Memory* gpuSub = reinterpret_cast<Memory*>(devSub);
          gpuSub->syncHostFromCache(gpu, syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because CPU was the last writer
    if (nullptr != owner()->getLastWriter()) {
      // Mark parent as up to date, set our version accordingly
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);
    // If device on the provided queue doesn't match the device memory was allocated,
    // then use blit manager on device
    const auto& bltMgr = (&gpu->dev() != &dev()) ? dev().xferMgr() : gpu->blitMgr();

    // If backing store was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      Memory& pinned = *dev().getRocMemory(pinnedMemory_);
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = bltMgr.copyBuffer(*this, pinned, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = bltMgr.copyImageToBuffer(*this, pinned, origin, origin, image.getRegion(), Entire,
                                          image.getRowPitch(), image.getSlicePitch());
      }
    }

    // Just do a basic host read
    if (!result) {
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = bltMgr.readBuffer(*this, owner()->getHostMem(), origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = bltMgr.readImage(*this, owner()->getHostMem(), origin, image.getRegion(),
                                  image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}

void Memory::mgpuCacheWriteBack(VirtualGPU& gpu) {
  // Lock memory object, so only one write back can occur
  amd::ScopedLock lock(owner()->lockMemoryOps());

  // Attempt to allocate a staging buffer if don't have any
  if (owner()->getHostMem() == nullptr) {
    if (nullptr != owner()->getSvmPtr()) {
      owner()->commitSvmMemory();
      owner()->setHostMem(owner()->getSvmPtr());
    } else {
      static const bool forceAllocHostMem = true;
      owner()->allocHostMemory(nullptr, forceAllocHostMem);
    }
  }

  // Make synchronization
  if (owner()->getHostMem() != nullptr) {
    //! \note Ignore pinning result
    bool ok = pinSystemMemory(owner()->getHostMem(), owner()->getSize());
    owner()->cacheWriteBack(&gpu);
  }
}

// ==================================== roc::Buffer ===============================================
Buffer::Buffer(const roc::Device& dev, amd::Memory& owner) : roc::Memory(dev, owner) {}

Buffer::Buffer(const roc::Device& dev, size_t size) : roc::Memory(dev, size) {}

Buffer::~Buffer() {
  if (owner() == nullptr) {
    dev().memFree(deviceMemory_, size());
  } else {
    destroy();

    if (owner()->ipcShared()) {
      // Detach the memory from HSA
      auto hsa_status = Hsa::ipc_memory_detach(owner()->getSvmPtr());
      if (hsa_status != HSA_STATUS_SUCCESS) {
        LogPrintfError("HSA failed to detach memory with status: %d", hsa_status);
      }
    }
  }
}

// ================================================================================================
void Buffer::destroy() {
  if (owner()->parent() != nullptr) {
    return;
  }

  if (kind_ == MEMORY_KIND_INTEROP) {
    destroyInteropBuffer();
    return;
  }

  cl_mem_flags memFlags = owner()->getMemFlags();

  if (owner()->getSvmPtr() != nullptr) {
    if (dev().forceFineGrain(owner()) || dev().isFineGrainedSystem(true)) {
      memFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }
    const bool isFineGrain = memFlags & CL_MEM_SVM_FINE_GRAIN_BUFFER;

    if (memFlags & ROCCLR_MEM_PHYMEM) {
      // If this is physical memory, dont call hsa free function, since device mem was never created
      dev().deviceVmemRelease(owner()->getUserData().hsa_handle);
      return;
    }

    if (kind_ != MEMORY_KIND_PTRGIVEN) {
      if (isFineGrain) {
        if (memFlags & (CL_MEM_ALLOC_HOST_PTR)) {
          if (dev().info().hmmSupported_) {
            // AMD HMM path. Release reserved system memory
            dev().releaseMemory(deviceMemory_, size());
          } else {
            dev().memFree(deviceMemory_, size());
          }
        } else if (memFlags & ROCCLR_MEM_HSA_SIGNAL_MEMORY) {
          if (HSA_STATUS_SUCCESS != Hsa::signal_destroy(signal_)) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM, "hsa_signal_destroy failed");
          }
          deviceMemory_ = nullptr;
        } else {
          dev().memFree(deviceMemory_, size());
        }
      } else {
        dev().memFree(deviceMemory_, size());
      }
    } else {
      if (memFlags & CL_MEM_USE_HOST_PTR) {
        // unlock svm host pointer from memory pool
        if (!dev().info().hmmSupported_) {
          Hsa::memory_unlock(owner()->getSvmPtr());
        }
        // destroy system memory
        if (!(amd::Os::releaseMemory(deviceMemory_, size()))) {
          ClPrint(amd::LOG_ERROR, amd::LOG_MEM, "munmap failed");
        }
      }
    }

    if ((deviceMemory_ != nullptr) && (dev().settings().apuSystem_ || !isFineGrain)) {
      const_cast<Device&>(dev()).updateFreeMemory(size(), true);
    }

    return;
  }

  if (deviceMemory_ != nullptr) {
    bool needUnlockHostMem = false;
    if (deviceMemory_ != owner()->getHostMem()) {
      // if they are identical, the host pointer will be
      // deallocated later on => avoid double deallocation
      if (isHostMemDirectAccess()) {
        needUnlockHostMem = true;
      } else {
        dev().memFree(deviceMemory_, size());
        const_cast<Device&>(dev()).updateFreeMemory(size(), true);
      }
    } else {
      if (!(memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
        dev().memFree(deviceMemory_, size());
        if (dev().settings().apuSystem_) {
          const_cast<Device&>(dev()).updateFreeMemory(size(), true);
        }
      } else if ((memFlags & CL_MEM_ALLOC_HOST_PTR) &&
                 (owner()->getContext().devices().size() == 1)) {
        dev().hostFree(deviceMemory_, size());
      } else if (isHostMemDirectAccess()) {
        needUnlockHostMem = true;
      }
    }

    if (needUnlockHostMem) {
      if (memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
        if (dev().agent_profile() != HSA_PROFILE_FULL) Hsa::memory_unlock(owner()->getHostMem());
      }
    }
  }

  if (memFlags & CL_MEM_USE_HOST_PTR) {
    if (dev().agent_profile() == HSA_PROFILE_FULL) {
      Hsa::memory_deregister(owner()->getHostMem(), size());
    }
  }
}

// ================================================================================================
bool Buffer::create(bool alloc_local) {
  if (owner() == nullptr) {
    if (alloc_local) {
      deviceMemory_ = dev().deviceLocalAlloc(size());
      if (deviceMemory_ != nullptr) {
        flags_ |= HostMemoryDirectAccess;
        return true;
      }
    } else {
      deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
      if (deviceMemory_ != nullptr) {
        flags_ |= HostMemoryDirectAccess;
        return true;
      }
    }
    return false;
  }

  if (owner()->ipcShared()) {
    void* orig_dev_ptr = nullptr;
    // Extra 1 for the current device
    const uint32_t ipc_agents_num = dev().p2pAgents().size() + 1;
    // Retrieve the devPtr from the handle
    auto hsa_status = Hsa::ipc_memory_attach(
        reinterpret_cast<const hsa_amd_ipc_memory_t*>(
            reinterpret_cast<const amd::IpcBuffer*>(owner())->Handle()),
        owner()->getSize(), ipc_agents_num, dev().IpcAgents(), &orig_dev_ptr);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      LogPrintfError("HSA failed to attach IPC memory with status: %d", hsa_status);
      return false;
    }
    owner()->setSvmPtr(orig_dev_ptr);
  }


  // Allocate backing storage in device local memory unless UHP or AHP are set
  cl_mem_flags memFlags = owner()->getMemFlags();

  if (memFlags & ROCCLR_MEM_PHYMEM) {
    if (memFlags & ROCCLR_MEM_INTERPROCESS) {
      // if interprocess flag is set, then the memory is importable.
      if (!dev().ImportShareableHSAHandle(owner()->getSvmPtr(),
                                          &owner()->getUserData().hsa_handle)) {
        LogPrintfError("Importing Shareable Memory failed with os_handle: 0x%x",
                       owner()->getSvmPtr());
        return false;
      }
    } else {
      // If this is physical memory request, then get an handle and store it in user data
      owner()->getUserData().hsa_handle = dev().deviceVmemAlloc(owner()->getSize(), 0);
    }

    if (owner()->getUserData().hsa_handle == 0) {
      LogError("HSA Opaque Handle returned was null");
      return false;
    }

    owner()->setSvmPtr(reinterpret_cast<void*>(owner()->getUserData().hsa_handle));

    return true;
  }

  if ((owner()->parent() == nullptr) && (owner()->getSvmPtr() != nullptr)) {
    if (dev().forceFineGrain(owner()) || dev().isFineGrainedSystem(true)) {
      memFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }
    const bool isFineGrain = memFlags & CL_MEM_SVM_FINE_GRAIN_BUFFER;

    if (isFineGrain && !(memFlags & CL_MEM_VA_RANGE_AMD)) {
      // Use CPU direct access for the fine grain buffer
      flags_ |= HostMemoryDirectAccess;
    }

    if (owner()->getSvmPtr() == reinterpret_cast<void*>(amd::Memory::MemoryType::kSvmMemoryPtr)) {
      if (isFineGrain) {
        if (memFlags & CL_MEM_ALLOC_HOST_PTR) {
          if (dev().info().hmmSupported_) {
            // AMD HMM path. ROCr allocates system memory and KFD will manage it
            deviceMemory_ = dev().reserveMemory(size(), amd::Os::pageSize());
            if (deviceMemory_ == NULL) {
              return false;
            }
            // Currently HMM requires certain initial calls to mark sysmem allocation as
            // GPU accessible or prefetch memory into GPU
            if (!dev().SvmAllocInit(deviceMemory_, size())) {
              ClPrint(amd::LOG_ERROR, amd::LOG_MEM, "SVM init in ROCr failed!");
              return false;
            }
          } else {
            deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
          }
        } else if (memFlags & CL_MEM_FOLLOW_USER_NUMA_POLICY) {
          deviceMemory_ = dev().hostNumaAlloc(size(), 1, getHostMemorySegment(memFlags));
        } else if (memFlags & ROCCLR_MEM_HSA_SIGNAL_MEMORY) {
          // TODO: ROCr will introduce a new attribute enum that implies a non-blocking signal,
          // replace "HSA_AMD_SIGNAL_AMD_GPU_ONLY" with this new enum when it is ready.
          if (HSA_STATUS_SUCCESS != Hsa::signal_create(kInitSignalValueOne, 0, nullptr,
                                                       HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal_)) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM,
                    "hsa_amd_signal_create signal failed");
            return false;
          }
          volatile hsa_signal_value_t* signalValuePtr = nullptr;
          if (HSA_STATUS_SUCCESS != Hsa::signal_value_pointer(signal_, &signalValuePtr)) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM,
                    "hsa_amd_signal_value_pointer failed");
            return false;
          }

          deviceMemory_ = const_cast<void*>(reinterpret_cast<volatile void*>(signalValuePtr));

          // Disable host access to force blit path for memeory writes.
          flags_ &= ~HostMemoryDirectAccess;
        } else {
          deviceMemory_ = dev().hostAlloc(size(), 1, getHostMemorySegment(memFlags));
        }
      } else {
        assert(!isHostMemDirectAccess() && "Runtime doesn't support direct access to GPU memory!");
        amd::Device::AllocationFlags flags = {};
        flags.atomics_ = (memFlags & CL_MEM_SVM_ATOMICS) != 0;
        flags.pseudo_fine_grain_ = (memFlags & ROCCLR_MEM_HSA_UNCACHED) != 0;
        flags.contiguous_ = (memFlags & ROCCLR_MEM_HSA_CONTIGUOUS) != 0;
        deviceMemory_ = dev().deviceLocalAlloc(size(), flags);
      }
      owner()->setSvmPtr(deviceMemory_);
    } else {
      deviceMemory_ = owner()->getSvmPtr();
      if (owner()->getSvmPtr() ==
          reinterpret_cast<void*>(amd::Memory::MemoryType ::kArenaMemoryPtr)) {
        kind_ = MEMORY_KIND_ARENA;
        flags_ |= HostMemoryDirectAccess;
      } else {
        kind_ = MEMORY_KIND_PTRGIVEN;
      }
      if (memFlags & CL_MEM_USE_HOST_PTR) {
        if (dev().info().hmmSupported_) {
          // Currently HMM requires certain initial calls to mark sysmem allocation as
          // GPU accessible or prefetch memory into GPU
          if (!dev().SvmAllocInit(deviceMemory_, size())) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM, "SVM init in ROCr failed!");
            return false;
          }
        } else {
          deviceMemory_ =
              dev().hostLock(owner()->getSvmPtr(), size(), getHostMemorySegment(memFlags));
        }
      }
    }

    if ((deviceMemory_ != nullptr) && (dev().settings().apuSystem_ || !isFineGrain) &&
        (kind_ != MEMORY_KIND_ARENA)) {
      const_cast<Device&>(dev()).updateFreeMemory(size(), false);
    }

    return deviceMemory_ != nullptr;
  }

  // Interop buffer
  if (owner()->isInterop()) {
    amd::InteropObject* interop = owner()->getInteropObj();
    auto ext_memory = interop->asExternalMemory();
    amd::GLObject* glObject = interop->asGLObject();
    if (ext_memory != nullptr) {
      hsa_status_t status = interopMapBuffer(ext_memory->Handle());
      if (status != HSA_STATUS_SUCCESS) return false;
      return true;
    } else if (glObject != nullptr) {
      return createInteropBuffer(GL_ARRAY_BUFFER, 0);
    }
  }
  if (nullptr != owner()->parent()) {
    amd::Memory& parent = *owner()->parent();
    // Sub-Buffer creation.
    roc::Memory* parentBuffer = static_cast<roc::Memory*>(parent.getDeviceMemory(dev_));

    if (parentBuffer == nullptr) {
      LogError("[OCL] Fail to allocate parent buffer");
      return false;
    }

    const size_t offset = owner()->getOrigin();
    deviceMemory_ = parentBuffer->getDeviceMemory() + offset;

    flags_ |= parentBuffer->isHostMemDirectAccess() ? HostMemoryDirectAccess : 0;
    flags_ |= parentBuffer->isCpuUncached() ? MemoryCpuUncached : 0;

    // Explicitly set the host memory location,
    // because the parent location could change after reallocation
    if (nullptr != parent.getHostMem()) {
      owner()->setHostMem(reinterpret_cast<char*>(parent.getHostMem()) + offset);
    } else {
      owner()->setHostMem(nullptr);
    }

    return true;
  }

#ifdef WITH_AMDGPU_PRO
  if ((memFlags & CL_MEM_USE_PERSISTENT_MEM_AMD) && dev().ProEna()) {
    void* host_ptr = nullptr;
    deviceMemory_ = dev().iPro().AllocDmaBuffer(dev().getBackendDevice(), size(), &host_ptr);
    if (deviceMemory_ == nullptr) {
      return false;
    }
    persistent_host_ptr_ = host_ptr;
    return true;
  }
#endif

  if (!(memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) {
    deviceMemory_ = dev().deviceLocalAlloc(size());

    if (deviceMemory_ == nullptr) {
      // TODO: device memory is not enabled yet.
      // Fallback to system memory if exist.
      flags_ |= HostMemoryDirectAccess;
      if (dev().agent_profile() == HSA_PROFILE_FULL && owner()->getHostMem() != nullptr) {
        deviceMemory_ = owner()->getHostMem();
        assert(
            amd::isMultipleOf(deviceMemory_, static_cast<size_t>(dev().info().memBaseAddrAlign_)));
        return true;
      }

      deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
      owner()->setHostMem(deviceMemory_);

      if ((deviceMemory_ != nullptr) && dev().settings().apuSystem_) {
        const_cast<Device&>(dev()).updateFreeMemory(size(), false);
      }
    } else {
      const_cast<Device&>(dev()).updateFreeMemory(size(), false);
    }

    assert(amd::isMultipleOf(deviceMemory_, static_cast<size_t>(dev().info().memBaseAddrAlign_)));

    // Transfer data only if OCL context has one device.
    // Cache coherency layer will update data for multiple devices
    if (deviceMemory_ && (memFlags & CL_MEM_COPY_HOST_PTR) &&
        (owner()->getContext().devices().size() == 1)) {
      // To avoid recurssive call to Device::createMemory, we perform
      // data transfer to the view of the buffer.
      amd::Buffer* bufferView = new (owner()->getContext())
          amd::Buffer(*owner(), 0, owner()->getOrigin(), owner()->getSize());
      bufferView->create(nullptr, false, true);

      roc::Buffer* devBufferView = new roc::Buffer(dev_, *bufferView);
      devBufferView->deviceMemory_ = deviceMemory_;

      bufferView->replaceDeviceMemory(&dev_, devBufferView);

      bool ret = dev().xferMgr().writeBuffer(owner()->getHostMem(), *devBufferView, amd::Coord3D(0),
                                             amd::Coord3D(size()), true);

      // Release host memory, since runtime copied data
      owner()->setHostMem(nullptr);
      bufferView->release();
      return ret;
    }

    return deviceMemory_ != nullptr;
  }
  assert(owner()->getHostMem() != nullptr || (owner()->getContext().devices().size() == 1));

  flags_ |= HostMemoryDirectAccess;

  if (dev().agent_profile() == HSA_PROFILE_FULL) {
    deviceMemory_ = owner()->getHostMem();

    if (memFlags & CL_MEM_USE_HOST_PTR) {
      Hsa::memory_register(deviceMemory_, size());
    }

    return deviceMemory_ != nullptr;
  }

  // Just one device and allocation must be done in the backend
  if ((memFlags & CL_MEM_ALLOC_HOST_PTR) && (owner()->getContext().devices().size() == 1)) {
    deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
    // Copy original data to the allocated host memory
    if (memFlags & CL_MEM_COPY_HOST_PTR) {
      memcpy(deviceMemory_, owner()->getHostMem(), owner()->getSize());
    }
    owner()->setHostMem(deviceMemory_);
  } else if (owner()->getSvmPtr() != owner()->getHostMem()) {
    if (memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
      deviceMemory_ =
          dev().hostLock(owner()->getHostMem(), owner()->getSize(), getHostMemorySegment(memFlags));
    } else {
      deviceMemory_ = owner()->getHostMem();
    }
  } else {
    deviceMemory_ = owner()->getHostMem();
  }

  return deviceMemory_ != nullptr;
}

// ================================================================================================
bool Buffer::ExportHandle(void* handle) const {
  void* orig_dev_ptr = nullptr;
  if (owner()->getSvmPtr() != nullptr) {
    orig_dev_ptr = owner()->getSvmPtr();
  } else if (owner()->getHostMem() != nullptr) {
    orig_dev_ptr = owner()->getHostMem();
  }

  auto hsa_status = Hsa::ipc_memory_create(orig_dev_ptr, owner()->getSize(),
                                              reinterpret_cast<hsa_amd_ipc_memory_t*>(handle));
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed to create memory for IPC, failed with hsa_status: %d", hsa_status);
    return false;
  }
  return true;
}

// ================================================================================================
bool Buffer::GetFDHandleForMem(void* dev_ptr, size_t size, bool vmm, void* handle) {
  int dmabuffd = -1;
  size_t offset = 0;

  // In case of vmm, we use a different set of APIs for retrieving the dmabuffd.
  if (vmm) {
    hsa_amd_vmem_alloc_handle_t mem_handle;

    // Retrieve the corresponding phys_mem handle for the mapped dev_ptr.
    hsa_status_t hsa_status = Hsa::vmem_retain_alloc_handle(&mem_handle, dev_ptr);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      LogPrintfError("Cannot retain alloc handle for dev_ptr: 0x%x hsa returned status: %d",
                     dev_ptr, hsa_status);
      return false;
    }

    // Now, retrieve the shareable handle (fd in linux) for the phys_mem handle.
    hsa_status = Hsa::vmem_export_shareable_handle(&dmabuffd, mem_handle, 0);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      LogPrintfError("Cannot get shareable handle for mem_handle: %lu, hsa returned status: %d",
                     mem_handle, hsa_status);
      return false;
    }
  } else {
    // Retrieve a shareable handle for the device ptr.
    hsa_status_t hsa_status = Hsa::portable_export_dmabuf(dev_ptr, size, &dmabuffd, &offset);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      LogPrintfError(
          "Cannot export a portable fd for dev_ptr: 0x%x with size: %lu,"
          "hsa returned status: %d",
          dev_ptr, size, hsa_status);
      return false;
    }
  }
  if (dmabuffd <= 0) {
    LogPrintfError("Invalid file descriptor handle: %d returned", dmabuffd);
    return false;
  }

  // As per spec, handle passed through HIP API is ptr to int.
  *(reinterpret_cast<int*>(handle)) = dmabuffd;
  return true;
}

// ======================================= roc::Image =============================================
typedef struct ChannelOrderMap {
  uint32_t cl_channel_order;
  hsa_ext_image_channel_order_t hsa_channel_order;
} ChannelOrderMap;

typedef struct ChannelTypeMap {
  uint32_t cl_channel_type;
  hsa_ext_image_channel_type_t hsa_channel_type;
} ChannelTypeMap;

static constexpr ChannelOrderMap kChannelOrderMapping[] = {
    {CL_R, HSA_EXT_IMAGE_CHANNEL_ORDER_R},
    {CL_A, HSA_EXT_IMAGE_CHANNEL_ORDER_A},
    {CL_RG, HSA_EXT_IMAGE_CHANNEL_ORDER_RG},
    {CL_RA, HSA_EXT_IMAGE_CHANNEL_ORDER_RA},
    {CL_RGB, HSA_EXT_IMAGE_CHANNEL_ORDER_RGB},
    {CL_RGBA, HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA},
    {CL_BGRA, HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA},
    {CL_ARGB, HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB},
    {CL_INTENSITY, HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY},
    {CL_LUMINANCE, HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE},
    {CL_Rx, HSA_EXT_IMAGE_CHANNEL_ORDER_RX},
    {CL_RGx, HSA_EXT_IMAGE_CHANNEL_ORDER_RGX},
    {CL_RGBx, HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX},
    {CL_DEPTH, HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH},
    {CL_DEPTH_STENCIL, HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL},
    {CL_sRGB, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB},
    {CL_sRGBx, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX},
    {CL_sRGBA, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA},
    {CL_sBGRA, HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA},
    {CL_ABGR, HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR},
};

static constexpr ChannelTypeMap kChannelTypeMapping[] = {
    {CL_SNORM_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8},
    {CL_SNORM_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16},
    {CL_UNORM_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8},
    {CL_UNORM_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16},
    {CL_UNORM_SHORT_565, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565},
    {CL_UNORM_SHORT_555, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555},
    {CL_UNORM_INT_101010, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010},
    {CL_SIGNED_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8},
    {CL_SIGNED_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16},
    {CL_SIGNED_INT32, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32},
    {CL_UNSIGNED_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8},
    {CL_UNSIGNED_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16},
    {CL_UNSIGNED_INT32, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32},
    {CL_HALF_FLOAT, HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT},
    {CL_FLOAT, HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT},
    {CL_UNORM_INT24, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24},
};


static hsa_access_permission_t GetHsaAccessPermission(const cl_mem_flags flags) {
  if (flags & CL_MEM_READ_ONLY)
    return HSA_ACCESS_PERMISSION_RO;
  else if (flags & CL_MEM_WRITE_ONLY)
    return HSA_ACCESS_PERMISSION_WO;
  else
    return HSA_ACCESS_PERMISSION_RW;
}

Image::Image(const roc::Device& dev, amd::Memory& owner) : roc::Memory(dev, owner) {
  flags_ &= (~HostMemoryDirectAccess & ~HostMemoryRegistered);
  populateImageDescriptor();
  hsaImageObject_.handle = 0;
  originalDeviceMemory_ = nullptr;
}

void Image::populateImageDescriptor() {
  amd::Image* image = owner()->asImage();

  // build HSA runtime image descriptor
  imageDescriptor_.width = image->getWidth();
  imageDescriptor_.height = image->getHeight();
  imageDescriptor_.depth = image->getDepth();
  imageDescriptor_.array_size = 0;

  switch (image->getType()) {
    case CL_MEM_OBJECT_IMAGE1D:
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
      imageDescriptor_.height = 0;
      imageDescriptor_.depth = 0;
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1DB;
      imageDescriptor_.height = 0;
      imageDescriptor_.depth = 0;
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      //@todo - arraySize = height ?!
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1DA;
      imageDescriptor_.height = 0;
      imageDescriptor_.array_size = image->getHeight();
      break;
    case CL_MEM_OBJECT_IMAGE2D:
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
      imageDescriptor_.depth = 0;
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      //@todo - arraySize = depth ?!
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
      imageDescriptor_.depth = 0;
      imageDescriptor_.array_size = image->getDepth();
      break;
    case CL_MEM_OBJECT_IMAGE3D:
      imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
      break;
  }

  const int kChannelOrderCount = sizeof(kChannelOrderMapping) / sizeof(ChannelOrderMap);
  for (int i = 0; i < kChannelOrderCount; i++) {
    if (image->getImageFormat().image_channel_order == kChannelOrderMapping[i].cl_channel_order) {
      imageDescriptor_.format.channel_order = kChannelOrderMapping[i].hsa_channel_order;
      break;
    }
  }

  const int kChannelTypeCount = sizeof(kChannelTypeMapping) / sizeof(ChannelTypeMap);
  for (int i = 0; i < kChannelTypeCount; i++) {
    if (image->getImageFormat().image_channel_data_type == kChannelTypeMapping[i].cl_channel_type) {
      imageDescriptor_.format.channel_type = kChannelTypeMapping[i].hsa_channel_type;
      break;
    }
  }

  permission_ = GetHsaAccessPermission(owner()->getMemFlags());
}

bool Image::createInteropImage() {
  auto obj = owner()->getInteropObj()->asGLObject();
  assert(obj->getCLGLObjectType() != CL_GL_OBJECT_BUFFER &&
         "Non-image OpenGL object used with interop image API.");

  GLenum glTarget = obj->getGLTarget();
  if (glTarget == GL_TEXTURE_CUBE_MAP) {
    glTarget = obj->getCubemapFace();
  }

  if (!createInteropBuffer(glTarget, obj->getGLMipLevel())) {
    assert(false && "Failed to map image buffer.");
    return false;
  }

  originalDeviceMemory_ = deviceMemory_;

  if (obj->getGLTarget() == GL_TEXTURE_BUFFER) {
    hsa_status_t err = Hsa::image_create(dev().getBackendDevice(), &imageDescriptor_,
                                         originalDeviceMemory_, permission_, &hsaImageObject_);
    return (err == HSA_STATUS_SUCCESS);
  }

  image_metadata desc;
  if (!desc.create(amdImageDesc_)) {
    return false;
  }

  if (!desc.setMipLevel(obj->getGLMipLevel())) {
    return false;
  }

  if (obj->getGLTarget() == GL_TEXTURE_CUBE_MAP) {
    desc.setFace(obj->getCubemapFace(), dev().isa().versionMajor());
  }

  hsa_status_t err =
      Hsa::image_create(dev().getBackendDevice(), &imageDescriptor_, amdImageDesc_,
                        originalDeviceMemory_, permission_, &hsaImageObject_);
  if (err != HSA_STATUS_SUCCESS) return false;

  return true;
}

bool Image::create(bool alloc_local) {
  if (owner()->parent() != nullptr) {
    // Image view creation
    roc::Memory* parent = static_cast<roc::Memory*>(owner()->parent()->getDeviceMemory(dev_));

    if (parent == nullptr) {
      LogError("[OCL] Fail to allocate parent image");
      return false;
    }

    return createView(*parent);
  }

  // Interop image
  if (owner()->isInterop()) {
    return createInteropImage();
  }

  // Checking if original device memory can be accessed by peer devices
  device::Memory* orgDevMem = owner()->getOriginalDeviceMemory();
  if (amd::IS_HIP && orgDevMem != nullptr && orgDevMem->getAllowedPeerAccess()) {
    roc::Image* orgImage = static_cast<roc::Image*>(orgDevMem);
    // fill all required values
    deviceImageInfo_ = orgImage->deviceImageInfo_;
    permission_ = orgImage->permission_;
    deviceMemory_ = orgImage->deviceMemory_;
    hsaImageObject_ = orgImage->hsaImageObject_;
    ownsHsaImageObject_ = false;
    return true;
  }

  // Get memory size requirement for device specific image.
  hsa_status_t status = Hsa::image_data_get_info(dev().getBackendDevice(), &imageDescriptor_,
                                                 permission_, &deviceImageInfo_);

  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail to allocate image memory, failed with hsa_status: %d", status);
    return false;
  }

  // roc::Device::hostAlloc and deviceLocalAlloc implementation does not
  // support alignment larger than HSA memory region allocation granularity.
  // In this case, the user manages the alignment.
  const size_t alloc_size = (deviceImageInfo_.alignment <= dev().alloc_granularity())
                                ? deviceImageInfo_.size
                                : deviceImageInfo_.size + deviceImageInfo_.alignment;

  if (!(owner()->getMemFlags() & CL_MEM_ALLOC_HOST_PTR)) {
    originalDeviceMemory_ = dev().deviceLocalAlloc(alloc_size);
  }

  if (originalDeviceMemory_ == nullptr) {
    originalDeviceMemory_ = dev().hostAlloc(alloc_size, 1, Device::MemorySegment::kNoAtomics);
    if (originalDeviceMemory_ != nullptr) {
      kind_ = MEMORY_KIND_HOST;
      if (dev().settings().apuSystem_) {
        const_cast<Device&>(dev()).updateFreeMemory(alloc_size, false);
      }
    }
  } else {
    const_cast<Device&>(dev()).updateFreeMemory(alloc_size, false);
  }
  // record real size of the buffer so we will release and count it correctly.
  deviceImageInfo_.size = alloc_size;

  deviceMemory_ = reinterpret_cast<void*>(
      amd::alignUp(reinterpret_cast<uintptr_t>(originalDeviceMemory_), deviceImageInfo_.alignment));

  assert(amd::isMultipleOf(deviceMemory_, static_cast<size_t>(deviceImageInfo_.alignment)));

  status = Hsa::image_create(dev().getBackendDevice(), &imageDescriptor_, deviceMemory_,
                             permission_, &hsaImageObject_);

  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("[OCL] Fail to allocate image memory, failed with hsa_status: %d \n", status);
    return false;
  }

  return true;
}

bool Image::createView(const Memory& parent) {
  deviceMemory_ = parent.getDeviceMemory();

  originalDeviceMemory_ = (parent.owner()->asBuffer() != nullptr)
                              ? deviceMemory_
                              : static_cast<const Image&>(parent).originalDeviceMemory_;

  // Detect image view from buffer to distinguish linear paths from tiled.
  amd::Memory* ancestor = parent.owner();
  while ((ancestor->asBuffer() == nullptr) && (ancestor->parent() != nullptr)) {
    ancestor = ancestor->parent();
  }
  bool linearLayout = (ancestor->asBuffer() != nullptr);

  kind_ = parent.getKind();
  version_ = parent.version();

  if (parent.isHostMemDirectAccess()) {
    flags_ |= HostMemoryDirectAccess;
  }

  hsa_status_t status;
  if (linearLayout) {
    size_t rowPitch;
    amd::Image& ownerImage = *owner()->asImage();
    size_t elementSize = ownerImage.getImageFormat().getElementSize();
    // First get the row pitch in pixels
    if (ownerImage.getRowPitch() != 0) {
      rowPitch = ownerImage.getRowPitch() / elementSize;
    } else {
      rowPitch = ownerImage.getWidth();
    }

    // Make sure the row pitch is aligned to pixels
    rowPitch =
        elementSize * amd::alignUp(rowPitch, (dev().info().imagePitchAlignment_ / elementSize));

    status = Hsa::image_create_with_layout(
        dev().getBackendDevice(), &imageDescriptor_, deviceMemory_, permission_,
        HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, rowPitch, 0, &hsaImageObject_);

    if (!amd::IS_HIP && dev().settings().imageBufferWar_ &&
        ((ownerImage.getWidth() * ownerImage.getImageFormat().getElementSize()) <
         ownerImage.getRowPitch())) {
      bool workaround = false;
      if (status == static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED)) {
        workaround = true;
      }
      if (status == HSA_STATUS_SUCCESS) {
        // There are corner cases which still need workaround.
        const size_t kAlignments[] = {16, 32, 64, 128, 256};
        size_t tryPitch;
        for (int i = 0; i < sizeof(kAlignments) / sizeof(kAlignments[0]); i++) {
          tryPitch = amd::alignUp(ownerImage.getWidth(), kAlignments[i]) * elementSize;
          if (tryPitch >= rowPitch) {
            break;
          }
          hsa_ext_image_t hsaImage;
          if (HSA_STATUS_SUCCESS == Hsa::image_create_with_layout(
                                        dev().getBackendDevice(), &imageDescriptor_, deviceMemory_,
                                        permission_, HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, tryPitch, 0,
                                        &hsaImage)) {
            // The image pitch from app is not expectation of the GPU
            LogWarning("[OCL] will use copy image");
            workaround = true;
            // Free the image.
            Hsa::image_destroy(dev().getBackendDevice(), hsaImage);
            Hsa::image_destroy(dev().getBackendDevice(), hsaImageObject_);
            hsaImageObject_.handle = 0;
            break;
          }
        }
      }

      if (workaround) {
        if (ValidateMemory()) {
          status = HSA_STATUS_SUCCESS;
        } else {
          LogWarning("[OCL] copy image fail during validation");
          status = HSA_STATUS_ERROR;
        }
      }
    }
  } else if (kind_ == MEMORY_KIND_INTEROP) {
    amdImageDesc_ = static_cast<Image*>(parent.owner()->getDeviceMemory(dev()))->amdImageDesc_;
    status = Hsa::image_create(dev().getBackendDevice(), &imageDescriptor_, amdImageDesc_,
                               deviceMemory_, permission_, &hsaImageObject_);
  } else {
    status = Hsa::image_create(dev().getBackendDevice(), &imageDescriptor_, deviceMemory_,
                               permission_, &hsaImageObject_);
  }

  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("[OCL] Fail to allocate image memory with status: %d \n", status);
    return false;
  }

  // Explicitly set the host memory location,
  // because the parent location could change after reallocation
  if (nullptr != parent.owner()->getHostMem()) {
    owner()->setHostMem(reinterpret_cast<char*>(parent.owner()->getHostMem()) +
                        owner()->getOrigin());
  } else {
    owner()->setHostMem(nullptr);
  }

  return true;
}

void* Image::allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region, uint mapFlags,
                            size_t* rowPitch, size_t* slicePitch) {
  amd::ScopedLock lock(owner()->lockMemoryOps());

  incIndMapCount();

  void* pHostMem = owner()->getHostMem();

  amd::Image* image = owner()->asImage();

  size_t elementSize = image->getImageFormat().getElementSize();

  size_t offset = origin[0] * elementSize;
  if (pHostMem == nullptr) {
    if (indirectMapCount_ == 1) {
      if (!allocateMapMemory(owner()->getSize())) {
        decIndMapCount();
        return nullptr;
      }
    } else {
      // Did the map resource allocation fail?
      if (mapMemory_ == nullptr) {
        DevLogError("Could not map target resource");
        return nullptr;
      }
    }

    pHostMem = mapMemory_->getHostMem();

    size_t rowPitchTemp = 0;
    if (rowPitch != nullptr) {
      *rowPitch = region[0] * elementSize;
      rowPitchTemp = *rowPitch;
    }

    size_t slicePitchTmp = 0;

    if (imageDescriptor_.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA) {
      slicePitchTmp = rowPitchTemp;
    } else {
      slicePitchTmp = rowPitchTemp * region[1];
    }
    if (slicePitch != nullptr) {
      *slicePitch = slicePitchTmp;
    }

    return pHostMem;
  }

  // Adjust offset with Y dimension
  offset += image->getRowPitch() * origin[1];

  // Adjust offset with Z dimension
  offset += image->getSlicePitch() * origin[2];

  if (rowPitch != nullptr) {
    *rowPitch = image->getRowPitch();
  }

  if (slicePitch != nullptr) {
    *slicePitch = image->getSlicePitch();
  }

  return (static_cast<uint8_t*>(pHostMem) + offset);
}

Image::~Image() { destroy(); }

// ================================================================================================
void Image::destroy() {
  for (auto it : view_cache_) {
    it->release();
  }

  delete copyImageBuffer_;

  if (hsaImageObject_.handle != 0 && ownsHsaImageObject_) {
    hsa_status_t status = Hsa::image_destroy(dev().getBackendDevice(), hsaImageObject_);
    assert(status == HSA_STATUS_SUCCESS);
  }
  // Don't destroy memory if it's a view. Parent will destroy the original allocation.
  if ((owner()->parent() != nullptr) || owner()->ImageView()) {
    return;
  }

  delete[] amdImageDesc_;
  amdImageDesc_ = nullptr;

  if (kind_ == MEMORY_KIND_INTEROP) {
    destroyInteropBuffer();
    return;
  }

  if (originalDeviceMemory_ != nullptr) {
    dev().memFree(originalDeviceMemory_, deviceImageInfo_.size);
    if (kind_ == MEMORY_KIND_HOST) {
      if (dev().settings().apuSystem_) {
        const_cast<Device&>(dev()).updateFreeMemory(deviceImageInfo_.size, true);
      }
    } else {
      const_cast<Device&>(dev()).updateFreeMemory(deviceImageInfo_.size, true);
    }
  }
}

// ================================================================================================
bool Image::ValidateMemory() {
  amd::Image* img = owner()->asImage();
  // Create a native image without pitch for validation
  copyImageBuffer_ = new (dev().context())
      amd::Image(dev().context(), CL_MEM_OBJECT_IMAGE2D, 0, img->getImageFormat(), img->getWidth(),
                 img->getHeight(), 1, 0, 0);

  if ((copyImageBuffer_ == nullptr) || !copyImageBuffer_->create()) {
    return false;
  } else {
    return true;
  }
}

// ================================================================================================
bool Image::AddView(amd::Image* image) {
  amd::ScopedLock l(owner()->lockMemoryOps());
  for (auto it : view_cache_) {
    if ((it->getImageFormat().image_channel_data_type ==
         image->getImageFormat().image_channel_data_type) &&
        (it->getImageFormat().image_channel_order == image->getImageFormat().image_channel_order)) {
      return false;
    }
  }
  view_cache_.push_back(image);
  // Remove parent dependency on the child, since cache will be destroyed within the parent
  owner()->release();
  image->SetParent(nullptr);
  return true;
}

// ================================================================================================
amd::Image* Image::FindView(cl_image_format format) const {
  amd::ScopedLock l(owner()->lockMemoryOps());
  for (auto it : view_cache_) {
    if ((it->getImageFormat().image_channel_data_type == format.image_channel_data_type) &&
        (it->getImageFormat().image_channel_order == format.image_channel_order)) {
      return it;
    }
  }
  return nullptr;
}

}  // namespace amd::roc

