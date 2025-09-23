/* Copyright (c) 2016 - 2023 Advanced Micro Devices, Inc.

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

#pragma once

#include "top.hpp"
#include "platform/memory.hpp"
#include "utils/debug.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocglinterop.hpp"

namespace amd::roc {
class Memory : public device::Memory {
 public:
  enum MEMORY_KIND {
    MEMORY_KIND_NORMAL = 0,

    // The memory is allocated by ROCclr in host memory. This flag is used by
    // image class now. It may find more use in buffer class in future
    MEMORY_KIND_HOST,

    MEMORY_KIND_INTEROP,

    MEMORY_KIND_PTRGIVEN,

    MEMORY_KIND_ARENA
  };

  Memory(const roc::Device& dev, amd::Memory& owner);

  Memory(const roc::Device& dev, size_t size);

  virtual ~Memory();

  // Getter for deviceMemory_
  address getDeviceMemory() const { return reinterpret_cast<address>(deviceMemory_); }

  // Gets a pointer to a region of host-visible memory for use as the target
  // of an indirect map for a given memory object
  void* allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region, uint mapFlags,
                       size_t* rowPitch, size_t* slicePitch) override;

  // Create device memory according to OpenCL memory flag.
  virtual bool create(bool local_alloc = false) = 0;

  // Pins system memory associated with this memory object.
  bool pinSystemMemory(void* hostPtr,  // System memory address
                       size_t size     // Size of allocated system memory
                       ) override;

  //! Updates device memory from the owner's host allocation
  void syncCacheFromHost(VirtualGPU& gpu,  //!< Virtual GPU device object
                                           //! Synchronization flags
                         device::Memory::SyncFlags syncFlags = device::Memory::SyncFlags());

  // Immediate blocking write from device cache to owners's backing store.
  // Marks owner as "current" by resetting the last writer to nullptr.
  void syncHostFromCache(device::VirtualDevice* vDev, SyncFlags syncFlags = SyncFlags()) override;

  //! Allocates host memory for synchronization with MGPU context
  void mgpuCacheWriteBack(VirtualGPU& gpu);

  // Releases indirect map surface
  void releaseIndirectMap() override { decIndMapCount(); }

  //! Map the device memory to CPU visible
  void* cpuMap(device::VirtualDevice& vDev,  //!< Virtual device for map operaiton
               uint flags = 0,               //!< flags for the map operation
               // Optimization for multilayer map/unmap
               uint startLayer = 0,          //!< Start layer for multilayer map
               uint numLayers = 0,           //!< End layer for multilayer map
               size_t* rowPitch = nullptr,   //!< Row pitch for the device memory
               size_t* slicePitch = nullptr  //!< Slice pitch for the device memory
               ) override;

  //! Unmap the device memory
  void cpuUnmap(device::VirtualDevice& vDev  //!< Virtual device for unmap operaiton
                ) override;

  // Mesa has already decomressed if needed and also does acquire at the start of every command
  // batch.
  bool processGLResource(GLResourceOP operation) override { return true; }

  uint64_t virtualAddress() const override { return reinterpret_cast<uint64_t>(getDeviceMemory()); }

  uint64_t originalDeviceAddress() const override { return virtualAddress(); }

  // Accessors for indirect map memory object
  amd::Memory* mapMemory() const { return mapMemory_; }

  MEMORY_KIND getKind() const { return kind_; }

  const roc::Device& dev() const { return dev_; }

  size_t version() const { return version_; }

  bool IsPersistentDirectMap() const { return (persistent_host_ptr_ != nullptr); }

  void* PersistentHostPtr() const { return persistent_host_ptr_; }

  //! Validates allocated memory for possible workarounds
  virtual bool ValidateMemory() { return true; }

 protected:
  bool allocateMapMemory(size_t allocationSize);

  // Decrement map count
  void decIndMapCount() override;

  // Free / deregister device memory.
  virtual void destroy() = 0;
  hsa_status_t interopMapBuffer(amd::Os::FileDesc fdn);

  // Place interop object into HSA's flat address space
  bool createInteropBuffer(GLenum targetType, int miplevel);

  void destroyInteropBuffer();

  // Pointer to the device associated with this memory object.
  const roc::Device& dev_;

  // Pointer to the device memory. This could be in system or device local mem.
  void* deviceMemory_;

  // Pointer to the interop device memory, which has an offset from deviceMemory_
  void* interop_deviceMemory_;

  // Track if this memory is interop, lock, gart, or normal.
  MEMORY_KIND kind_;

  hsa_amd_image_descriptor_t* amdImageDesc_;

  void* persistent_host_ptr_;  //!< Host accessible pointer for persistent memory

  // Get MemorySegment type in terms of host memory allocation flags
  Device::MemorySegment getHostMemorySegment(const unsigned int memFlags) {
    return (memFlags & CL_MEM_SVM_ATOMICS) == 0 ? Device::MemorySegment::kNoAtomics
                                                : ((memFlags & ROCCLR_MEM_HSA_UNCACHED) != 0
                                                       ? Device::MemorySegment::kUncachedAtomics
                                                       : Device::MemorySegment::kAtomics);
  }

 private:
  // Disable copy constructor
  Memory(const Memory&);

  // Disable operator=
  Memory& operator=(const Memory&);

  amd::Memory* pinnedMemory_;  //!< Memory used as pinned system memory
};

class Buffer : public roc::Memory {
 public:
  Buffer(const roc::Device& dev, amd::Memory& owner);
  Buffer(const roc::Device& dev, size_t size);

  virtual ~Buffer();

  // Create device memory according to OpenCL memory flag.
  virtual bool create(bool local_alloc = false);

  virtual bool ExportHandle(void* handle) const final;

  virtual bool GetFDHandleForMem(void* dev_ptr, size_t size, bool vmm, void* handle) final;

  // Recreate the device memory using new size and alignment.
  bool recreate(size_t newSize, size_t newAlignment, bool forceSystem);

  // Returns the HSA signal associated with this Memory object.
  hsa_signal_t getSignal() const { return signal_; }

 private:
  // signal object used when ROCCLR_MEM_HSA_SIGNAL_MEMORY is set
  hsa_signal_t signal_;

  // Disable copy constructor
  Buffer(const Buffer&);

  // Disable operator=
  Buffer& operator=(const Buffer&);

  // Free device memory.
  void destroy();
};

class Image : public roc::Memory {
 public:
  Image(const roc::Device& dev, amd::Memory& owner);

  virtual ~Image();

  //! Create device memory according to OpenCL memory flag.
  virtual bool create(bool local_alloc = false);

  //! Create an image view
  bool createView(const Memory& parent);

  //! Gets a pointer to a region of host-visible memory for use as the target
  //! of an indirect map for a given memory object
  virtual void* allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region,
                               uint mapFlags, size_t* rowPitch, size_t* slicePitch);

  size_t getDeviceDataSize() { return deviceImageInfo_.size; }
  size_t getDeviceDataAlignment() { return deviceImageInfo_.alignment; }

  hsa_ext_image_t getHsaImageObject() const { return hsaImageObject_; }
  const hsa_ext_image_descriptor_t& getHsaImageDescriptor() const { return imageDescriptor_; }

  virtual const address cpuSrd() const {
    return reinterpret_cast<const address>(getHsaImageObject().handle);
  }

  //! Validates allocated memory for possible workarounds
  bool ValidateMemory() final;

  amd::Image* CopyImageBuffer() const { return copyImageBuffer_; }

  virtual uint64_t originalDeviceAddress() const {
    return reinterpret_cast<uint64_t>(originalDeviceMemory_);
  }

  //! Adds an image view to the view cache for the fast blit manager operations
  bool AddView(amd::Image* image);

  //! Finds an image view of this original image from the cache
  amd::Image* FindView(cl_image_format format) const;

 private:
  //! Disable copy constructor
  Image(const Buffer&);

  //! Disable operator=
  Image& operator=(const Buffer&);

  // Setup an interop image
  bool createInteropImage();

  // Free / deregister device memory.
  void destroy();

  void populateImageDescriptor();

  hsa_ext_image_descriptor_t imageDescriptor_;
  hsa_access_permission_t permission_;
  hsa_ext_image_data_info_t deviceImageInfo_;
  hsa_ext_image_t hsaImageObject_;
  bool ownsHsaImageObject_ = true;

  void* originalDeviceMemory_;
  amd::Image* copyImageBuffer_ = nullptr;
  std::vector<amd::Image*> view_cache_;  //!< Cache of views for fast access
};
}  // namespace amd::roc

