/* Copyright (c) 2010 - 2025 Advanced Micro Devices, Inc.

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

#include "amdocl/cl_common.hpp"

#include "os/alloc.hpp"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "device/device.hpp"

#include <atomic>

namespace amd {

// Stores the no. of memory allocations
std::atomic<uint32_t> numAllocs = {0};

bool BufferRect::create(const size_t* bufferOrigin, const size_t* region, size_t bufferRowPitch,
                        size_t bufferSlicePitch) {
  bool valid = false;
  // Find the buffer's row pitch
  rowPitch_ = (bufferRowPitch != 0) ? bufferRowPitch : region[0];
  // Find the buffer's slice pitch
  slicePitch_ = (bufferSlicePitch != 0) ? bufferSlicePitch : rowPitch_ * region[1];
  // Find the region start offset
  start_ = bufferOrigin[2] * slicePitch_ + bufferOrigin[1] * rowPitch_ + bufferOrigin[0];
  // Find the region relative end offset
  end_ = (region[2] - 1) * slicePitch_ + (region[1] - 1) * rowPitch_ + region[0];
  // Make sure we have a valid region
  if ((rowPitch_ >= region[0]) && (slicePitch_ >= (region[1] * rowPitch_)) &&
      ((slicePitch_ % rowPitch_) == 0)) {
    valid = true;
  }
  return valid;
}

bool HostMemoryReference::allocateMemory(size_t size, const Context& context) {
  assert(!alloced_ && "Runtime should not reallocate system memory!");
  size_t memoryAlignment = (CPU_MEMORY_ALIGNMENT_SIZE <= 0) ? 256 : CPU_MEMORY_ALIGNMENT_SIZE;
  size_ = amd::alignUp(size, memoryAlignment);
  //! \note memory size must be aligned for CAL pinning
  hostMem_ = CPU_MEMORY_GUARD_PAGES ? GuardedMemory::allocate(size_, MEMOBJ_BASE_ADDR_ALIGN,
                                                              CPU_MEMORY_GUARD_PAGE_SIZE * Ki)
                                    : context.hostAlloc(size_, MEMOBJ_BASE_ADDR_ALIGN);
  alloced_ = (hostMem_ != NULL);
  return alloced_;
}

// Frees system memory if it was allocated
void HostMemoryReference::deallocateMemory(const Context& context) {
  if (alloced_) {
    if (CPU_MEMORY_GUARD_PAGES)
      GuardedMemory::deallocate(hostMem_);
    else
      context.hostFree(hostMem_);
    size_ = 0;
    alloced_ = false;
    hostMem_ = NULL;
  }
}

Memory::Memory(Context& context, Type type, Flags flags, size_t size, void* svmPtr,
               size_t alignment)
    : numDevices_(0),
      deviceMemories_(NULL),
      destructorCallbacks_(NULL),
      context_(context),
      parent_(NULL),
      type_(type),
      hostMemRef_(NULL),
      origin_(0),
      size_(size),
      flags_(flags),
      version_(0),
      lastWriter_(NULL),
      interopObj_(NULL),
      vDev_(NULL),
      mapCount_(0),
      svmHostAddress_(svmPtr),
      resOffset_(0),
      flagsEx_(0),
      lockMemoryOps_(true),
      alignment_(alignment) /* Memory Ops Lock */ {
  svmPtrCommited_ = (flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) ? true : false;
  canBeCached_ = true;
}

Memory::Memory(Memory& parent, Flags flags, size_t origin, size_t size, Type type, Context* new_ctx)
    : numDevices_(0),
      deviceMemories_(NULL),
      destructorCallbacks_(NULL),
      context_((new_ctx != nullptr) ? *new_ctx : parent.getContext()),
      parent_(&parent),
      type_((type == 0) ? parent.type_ : type),
      hostMemRef_(NULL),
      origin_(origin),
      size_(size),
      flags_(flags),
      version_(parent.getVersion()),
      lastWriter_(parent.getLastWriter()),
      interopObj_(amd::IS_HIP ? nullptr : parent.getInteropObj()),
      vDev_(NULL),
      mapCount_(0),
      svmHostAddress_(parent.getSvmPtr()),
      resOffset_(0),
      flagsEx_(0),
      lockMemoryOps_(true) /* Memory Ops Lock */ {
  svmPtrCommited_ = parent.isSvmPtrCommited();
  canBeCached_ = true;
  parent_->retain();
  parent_->isParent_ = true;

  if (parent.getHostMem() != nullptr) {
    setHostMem(reinterpret_cast<address>(parent.getHostMem()) + origin);
  }

  if (parent.getSvmPtr() != nullptr) {
    setSvmPtr(reinterpret_cast<address>(parent.getSvmPtr()) + origin);
  }

  // Inherit memory flags from the parent
  if ((flags_ & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY)) == 0) {
    flags_ |= parent_->getMemFlags() & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
  }

  flags_ |=
      parent_->getMemFlags() & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);

  if ((flags_ & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) == 0) {
    flags_ |= parent_->getMemFlags() &
              (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS);
  }
}

uint32_t Memory::NumDevicesWithP2P() {
  uint32_t devices = context_().devices().size();
  if (devices == 1) {
    // Add p2p devices for allocation
    devices = context_().devices().size() + context_().devices()[0]->P2PAccessDevices().size();
    if (devices > 1) {
      p2pAccess_ = true;
    }
  }
  return devices;
}

void Memory::initDeviceMemory() {
  deviceMemories_ = reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Memory));
  memset(deviceMemories_, 0, NumDevicesWithP2P() * sizeof(DeviceMemory));
}

// ================================================================================================
void Memory::resetAllocationState() {
  // Reset device memory allocation state
  for (size_t i = 0; i < context_().devices().size(); i++) {
    deviceAlloced_[context_().devices()[i]].store(AllocInit, std::memory_order_relaxed);
  }
}

// ================================================================================================
void* Memory::operator new(size_t size, const Context& context) {
  uint32_t devices = context.devices().size();
  if (devices == 1) {
    // Add p2p devices for allocation
    devices = context.devices().size() + context.devices()[0]->P2PAccessDevices().size();
  }

  return RuntimeObject::operator new(size + devices * sizeof(DeviceMemory));
}

void Memory::operator delete(void* p) { RuntimeObject::operator delete(p); }

void Memory::operator delete(void* p, const Context& context) { Memory::operator delete(p); }


void Memory::addSubBuffer(Memory* view) {
  amd::ScopedLock lock(lockMemoryOps());
  subBuffers_.emplace(view);
}

void Memory::removeSubBuffer(Memory* view) {
  amd::ScopedLock lock(lockMemoryOps());
  subBuffers_.erase(view);
}

bool Memory::allocHostMemory(void* initFrom, bool allocHostMem, bool forceCopy) {
  // Sanity checks (the parameters should have been prevalidated by the API)
  assert(!(flags_ & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR) && (initFrom == NULL) &&
           !allocHostMem && !isSvmPtrCommited()));
  assert(
      !((initFrom != NULL) && !forceCopy &&
        !(flags_ & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_EXTERNAL_PHYSICAL_AMD))));
  assert(!(flags_ & CL_MEM_COPY_HOST_PTR && flags_ & CL_MEM_USE_HOST_PTR));

  const std::vector<Device*>& devices = context_().devices();

  // This allocation is necessary to use coherency mechanism
  // for the initialization
  if (getMemFlags() & (CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
    // Extra system memory allocation and copy can be very expensive.
    // Thus, avoid it if runtime doesn't perform deferred allocations
    if ((devices.size() == 1) || DISABLE_DEFERRED_ALLOC) {
      allocHostMem = false;
      setHostMem(initFrom);
    } else {
      allocHostMem = true;
    }
  }

  // Did application request to use host memory?
  if (getMemFlags() & CL_MEM_USE_HOST_PTR) {
    setHostMem(initFrom);

    // Recalculate image size according to pitch
    Image* image = asImage();
    if (image != NULL) {
      if (image->getDims() < 3) {
        size_ = image->getRowPitch() * image->getHeight();
      } else {
        size_ = image->getSlicePitch() * image->getDepth();
      }
    }
  }
  // Allocate host memory buffer if needed.
  // @note: SVM host memory allocation should be done in the device backend
  else if (allocHostMem && !isInterop() && !(getMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
    if (!hostMemRef_.allocateMemory(size_, context_())) {
      DevLogError("Cannot allocate Host Memory Buffer \n");
      return false;
    }

    // Copy data to the backing store if the app has requested
    if (((flags_ & CL_MEM_COPY_HOST_PTR) || forceCopy) && (initFrom != NULL)) {
      copyToBackingStore(initFrom);
    }
  }

  if (allocHostMem && type_ == CL_MEM_OBJECT_PIPE) {
    // Initialize the pipe for a CPU device
    clk_pipe_t* pipe = reinterpret_cast<clk_pipe_t*>(getHostMem());
    pipe->read_idx = 0;
    pipe->write_idx = 0;
    pipe->end_idx = asPipe()->getMaxNumPackets();
  }

  if ((flags_ & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) && (NULL == lastWriter_)) {
    // Signal write, so coherency mechanism will initialize
    // memory on all devices
    signalWrite(NULL);
  }

  return true;
}

// ================================================================================================
bool Memory::create(void* initFrom, bool sysMemAlloc, bool skipAlloc, bool forceAlloc) {
  static const bool forceAllocHostMem = false;

  // Sanity checks (can't defer and force allocations at the same time)
  assert(!(skipAlloc && forceAlloc));

  initDeviceMemory();

  // Check if it's a subbuffer allocation
  if (parent_ != NULL) {
    // Find host memory pointer for subbuffer
    if (parent_->getHostMem() != NULL) {
      setHostMem((address)parent_->getHostMem() + origin_);
    }

    // Add a new subbuffer to the list
    parent_->addSubBuffer(this);
  }
  // Allocate host memory if requested
  else if (!allocHostMemory(initFrom, forceAllocHostMem)) {
    DevLogError("Cannot allocate Host Memory \n");
    return false;
  }

  const std::vector<Device*>& devices = context_().devices();
  if (IS_LINUX && (devices.size() == 1) && devices[0]->info().largeBar_) {
    largeBarSystem_ = 1;
  }

  // Forces system memory allocation on the device,
  // instead of device memory
  forceSysMemAlloc_ = sysMemAlloc;

  // Create memory on all available devices
  for (size_t i = 0; i < devices.size(); i++) {
    deviceAlloced_[devices[i]].store(AllocInit, std::memory_order_relaxed);

    deviceMemories_[i].ref_ = devices[i];
    deviceMemories_[i].value_ = NULL;

    if (forceAlloc || (!skipAlloc && ((devices.size() == 1) || DISABLE_DEFERRED_ALLOC))) {
      device::Memory* mem = getDeviceMemory(*devices[i]);
      if (NULL == mem) {
        LogPrintfError("Can't allocate memory size - 0x%08X bytes!", getSize());
        return false;
      }
    }
  }

  // Store the unique id for each memory allocation
  uniqueId_ = ++numAllocs;
  return true;
}

// ================================================================================================
bool Memory::addDeviceMemory(const Device* dev) {
  bool result = false;
  AllocState create = AllocCreate;
  AllocState init = AllocInit;

  amd::ScopedLock lock(lockMemoryOps());
  if (deviceAlloced_[dev].compare_exchange_strong(init, create, std::memory_order_acq_rel)) {
    // Check if runtime already allocated all available slots for device memory
    if (numDevices() == NumDevicesWithP2P()) {
      // Mark the allocation as an empty
      deviceAlloced_[dev].store(AllocInit, std::memory_order_release);
      return result;
    }
    device::Memory* dm = dev->createMemory(*this);

    // Add the new memory allocation to the device map
    if (NULL != dm) {
      deviceMemories_[numDevices_].ref_ = dev;
      deviceMemories_[numDevices_].value_ = dm;
      numDevices_++;
      assert((numDevices() <= NumDevicesWithP2P()) && "Too many device objects");

      // Mark the allocation with the complete flag
      deviceAlloced_[dev] = AllocComplete;
      if (getSvmPtr() != nullptr) {
        svmBase_ = dm;
      }
    } else {
      LogError("Video memory allocation failed!");
      // Mark the allocation as an empty
      deviceAlloced_[dev].store(AllocInit, std::memory_order_release);
      return result;
    }
  }

  // Make sure runtime finished memory allocation.
  // Loop if in the create state
  while (deviceAlloced_[dev].load(std::memory_order_acquire) == AllocCreate) {
    Os::yield();
  }

  if (deviceAlloced_[dev].load(std::memory_order_acquire) == AllocComplete) {
    result = true;
  }

  return result;
}

void Memory::replaceDeviceMemory(const Device* dev, device::Memory* dm) {
  uint i;
  for (i = 0; i < numDevices_; ++i) {
    if (deviceMemories_[i].ref_ == dev) {
      delete deviceMemories_[i].value_;
      break;
    }
  }

  if (numDevices_ == 0) {
    ++numDevices_;
    deviceMemories_[0].ref_ = dev;
  }

  deviceMemories_[i].value_ = dm;
  deviceAlloced_[dev].store(AllocRealloced, std::memory_order_release);
}

device::Memory* Memory::getDeviceMemory(const Device& dev, bool alloc) {
  device::Memory* dm = NULL;
  for (uint i = 0; i < numDevices_; ++i) {
    if (deviceMemories_[i].ref_ == &dev) {
      dm = deviceMemories_[i].value_;
      break;
    }
  }

  if ((NULL == dm) && alloc) {
    if (!addDeviceMemory(&dev)) {
      return NULL;
    }
    dm = deviceMemories_[numDevices() - 1].value_;
  }

  return dm;
}

// ================================================================================================
Memory::~Memory() {
  if (ipcShared()) {
    amd::MemObjMap::RemoveIpcHandleMemObj(this);
    auto device = context_().devices()[0];
    if (device != nullptr) {
      device->IpcDetach(this);
    }
  }
  // For_each destructor callback:
  DestructorCallBackEntry* entry;
  for (entry = destructorCallbacks_; entry != nullptr; entry = entry->next_) {
    // invoke the callback function.
    entry->callback_(const_cast<cl_mem>(as_cl(this)), entry->data_);
  }

  // Release the parent.
  if (parent_ != nullptr) {
    // Update cache if runtime destroys a subbuffer
    if (parent_->getHostMem() != nullptr && (vDev_ == nullptr)) {
      cacheWriteBack(nullptr);
    }
    parent_->removeSubBuffer(this);
  }

  if (deviceMemories_ != nullptr) {
    // Destroy all device memory objects
    for (uint i = 0; i < numDevices_; ++i) {
      delete deviceMemories_[i].value_;
    }
  }

  // Sanity check
  if (subBuffers_.size() != 0) {
    LogError("Can't have views if parent is destroyed!");
  }

  // Destroy the destructor callback entries
  DestructorCallBackEntry* callback = destructorCallbacks_;
  while (callback != nullptr) {
    DestructorCallBackEntry* next = callback->next_;
    delete callback;
    callback = next;
  }

  // Make sure runtime destroys the parent only after subbuffer destruction
  if (parent_ != nullptr) {
    parent_->release();
  }
  hostMemRef_.deallocateMemory(context_());
  if (parent_ == nullptr && (getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
    // The mapping may manually be removed prior to objects destruction
    if (amd::MemObjMap::FindVirtualMemObj(getSvmPtr())) {
      amd::MemObjMap::RemoveVirtualMemObj(getSvmPtr());
    }
    // If runtime executes graph mempool with VM, then VA can be mapped in space
    // for graph validation logic during execution. And the reason it's not unmaped
    // in graph itself because the app can have a graph without a free node
    if (amd::MemObjMap::FindMemObj(getSvmPtr())) {
      amd::MemObjMap::RemoveMemObj(getSvmPtr());
    }
  }
}

// ================================================================================================
bool Memory::setDestructorCallback(DestructorCallBackFunction callback, void* data) {
  DestructorCallBackEntry* entry = new DestructorCallBackEntry(callback, data);
  if (entry == NULL) {
    return false;
  }

  entry->next_ = destructorCallbacks_;
  while (!destructorCallbacks_.compare_exchange_weak(
      entry->next_, entry));  // Someone else is also updating the head of the linked list! reload.

  return true;
}

void Memory::signalWrite(const Device* writer) {
  // Disable cache coherency layer for HIP
  if (!amd::IS_HIP) {
    // (The potential race condition below doesn't matter, no critical
    // section needed)
    ++version_;
    lastWriter_ = writer;
    // Update all subbuffers for this object
    for (auto buf : subBuffers_) {
      buf->signalWrite(writer);
    }
  }
}

void Memory::cacheWriteBack(device::VirtualDevice* vDev) {
  if (NULL != lastWriter_) {
    device::Memory* dmem = getDeviceMemory(*lastWriter_);
    //! @note It's a special condition, when a subbuffer was created,
    //! but never used. Thus dev memory is still NULL and lastWriter_
    //! was passed from the parent.
    if (NULL != dmem) {
      dmem->syncHostFromCache(vDev);
    }
  } else if (isParent()) {
    // On CPU parent can't be synchronized, because lastWriter_ could be NULL
    // and syncHostFromCache() won't be called.
    for (uint i = 0; i < numDevices_; ++i) {
      deviceMemories_[i].value_->syncHostFromCache(vDev);
    }
  }
}

void Memory::copyToBackingStore(void* initFrom) { memcpy(getHostMem(), initFrom, size_); }

bool Memory::usesSvmPointer() const {
  if (!(flags_ & CL_MEM_USE_HOST_PTR)) {
    return false;
  }
  // If the application host pointer lies within a SVM region, so does the
  // sub-buffer host pointer - so the following check works in both cases
  return (SvmBuffer::malloced(getHostMem()) || NULL != svmHostAddress_);
}

void Memory::commitSvmMemory() {
  ScopedLock lock(lockMemoryOps_);
  // if VRAM is visible for host, it is not necessary to mmap again
  if (!svmPtrCommited_ && !largeBarSystem_) {
    if (amd::Os::commitMemory(svmHostAddress_, size_, amd::Os::MEM_PROT_RW)) {
      svmPtrCommited_ = true;
    } else {
      LogPrintfError("Mem Map failed for the host address 0x%x", svmHostAddress_);
    }
  }
}

void Memory::uncommitSvmMemory() {
  ScopedLock lock(lockMemoryOps_);
  if (svmPtrCommited_ && !(flags_ & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
    if (amd::Os::uncommitMemory(svmHostAddress_, size_)) {
      svmPtrCommited_ = false;
    } else {
      LogPrintfError("Mem Unmap failed for the host address 0x%x", svmHostAddress_);
    }
  }
}

Device* Memory::GetDeviceById() {
  size_t device_idx = (userData_.deviceId < getContext().devices().size()) ? userData_.deviceId : 0;
  return getContext().devices()[device_idx];
}

// =================================================================================================
bool Memory::ValidateMemAccess(const Device& dev, bool read_write) {
  if (flags_ & CL_MEM_VA_RANGE_AMD) {
    return dev.ValidateMemAccess(*this, read_write);
  }
  return true;
}

void Buffer::initDeviceMemory() {
  deviceMemories_ = reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Buffer));
  memset(deviceMemories_, 0, NumDevicesWithP2P() * sizeof(DeviceMemory));
}

bool Buffer::create(void* initFrom, bool sysMemAlloc, bool skipAlloc, bool forceAlloc) {
  if ((getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD) && (initFrom != NULL)) {
    busAddress_ = *(reinterpret_cast<cl_bus_address_amd*>(initFrom));
    initFrom = NULL;
  } else {
    busAddress_.surface_bus_address = 0;
    busAddress_.marker_bus_address = 0;
  }
  return Memory::create(initFrom, sysMemAlloc, skipAlloc, forceAlloc);
}

bool Buffer::isEntirelyCovered(const Coord3D& origin, const Coord3D& region) const {
  return ((origin[0] == 0) && (region[0] == getSize())) ? true : false;
}

bool Buffer::validateRegion(const Coord3D& origin, const Coord3D& region) const {
  return ((region[0] > 0) && (origin[0] < getSize()) && ((origin[0] + region[0]) <= getSize()))
             ? true
             : false;
}

void Pipe::initDeviceMemory() {
  deviceMemories_ = reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Pipe));
  memset(deviceMemories_, 0, NumDevicesWithP2P() * sizeof(DeviceMemory));
}

#define GETMIPDIM(dim, mip) (((dim >> mip) > 0) ? (dim >> mip) : 1)

Image::Image(const Format& format, Image& parent, uint baseMipLevel, cl_mem_flags flags,
             bool isMipmapView)
    : Memory(parent, flags, 0,
             parent.getWidth() * parent.getHeight() * parent.getDepth() * format.getElementSize()),
      impl_(format,
            Coord3D(parent.getWidth() * parent.getImageFormat().getElementSize() /
                        format.getElementSize(),
                    parent.getHeight(), parent.getDepth()),
            parent.getRowPitch(), parent.getSlicePitch(), parent.getBytePitch()),
      mipLevels_(isMipmapView ? parent.getMipLevels() : 1),
      baseMipLevel_(baseMipLevel) {
  if (baseMipLevel > 0) {
    impl_.region_.c[0] = GETMIPDIM(parent.getWidth(), baseMipLevel) *
                         parent.getImageFormat().getElementSize() / format.getElementSize();
    impl_.region_.c[1] = GETMIPDIM(parent.getHeight(), baseMipLevel);
    impl_.region_.c[2] = GETMIPDIM(parent.getDepth(), baseMipLevel);

    if (parent.getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
      impl_.region_.c[1] = parent.getHeight();
    } else if (parent.getType() == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
      impl_.region_.c[2] = parent.getDepth();
    }
    size_ = getWidth() * getHeight() * parent.getDepth() * format.getElementSize();
  }
  initDimension();
  image_view_ = true;
}

Image::Image(Context& context, Type type, Flags flags, const Format& format, size_t width,
             size_t height, size_t depth, size_t rowPitch, size_t slicePitch, uint mipLevels)
    : Memory(context, type, flags, width * height * depth * format.getElementSize()),
      impl_(format, Coord3D(width, height, depth), rowPitch, slicePitch),
      mipLevels_(mipLevels),
      baseMipLevel_(0) {
  initDimension();
}

Image::Image(Buffer& buffer, Type type, Flags flags, const Format& format, size_t width,
             size_t height, size_t depth, size_t rowPitch, size_t slicePitch, uint mipLevels,
             size_t offset)
    : Memory(buffer, flags, offset, buffer.getSize(), type),
      impl_(format, Coord3D(width, height, depth), rowPitch, slicePitch),
      mipLevels_(mipLevels),
      baseMipLevel_(0) {
  initDimension();
}

bool Image::validateDimensions(const std::vector<amd::Device*>& devices, cl_mem_object_type type,
                               size_t width, size_t height, size_t depth, size_t arraySize) {
  bool sizePass = false;
  switch (type) {
    case CL_MEM_OBJECT_IMAGE3D:
      if ((width == 0) || (height == 0) || (depth < 1)) {
        DevLogPrintfError("Invalid Dimenstions, width: %u height: %u depth: %u \n", width, height,
                          depth);
        return false;
      }
      for (const auto& dev : devices) {
        if ((dev->info().image3DMaxWidth_ >= width) && (dev->info().image3DMaxHeight_ >= height) &&
            (dev->info().image3DMaxDepth_ >= depth)) {
          return true;
        }
      }
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      if (arraySize == 0) {
        DevLogError("Array is empty \n");
        return false;
      }
      for (const auto& dev : devices) {
        if (dev->info().imageMaxArraySize_ >= arraySize) {
          sizePass = true;
          break;
        }
      }
      if (!sizePass) {
        DevLogPrintfError("Cannot allocate image of size: %u \n", arraySize);
        return false;
      }
    // Fall through...
    case CL_MEM_OBJECT_IMAGE2D:
      for (const auto dev : devices) {
        if ((dev->info().image2DMaxHeight_ >= height) && (dev->info().image2DMaxWidth_ >= width)) {
          return true;
        }
      }
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      if (arraySize == 0) {
        DevLogError("Array size cannot be empty \n");
        return false;
      }

      for (const auto& dev : devices) {
        if (dev->info().imageMaxArraySize_ >= arraySize) {
          sizePass = true;
          break;
        }
      }
      if (!sizePass) {
        DevLogPrintfError("Cannot allocate image of size: %u \n", arraySize);
        return false;
      }
    // Fall through...
    case CL_MEM_OBJECT_IMAGE1D:
      if (width == 0) {
        DevLogError("Invalid dimension \n");
        return false;
      }
      for (const auto& dev : devices) {
        if (dev->info().image2DMaxWidth_ >= width) {
          return true;
        }
      }
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      for (const auto& dev : devices) {
        if (dev->info().imageMaxBufferSize_ >= width) {
          return true;
        }
      }
      break;
    default:
      break;
  }

  DevLogError("Dimension Validation failed \n");
  return false;
}

void Image::initDimension() {
  const size_t elemSize = impl_.format_.getElementSize();
  if (impl_.rp_ == 0) {
    impl_.rp_ = impl_.region_[0] * elemSize;
  }
  switch (type_) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      dim_ = 3;
      if (impl_.sp_ == 0) {
        impl_.sp_ = impl_.region_[0] * impl_.region_[1] * elemSize;
      }
      break;
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      dim_ = 2;
      if ((impl_.sp_ == 0) && (type_ == CL_MEM_OBJECT_IMAGE1D_ARRAY)) {
        impl_.sp_ = impl_.rp_;
      }
      break;
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    default:
      dim_ = 1;
      break;
  }
}

void Image::initDeviceMemory() {
  deviceMemories_ = reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image));
  memset(deviceMemories_, 0, NumDevicesWithP2P() * sizeof(DeviceMemory));
}

size_t Image::Format::getNumChannels() const {
  switch (image_channel_order) {
    case CL_RG:
    case CL_RA:
      return 2;

    case CL_RGB:
    case CL_sRGB:
    case CL_sRGBx:
      return 3;

    case CL_RGBA:
    case CL_BGRA:
    case CL_ARGB:
    case CL_sRGBA:
    case CL_sBGRA:
      return 4;
  }
  return 1;
}

size_t Image::Format::getElementSize() const {
  size_t bytesPerPixel = getNumChannels();
  switch (image_channel_data_type) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      break;

    case CL_UNORM_INT_101010:
      bytesPerPixel = 4;
      break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      bytesPerPixel *= 4;
      break;

    default:
      bytesPerPixel *= 2;
      break;
  }
  return bytesPerPixel;
}

bool Image::Format::isValid() const {
  switch (image_channel_data_type) {
    case CL_SNORM_INT8:
    case CL_SNORM_INT16:
    case CL_UNORM_INT8:
    case CL_UNORM_INT16:
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
    case CL_UNORM_INT_101010:
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
    case CL_HALF_FLOAT:
    case CL_FLOAT:
      break;

    default: {
      DevLogPrintfError("Invalid Image format: %u \n", image_channel_data_type);
      return false;
    }
  }

  switch (image_channel_order) {
    case CL_R:
    case CL_A:
    case CL_RG:
    case CL_RA:
    case CL_RGBA:
      break;

    case CL_INTENSITY:
    case CL_LUMINANCE:
      switch (image_channel_data_type) {
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
          break;

        default: {
          DevLogPrintfError("Invalid Luminance: %u \n", image_channel_data_type);
          return false;
        }
      }
      break;

    case CL_RGB:
      switch (image_channel_data_type) {
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_UNORM_INT_101010:
          break;

        default: {
          DevLogPrintfError("Invalid RGB: %u \n", image_channel_data_type);
          return false;
        }
      }
      break;

    case CL_BGRA:
    case CL_ARGB:
      switch (image_channel_data_type) {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
          break;

        default: {
          DevLogPrintfError("Invalid BGRA/ARGB: %u \n", image_channel_data_type);
          return false;
        }
      }
      break;

    case CL_sRGB:
    case CL_sRGBx:
    case CL_sRGBA:
    case CL_sBGRA:
      switch (image_channel_data_type) {
        case CL_UNORM_INT8:
          break;
        default: {
          DevLogPrintfError("Invalid sBGRA: %u \n", image_channel_data_type);
          return false;
        }
      }
      break;

    case CL_DEPTH:
      switch (image_channel_data_type) {
        case CL_UNORM_INT16:
        case CL_FLOAT:
          break;
        default: {
          DevLogPrintfError("Invalid CL Depth: %u \n", image_channel_data_type);
          return false;
        }
      }
      break;

    default: {
      DevLogPrintfError("Invalid image_channel_order: %u \n", image_channel_order);
      return false;
    }
  }
  return true;
}

// definition of list of supported formats
const cl_image_format Image::supportedFormats[] = {
    // R
    {CL_R, CL_SNORM_INT8},
    {CL_R, CL_SNORM_INT16},
    {CL_R, CL_UNORM_INT8},
    {CL_R, CL_UNORM_INT16},

    {CL_R, CL_SIGNED_INT8},
    {CL_R, CL_SIGNED_INT16},
    {CL_R, CL_SIGNED_INT32},
    {CL_R, CL_UNSIGNED_INT8},
    {CL_R, CL_UNSIGNED_INT16},
    {CL_R, CL_UNSIGNED_INT32},

    {CL_R, CL_HALF_FLOAT},
    {CL_R, CL_FLOAT},

    // A
    {CL_A, CL_SNORM_INT8},
    {CL_A, CL_SNORM_INT16},
    {CL_A, CL_UNORM_INT8},
    {CL_A, CL_UNORM_INT16},

    {CL_A, CL_SIGNED_INT8},
    {CL_A, CL_SIGNED_INT16},
    {CL_A, CL_SIGNED_INT32},
    {CL_A, CL_UNSIGNED_INT8},
    {CL_A, CL_UNSIGNED_INT16},
    {CL_A, CL_UNSIGNED_INT32},

    {CL_A, CL_HALF_FLOAT},
    {CL_A, CL_FLOAT},

    // RG
    {CL_RG, CL_SNORM_INT8},
    {CL_RG, CL_SNORM_INT16},
    {CL_RG, CL_UNORM_INT8},
    {CL_RG, CL_UNORM_INT16},

    {CL_RG, CL_SIGNED_INT8},
    {CL_RG, CL_SIGNED_INT16},
    {CL_RG, CL_SIGNED_INT32},
    {CL_RG, CL_UNSIGNED_INT8},
    {CL_RG, CL_UNSIGNED_INT16},
    {CL_RG, CL_UNSIGNED_INT32},

    {CL_RG, CL_HALF_FLOAT},
    {CL_RG, CL_FLOAT},

    // RGBA
    {CL_RGBA, CL_SNORM_INT8},
    {CL_RGBA, CL_SNORM_INT16},
    {CL_RGBA, CL_UNORM_INT8},
    {CL_RGBA, CL_UNORM_INT16},

    {CL_RGBA, CL_SIGNED_INT8},
    {CL_RGBA, CL_SIGNED_INT16},
    {CL_RGBA, CL_SIGNED_INT32},
    {CL_RGBA, CL_UNSIGNED_INT8},
    {CL_RGBA, CL_UNSIGNED_INT16},
    {CL_RGBA, CL_UNSIGNED_INT32},

    {CL_RGBA, CL_HALF_FLOAT},
    {CL_RGBA, CL_FLOAT},

    // ARGB
    {CL_ARGB, CL_SNORM_INT8},
    {CL_ARGB, CL_UNORM_INT8},
    {CL_ARGB, CL_SIGNED_INT8},
    {CL_ARGB, CL_UNSIGNED_INT8},

    // BGRA
    {CL_BGRA, CL_SNORM_INT8},
    {CL_BGRA, CL_UNORM_INT8},
    {CL_BGRA, CL_SIGNED_INT8},
    {CL_BGRA, CL_UNSIGNED_INT8},

    // LUMINANCE
    {CL_LUMINANCE, CL_SNORM_INT8},
    {CL_LUMINANCE, CL_SNORM_INT16},
    {CL_LUMINANCE, CL_UNORM_INT8},
    {CL_LUMINANCE, CL_UNORM_INT16},
    {CL_LUMINANCE, CL_HALF_FLOAT},
    {CL_LUMINANCE, CL_FLOAT},

    // INTENSITY
    {CL_INTENSITY, CL_SNORM_INT8},
    {CL_INTENSITY, CL_SNORM_INT16},
    {CL_INTENSITY, CL_UNORM_INT8},
    {CL_INTENSITY, CL_UNORM_INT16},
    {CL_INTENSITY, CL_HALF_FLOAT},
    {CL_INTENSITY, CL_FLOAT},

    // RGB
    {CL_RGB, CL_UNORM_INT_101010},

    // sRGB
    {CL_sRGBA, CL_UNORM_INT8},

    // DEPTH
    {CL_DEPTH, CL_UNORM_INT16},
    {CL_DEPTH, CL_FLOAT},
};

const uint32_t NUM_CHANNEL_ORDER_OF_RGB = 1;   // The number of channel orders of RGB at the end of
                                               // the table supportedFormats above and before sRGB
                                               // and depth.
const uint32_t NUM_CHANNEL_ORDER_OF_sRGB = 1;  // The number of channel orders of sRGB at the end of
                                               // the table supportedFormats above and before depth.
const uint32_t NUM_CHANNEL_ORDER_OF_DEPTH =
    2;  // The number of channel orders of DEPTH at the end of the table supportedFormats above.

// definition of list of supported RA formats
const cl_image_format Image::supportedFormatsRA[] = {
    {CL_RA, CL_SNORM_INT8},     {CL_RA, CL_SNORM_INT16},   {CL_RA, CL_UNORM_INT8},
    {CL_RA, CL_UNORM_INT16},    {CL_RA, CL_SIGNED_INT8},   {CL_RA, CL_SIGNED_INT16},
    {CL_RA, CL_SIGNED_INT32},   {CL_RA, CL_UNSIGNED_INT8}, {CL_RA, CL_UNSIGNED_INT16},
    {CL_RA, CL_UNSIGNED_INT32}, {CL_RA, CL_HALF_FLOAT},    {CL_RA, CL_FLOAT},
};

const cl_image_format Image::supportedDepthStencilFormats[] = {
    // DEPTH STENCIL
    {CL_DEPTH_STENCIL, CL_FLOAT},
    {CL_DEPTH_STENCIL, CL_UNORM_INT24}};

uint32_t Image::numSupportedFormats(const Context& context, cl_mem_object_type image_type,
                                    cl_mem_flags flags) {
  const std::vector<amd::Device*>& devices = context.devices();
  uint numFormats = sizeof(supportedFormats) / sizeof(cl_image_format);

  bool supportRA = false;
  bool supportDepthsRGB = false;
  bool supportDepthStencil = false;

  // Add RA if RA is supported.
  for (uint i = 0; i < devices.size(); i++) {
    if (devices[i]->settings().supportRA_) {
      supportRA = true;
    }
    if (devices[i]->settings().supportDepthsRGB_) {
      supportDepthsRGB = true;
    }
    if (devices[i]->settings().checkExtension(ClKhrGLDepthImages) &&
        (context.info().flags_ & Context::GLDeviceKhr)) {
      supportDepthStencil = true;
    }
  }

  if (supportDepthsRGB) {
    if ((image_type != CL_MEM_OBJECT_IMAGE2D) && (image_type != CL_MEM_OBJECT_IMAGE2D_ARRAY) &&
        (image_type != 0)) {
      numFormats -= NUM_CHANNEL_ORDER_OF_DEPTH;  // substract channel order of DEPTH type.
    }
    // Currently we are not supported sRGB for write_imagef (extension cl_khr_srgb_image_writes)
    if ((image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
        ((flags & (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_KERNEL_READ_AND_WRITE)) != 0)) {
      numFormats -= NUM_CHANNEL_ORDER_OF_sRGB;
    }
  } else {
    numFormats -= NUM_CHANNEL_ORDER_OF_RGB;    // substract channel order of RGB type.
    numFormats -= NUM_CHANNEL_ORDER_OF_sRGB;   // substract channel order of sRGB type.
    numFormats -= NUM_CHANNEL_ORDER_OF_DEPTH;  // substract channel order of DEPTH type.
  }

  // Add RA if RA is supported. RA isn't supported on SI.
  if (supportRA) {
    numFormats +=
        sizeof(supportedFormatsRA) / sizeof(cl_image_format);  // Add channel order of RA type.
  }

  if (supportDepthStencil) {
    if (flags & CL_MEM_READ_ONLY) {
      numFormats += sizeof(supportedDepthStencilFormats) / sizeof(cl_image_format);
    }
  }

  return numFormats;
}

uint32_t Image::getSupportedFormats(const Context& context, cl_mem_object_type image_type,
                                    const uint32_t num_entries, cl_image_format* image_formats,
                                    cl_mem_flags flags) {
  const std::vector<amd::Device*>& devices = context.devices();
  uint numFormats = 0;

  bool supportRA = false;
  bool supportDepthsRGB = false;
  bool supportDepthStencil = false;

  // Add RA if RA is supported.
  for (uint i = 0; i < devices.size(); i++) {
    if (devices[i]->settings().supportRA_) {
      supportRA = true;
    }
    if (devices[i]->settings().supportDepthsRGB_) {
      supportDepthsRGB = true;
    }
    if (devices[i]->settings().checkExtension(ClKhrGLDepthImages) &&
        (context.info().flags_ & Context::GLDeviceKhr)) {
      supportDepthStencil = true;
    }
  }

  cl_image_format* format = image_formats;
  uint numSupportedFormats = sizeof(supportedFormats) / sizeof(cl_image_format);

  bool srgbWriteSupported = true;
  if (supportDepthsRGB) {
    if ((image_type != CL_MEM_OBJECT_IMAGE2D) && (image_type != CL_MEM_OBJECT_IMAGE2D_ARRAY) &&
        (image_type != 0)) {
      numSupportedFormats -= NUM_CHANNEL_ORDER_OF_DEPTH;
    }
    // Currently we are not supported sRGB for write_imagef (extension cl_khr_srgb_image_writes)
    if ((image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
        ((flags & (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_KERNEL_READ_AND_WRITE)) != 0)) {
      srgbWriteSupported = false;
    }
  } else {
    numSupportedFormats -= NUM_CHANNEL_ORDER_OF_RGB;    // substract channel order of RGB type.
    numSupportedFormats -= NUM_CHANNEL_ORDER_OF_sRGB;   // substract channel order of sRGB type.
    numSupportedFormats -= NUM_CHANNEL_ORDER_OF_DEPTH;  // substract channel order of DEPTH type.
  }

  for (uint i = 0; i < numSupportedFormats; i++) {
    if (numFormats == num_entries) {
      break;
    }
    if (!srgbWriteSupported) {
      if ((amd::Image::supportedFormats[i].image_channel_order == CL_sRGBA) ||
          (amd::Image::supportedFormats[i].image_channel_order == CL_sRGB) ||
          (amd::Image::supportedFormats[i].image_channel_order == CL_sRGBx) ||
          (amd::Image::supportedFormats[i].image_channel_order == CL_sBGRA)) {
        continue;
      }
    }
    *format++ = amd::Image::supportedFormats[i];
    numFormats++;
  }

  // Add RA if RA is supported.
  if (supportRA) {
    for (uint i = 0; i < sizeof(supportedFormatsRA) / sizeof(cl_image_format); i++) {
      if (numFormats == num_entries) {
        break;
      }
      *format++ = amd::Image::supportedFormatsRA[i];
      numFormats++;
    }
  }

  if (supportDepthStencil) {
    if (flags & CL_MEM_READ_ONLY) {
      for (uint i = 0; i < sizeof(supportedDepthStencilFormats) / sizeof(cl_image_format); i++) {
        if (numFormats == num_entries) {
          break;
        }
        *format++ = amd::Image::supportedDepthStencilFormats[i];
        numFormats++;
      }
    }
  }
  return numFormats;
}

bool Image::Format::isSupported(const Context& context, cl_mem_object_type image_type,
                                cl_mem_flags flags) const {
  const cl_image_format RGBA10 = {CL_RGBA, CL_UNORM_INT_101010};

  uint numFormats = numSupportedFormats(context, image_type, flags);

  std::vector<cl_image_format> image_formats(numFormats);

  getSupportedFormats(context, image_type, numFormats, image_formats.data(), flags);

  for (uint i = 0; i < numFormats; i++) {
    if (*this == image_formats[i]) {
      return true;
    }
  }
  if (*this == RGBA10) {
    return true;
  }

  return false;
}

// ================================================================================================
Image* Image::createView(const Context& context, const Format& format, device::VirtualDevice* vDev,
                         uint baseMipLevel, cl_mem_flags flags, bool createMipmapView,
                         bool forceAlloc) {
  // Find the image dimensions and create a corresponding object
  Image* view = new (context) Image(format, *this, baseMipLevel, flags, createMipmapView);

  if (view != nullptr) {
    // Set GPU virtual device for this view
    view->setVirtualDevice(vDev);

    view->resetAllocationState();

    // Initialize array of the device memory pointers
    view->initDeviceMemory();

    // Check if runtime has to allocate memory
    if ((context.devices().size() == 1) || DISABLE_DEFERRED_ALLOC || forceAlloc) {
      for (uint i = 0; i < numDevices_; ++i) {
        // Make sure the parent's device memory is avaialbe
        if ((deviceMemories_[i].ref_ != nullptr) && (deviceMemories_[i].value_ != nullptr)) {
          device::Memory* mem = view->getDeviceMemory(*(deviceMemories_[i].ref_));
        }
      }
    }
  }

  return view;
}

// ================================================================================================
bool Image::isEntirelyCovered(const Coord3D& origin, const Coord3D& region) const {
  return (origin[0] == 0 && origin[1] == 0 && origin[2] == 0 && region[0] == getWidth() &&
          region[1] == getHeight() && region[2] == getDepth())
             ? true
             : false;
}

bool Image::validateRegion(const Coord3D& origin, const Coord3D& region) const {
  return ((region[0] > 0) && (region[1] > 0) && (region[2] > 0) && (origin[0] < getWidth()) &&
          (region[0] != 0) && (origin[1] < getHeight()) && (region[1] != 0) &&
          (origin[2] < getDepth()) && (region[2] != 0) && ((origin[0] + region[0]) <= getWidth()) &&
          ((origin[1] + region[1]) <= getHeight()) && ((origin[2] + region[2]) <= getDepth()))
             ? true
             : false;
}

bool Image::isRowSliceValid(size_t rowPitch, size_t slice, size_t width, size_t height) const {
  size_t tmpHeight = (getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? 1 : height;

  bool valid = (rowPitch == 0) ||
               ((rowPitch != 0) && (rowPitch >= width * getImageFormat().getElementSize()));

  return ((slice == 0) || ((slice != 0) && (slice >= rowPitch * tmpHeight))) ? valid : false;
}

void Image::copyToBackingStore(void* initFrom) {
  char* dst = reinterpret_cast<char*>(getHostMem());
  size_t cpySize = getWidth() * getImageFormat().getElementSize();

  for (uint z = 0; z < getDepth(); ++z) {
    char* src = reinterpret_cast<char*>(initFrom) + z * getSlicePitch();
    for (uint y = 0; y < getHeight(); ++y) {
      memcpy(dst, src, cpySize);
      dst += cpySize;
      src += getRowPitch();
    }
  }

  impl_.rp_ = cpySize;
  if (impl_.sp_ != 0) {
    impl_.sp_ = impl_.rp_;
    if (getDims() == 3) {
      impl_.sp_ *= getHeight();
    }
  }
}

template <typename In, typename Out> static Out interpret_value(In in) {
  static_assert(sizeof(In) == sizeof(Out));
  union {
    In in_val;
    Out out_val;
  } u{in};
  return u.out_val;
}

static int round_to_even(float v) {
  // clamp overflow
  if (v >= -(float)std::numeric_limits<int>::min()) {
    return std::numeric_limits<int>::max();
  }
  if (v <= (float)std::numeric_limits<int>::min()) {
    return std::numeric_limits<int>::min();
  }
  static const unsigned int magic[2] = {0x4b000000u, 0xcb000000u};

  // round fractional values to integer value
  if (fabsf(v) < interpret_value<unsigned, float>(magic[0])) {
    float magicVal = interpret_value<unsigned, float>(magic[v < 0.0f]);
    v += magicVal;
    v -= magicVal;
  }

  return static_cast<int>(v);
}

static uint16_t float2half_rtz(float f) {
  union {
    float f;
    uint32_t u;
  } u = {f};
  uint32_t sign = (u.u >> 16) & 0x8000;
  float x = fabsf(f);

  // Nan
  if (x != x) {
    u.u >>= (24 - 11);
    u.u &= 0x7fff;
    u.u |= 0x0200;  // silence the NaN
    return u.u | sign;
  }
  int values[5] = {0x47800000, 0x33800000, 0x38800000, 0x4b800000, 0x7f800000};
  // overflow
  if (x >= interpret_value<int, float>(values[0])) {
    if (x == interpret_value<int, float>(values[4])) {
      return 0x7c00 | sign;
    }
    return 0x7bff | sign;
  }

  // underflow
  if (x < interpret_value<int, float>(values[1])) {
    return sign;  // The halfway case can return 0x0001 or 0. 0 is even.
  }

  // half denormal
  if (x < interpret_value<int, float>(values[2])) {
    x *= interpret_value<int, float>(values[3]);
    return static_cast<uint16_t>((int)x | sign);
  }

  u.u &= 0xFFFFE000U;
  u.u -= 0x38000000U;

  return (u.u >> (24 - 11)) | sign;
}

void Image::Format::getChannelOrder(uint8_t* channelOrder) const {
  enum { CH_ORDER_R = 0, CH_ORDER_G, CH_ORDER_B, CH_ORDER_A };
  switch (image_channel_order) {
    case CL_A:
      channelOrder[0] = CH_ORDER_A;
      break;

    case CL_RA:
      channelOrder[0] = CH_ORDER_R;
      channelOrder[1] = CH_ORDER_A;
      break;

    case CL_BGRA:
      channelOrder[0] = CH_ORDER_B;
      channelOrder[1] = CH_ORDER_G;
      channelOrder[2] = CH_ORDER_R;
      channelOrder[3] = CH_ORDER_A;
      break;

    case CL_ARGB:
      channelOrder[0] = CH_ORDER_A;
      channelOrder[1] = CH_ORDER_R;
      channelOrder[2] = CH_ORDER_G;
      channelOrder[3] = CH_ORDER_B;
      break;

    default:
      channelOrder[0] = CH_ORDER_R;
      channelOrder[1] = CH_ORDER_G;
      channelOrder[2] = CH_ORDER_B;
      channelOrder[3] = CH_ORDER_A;
      break;
  }
}

// "colorRGBA" is a four component RGBA floating-point color value if the image
// channel data type is not an unnormalized signed and unsigned integer type,
// is a four component signed integer value if the image channel data type is
// an unnormalized signed integer type and is a four component unsigned integer
// value if the image channel data type is an unormalized unsigned integer type.
void Image::Format::formatColor(const void* colorRGBA, void* colorFormat) const {
  union t565 {
    struct {
      uint16_t r_ : 5;
      uint16_t g_ : 6;
      uint16_t b_ : 5;
    };
    uint16_t rgba_;
  };

  union t555 {
    struct {
      uint16_t r_ : 5;
      uint16_t g_ : 5;
      uint16_t b_ : 5;
      uint16_t a_ : 1;
    };
    uint16_t rgba_;
  };

  union t101010 {
    struct {
      uint32_t b_ : 10;
      uint32_t g_ : 10;
      uint32_t r_ : 10;
      uint32_t a_ : 2;
    };
    uint32_t rgba_;
  };

  const float* colorRGBAf = reinterpret_cast<const float*>(colorRGBA);
  const int32_t* colorRGBAi = reinterpret_cast<const int32_t*>(colorRGBA);
  const uint32_t* colorRGBAui = reinterpret_cast<const uint32_t*>(colorRGBA);

  size_t chCount = getNumChannels();
  uint8_t chOrder[4];
  getChannelOrder(chOrder);

  bool allChannels = false;
  for (size_t i = 0; i < chCount && !allChannels; ++i) {
    switch (image_channel_data_type) {
      case CL_SNORM_INT8: {
        int8_t* color = reinterpret_cast<int8_t*>(colorFormat);
        color[i] = round_to_even(INT8_MAX * colorRGBAf[chOrder[i]]);
      } break;
      case CL_SNORM_INT16: {
        int16_t* color = reinterpret_cast<int16_t*>(colorFormat);
        color[i] = round_to_even(INT16_MAX * colorRGBAf[chOrder[i]]);
      } break;
      case CL_UNORM_INT8: {
        uint8_t* color = reinterpret_cast<uint8_t*>(colorFormat);
        color[i] = round_to_even(UINT8_MAX * colorRGBAf[chOrder[i]]);
      } break;
      case CL_UNORM_INT16: {
        uint16_t* color = reinterpret_cast<uint16_t*>(colorFormat);
        color[i] = round_to_even(UINT16_MAX * colorRGBAf[chOrder[i]]);
      } break;
      case CL_UNORM_SHORT_565: {
        t565* color = reinterpret_cast<t565*>(colorFormat);
        color->r_ = round_to_even(0x1F * colorRGBAf[0]);
        color->g_ = round_to_even(0x3F * colorRGBAf[1]);
        color->b_ = round_to_even(0x1F * colorRGBAf[2]);
        allChannels = true;
      } break;
      case CL_UNORM_SHORT_555: {
        t555* color = reinterpret_cast<t555*>(colorFormat);
        color->r_ = round_to_even(0x1F * colorRGBAf[0]);
        color->g_ = round_to_even(0x1F * colorRGBAf[1]);
        color->b_ = round_to_even(0x1F * colorRGBAf[2]);
        color->a_ = round_to_even(colorRGBAf[3]);
        allChannels = true;
      } break;
      case CL_UNORM_INT_101010: {
        t101010* color = reinterpret_cast<t101010*>(colorFormat);
        color->r_ = round_to_even(0x3FF * colorRGBAf[0]);
        color->g_ = round_to_even(0x3FF * colorRGBAf[1]);
        color->b_ = round_to_even(0x3FF * colorRGBAf[2]);
        color->a_ = round_to_even(0x3 * colorRGBAf[3]);
        allChannels = true;
      } break;
      case CL_SIGNED_INT8: {
        int8_t* color = reinterpret_cast<int8_t*>(colorFormat);
        color[i] = colorRGBAi[chOrder[i]];
      } break;
      case CL_SIGNED_INT16: {
        int16_t* color = reinterpret_cast<int16_t*>(colorFormat);
        color[i] = colorRGBAi[chOrder[i]];
      } break;
      case CL_SIGNED_INT32: {
        int32_t* color = reinterpret_cast<int32_t*>(colorFormat);
        color[i] = colorRGBAi[chOrder[i]];
      } break;
      case CL_UNSIGNED_INT8: {
        uint8_t* color = reinterpret_cast<uint8_t*>(colorFormat);
        color[i] = colorRGBAui[chOrder[i]];
      } break;
      case CL_UNSIGNED_INT16: {
        uint16_t* color = reinterpret_cast<uint16_t*>(colorFormat);
        color[i] = colorRGBAui[chOrder[i]];
      } break;
      case CL_UNSIGNED_INT32: {
        uint32_t* color = reinterpret_cast<uint32_t*>(colorFormat);
        color[i] = colorRGBAui[chOrder[i]];
      } break;
      case CL_HALF_FLOAT: {
        uint16_t* color = reinterpret_cast<uint16_t*>(colorFormat);
        color[i] = float2half_rtz(colorRGBAf[chOrder[i]]);
      } break;
      case CL_FLOAT: {
        float* color = reinterpret_cast<float*>(colorFormat);
        color[i] = colorRGBAf[chOrder[i]];
      } break;
    }
  }
}

Monitor SvmBuffer::AllocatedLock_ ROCCLR_INIT_PRIORITY(101)("Guards SVM allocation list");
std::map<uintptr_t, uintptr_t> SvmBuffer::Allocated_ ROCCLR_INIT_PRIORITY(101);

void SvmBuffer::Add(uintptr_t k, uintptr_t v) {
  ScopedLock lock(AllocatedLock_);
  Allocated_.insert(std::pair<uintptr_t, uintptr_t>(k, v));
}

void SvmBuffer::Remove(uintptr_t k) {
  ScopedLock lock(AllocatedLock_);
  Allocated_.erase(k);
}

bool SvmBuffer::Contains(uintptr_t ptr) {
  ScopedLock lock(AllocatedLock_);
  auto it = Allocated_.upper_bound(ptr);
  if (it == Allocated_.begin()) {
    return false;
  }
  --it;
  return ptr >= it->first && ptr < it->second;
}

// The allocation flags are ignored for now.
void* SvmBuffer::malloc(Context& context, cl_svm_mem_flags flags, size_t size, size_t alignment,
                        const amd::Device* curDev, void* hostptr) {
  void* ret = context.svmAlloc(size, alignment, flags, curDev, hostptr);
  if (ret == nullptr) {
    LogError("Unable to allocate aligned memory");
    return nullptr;
  }
  uintptr_t ret_u = reinterpret_cast<uintptr_t>(ret);
  Add(ret_u, ret_u + size);
  return ret;
}

void SvmBuffer::free(const Context& context, void* ptr) {
  Remove(reinterpret_cast<uintptr_t>(ptr));
  context.svmFree(ptr);
}

void SvmBuffer::memFill(void* dst, const void* src, size_t srcSize, size_t times) {
  address dstAddress = reinterpret_cast<address>(dst);
  const_address srcAddress = reinterpret_cast<const_address>(src);
  for (size_t i = 0; i < times; i++) {
    ::memcpy(dstAddress + i * srcSize, srcAddress, srcSize);
  }
}

// ================================================================================================
bool SvmBuffer::malloced(const void* ptr) { return Contains(reinterpret_cast<uintptr_t>(ptr)); }

// ================================================================================================
void IpcBuffer::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(IpcBuffer));
  memset(deviceMemories_, 0, NumDevicesWithP2P() * sizeof(DeviceMemory));
}

}  // namespace amd
