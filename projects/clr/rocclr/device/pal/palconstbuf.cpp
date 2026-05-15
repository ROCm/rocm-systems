/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "device/pal/palconstbuf.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palsettings.hpp"

namespace amd::pal {

// ================================================================================================
ManagedBuffer::ManagedBuffer(VirtualGPU& gpu, uint32_t size)
    : gpu_(gpu),
      pool_(MaxNumberOfBuffers),
      activeBuffer_(0),
      size_(size),
      wrtOffset_(0),
      wrtAddress_(nullptr) {}

// ================================================================================================
void ManagedBuffer::release() {
  for (auto it : pool_) {
    if ((it.buf != nullptr) && (it.buf->data() != nullptr)) {
      it.buf->unmap(&gpu_);
    }
    delete it.buf;
  }
}

// ================================================================================================
bool ManagedBuffer::create(Resource::MemoryType type) {
  for (uint i = 0; i < pool_.size(); ++i) {
    pool_[i].buf = new Memory(const_cast<pal::Device&>(gpu_.dev()), size_);
    if (nullptr == pool_[i].buf || !pool_[i].buf->create(type)) {
      LogPrintfError("We couldn't create HW constant buffer, size(%d)!", size_);
      return false;
    }
    // Assign virtual gpu to the allocation. Buffer will be used only on a particular queue
    pool_[i].buf->memRef()->gpu_ = &gpu_;
    void* wrtAddress = pool_[i].buf->map(&gpu_);
    if (wrtAddress == nullptr) {
      LogPrintfError("We couldn't map HW constant buffer, size(%d)!", size_);
      return false;
    }
    // Make sure OCL touches every buffer in the queue to avoid delays on the first submit
    uint dummy = 0;
    static constexpr bool Wait = true;
    // Write 0 for the buffer paging by VidMM
    pool_[i].buf->writeRawData(gpu_, 0, sizeof(dummy), &dummy, Wait);
  }
  wrtAddress_ = pool_[activeBuffer_].buf->data();
  return true;
}

// ================================================================================================
address ManagedBuffer::reserve(uint32_t size, uint64_t* gpu_address) {
  // Align to the maximum data size available in OpenCL
  static constexpr uint32_t MemAlignment = sizeof(double[32]);

  // Align reserve size on the vector's boundary
  uint32_t count = amd::alignUp(size, MemAlignment);

  // Save previous event
  pinGpuEvent();

  // Check if buffer has enough space for reservation
  if ((wrtOffset_ + count) > size_) {
    // Get the next buffer in the list
    ++activeBuffer_;
    activeBuffer_ %= MaxNumberOfBuffers;
    if (!gpu().dev().settings().disableSdma_) {
      // Make sure the buffer isn't busy
      gpu().waitForEvent(&pool_[activeBuffer_].events[SdmaEngine]);
    }
    gpu().waitForEvent(&pool_[activeBuffer_].events[MainEngine]);
    wrtAddress_ = pool_[activeBuffer_].buf->data();
    wrtOffset_ = 0;
  }

  *gpu_address = pool_[activeBuffer_].buf->vmAddress() + wrtOffset_;
  address cpu_address = wrtAddress_ + wrtOffset_;

  // Adjust the offset by the reserved size
  wrtOffset_ += count;

  return cpu_address;
}

// ================================================================================================
void ManagedBuffer::pinGpuEvent() {
  GpuEvent* event = activeMemory()->getGpuEvent(gpu());
  pool_[activeBuffer_].events[event->engineId_] = *event;
  activeMemory()->setBusy(gpu(), GpuEvent::InvalidID);
}

// ================================================================================================
ConstantBuffer::ConstantBuffer(ManagedBuffer& mbuf, uint32_t size)
    : mbuf_(mbuf), sys_mem_copy_(nullptr), size_(size) {}

// ================================================================================================
ConstantBuffer::~ConstantBuffer() { amd::AlignedMemory::deallocate(sys_mem_copy_); }

// ================================================================================================
bool ConstantBuffer::Create() {
  // Create sysmem copy for the constant buffer.
  sys_mem_copy_ = reinterpret_cast<address>(amd::AlignedMemory::allocate(size_, 256));
  if (sys_mem_copy_ == nullptr) {
    LogPrintfError("We couldn't allocate sysmem copy for constant buffer, size(%d)!", size_);
    return false;
  }
  memset(sys_mem_copy_, 0, size_);
  return true;
}

// ================================================================================================
uint64_t ConstantBuffer::UploadDataToHw(uint32_t size) const {
  uint64_t vm_address;
  address cpu_address = mbuf_.reserve(size, &vm_address);
  // Update memory with new CB data
  memcpy(cpu_address, sys_mem_copy_, size);
  return vm_address;
}

// ================================================================================================
uint64_t ConstantBuffer::UploadDataToHw(const void* sysmem, uint32_t size) const {
  uint64_t vm_address;
  address cpu_address = mbuf_.reserve(size, &vm_address);
  // Update memory with new CB data
  memcpy(cpu_address, sysmem, size);
  return vm_address;
}

// ================================================================================================
XferBuffer::XferBuffer(const Device& device, ManagedBuffer& mbuf, uint32_t size)
    : buffer_view_(device, size), mbuf_(mbuf), size_(size) {
  // Create a view for access
  Resource::ViewParams params = {};
  params.gpu_ = &mbuf_.gpu();
  params.offset_ = 0;
  params.size_ = size_;
  params.resource_ = mbuf_.activeMemory();
  bool result = buffer_view_.create(Resource::View, &params);
  assert(result && "View creaiton should never return an error!");
}

// ================================================================================================
Memory& XferBuffer::Acquire(uint32_t size) {
  uint64_t vm_address;
  // Reserve space in the managed buffer
  address cpu_address = mbuf_.reserve(size, &vm_address);
  // Update a view for access
  buffer_view_.updateView(mbuf_.activeMemory(), vm_address - mbuf_.vmAddress(), size);
  return buffer_view_;
}

}  // namespace amd::pal
