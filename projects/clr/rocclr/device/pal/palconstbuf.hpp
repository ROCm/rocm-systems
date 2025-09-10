/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "device/pal/palmemory.hpp"

//! \namespace amd::pal PAL Resource Implementation
namespace amd::pal {

//! Managed buffer (staging or constant)
class ManagedBuffer : public amd::EmbeddedObject {
 public:
  //! Constructor for the ConstBuffer class
  ManagedBuffer(VirtualGPU& gpu,  //!< Virtual GPU device object
                uint32_t size     //!< size of the managed buffers in bytes
  );
  ~ManagedBuffer() {}

  //! Creates the managed buffers
  bool create(Resource::MemoryType type);

  //! Release the managed buffers
  void release();

  /*! \brief Uploads current constant buffer data from sysMemCopy_ to HW
   *
   *  \return True if the data upload was succesful
   */
  address reserve(uint32_t size,  //!< real data size for upload
                  uint64_t* gpu_address);

  //! Returns CB size
  uint32_t size() const { return size_; }

  //! Returns current write offset for the managed buffer
  uint32_t wrtOffset() const { return wrtOffset_; }

  //! Returns active GPU buffer
  Memory* activeMemory() const { return pool_[activeBuffer_].buf; }

  //! Retruns VM address for the active buffer
  uint64_t vmAddress() const { return pool_[activeBuffer_].buf->vmAddress(); }

  //! Update the timestamp for the HW operation
  void pinGpuEvent();

  //! Returns VirtualGPU object this managed resource associated
  VirtualGPU& gpu() const { return gpu_; }

  void CpuReadBack() const {
    volatile auto tmp = *reinterpret_cast<uint64_t*>(pool_[activeBuffer_].buf->data());
  }

 private:
  struct TimeStampedBuffer {
    Memory* buf;
    GpuEvent events[AllEngines];
  };

  //! The maximum number of the managed buffers
  static constexpr uint32_t MaxNumberOfBuffers = 3;

  //! Disable copy constructor
  ManagedBuffer(const ManagedBuffer&) = delete;

  //! Disable operator=
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  VirtualGPU& gpu_;                      //!< Virtual GPU object
  std::vector<TimeStampedBuffer> pool_;  //!< Buffers for management
  uint32_t activeBuffer_;                //!< Current active buffer
  uint32_t size_;                        //!< Constant buffer size
  uint32_t wrtOffset_;                   //!< Current write offset
  address wrtAddress_;                   //!< Write address in CB
};

//! Constant buffer
class ConstantBuffer : public amd::HeapObject {
 public:
  //! Constructor for the ConstBuffer class
  ConstantBuffer(ManagedBuffer& mbuf,  //!< Managed buffer
                 uint32_t size         //!< Max size of the constant buffer
  );

  //! Destructor for the ConstBuffer class
  ~ConstantBuffer();

  //! Creates the HW constant buffer
  bool Create();

  /*! \brief Uploads current constant buffer data from sysMemCopy_ to HW
   *
   *  \return GPU address for the uploaded data
   */
  uint64_t UploadDataToHw(uint32_t size  //!< real data size for upload
  ) const;

  /*! \brief Uploads current constant buffer data from sysMemCopy_ to HW
   *
   *  \return GPU address for the uploaded data
   */
  uint64_t UploadDataToHw(const void* sysmem,  //!< Pointer to the data for upload
                          uint32_t size        //!< Real data size for upload
  ) const;

  //! Returns a pointer to the system memory copy for CB
  address SysMemCopy() const { return sys_mem_copy_; }

  //! Returns active GPU buffer
  Memory* ActiveMemory() const { return mbuf_.activeMemory(); }

 private:
  //! Disable copy constructor
  ConstantBuffer(const ConstantBuffer&) = delete;

  //! Disable operator=
  ConstantBuffer& operator=(const ConstantBuffer&) = delete;

  ManagedBuffer& mbuf_;   //!< Managed buffer on GPU
  address sys_mem_copy_;  //!< System memory copy
  uint32_t size_;         //!< Constant buffer size
};

//! Staging buffer
class XferBuffer : public amd::EmbeddedObject {
 public:
  //! Constructor for the ConstBuffer class
  XferBuffer(const Device& device,  //!< Active GPU device
             ManagedBuffer& mbuf,   //!< Managed buffer
             uint32_t size          //!< Maximum size of the transfer buffer
  );

  //! Destructor for the ConstBuffer class
  ~XferBuffer() {}

  /*! \brief Acquires free memory from the managed buffer
   *
   *  \return GPU memory object associated with free memory
   */
  Memory& Acquire(uint32_t size  //!< data size for transfers
  );

  //! Releases memory object used in the staging transfer
  void Release(Memory& mem  //!< Memory object for release
  ) {
    buffer_view_.updateView(nullptr, 0, 0);
  }

  size_t MaxSize() const { return static_cast<size_t>(size_); }

 private:
  //! Disable copy constructor
  XferBuffer(const XferBuffer&) = delete;

  //! Disable operator=
  XferBuffer& operator=(const XferBuffer&) = delete;

  Memory buffer_view_;   //!< Buffer view returned in the acquire
  ManagedBuffer& mbuf_;  //!< Managed buffer on GPU
  uint32_t size_;        //!< Mx staging buffer size
};
/*@}*/  // namespace amd::pal
}  // namespace amd::pal
