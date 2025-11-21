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

#include "top.hpp"
#include "platform/command.hpp"
#include "device/pal/paldefs.hpp"
#include "device/device.hpp"
#include "device/blit.hpp"

/*! \addtogroup PAL Blit Implementation
 *  @{
 */

//! PAL Blit Manager Implementation
namespace amd::pal {

class Device;
class Kernel;
class Memory;
class VirtualGPU;

//! DMA Blit Manager
class DmaBlitManager : public device::HostBlitManager {
 public:
  //! Constructor
  DmaBlitManager(VirtualGPU& gpu,       //!< Virtual GPU to be used for blits
                 Setup setup = Setup()  //!< Specifies HW accelerated blits
  );

  //! Destructor
  virtual ~DmaBlitManager() {}

  //! Creates DmaBlitManager object
  virtual bool create(amd::Device& device) { return true; }

  //! Copies a buffer object to system memory
  virtual bool readBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to system memory
  virtual bool readBufferRect(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destinaiton host memory
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to system memory
  virtual bool readImage(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBuffer(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBufferRect(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to an image object
  virtual bool writeImage(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to another buffer object
  virtual bool copyBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to another buffer object
  virtual bool copyBufferRect(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::BufferRect& srcRect,                       //!< Source rectangle
      const amd::BufferRect& dstRect,                       //!< Destination rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to a buffer object
  virtual bool copyImageToBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to an image object
  virtual bool copyBufferToImage(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to another image object
  virtual bool copyImage(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Stream memory write operation - Write a 'value' at 'memory'.
  virtual bool streamOpsWrite(device::Memory& memory,  //!< Memory to write the 'value'
                              uint64_t value, size_t offset, size_t sizeBytes) const {
    assert(!"Unimplemented");
    return false;
  }

  //! Stream memory ops- Waits for a 'value' at 'memory' and wait is released based on compare op.
  virtual bool streamOpsWait(device::Memory& memory,  //!< Memory to compare the 'value' against
                             uint64_t value, size_t offset, size_t sizeBytes, uint64_t flags,
                             uint64_t mask) const {
    assert(!"Unimplemented");
    return false;
  }

  virtual bool initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                        uint heap_size, uint number_of_initial_blocks) const {
    assert(!"Unimplemented");
    return false;
  }

 protected:
  static constexpr uint MaxPinnedBuffers = 4;

  //! Synchronizes the blit operations if necessary
  inline void synchronize() const;

  //! Returns the virtual GPU object
  VirtualGPU& gpu() const { return static_cast<VirtualGPU&>(vDev_); }

  //! Returns the GPU device object
  const Device& dev() const { return static_cast<const Device&>(dev_); };

  inline Memory& gpuMem(device::Memory& mem) const;

  //! Pins host memory for GPU access
  amd::Memory* pinHostMemory(const void* hostMem,  //!< Host memory pointer
                             size_t pinSize,       //!< Host memory size
                             size_t& partial       //!< Extra offset for memory alignment
  ) const;

  const size_t MinSizeForPinnedTransfer;
  bool completeOperation_;  //!< DMA blit manager must complete operation
  amd::Context* context_;   //!< A dummy context

 private:
  //! Disable copy constructor
  DmaBlitManager(const DmaBlitManager&);

  //! Disable operator=
  DmaBlitManager& operator=(const DmaBlitManager&);

  //! Reads video memory, using a staged buffer
  bool readMemoryStaged(Memory& srcMemory,  //!< Source memory object
                        void* dstHost,      //!< Destination host memory
                        Memory** xferBuf,   //!< Staged buffer for read
                        size_t origin,      //!< Original offset in the source memory
                        size_t& offset,     //!< Offset for the current copy pointer
                        size_t& totalSize,  //!< Total size for copy region
                        size_t xferSize     //!< Transfer size
  ) const;

  //! Write into video memory, using a staged buffer
  bool writeMemoryStaged(const void* srcHost,  //!< Source host memory
                         Memory& dstMemory,    //!< Destination memory object
                         Memory& xferBuf,      //!< Staged buffer for write
                         size_t origin,        //!< Original offset in the destination memory
                         size_t& offset,       //!< Offset for the current copy pointer
                         size_t& totalSize,    //!< Total size for the copy region
                         size_t xferSize       //!< Transfer size
  ) const;
};

//! Kernel Blit Manager
class KernelBlitManager : public DmaBlitManager {
 public:
  enum {
    BlitCopyImage = 0,
    BlitCopyImage1DA,
    BlitCopyImageToBuffer,
    BlitCopyBufferToImage,
    BlitCopyBufferRect,
    BlitCopyBufferRectAligned,
    BlitCopyBuffer,
    BlitCopyBufferAligned,
    FillBufferAligned,
    FillImage,
    Scheduler,
    GwsInit,
    StreamOpsWrite,
    StreamOpsWait,
    InitHeap,
    BlitTotal,
  };

  //! Constructor
  KernelBlitManager(VirtualGPU& gpu,       //!< Virtual GPU to be used for blits
                    Setup setup = Setup()  //!< Specifies HW accelerated blits
  );

  //! Destructor
  virtual ~KernelBlitManager();

  //! Creates DmaBlitManager object
  virtual bool create(amd::Device& device);

  //! Copies a buffer object to another buffer object
  virtual bool copyBufferRect(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::BufferRect& srcRectIn,                     //!< Source rectangle
      const amd::BufferRect& dstRectIn,                     //!< Destination rectangle
      const amd::Coord3D& sizeIn,                           //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to system memory
  virtual bool readBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to system memory
  virtual bool readBufferRect(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destinaiton host memory
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBuffer(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBufferRect(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to an image object
  virtual bool copyBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies a buffer object to an image object
  virtual bool copyBufferToImage(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to a buffer object
  virtual bool copyImageToBuffer(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to another image object
  virtual bool copyImage(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to system memory
  virtual bool readImage(
      device::Memory& srcMemory,                            //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies system memory to an image object
  virtual bool writeImage(
      const void* srcHost,                                  //!< Source host memory
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer(device::Memory& memory,       //!< Memory object to fill with pattern
                          const void* pattern,          //!< Pattern data
                          size_t patternSize,           //!< Pattern size
                          const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                          const amd::Coord3D& origin,   //!< Destination origin
                          const amd::Coord3D& size,     //!< Size of the fill region
                          bool entire = false,          //!< Entire buffer will be updated
                          bool forceBlit = false        //!< Force GPU Blit for fill
  ) const;

  //! Fills an image memory with a pattern data
  virtual bool fillImage(device::Memory& dstMemory,   //!< Memory object to fill with pattern
                         const void* pattern,         //!< Pattern data
                         const amd::Coord3D& origin,  //!< Destination origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         bool entire = false          //!< Entire buffer will be updated
  ) const;

  //! Runs a GPU scheduler for device enqueue
  bool runScheduler(device::Memory& vqueue,  //!< Memory object for virtual queue
                    device::Memory& params,  //!< Extra arguments for the scheduler
                    uint paramIdx,           //!< Parameter index
                    uint threads             //!< Number of scheduling threads
  ) const;

  //! Writes CPU raw data into GPU memory
  void writeRawData(device::Memory& memory,  //!< Memory object for data udpate
                    size_t size,             //!< Size of raw data
                    const void* data         //!< Raw data pointer
  ) const;

  //! Runs a blit kernel for GWS init
  bool RunGwsInit(uint32_t value  //!< Initial value for GWS resource
  ) const;

  virtual amd::Monitor* lockXfer() const { return &lockXferOps_; }

  //! Stream memory write operation - Write a 'value' at 'memory'.
  virtual bool streamOpsWrite(device::Memory& memory,  //!< Memory to write the 'value'
                              uint64_t value, size_t offset, size_t sizeBytes) const;

  //! Stream memory ops- Waits for a 'value' at 'memory' and wait is released based on compare op.
  virtual bool streamOpsWait(
      device::Memory& memory,  //!< Memory contents to compare the 'value' against
      uint64_t value, size_t offset, size_t sizeBytes, uint64_t flags, uint64_t mask) const;

  virtual bool initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                        uint heap_size, uint number_of_initial_blocks) const;

  //! Batch memory ops- Submits batch of streamWaits and streamWrite operations.
  virtual bool batchMemOps(const void* paramArray, size_t paramSize, uint32_t count) const {
    assert(!"Unimplemented");
    return false;
  }

 private:
  static constexpr size_t MaxXferBuffers = 2;
  static constexpr uint TransferSplitSize = 3;

  //! Copies a buffer object to an image object
  bool copyBufferToImageKernel(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Copies an image object to a buffer object
  bool copyImageToBufferKernel(
      device::Memory& srcMemory,                            //!< Source memory object
      device::Memory& dstMemory,                            //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const;

  //! Creates a program for all blit operations
  bool createProgram(Device& device  //!< Device object
  );

  inline void setArgument(amd::Kernel* kernel, size_t index, size_t size, const void* value,
                          size_t offset = 0, const device::Memory* dev_mem = nullptr,
                          bool writeVAImmediate = false) const;


  //! Creates a view memory object
  Memory* createView(const Memory& parent,         //!< Parent memory object
                     const cl_image_format format  //!< The new format for a view
  ) const;

  //! Disable copy constructor
  KernelBlitManager(const KernelBlitManager&);

  //! Disable operator=
  KernelBlitManager& operator=(const KernelBlitManager&);

  amd::Program* program_;                     //!< GPU program object
  amd::Kernel* kernels_[BlitTotal];           //!< GPU kernels for blit
  amd::Memory* xferBuffers_[MaxXferBuffers];  //!< Transfer buffers for images
  size_t xferBufferSize_;                     //!< Transfer buffer size
  mutable amd::Monitor lockXferOps_;          //!< Lock transfer operation
};

static const char* BlitName[KernelBlitManager::BlitTotal] = {
    "__amd_rocclr_copyImage",         "__amd_rocclr_copyImage1DA",
    "__amd_rocclr_copyImageToBuffer", "__amd_rocclr_copyBufferToImage",
    "__amd_rocclr_copyBufferRect",    "__amd_rocclr_copyBufferRectAligned",
    "__amd_rocclr_copyBuffer",        "__amd_rocclr_copyBufferAligned",
    "__amd_rocclr_fillBufferAligned", "__amd_rocclr_fillImage",
    "__amd_rocclr_scheduler",         "__amd_rocclr_gwsInit",
    "__amd_rocclr_streamOpsWrite",    "__amd_rocclr_streamOpsWait",
    "__amd_rocclr_initHeap"};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
