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
#include "platform/commandqueue.hpp"
#include "device/device.hpp"
#include "device/blit.hpp"
#include "device/rocm/rocdefs.hpp"
#include "device/rocm/rocsched.hpp"

/*! \addtogroup ROC Blit Implementation
 *  @{
 */

//! ROC Blit Manager Implementation
namespace amd::roc {

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
  struct BufferState {
    address buffer_;          //!< Staging Buffer or Pinned Host Mem Address
    amd::Memory* pinnedMem_;  //!< Pinned Memory
    size_t copySize_;         //!< last copy size
  };
  //! Synchronizes the blit operations if necessary
  inline void synchronize() const;

  //! Returns the virtual GPU object
  VirtualGPU& gpu() const { return static_cast<VirtualGPU&>(vDev_); }

  //! Returns the ROC device object
  const Device& dev() const { return static_cast<const Device&>(dev_); };

  inline Memory& gpuMem(device::Memory& mem) const;

  //! Pins host memory for GPU access
  amd::Memory* pinHostMemory(const void* hostMem,  //!< Host memory pointer
                             size_t pinSize,       //!< Host memory size
                             size_t& partial       //!< Extra offset for memory alignment
  ) const;

  //! Assits in transferring data from Host to Local or vice versa
  //! taking into account the Hsail profile supported by Hsa Agent
  bool hsaCopy(const Memory& srcMemory, const Memory& dstMemory, const amd::Coord3D& srcOrigin,
               const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
               amd::CopyMetadata& copyMetadata) const;

  inline bool rocrCopyBuffer(address dst, hsa_agent_t& dstAgent, const_address src,
                             hsa_agent_t& srcAgent, size_t size,
                             amd::CopyMetadata& copyMetadata) const;

  // Get Pinned Host Memory or Staging Buffer
  void getBuffer(const_address hostMem,  //!< Host Mem Address
                 size_t size,            //!< Transfer Size
                 bool enablePin,         //!< True when Pinning is enabled
                 bool first_tx,          //!< True for first copy
                 BufferState& buffer     //!< State of Buffer
  ) const;

  // Release Pinned host memory
  void releaseBuffer(BufferState& buff  //!< True if last copy used Pinned resource
  ) const;

  bool hsaCopyStagedOrPinned(const_address hostSrc,            //!< Src buffer address
                             address hostDst,                  //!< Dst Buffer address
                             size_t size,                      //!< Size in bytes
                             bool hostToDev,                   //!< True for H2D copy
                             amd::CopyMetadata& copyMetadata,  //!< copy MetaData
                             bool enPinning = false            //!< True if pinning required
  ) const;

  const size_t PinXferSize;           //!< Copy size for Pinned Copy
  const size_t MinSizeForPinnedXfer;  //!< Mininum copy size for Pinned Copy
  const size_t StagingXferSize;       //!< Copy size for Staging Buffer Copy

  bool completeOperation_;        //!< DMA blit manager must complete operation
  amd::Context* context_;         //!< A dummy context
  uint32_t sdmaEngineReadMask_;   //!< SDMA Engine Read Mask
  uint32_t sdmaEngineWriteMask_;  //!< SDMA Engine Write Mask

 private:
  //! Disable copy constructor
  DmaBlitManager(const DmaBlitManager&);

  //! Disable operator=
  DmaBlitManager& operator=(const DmaBlitManager&);
};

//! Kernel Blit Manager
class KernelBlitManager : public DmaBlitManager {
 public:
  enum {
    FillBufferAligned = 0,
    FillBufferAligned2D,
    BlitCopyBuffer,
    BlitCopyBufferAligned,
    BlitCopyBufferRect,
    BlitCopyBufferRectAligned,
    StreamOpsWrite,
    StreamOpsWait,
    Scheduler,
    GwsInit,
    InitHeap,
    BatchMemOp,
    BlitLinearTotal,
    FillImage = BlitLinearTotal,
    BlitCopyImage,
    BlitCopyImage1DA,
    BlitCopyImageToBuffer,
    BlitCopyBufferToImage,
    BlitTotal
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

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer1D(device::Memory& memory,       //!< Memory object to fill with pattern
                            const void* pattern,          //!< Pattern data
                            size_t patternSize,           //!< Pattern size
                            const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                            const amd::Coord3D& origin,   //!< Destination origin
                            const amd::Coord3D& size,     //!< Size of the fill region
                            bool entire = false,          //!< Entire buffer will be updated
                            bool forceBlit = false        //!< Force GPU Blit for fill
  ) const;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer2D(device::Memory& memory,       //!< Memory object to fill with pattern
                            const void* pattern,          //!< Pattern data
                            size_t patternSize,           //!< Pattern size
                            const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                            const amd::Coord3D& origin,   //!< Destination origin
                            const amd::Coord3D& size,     //!< Size of the fill region
                            bool entire = false,          //!< Entire buffer will be updated
                            bool forceBlit = false        //!< Force GPU Blit for fill
  ) const;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer3D(device::Memory& memory,       //!< Memory object to fill with pattern
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

  bool runScheduler(uint64_t vqVM, hsa_queue_t* schedulerQueue, uint threads, uint64_t aql_wrap);

  //! Runs a blit kernel for GWS init
  bool RunGwsInit(uint32_t value  //!< Initial value for GWS resource
  ) const;

  //! Stream memory write operation - Write a 'value' at 'memory'.
  virtual bool streamOpsWrite(device::Memory& memory,  //!< Memory to write the 'value'
                              uint64_t value, size_t offset, size_t sizeBytes) const;

  //! Stream memory ops- Waits for a 'value' at 'memory' and wait is released based on compare op.
  virtual bool streamOpsWait(
      device::Memory& memory,  //!< Memory contents to compare the 'value' against
      uint64_t value, size_t offset, size_t sizeBytes, uint64_t flags, uint64_t mask) const;

  //! Batch memory ops- Submits batch of streamWaits and streamWrite operations.
  virtual bool batchMemOps(const void* paramArray, size_t paramSize, uint32_t count) const;

  virtual amd::Monitor* lockXfer() const { return &lockXferOps_; }

  virtual bool initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                        uint heap_size, uint number_of_initial_blocks) const;

 private:
  static constexpr size_t MaxXferBuffers = 2;
  static constexpr uint TransferSplitSize = 1;
  static constexpr uint MaxNumIssuedTransfers = 3;

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

  //! Creates a view memory object
  Memory* createView(const Memory& parent,    //!< Parent memory object
                     cl_image_format format,  //!< The new format for a view
                     cl_mem_flags flags       //!< Memory flags
  ) const;

  address captureArguments(const amd::Kernel* kernel) const;
  void releaseArguments(address args) const;

  inline void setArgument(amd::Kernel* kernel, size_t index, size_t size, const void* value,
                          size_t offset = 0, const device::Memory* dev_mem = nullptr,
                          bool writeVAImmediate = false) const;

  static constexpr uint32_t kCBSize = 0x100;
  static constexpr size_t kCBAlignment = 0x100;

  inline uint32_t NumBlitKernels() {
    return (dev().info().imageSupport_) ? BlitTotal : BlitLinearTotal;
  }

  //! Copies a buffer using the shader path
  bool shaderCopyBuffer(address dst, address src, const amd::Coord3D& dstOrigin,
                        const amd::Coord3D& srcOrigin, const amd::Coord3D& size, bool entire,
                        const uint32_t blitWg, amd::CopyMetadata copyMetadata,
                        bool attachSignal = false) const;

  //! Disable copy constructor
  KernelBlitManager(const KernelBlitManager&);

  //! Disable operator=
  KernelBlitManager& operator=(const KernelBlitManager&);

  amd::Program* program_;             //!< GPU program object
  amd::Kernel* kernels_[BlitTotal];   //!< GPU kernels for blit
  size_t xferBufferSize_;             //!< Transfer buffer size
  mutable amd::Monitor lockXferOps_;  //!< Lock transfer operation
};

static const char* BlitName[KernelBlitManager::BlitTotal] = {
    "__amd_rocclr_fillBufferAligned", "__amd_rocclr_fillBufferAligned2D",
    "__amd_rocclr_copyBuffer",        "__amd_rocclr_copyBufferAligned",
    "__amd_rocclr_copyBufferRect",    "__amd_rocclr_copyBufferRectAligned",
    "__amd_rocclr_streamOpsWrite",    "__amd_rocclr_streamOpsWait",
    "__amd_rocclr_scheduler",         "__amd_rocclr_gwsInit",
    "__amd_rocclr_initHeap",          "__amd_rocclr_batchMemOp",
    "__amd_rocclr_fillImage",         "__amd_rocclr_copyImage",
    "__amd_rocclr_copyImage1DA",      "__amd_rocclr_copyImageToBuffer",
    "__amd_rocclr_copyBufferToImage"};

inline void KernelBlitManager::setArgument(amd::Kernel* kernel, size_t index, size_t size,
                                           const void* value, size_t offset,
                                           const device::Memory* dev_mem,
                                           bool writeVAImmediate) const {
  const amd::KernelParameterDescriptor& desc = kernel->signature().at(index);

  void* param = kernel->parameters().values() + desc.offset_;
  assert((desc.type_ == T_POINTER || value != NULL ||
          (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL)) &&
         "not a valid local mem arg");

  uint32_t uint32_value = 0;
  uint64_t uint64_value = 0;

  if (desc.type_ == T_POINTER && (desc.addressQualifier_ != CL_KERNEL_ARG_ADDRESS_LOCAL)) {
    if ((value == NULL) || (static_cast<const cl_mem*>(value) == NULL)) {
      LP64_SWITCH(uint32_value, uint64_value) = 0;
      reinterpret_cast<Memory**>(kernel->parameters().values() +
                                 kernel->parameters().memoryObjOffset())[desc.info_.arrayIndex_] =
          nullptr;
    } else {
      if (!writeVAImmediate) {
        // convert cl_mem to amd::Memory*, return false if invalid.
        amd::Memory* mem = as_amd(*static_cast<const cl_mem*>(value));

        reinterpret_cast<amd::Memory**>(
            kernel->parameters().values() +
            kernel->parameters().memoryObjOffset())[desc.info_.arrayIndex_] = mem;
        if (dev_mem == nullptr) {
          LP64_SWITCH(uint32_value, uint64_value) =
              static_cast<uintptr_t>(mem->getDeviceMemory(dev())->virtualAddress()) + offset;
        } else {
          LP64_SWITCH(uint32_value, uint64_value) =
              static_cast<uintptr_t>(dev_mem->virtualAddress()) + offset;
        }
      } else {
        reinterpret_cast<amd::Memory**>(
            kernel->parameters().values() +
            kernel->parameters().memoryObjOffset())[desc.info_.arrayIndex_] = nullptr;
        uintptr_t addr = reinterpret_cast<uintptr_t>(value);
        LP64_SWITCH(uint32_value, uint64_value) = addr + offset;
      }
    }
  } else if (desc.type_ == T_SAMPLER) {
    assert(false && "No sampler support in blit manager! Use internal samplers!");
  } else {
    switch (desc.size_) {
      case 4:
        if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
          uint32_value = size;
        } else {
          uint32_value = *static_cast<const uint32_t*>(value);
        }
        break;
      case 8:
        if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
          uint64_value = size;
        } else {
          uint64_value = *static_cast<const uint64_t*>(value);
        }
        break;
      default:
        break;
    }
  }
  switch (desc.size_) {
    case sizeof(uint32_t):
      *static_cast<uint32_t*>(param) = uint32_value;
      break;
    case sizeof(uint64_t):
      *static_cast<uint64_t*>(param) = uint64_value;
      break;
    default:
      ::memcpy(param, value, size);
      break;
  }
}


/*@}*/  // namespace amd::roc
}  // namespace amd::roc
