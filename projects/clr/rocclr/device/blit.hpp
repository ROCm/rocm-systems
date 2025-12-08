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

#ifndef BLIT_HPP_
#define BLIT_HPP_

#include "top.hpp"
#include "platform/command.hpp"
#include "device/device.hpp"

/*! \addtogroup GPU Blit Implementation
 *  @{
 */

//! GPU Blit Manager Implementation
namespace amd::device {

//! Blit Manager Abstraction class
class BlitManager : public amd::HeapObject {
 public:
  //! HW accelerated setup
  union Setup {
    struct {
      uint disableReadBuffer_ : 1;
      uint disableReadBufferRect_ : 1;
      uint disableReadImage_ : 1;
      uint disableWriteBuffer_ : 1;
      uint disableWriteBufferRect_ : 1;
      uint disableWriteImage_ : 1;
      uint disableCopyBuffer_ : 1;
      uint disableCopyBufferRect_ : 1;
      uint disableCopyImageToBuffer_ : 1;
      uint disableCopyBufferToImage_ : 1;
      uint disableCopyImage_ : 1;
      uint disableFillBuffer_ : 1;
      uint disableFillImage_ : 1;
      uint disableCopyBufferToImageOpt_ : 1;
      uint disableHwlCopyBuffer_ : 1;
    };
    uint32_t value_;
    Setup() : value_(0) {}
    void disableAll() { value_ = 0xffffffff; }
  };

 public:
  //! Constructor
  BlitManager(Setup setup = Setup()  //!< Specifies HW accelerated blits
              )
      : setup_(setup), syncOperation_(false) {}

  //! Destructor
  virtual ~BlitManager() {}

  //! Creates HostBlitManager object
  virtual bool create(amd::Device& device) { return true; }

  //! Copies a buffer object to system memory
  virtual bool readBuffer(
      Memory& srcMemory,                                    //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies a buffer object to system memory
  virtual bool readBufferRect(
      Memory& srcMemory,                                    //!< Source memory object
      void* dstHost,                                        //!< Destinaiton host memory
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies an image object to system memory
  virtual bool readImage(
      Memory& srcMemory,                                    //!< Source memory object
      void* dstHost,                                        //!< Destination host memory
      const amd::Coord3D& origin,                           //!< Source origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies system memory to a buffer object
  virtual bool writeBuffer(
      const void* srcHost,                                  //!< Source host memory
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies system memory to a buffer object
  virtual bool writeBufferRect(
      const void* srcHost,                                  //!< Source host memory
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::BufferRect& hostRect,                      //!< Destination rectangle
      const amd::BufferRect& bufRect,                       //!< Source rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies system memory to an image object
  virtual bool writeImage(
      const void* srcHost,                                  //!< Source host memory
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& origin,                           //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      size_t rowPitch,                                      //!< Row pitch for host memory
      size_t slicePitch,                                    //!< Slice pitch for host memory
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies a buffer object to another buffer object
  virtual bool copyBuffer(
      Memory& srcMemory,                                    //!< Source memory object
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies a buffer object to another buffer object
  virtual bool copyBufferRect(
      Memory& srcMemory,                                    //!< Source memory object
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::BufferRect& srcRect,                       //!< Source rectangle
      const amd::BufferRect& dstRect,                       //!< Destination rectangle
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies an image object to a buffer object
  virtual bool copyImageToBuffer(
      Memory& srcMemory,                                    //!< Source memory object
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies a buffer object to an image object
  virtual bool copyBufferToImage(
      Memory& srcMemory,                                    //!< Source memory object
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      size_t rowPitch = 0,                                  //!< Pitch for buffer
      size_t slicePitch = 0,                                //!< Slice for buffer
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Copies an image object to another image object
  virtual bool copyImage(
      Memory& srcMemory,                                    //!< Source memory object
      Memory& dstMemory,                                    //!< Destination memory object
      const amd::Coord3D& srcOrigin,                        //!< Source origin
      const amd::Coord3D& dstOrigin,                        //!< Destination origin
      const amd::Coord3D& size,                             //!< Size of the copy region
      bool entire = false,                                  //!< Entire buffer will be updated
      amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  ) const = 0;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer(Memory& memory,               //!< Memory object to fill with pattern
                          const void* pattern,          //!< Pattern data
                          size_t patternSize,           //!< Pattern size
                          const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                          const amd::Coord3D& origin,   //!< Destination origin
                          const amd::Coord3D& size,     //!< Size of the fill region
                          bool entire = false,          //!< Entire buffer will be updated
                          bool forceBlit = false        //!< Force GPU Blit for fill
  ) const = 0;

  //! Fills an image memory with a pattern data
  virtual bool fillImage(Memory& dstMemory,           //!< Memory object to fill with pattern
                         const void* pattern,         //!< Pattern data
                         const amd::Coord3D& origin,  //!< Destination origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         bool entire = false          //!< Entire buffer will be updated
  ) const = 0;


  //! Stream memory write operation - Write a 'value' at 'memory'.
  virtual bool streamOpsWrite(device::Memory& memory,  //!< Memory to write the 'value'
                              uint64_t value, size_t offset, size_t sizeBytes) const = 0;


  //! Stream memory ops- Waits for a 'value' at 'memory' and wait is released based on compare op.
  virtual bool streamOpsWait(
      device::Memory& memory,  //!< Memory contents to compare the 'value' against
      uint64_t value, size_t offset, size_t sizeBytes, uint64_t flags, uint64_t mask) const = 0;

  //! Stream batch memory operation
  virtual bool batchMemOps(const void* paramArray, size_t paramSize, uint32_t count) const = 0;

  //! Enables synchronization on blit operations
  void enableSynchronization() { syncOperation_ = true; }

  //! Returns Xfer queue lock
  virtual amd::Monitor* lockXfer() const { return nullptr; }

  virtual bool initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                        uint heap_size, uint number_of_initial_blocks) const = 0;

 protected:
  const Setup setup_;   //!< HW accelerated blit requested
  bool syncOperation_;  //!< Blit operations are synchronized

 private:
  //! Disable copy constructor
  BlitManager(const BlitManager&);

  //! Disable operator=
  BlitManager& operator=(const BlitManager&);
};

//! Host Blit Manager
class HostBlitManager : public device::BlitManager {
 public:
  //! Constructor
  HostBlitManager(VirtualDevice& vdev,   //!< Virtual GPU to be used for blits
                  Setup setup = Setup()  //!< Specifies HW accelerated blits
  );

  //! Destructor
  virtual ~HostBlitManager() {}

  //! Creates HostBlitManager object
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

  uint32_t sRGBmap(float fc) const;

 protected:
  VirtualDevice& vDev_;     //!< Virtual device object
  const amd::Device& dev_;  //!< Physical device

  // Packed Fill Buffer
  struct FillBufferInfo {
    static constexpr uint32_t kExtendedSize = 2 * sizeof(uint64_t);

    static void PackInfo(const device::Memory& memory, size_t fill_size, size_t fill_origin,
                         const void* pattern, size_t pattern_size,
                         std::vector<FillBufferInfo>& packed_info);

    FillBufferInfo(size_t fill_size) : fill_size_(fill_size), pattern_expanded_(false) {
      memset(&expanded_pattern_, 0, sizeof(expanded_pattern_));
    }

    void ExpandPattern(uint32_t pattern_size, const void* pattern);

    size_t fill_size_;                         //!< Fill size for this command
    uint8_t expanded_pattern_[kExtendedSize];  //!< Pattern for this command - 16 bytes
    bool pattern_expanded_;                    //!< Boolean to check if pattern is expanded
  };

 private:
  //! Disable copy constructor
  HostBlitManager(const HostBlitManager&);

  //! Disable operator=
  HostBlitManager& operator=(const HostBlitManager&);
};

/*@}*/  // namespace amd::device
}  // namespace amd::device

#endif /*BLIT_HPP_*/
