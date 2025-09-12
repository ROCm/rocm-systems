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

#include "platform/commandqueue.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/palvirtual.hpp"
#include "utils/debug.hpp"
#include <algorithm>

namespace amd::pal {

DmaBlitManager::DmaBlitManager(VirtualGPU& gpu, Setup setup)
    : HostBlitManager(gpu, setup),
      MinSizeForPinnedTransfer(dev().settings().pinnedMinXferSize_),
      completeOperation_(false),
      context_(NULL) {}

inline void DmaBlitManager::synchronize() const {
  if (syncOperation_) {
    gpu().waitAllEngines();
  }
}

inline Memory& DmaBlitManager::gpuMem(device::Memory& mem) const {
  return static_cast<Memory&>(mem);
}

bool DmaBlitManager::readMemoryStaged(Memory& srcMemory, void* dstHost, Memory** xferBuf,
                                      size_t origin, size_t& offset, size_t& totalSize,
                                      size_t xferSize) const {
  amd::Coord3D dst(0, 0, 0);
  size_t tmpSize;
  uint idxWrite = 0;
  uint idxRead = 0;
  size_t chunkSize;
  static const bool CopyRect = false;
  // Flush DMA for ASYNC copy
  static const bool FlushDMA = true;

  if (dev().xferRead().bufSize() < 128 * Ki) {
    chunkSize = dev().xferRead().bufSize();
  } else {
    chunkSize = std::min(amd::alignUp(xferSize / 4, 256), dev().xferRead().bufSize());
    chunkSize = std::max(chunkSize, 128 * Ki);
  }

  // Find the partial transfer size
  tmpSize = std::min(chunkSize, xferSize);

  amd::Coord3D srcLast(origin + offset, 0, 0);
  amd::Coord3D copySizeLast(tmpSize, 0, 0);

  // Copy data into the temporary surface
  if (!srcMemory.partialMemCopyTo(gpu(), srcLast, dst, copySizeLast, *xferBuf[idxWrite], CopyRect,
                                  FlushDMA)) {
    return false;
  }

  totalSize -= tmpSize;
  xferSize -= tmpSize;
  offset += tmpSize;

  while (xferSize != 0) {
    // Find the partial transfer size
    tmpSize = std::min(chunkSize, xferSize);

    amd::Coord3D src(origin + offset, 0, 0);
    amd::Coord3D copySize(tmpSize, 0, 0);

    idxWrite = (idxWrite + 1) % 2;
    // Copy data into the temporary surface
    if (!srcMemory.partialMemCopyTo(gpu(), src, dst, copySize, *xferBuf[idxWrite], CopyRect,
                                    FlushDMA)) {
      return false;
    }

    // Read previous buffer
    if (!xferBuf[idxRead]->hostRead(&gpu(),
                                    reinterpret_cast<char*>(dstHost) + offset - copySizeLast[0],
                                    dst, copySizeLast)) {
      return false;
    }
    idxRead = (idxRead + 1) % 2;
    copySizeLast = copySize;

    totalSize -= tmpSize;
    xferSize -= tmpSize;
    offset += tmpSize;
  }

  // Last read
  if (!xferBuf[idxRead]->hostRead(
          &gpu(), reinterpret_cast<char*>(dstHost) + offset - copySizeLast[0], dst, copySizeLast)) {
    return false;
  }
  return true;
}

bool DmaBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (setup_.disableReadBuffer_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable())) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
  } else {
    size_t srcSize = size[0];
    size_t offset = 0;
    size_t pinSize = dev().settings().pinnedXferSize_;
    pinSize = std::min(pinSize, srcSize);

    // Check if a pinned transfer can be executed
    if (pinSize && (srcSize > MinSizeForPinnedTransfer)) {
      // Allign offset to 4K boundary (Vista/Win7 limitation)
      char* tmpHost = const_cast<char*>(
          amd::alignDown(reinterpret_cast<const char*>(dstHost), PinnedMemoryAlignment));

      // Find the partial size for unaligned copy
      size_t partial = reinterpret_cast<const char*>(dstHost) - tmpHost;

      amd::Memory* pinned = NULL;
      bool first = true;
      size_t tmpSize;
      size_t pinAllocSize;

      // Copy memory, using pinning
      while (srcSize > 0) {
        // If it's the first iterarion, then readjust the copy size
        // to include alignment
        if (first) {
          pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);
          tmpSize = std::min(pinAllocSize - partial, srcSize);
          first = false;
        } else {
          tmpSize = std::min(pinSize, srcSize);
          pinAllocSize = amd::alignUp(tmpSize, PinnedMemoryAlignment);
          partial = 0;
        }
        amd::Coord3D dst(partial, 0, 0);
        amd::Coord3D srcPin(origin[0] + offset, 0, 0);
        amd::Coord3D copySizePin(tmpSize, 0, 0);
        size_t partial2;

        // Allocate a GPU resource for pinning
        pinned = pinHostMemory(tmpHost, pinAllocSize, partial2);

        if (pinned != NULL) {
          // Get device memory for this virtual device
          Memory* dstMemory = dev().getGpuMemory(pinned);

          if (!gpuMem(srcMemory).partialMemCopyTo(gpu(), srcPin, dst, copySizePin, *dstMemory)) {
            LogWarning("DmaBlitManager::readBuffer failed a pinned copy!");
            gpu().addPinnedMem(pinned);
            break;
          }
          gpu().addPinnedMem(pinned);
        } else {
          LogWarning("DmaBlitManager::readBuffer failed to pin a resource!");
          break;
        }
        srcSize -= tmpSize;
        offset += tmpSize;
        tmpHost = reinterpret_cast<char*>(tmpHost) + tmpSize + partial;
      }
    }

    if (0 != srcSize) {
      Memory& xferBuf0 = dev().xferRead().acquire();
      Memory& xferBuf1 = dev().xferRead().acquire();
      Memory* xferBuf[2] = {&xferBuf0, &xferBuf1};

      // Read memory using a staged resource
      if (!readMemoryStaged(gpuMem(srcMemory), dstHost, xferBuf, origin[0], offset, srcSize,
                            srcSize)) {
        LogError("DmaBlitManager::readBuffer failed!");
        return false;
      }

      dev().xferRead().release(gpu(), xferBuf1);
      dev().xferRead().release(gpu(), xferBuf0);
    }
  }

  return true;
}

bool DmaBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                    const amd::BufferRect& bufRect, const amd::BufferRect& hostRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable())) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                           copyMetadata);
  } else {
    Memory& xferBuf = dev().xferRead().acquire();

    amd::Coord3D dst(0, 0, 0);
    size_t bufOffset;
    size_t hostOffset;
    size_t srcSize;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        srcSize = size[0];
        bufOffset = bufRect.offset(0, y, z);
        hostOffset = hostRect.offset(0, y, z);

        while (srcSize != 0) {
          // Find the partial transfer size
          size_t tmpSize = std::min(dev().xferRead().bufSize(), srcSize);

          amd::Coord3D src(bufOffset, 0, 0);
          amd::Coord3D copySize(tmpSize, 0, 0);

          // Copy data into the temporary surface
          if (!gpuMem(srcMemory).partialMemCopyTo(gpu(), src, dst, copySize, xferBuf, true)) {
            LogError("DmaBlitManager::readBufferRect failed!");
            return false;
          }

          if (!xferBuf.hostRead(&gpu(), reinterpret_cast<char*>(dstHost) + hostOffset, dst,
                                copySize)) {
            LogError("DmaBlitManager::readBufferRect failed!");
            return false;
          }

          srcSize -= tmpSize;
          bufOffset += tmpSize;
          hostOffset += tmpSize;
        }
      }
    }
    dev().xferRead().release(gpu(), xferBuf);
  }

  return true;
}

bool DmaBlitManager::readImage(device::Memory& srcMemory, void* dstHost, const amd::Coord3D& origin,
                               const amd::Coord3D& size, size_t rowPitch, size_t slicePitch,
                               bool entire, amd::CopyMetadata copyMetadata) const {
  gpu().releaseGpuMemoryFence();

  if (setup_.disableReadImage_) {
    return HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                      entire, copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                      entire, copyMetadata);
  }

  return true;
}

bool DmaBlitManager::writeMemoryStaged(const void* srcHost, Memory& dstMemory, Memory& xferBuf,
                                       size_t origin, size_t& offset, size_t& totalSize,
                                       size_t xferSize) const {
  size_t chunkSize;
  static const bool CopyRect = false;
  // Flush DMA for ASYNC copy
  // @todo Blocking write requires a flush to start earlier,
  // but currently VDI doesn't provide that info
  bool flushDMA = false;

  if (gpu().xferWrite().MaxSize() < 128 * Ki) {
    chunkSize = gpu().xferWrite().MaxSize();
  } else {
    chunkSize = std::min(amd::alignUp(xferSize / 4, 256), gpu().xferWrite().MaxSize());
    chunkSize = std::max(chunkSize, 64 * Ki);
    flushDMA = (xferSize > chunkSize);
  }
  size_t srcOffset = 0;
  uint32_t flags = Resource::NoWait;
  while (xferSize != 0) {
    // Find the partial transfer size
    size_t tmpSize = std::min(chunkSize, xferSize);
    amd::Coord3D src(srcOffset, 0, 0);
    amd::Coord3D dst(origin + offset, 0, 0);
    amd::Coord3D copySize(tmpSize, 0, 0);

    // Copy data into the temporary buffer, using CPU
    if (!xferBuf.hostWrite(&gpu(), reinterpret_cast<const char*>(srcHost) + offset, src, copySize,
                           flags)) {
      return false;
    }

    // Copy data into the original destination memory
    if (!xferBuf.partialMemCopyTo(gpu(), src, dst, copySize, dstMemory, CopyRect, flushDMA)) {
      return false;
    }

    totalSize -= tmpSize;
    offset += tmpSize;
    xferSize -= tmpSize;
    srcOffset += tmpSize;
    if ((srcOffset + tmpSize) > gpu().xferWrite().MaxSize()) {
      srcOffset = 0;
      flags = 0;
    } else {
      flags = Resource::NoWait;
    }
  }
  return true;
}

bool DmaBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableWriteBuffer_ ||
      (gpuMem(dstMemory).isHostMemDirectAccess() &&
       (gpuMem(dstMemory).memoryType() != Resource::ExternalPhysical)) ||
      gpuMem(dstMemory).isPersistentDirectMap()) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
  } else {
    size_t dstSize = size[0];
    size_t offset = 0;
    size_t pinSize = dev().settings().pinnedXferSize_;
    pinSize = std::min(pinSize, dstSize);

    // Check if a pinned transfer can be executed
    if (pinSize && (dstSize > MinSizeForPinnedTransfer)) {
      // Allign offset to 4K boundary (Vista/Win7 limitation)
      char* tmpHost = const_cast<char*>(
          amd::alignDown(reinterpret_cast<const char*>(srcHost), PinnedMemoryAlignment));

      // Find the partial size for unaligned copy
      size_t partial = reinterpret_cast<const char*>(srcHost) - tmpHost;

      amd::Memory* pinned = NULL;
      bool first = true;
      size_t pinAllocSize;

      // Copy memory, using pinning
      while (dstSize > 0) {
        size_t tmpSize;
        // If it's the first iterarion, then readjust the copy size
        // to include alignment
        if (first) {
          pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);
          tmpSize = std::min(pinAllocSize - partial, dstSize);
          first = false;
        } else {
          tmpSize = std::min(pinSize, dstSize);
          pinAllocSize = amd::alignUp(tmpSize, PinnedMemoryAlignment);
          partial = 0;
        }
        amd::Coord3D src(partial, 0, 0);
        amd::Coord3D dstPin(origin[0] + offset, 0, 0);
        amd::Coord3D copySizePin(tmpSize, 0, 0);
        size_t partial2;

        // Allocate a GPU resource for pinning
        pinned = pinHostMemory(tmpHost, pinAllocSize, partial2);

        if (pinned != NULL) {
          // Get device memory for this virtual device
          Memory* srcMemory = dev().getGpuMemory(pinned);

          if (!srcMemory->partialMemCopyTo(gpu(), src, dstPin, copySizePin, gpuMem(dstMemory))) {
            LogWarning("DmaBlitManager::writeBuffer failed a pinned copy!");
            gpu().addPinnedMem(pinned);
            break;
          }
          gpu().addPinnedMem(pinned);
        } else {
          LogWarning("DmaBlitManager::writeBuffer failed to pin a resource!");
          break;
        }
        dstSize -= tmpSize;
        offset += tmpSize;
        tmpHost = reinterpret_cast<char*>(tmpHost) + tmpSize + partial;
      }
    }

    while (dstSize > 0) {
      auto xfer_size = std::min(dstSize, gpu().xferWrite().MaxSize());
      Memory& xferBuf = gpu().xferWrite().Acquire(xfer_size);
      // Write memory using a staged resource
      if (!writeMemoryStaged(srcHost, gpuMem(dstMemory), xferBuf, origin[0], offset, dstSize,
                             xfer_size)) {
        LogError("DmaBlitManager::writeBuffer failed!");
        return false;
      }

      gpu().xferWrite().Release(xferBuf);
    }
  }

  return true;
}

bool DmaBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                     const amd::BufferRect& hostRect,
                                     const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                     bool entire, amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableWriteBufferRect_ ||
      (dstMemory.isHostMemDirectAccess() &&
       (gpuMem(dstMemory).memoryType() != Resource::ExternalPhysical)) ||
      gpuMem(dstMemory).isPersistentDirectMap()) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                            copyMetadata);
  } else {
    Memory& xferBuf = gpu().xferWrite().Acquire(std::min(gpu().xferWrite().MaxSize(), size[0]));

    amd::Coord3D src(0, 0, 0);
    size_t tmpSize = 0;
    size_t bufOffset;
    size_t hostOffset;
    size_t dstSize;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        dstSize = size[0];
        bufOffset = bufRect.offset(0, y, z);
        hostOffset = hostRect.offset(0, y, z);

        while (dstSize != 0) {
          // Find the partial transfer size
          tmpSize = std::min(gpu().xferWrite().MaxSize(), dstSize);

          amd::Coord3D dst(bufOffset, 0, 0);
          amd::Coord3D copySize(tmpSize, 0, 0);

          // Copy data into the temporary buffer, using CPU
          if (!xferBuf.hostWrite(&gpu(), reinterpret_cast<const char*>(srcHost) + hostOffset, src,
                                 copySize)) {
            LogError("DmaBlitManager::writeBufferRect failed!");
            return false;
          }

          // Copy data into the original destination memory
          if (!xferBuf.partialMemCopyTo(gpu(), src, dst, copySize, gpuMem(dstMemory))) {
            LogError("DmaBlitManager::writeBufferRect failed!");
            return false;
          }

          dstSize -= tmpSize;
          bufOffset += tmpSize;
          hostOffset += tmpSize;
        }
      }
    }
    gpu().xferWrite().Release(xferBuf);
  }

  return true;
}

bool DmaBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                size_t rowPitch, size_t slicePitch, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  gpu().releaseGpuMemoryFence();
  if (setup_.disableWriteImage_) {
    return HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                       entire, copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                       entire, copyMetadata);
  }

  return true;
}

bool DmaBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBuffer_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable() &&
       !dev().settings().apuSystem_ && gpuMem(dstMemory).isHostMemDirectAccess())) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size, false,
                                       copyMetadata);
  } else {
    return gpuMem(srcMemory).partialMemCopyTo(gpu(), srcOrigin, dstOrigin, size, gpuMem(dstMemory));
  }

  return true;
}

bool DmaBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                    const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBufferRect_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable() &&
       gpuMem(dstMemory).isHostMemDirectAccess())) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBufferRect(srcMemory, dstMemory, srcRect, dstRect, size, entire,
                                           copyMetadata);
  } else {
    size_t srcOffset;
    size_t dstOffset;

    uint bytesPerElement = 16;
    bool optimalElementSize = false;
    bool subWindowRectCopy = true;

    srcOffset = srcRect.offset(0, 0, 0);
    dstOffset = dstRect.offset(0, 0, 0);

    while (bytesPerElement >= 1) {
      if (((srcOffset % 4) == 0) && ((dstOffset % 4) == 0) && ((size[0] % bytesPerElement) == 0) &&
          ((srcRect.rowPitch_ % bytesPerElement) == 0) &&
          ((srcRect.slicePitch_ % bytesPerElement) == 0) &&
          ((dstRect.rowPitch_ % bytesPerElement) == 0) &&
          ((dstRect.slicePitch_ % bytesPerElement) == 0)) {
        optimalElementSize = true;
        break;
      }
      bytesPerElement = bytesPerElement >> 1;
    }

    // 19 bit limit in HW in SI and 16 bit limit in CI+
    // (we adjust the ElementSize to 4bytes but the packet still has 14bits)
    size_t pitchLimit = (0x3FFF * bytesPerElement) | 0xF;
    size_t sizeLimit = (0x3FFF * bytesPerElement) | 0xF;

    if (!optimalElementSize || (srcRect.rowPitch_ > pitchLimit) ||
        (dstRect.rowPitch_ > pitchLimit) || (size[0] > sizeLimit) ||  // See above
        (size[1] > 0x3fff) ||                                         // 14 bits limit in HW
        (size[2] > 0x7ff)) {                                          // 11 bits limit in HW
      // Restriction with rectLinearDRMDMA packet
      subWindowRectCopy = false;
    }

    if (subWindowRectCopy) {
      // Copy data with subwindow copy packet
      if (!gpuMem(srcMemory).partialMemCopyTo(
              gpu(), amd::Coord3D(srcOffset, srcRect.rowPitch_, srcRect.slicePitch_),
              amd::Coord3D(dstOffset, dstRect.rowPitch_, dstRect.slicePitch_), size,
              gpuMem(dstMemory), true, false, bytesPerElement)) {
        LogError("copyBufferRect failed!");
        return false;
      }
    } else {
      for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
          srcOffset = srcRect.offset(0, y, z);
          dstOffset = dstRect.offset(0, y, z);

          amd::Coord3D src(srcOffset, 0, 0);
          amd::Coord3D dst(dstOffset, 0, 0);
          amd::Coord3D copySize(size[0], 0, 0);

          // Copy data
          if (!gpuMem(srcMemory).partialMemCopyTo(gpu(), src, dst, copySize, gpuMem(dstMemory))) {
            LogError("copyBufferRect failed!");
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool DmaBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  bool result = false;
  if (setup_.disableCopyImageToBuffer_) {
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    // Use PAL path for a transfer
    result =
        gpuMem(srcMemory).partialMemCopyTo(gpu(), srcOrigin, dstOrigin, size, gpuMem(dstMemory));

    // Check if a HostBlit transfer is required
    if (completeOperation_ && !result) {
      gpu().releaseGpuMemoryFence();
      result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

bool DmaBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  bool result = false;
  if (setup_.disableCopyBufferToImage_) {
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    // Use PAL path for a transfer
    result =
        gpuMem(srcMemory).partialMemCopyTo(gpu(), srcOrigin, dstOrigin, size, gpuMem(dstMemory));

    // Check if a HostBlit transfer is required
    if (completeOperation_ && !result) {
      gpu().releaseGpuMemoryFence();
      result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

bool DmaBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                               const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                               const amd::Coord3D& size, bool entire,
                               amd::CopyMetadata copyMetadata) const {
  bool result = false;
  gpu().releaseGpuMemoryFence();


  if (setup_.disableCopyImage_) {
    return HostBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                      copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                      copyMetadata);
  }

  return result;
}

KernelBlitManager::KernelBlitManager(VirtualGPU& gpu, Setup setup)
    : DmaBlitManager(gpu, setup),
      program_(NULL),
      xferBufferSize_(0),
      lockXferOps_(true) /* Transfer Ops Lock */ {
  for (uint i = 0; i < BlitTotal; ++i) {
    kernels_[i] = NULL;
  }

  for (uint i = 0; i < MaxXferBuffers; ++i) {
    xferBuffers_[i] = NULL;
  }

  completeOperation_ = false;
}

KernelBlitManager::~KernelBlitManager() {
  for (uint i = 0; i < BlitTotal; ++i) {
    if (NULL != kernels_[i]) {
      kernels_[i]->release();
    }
  }
  if (NULL != program_) {
    program_->release();
  }

  if (NULL != context_) {
    // Release a dummy context
    context_->release();
  }

  for (uint i = 0; i < MaxXferBuffers; ++i) {
    if (NULL != xferBuffers_[i]) {
      xferBuffers_[i]->release();
    }
  }
}

bool KernelBlitManager::create(amd::Device& device) {
  if (!createProgram(static_cast<Device&>(device))) {
    return false;
  }
  return true;
}

bool KernelBlitManager::createProgram(Device& device) {
  if (device.blitProgram() == nullptr) {
    if (!device.createBlitProgram()) {
      return false;
    }
  }

  std::vector<amd::Device*> devices;
  devices.push_back(&device);

  // Save context and program for this device
  context_ = device.blitProgram()->context_;
  context_->retain();
  program_ = device.blitProgram()->program_;
  program_->retain();

  bool result = false;
  do {
    // Create kernel objects for all blits
    for (uint i = 0; i < BlitTotal; ++i) {
      const amd::Symbol* symbol = program_->findSymbol(BlitName[i]);
      if (symbol == NULL) {
        // Not all blit kernels are needed in some setup, so continue with the rest
        continue;
      }
      kernels_[i] = new amd::Kernel(*program_, *symbol, BlitName[i]);
      if (kernels_[i] == NULL) {
        break;
      }
      // Validate blit kernels for the scratch memory usage (pre SI)
      if (!device.validateKernel(*kernels_[i], &gpu())) {
        break;
      }
    }

    result = true;
  } while (!result);

  if (dev().settings().xferBufSize_ > 0) {
    xferBufferSize_ = dev().settings().xferBufSize_;
    for (uint i = 0; i < MaxXferBuffers; ++i) {
      // Create internal xfer buffers for image copy optimization
      xferBuffers_[i] = new (*context_) amd::Buffer(*context_, 0, xferBufferSize_);

      // Assign the xfer buffer to the current virtual GPU
      xferBuffers_[i]->setVirtualDevice(&gpu());

      if ((xferBuffers_[i] != NULL) && !xferBuffers_[i]->create(NULL)) {
        xferBuffers_[i]->release();
        xferBuffers_[i] = NULL;
        return false;
      } else if (xferBuffers_[i] == NULL) {
        return false;
      }

      //! @note Workaround for conformance allocation test.
      //! Force GPU mem alloc.
      //! Unaligned images require xfer optimization,
      //! but deferred memory allocation can cause
      //! virtual heap fragmentation for big allocations and
      //! then fail the following test with 32 bit ISA, because
      //! runtime runs out of 4GB space.
      dev().getGpuMemory(xferBuffers_[i]);
    }
  }

  return result;
}

// The following data structures will be used for the view creations.
// Some formats has to be converted before a kernel blit operation
struct FormatConvertion {
  uint32_t clOldType_;
  uint32_t clNewType_;
};

// The list of rejected data formats and corresponding conversion
static constexpr FormatConvertion RejectedData[] = {
    {CL_UNORM_INT8, CL_UNSIGNED_INT8},       {CL_UNORM_INT16, CL_UNSIGNED_INT16},
    {CL_SNORM_INT8, CL_UNSIGNED_INT8},       {CL_SNORM_INT16, CL_UNSIGNED_INT16},
    {CL_HALF_FLOAT, CL_UNSIGNED_INT16},      {CL_FLOAT, CL_UNSIGNED_INT32},
    {CL_SIGNED_INT8, CL_UNSIGNED_INT8},      {CL_SIGNED_INT16, CL_UNSIGNED_INT16},
    {CL_UNORM_INT_101010, CL_UNSIGNED_INT8}, {CL_SIGNED_INT32, CL_UNSIGNED_INT32}};

// The list of rejected channel's order and corresponding conversion
static constexpr FormatConvertion RejectedOrder[] = {
    {CL_A, CL_R},        {CL_RA, CL_RG},      {CL_LUMINANCE, CL_R}, {CL_INTENSITY, CL_R},
    {CL_RGB, CL_RGBA},   {CL_BGRA, CL_RGBA},  {CL_ARGB, CL_RGBA},   {CL_sRGB, CL_RGBA},
    {CL_sRGBx, CL_RGBA}, {CL_sRGBA, CL_RGBA}, {CL_sBGRA, CL_RGBA},  {CL_DEPTH, CL_R}};

const uint RejectedFormatDataTotal = sizeof(RejectedData) / sizeof(FormatConvertion);
const uint RejectedFormatChannelTotal = sizeof(RejectedOrder) / sizeof(FormatConvertion);

bool KernelBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  static const bool CopyRect = false;
  // Flush DMA for ASYNC copy
  static const bool FlushDMA = true;
  size_t imgRowPitch = size[0] * gpuMem(dstMemory).elementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyBufferToImage_) {
    result = DmaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                               entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (gpuMem(srcMemory).isHostMemDirectAccess() &&
           (((rowPitch == 0) && (slicePitch == 0)) ||
            ((rowPitch == imgRowPitch) && ((slicePitch == 0) || (slicePitch == imgSlicePitch))))) {
    // First attempt to do this all with DMA,
    // but there are restriciton with older hardware
    if (dev().settings().imageDMA_) {
      result = DmaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                 entire, rowPitch, slicePitch, copyMetadata);
      if (result) {
        synchronize();
        return result;
      }
    }

    if (!setup_.disableCopyBufferToImageOpt_) {
      // Find the overall copy size
      size_t copySize = size[0] * size[1] * size[2] * gpuMem(dstMemory).elementSize();

      // Check if double copy was requested
      if (xferBufferSize_ != 0) {
        amd::Coord3D src(srcOrigin);
        amd::Coord3D xferSrc(0, 0, 0);
        amd::Coord3D dst(dstOrigin);
        amd::Coord3D xferRect(size);
        // Find transfer size in pixels
        size_t xferSizePix = xferBufferSize_ / gpuMem(dstMemory).elementSize();
        bool transfer = true;

        // Find transfer rectangle
        if (xferRect[0] > xferSizePix) {
          // The algorithm can't break a line.
          // It requires multiple rectangles tracking
          transfer = false;
        } else {
          xferRect.c[1] = xferSizePix / xferRect[0];
        }
        // Check if we exceeded the original size boundary in Y
        if (xferRect[1] > size[1]) {
          xferRect.c[1] = size[1];
          xferRect.c[2] = xferSizePix / (xferRect[0] * xferRect[1]);
        } else {
          xferRect.c[2] = 1;
        }
        // Check if we exceeded the original size boundary in Z
        if (xferRect[2] > size[2]) {
          xferRect.c[2] = size[2];
        }
        // Make sure size in Y dimension is divided by the rectangle size
        if (size[2] > 1) {
          while ((size[1] % xferRect[1]) != 0) {
            xferRect.c[1]--;
          }
        }

        // Find one step copy size, based on the copy rectange
        amd::Coord3D oneStepSize(xferRect[0] * xferRect[1] * xferRect[2] *
                                 gpuMem(dstMemory).elementSize());

        // Initialize transfer buffer array
        Memory* xferBuf[MaxXferBuffers];
        for (uint i = 0; i < MaxXferBuffers; ++i) {
          xferBuf[i] = dev().getGpuMemory(xferBuffers_[i]);
          if (xferBuf[i] == NULL) {
            transfer = false;
            break;
          }
        }

        // Loop until we transfer all data
        while (transfer && (copySize > 0)) {
          size_t copySizeTmp = copySize;
          amd::Coord3D srcTmp(src);
          amd::Coord3D oneStepSizeTmp(oneStepSize);
          // Step 1. Initiate DRM transfer with all staging buffers
          for (uint i = 0; i < MaxXferBuffers; ++i) {
            // Make sure we don't transfer more than copy size
            if (copySizeTmp > 0) {
              if (!gpuMem(srcMemory).partialMemCopyTo(gpu(), srcTmp, xferSrc, oneStepSizeTmp,
                                                      *xferBuf[i], CopyRect, FlushDMA)) {
                transfer = false;
                break;
              }

              copySizeTmp -= oneStepSizeTmp[0];
              // Change buffer offset
              srcTmp.c[0] += oneStepSizeTmp[0];

              if (copySizeTmp < oneStepSizeTmp[0]) {
                oneStepSizeTmp.c[0] = copySizeTmp;
              }
            } else {
              break;
            }
          }

          // Step 2. Initiate compute transfer with all staging buffers
          for (uint i = 0; i < MaxXferBuffers; ++i) {
            if (copySize > 0) {
              if (!copyBufferToImageKernel(*xferBuf[i], dstMemory, xferSrc, dst, xferRect, false,
                                           0UL, 0UL, copyMetadata)) {
                transfer = false;
                break;
              }
              gpu().flushDMA(MainEngine);

              copySize -= oneStepSize[0];
              // Change buffer offset
              src.c[0] += oneStepSize[0];
              // Change image offset, ignore X offset
              for (uint j = 1; j < 3; ++j) {
                dst.c[j] += xferRect[j];
                if ((dst[j] - dstOrigin[j]) >= size[j]) {
                  dst.c[j] = dstOrigin[j];
                } else {
                  break;
                }
              }
              // Recalculate rectangle size if the remain data is smaller
              if (copySize < oneStepSize[0]) {
                for (uint j = 0; j < 3; ++j) {
                  xferRect.c[j] = size[j] - (dst[j] - dstOrigin[j]);
                }
                oneStepSize.c[0] = copySize;
              }
            } else {
              break;
            }
          }
        }

        if (copySize == 0) {
          result = true;
        } else {
          LogWarning("2 step transfer in copyBufferToImage failed");
        }
      }
    }
  }

  if (!result) {
    result = copyBufferToImageKernel(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                     rowPitch, slicePitch, copyMetadata);
  }

  synchronize();

  return result;
}

void CalcRowSlicePitches(uint64_t* pitch, const int32_t* copySize, size_t rowPitch,
                         size_t slicePitch, const Memory& mem) {
  uint32_t memFmtSize = mem.elementSize();
  bool img1Darray = (mem.desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? true : false;

  if (rowPitch == 0) {
    pitch[0] = copySize[0];
  } else {
    pitch[0] = rowPitch / memFmtSize;
  }
  if (slicePitch == 0) {
    pitch[1] = pitch[0] * (img1Darray ? 1 : copySize[1]);
  } else {
    pitch[1] = slicePitch / memFmtSize;
  }
  assert((pitch[0] <= pitch[1]) && "rowPitch must be <= slicePitch");

  if (img1Darray) {
    // For 1D array rowRitch = slicePitch
    pitch[0] = pitch[1];
  }
}

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
  size_t argSize = size;

  if (desc.type_ == T_POINTER && (desc.addressQualifier_ != CL_KERNEL_ARG_ADDRESS_LOCAL)) {
    if ((value == NULL) || (static_cast<const cl_mem*>(value) == NULL)) {
      reinterpret_cast<Memory**>(kernel->parameters().values() +
                                 kernel->parameters().memoryObjOffset())[desc.info_.arrayIndex_] =
          nullptr;
    } else {
      // convert cl_mem to amd::Memory*, return false if invalid.
      LP64_SWITCH(uint32_value, uint64_value) =
          static_cast<uintptr_t>((*static_cast<Memory* const*>(value))->virtualAddress()) + offset;
      reinterpret_cast<Memory**>(kernel->parameters().values() +
                                 kernel->parameters().memoryObjOffset())[desc.info_.arrayIndex_] =
          *static_cast<Memory* const*>(value);
      // Note: Special case for image SRD, which is 64 bit always
      if (LP64_SWITCH(true, false) &&
          (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject)) {
        uint64_value = uint32_value;
        argSize = sizeof(uint64_t);
      }
    }
  } else if (desc.type_ == T_SAMPLER) {
    assert(false && "No sampler support in blit manager! Use internal samplers!");
  } else
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

  switch (argSize) {
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

bool KernelBlitManager::copyBufferToImageKernel(
    device::Memory& srcMemory, device::Memory& dstMemory, const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin, const amd::Coord3D& size, bool entire, size_t rowPitch,
    size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  bool rejected = false;
  Memory* dstView = &gpuMem(dstMemory);
  bool releaseView = false;
  bool result = false;
  amd::Image::Format newFormat(gpuMem(dstMemory).desc().format_);
  bool swapLayer = dstView->desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY;

  // Find unsupported formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
      newFormat.image_channel_data_type = RejectedData[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // Find unsupported channel's order
  for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
    if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
      newFormat.image_channel_order = RejectedOrder[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // If the image format was rejected, then attempt to create a view
  if (rejected) {
    dstView = createView(gpuMem(dstMemory), newFormat);
    if (dstView != NULL) {
      rejected = false;
      releaseView = true;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                              entire, 0UL, 0UL, copyMetadata);
  }

  // Use a common blit type with three dimensions by default
  uint blitType = BlitCopyBufferToImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  if (gpuMem(dstMemory).desc().dimSize_ == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (gpuMem(dstMemory).desc().dimSize_ == 2) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  Memory* mem = &gpuMem(srcMemory);
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = dstView;
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);
  uint32_t memFmtSize = gpuMem(dstMemory).elementSize();
  uint32_t components = gpuMem(dstMemory).numComponents();

  // 1 element granularity for writes by default
  int32_t granularity = 1;
  if (memFmtSize == 2) {
    granularity = 2;
  } else if (memFmtSize >= 4) {
    granularity = 4;
  }
  CondLog(((srcOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  uint64_t srcOrg[4] = {srcOrigin[0] / granularity, srcOrigin[1], srcOrigin[2], 0};
  setArgument(kernels_[blitType], 2, sizeof(srcOrg), srcOrg);

  int32_t dstOrg[4] = {(int32_t)dstOrigin[0], (int32_t)dstOrigin[1], (int32_t)dstOrigin[2], 0};
  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};

  if (swapLayer) {
    dstOrg[2] = dstOrg[1];
    dstOrg[1] = 0;
    copySize[2] = copySize[1];
    copySize[1] = 1;
  }

  setArgument(kernels_[blitType], 3, sizeof(dstOrg), dstOrg);
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = memFmtSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  uint32_t format[4] = {components, memFmtSize / components, multiplier, 0};
  setArgument(kernels_[blitType], 5, sizeof(format), format);

  // Program row and slice pitches
  uint64_t pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, gpuMem(dstMemory));
  setArgument(kernels_[blitType], 6, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);
  if (releaseView) {
    delete dstView;
  }

  return result;
}

bool KernelBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  static const bool CopyRect = false;
  // Flush DMA for ASYNC copy
  static const bool FlushDMA = true;
  size_t imgRowPitch = size[0] * gpuMem(srcMemory).elementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyImageToBuffer_) {
    result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (gpuMem(dstMemory).isHostMemDirectAccess() &&
           (((rowPitch == 0) && (slicePitch == 0)) ||
            ((rowPitch == imgRowPitch) && ((slicePitch == 0) || (slicePitch == imgSlicePitch))))) {
    // First attempt to do this all with DMA,
    // but there are restriciton with older hardware
    // If the dest buffer is external physical(SDI), copy two step as
    // single step SDMA is causing corruption and the cause is under investigation
    if (dev().settings().imageDMA_ &&
        gpuMem(dstMemory).memoryType() != Resource::ExternalPhysical) {
      result = DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                 entire, rowPitch, slicePitch, copyMetadata);
      if (result) {
        synchronize();
        return result;
      }
    }

    // Find the overall copy size
    size_t copySize = size[0] * size[1] * size[2] * gpuMem(srcMemory).elementSize();

    // Check if double copy was requested
    if (xferBufferSize_ != 0) {
      amd::Coord3D src(srcOrigin);
      amd::Coord3D dst(dstOrigin);
      amd::Coord3D xferDst(0, 0, 0);
      amd::Coord3D xferRect(size);
      // Find transfer size in pixels
      size_t xferSizePix = xferBufferSize_ / gpuMem(srcMemory).elementSize();
      bool transfer = true;

      // Find transfer rectangle
      if (xferRect[0] > xferSizePix) {
        // The algorithm can't break a line.
        // It requires multiple rectangles tracking
        transfer = false;
      } else {
        xferRect.c[1] = xferSizePix / xferRect[0];
      }
      // Check if we exceeded the original size boundary in Y
      if (xferRect[1] > size[1]) {
        xferRect.c[1] = size[1];
        xferRect.c[2] = xferSizePix / (xferRect[0] * xferRect[1]);
      } else {
        xferRect.c[2] = 1;
      }
      // Check if we exceeded the original size boundary in Z
      if (xferRect[2] > size[2]) {
        xferRect.c[2] = size[2];
      }
      // Make sure size in Y dimension is divided by the rectangle size
      if (size[2] > 1) {
        while ((size[1] % xferRect[1]) != 0) {
          xferRect.c[1]--;
        }
      }

      // Find one step copy size, based on the copy rectange
      amd::Coord3D oneStepSize(xferRect[0] * xferRect[1] * xferRect[2] *
                               gpuMem(srcMemory).elementSize());

      // Initialize transfer buffer array
      Memory* xferBuf[MaxXferBuffers];
      for (uint i = 0; i < MaxXferBuffers; ++i) {
        xferBuf[i] = dev().getGpuMemory(xferBuffers_[i]);
        if (xferBuf[i] == NULL) {
          transfer = false;
          break;
        }
      }

      // Loop until we transfer all data
      while (transfer && (copySize > 0)) {
        size_t copySizeTmp = copySize;
        amd::Coord3D srcTmp(src);
        amd::Coord3D oneStepSizeTmp(oneStepSize);
        amd::Coord3D xferRectTmp(xferRect);

        // Step 1. Initiate compute transfer with all staging buffers
        for (uint i = 0; i < MaxXferBuffers; ++i) {
          if (copySizeTmp > 0) {
            if (!copyImageToBufferKernel(srcMemory, *xferBuf[i], srcTmp, xferDst, xferRectTmp,
                                         false, 0UL, 0UL, copyMetadata)) {
              transfer = false;
              break;
            }
            gpu().flushDMA(MainEngine);

            copySizeTmp -= oneStepSizeTmp[0];
            // Change image offset, ignore X offset
            for (uint j = 1; j < 3; ++j) {
              srcTmp.c[j] += xferRectTmp[j];
              if ((srcTmp[j] - srcOrigin[j]) >= size[j]) {
                srcTmp.c[j] = srcOrigin[j];
              } else {
                break;
              }
            }
            // Recalculate rectangle size if the remain data is smaller
            if (copySizeTmp < oneStepSizeTmp[0]) {
              for (uint j = 0; j < 3; ++j) {
                xferRectTmp.c[j] = size[j] - (srcTmp[j] - srcOrigin[j]);
              }
            }
          } else {
            break;
          }
        }

        // Step 2. Initiate DRM transfer with all staging buffers
        for (uint i = 0; i < MaxXferBuffers; ++i) {
          // Make sure we don't transfer more than copy size
          if (copySize > 0) {
            if (!xferBuf[i]->partialMemCopyTo(gpu(), xferDst, dst, oneStepSize, gpuMem(dstMemory),
                                              CopyRect, FlushDMA)) {
              transfer = false;
              break;
            }

            copySize -= oneStepSize[0];
            // Change buffer offset
            dst.c[0] += oneStepSize[0];
            // Change image offset, ignore X offset
            for (uint j = 1; j < 3; ++j) {
              src.c[j] += xferRect[j];
              if ((src[j] - srcOrigin[j]) >= size[j]) {
                src.c[j] = srcOrigin[j];
              } else {
                break;
              }
            }
            // Recalculate rectangle size if the remain data is smaller
            if (copySize < oneStepSize[0]) {
              for (uint j = 0; j < 3; ++j) {
                xferRect.c[j] = size[j] - (src[j] - srcOrigin[j]);
              }
              oneStepSize.c[0] = copySize;
            }
          } else {
            break;
          }
        }
      }

      if (copySize == 0) {
        result = true;
      } else {
        LogWarning("2 step transfer in copyBufferToImage failed");
      }
    }
  }

  if (!result) {
    result = copyImageToBufferKernel(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                     rowPitch, slicePitch, copyMetadata);
  }

  synchronize();

  return result;
}

bool KernelBlitManager::copyImageToBufferKernel(
    device::Memory& srcMemory, device::Memory& dstMemory, const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin, const amd::Coord3D& size, bool entire, size_t rowPitch,
    size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  bool rejected = false;
  Memory* srcView = &gpuMem(srcMemory);
  bool releaseView = false;
  bool result = false;
  amd::Image::Format newFormat(gpuMem(srcMemory).desc().format_);
  bool swapLayer = srcView->desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY;

  // Find unsupported formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
      newFormat.image_channel_data_type = RejectedData[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // Find unsupported channel's order
  for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
    if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
      newFormat.image_channel_order = RejectedOrder[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // If the image format was rejected, then attempt to create a view
  if (rejected) {
    srcView = createView(gpuMem(srcMemory), newFormat);
    if (srcView != NULL) {
      rejected = false;
      releaseView = true;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                              entire, 0UL, 0UL, copyMetadata);
  }

  uint blitType = BlitCopyImageToBuffer;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if (gpuMem(srcMemory).desc().dimSize_ == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (gpuMem(srcMemory).desc().dimSize_ == 2) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  Memory* mem = srcView;
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = &gpuMem(dstMemory);
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);

  // Update extra paramters for USHORT and UBYTE pointers.
  // Only then compiler can optimize the kernel to use
  // UAV Raw for other writes
  setArgument(kernels_[blitType], 2, sizeof(cl_mem), &mem);
  setArgument(kernels_[blitType], 3, sizeof(cl_mem), &mem);

  int32_t srcOrg[4] = {(int32_t)srcOrigin[0], (int32_t)srcOrigin[1], (int32_t)srcOrigin[2], 0};
  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  if (swapLayer) {
    srcOrg[2] = srcOrg[1];
    srcOrg[1] = 0;
    copySize[2] = copySize[1];
    copySize[1] = 1;
  }
  setArgument(kernels_[blitType], 4, sizeof(srcOrg), srcOrg);
  uint32_t memFmtSize = gpuMem(srcMemory).elementSize();
  uint32_t components = gpuMem(srcMemory).numComponents();

  // 1 element granularity for writes by default
  int32_t granularity = 1;
  if (memFmtSize == 2) {
    granularity = 2;
  } else if (memFmtSize >= 4) {
    granularity = 4;
  }
  CondLog(((dstOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  uint64_t dstOrg[4] = {dstOrigin[0] / granularity, dstOrigin[1], dstOrigin[2], 0};
  setArgument(kernels_[blitType], 5, sizeof(dstOrg), dstOrg);
  setArgument(kernels_[blitType], 6, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = memFmtSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  uint32_t format[4] = {components, memFmtSize / components, multiplier, 0};
  setArgument(kernels_[blitType], 7, sizeof(format), format);

  // Program row and slice pitches
  uint64_t pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, gpuMem(srcMemory));
  setArgument(kernels_[blitType], 8, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);
  if (releaseView) {
    delete srcView;
  }

  return result;
}

bool KernelBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                  const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                  const amd::Coord3D& size, bool entire,
                                  amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  Memory* srcView = &gpuMem(srcMemory);
  Memory* dstView = &gpuMem(dstMemory);
  amd::Image::Format srcFormat(srcView->desc().format_);
  amd::Image::Format dstFormat(dstView->desc().format_);
  bool srcRejected = false, dstRejected = false;
  bool srcReleaseView = false, dstReleaseView = false;

  // Find unsupported source formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == srcFormat.image_channel_data_type) {
      srcFormat.image_channel_data_type = RejectedData[i].clNewType_;
      srcRejected = true;
      break;
    }
  }

  // Search for the rejected source channel's order only if the format was rejected
  // Note: Image blit is independent from the channel order
  if (srcRejected) {
    for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
      if (RejectedOrder[i].clOldType_ == srcFormat.image_channel_order) {
        srcFormat.image_channel_order = RejectedOrder[i].clNewType_;
        break;
      }
    }
  }

  // Find unsupported destination formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == dstFormat.image_channel_data_type) {
      dstFormat.image_channel_data_type = RejectedData[i].clNewType_;
      dstRejected = true;
      break;
    }
  }

  // Search for the rejected destination channel's order only if the format was rejected
  // Note: Image blit is independent from the channel order
  if (dstRejected) {
    for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
      if (RejectedOrder[i].clOldType_ == dstFormat.image_channel_order) {
        dstFormat.image_channel_order = RejectedOrder[i].clNewType_;
        break;
      }
    }
  }

  if (srcFormat.image_channel_order != dstFormat.image_channel_order ||
      srcFormat.image_channel_data_type != dstFormat.image_channel_data_type) {
    // Give hint if any related test fails
    LogPrintfInfo("srcFormat(order=0x%xh, type=0x%xh) != dstFormat(order=0x%xh, type=0x%xh)",
                  srcFormat.image_channel_order, srcFormat.image_channel_data_type,
                  dstFormat.image_channel_order, dstFormat.image_channel_data_type);
  }

  // Attempt to create a view if the format was rejected
  if (srcRejected) {
    srcView = createView(gpuMem(srcMemory), srcFormat);
    if (srcView) {
      srcRejected = false;
      srcReleaseView = true;
    }
  }

  if (dstRejected) {
    dstView = createView(gpuMem(dstMemory), dstFormat);
    if (dstView) {
      dstRejected = false;
      dstReleaseView = true;
    }
  }
  // Fall into the host path for the copy if the image format was rejected
  if (srcRejected || dstRejected) {
    result = HostBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                        copyMetadata);
    if (srcReleaseView) {
      delete srcView;
    }
    if (dstReleaseView) {
      delete dstView;
    }
    synchronize();
    return result;
  }

  uint blitType = BlitCopyImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if ((gpuMem(srcMemory).desc().dimSize_ == 1) || (gpuMem(dstMemory).desc().dimSize_ == 1)) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if ((gpuMem(srcMemory).desc().dimSize_ == 2) || (gpuMem(dstMemory).desc().dimSize_ == 2)) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // The current OpenCL spec allows "copy images from a 1D image
  // array object to a 1D image array object" only.
  if ((gpuMem(srcMemory).desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
      (gpuMem(dstMemory).desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY)) {
    blitType = BlitCopyImage1DA;
  }

  // Program kernels arguments for the blit operation
  Memory* mem = srcView;
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = dstView;
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);

  // Program source origin
  int32_t srcOrg[4] = {(int32_t)srcOrigin[0], (int32_t)srcOrigin[1], (int32_t)srcOrigin[2], 0};
  if (gpuMem(srcMemory).desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    srcOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 2, sizeof(srcOrg), srcOrg);

  // Program destinaiton origin
  int32_t dstOrg[4] = {(int32_t)dstOrigin[0], (int32_t)dstOrigin[1], (int32_t)dstOrigin[2], 0};
  if (gpuMem(dstMemory).desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    dstOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 3, sizeof(dstOrg), dstOrg);

  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);

  if (srcReleaseView) {
    delete srcView;
  }
  if (dstReleaseView) {
    delete dstView;
  }
  synchronize();

  return result;
}

void FindPinSize(size_t& pinSize, const amd::Coord3D& size, size_t& rowPitch, size_t& slicePitch,
                 const Memory& mem) {
  pinSize = size[0] * mem.elementSize();
  if ((rowPitch == 0) || (rowPitch == pinSize)) {
    rowPitch = 0;
  } else {
    pinSize = rowPitch;
  }

  // Calculate the pin size, which should be equal to the copy size
  for (uint i = 1; i < mem.desc().dimSize_; ++i) {
    pinSize *= size[i];
    if (i == 1) {
      if ((slicePitch == 0) || (slicePitch == pinSize)) {
        slicePitch = 0;
      } else {
        if (mem.desc().topology_ != CL_MEM_OBJECT_IMAGE1D_ARRAY) {
          pinSize = slicePitch;
        } else {
          pinSize = slicePitch * size[i];
        }
      }
    }
  }
}

bool KernelBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  size_t rowPitch, size_t slicePitch, bool entire,
                                  amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableReadImage_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable())) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                        entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize;
    FindPinSize(pinSize, size, rowPitch, slicePitch, gpuMem(srcMemory));

    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

    if (amdMemory == NULL) {
      // Force SW copy
      result = HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                          entire, copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D dstOrigin(partial);

    // Get device memory for this virtual device
    Memory* dstMemory = dev().getGpuMemory(amdMemory);

    // Copy image to buffer
    result = copyImageToBuffer(srcMemory, *dstMemory, origin, dstOrigin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

bool KernelBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   size_t rowPitch, size_t slicePitch, bool entire,
                                   amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableWriteImage_ || gpuMem(dstMemory).isHostMemDirectAccess() ||
      gpuMem(dstMemory).isPersistentDirectMap()) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                         entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize;
    FindPinSize(pinSize, size, rowPitch, slicePitch, gpuMem(dstMemory));
    size_t partial = 0;
    bool pinned;

    amd::Memory* amdMemory = nullptr;
    Memory* srcMemory;
    if (pinSize > gpu().xferWrite().MaxSize()) {
      amdMemory = pinHostMemory(srcHost, pinSize, partial);
      if (amdMemory == nullptr) {
        // Force SW copy
        result = HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                             entire, copyMetadata);
        synchronize();
        return result;
      }
      // Get device memory for this virtual device
      srcMemory = dev().getGpuMemory(amdMemory);
      pinned = true;
    } else {
      srcMemory = &gpu().xferWrite().Acquire(pinSize);
      srcMemory->hostWrite(&gpu(), srcHost, 0, pinSize, Resource::NoWait);
      pinned = false;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Copy image to buffer
    result = copyBufferToImage(*srcMemory, dstMemory, srcOrigin, origin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    if (pinned) {
      // Add pinned memory for a later release
      gpu().addPinnedMem(amdMemory);
    } else {
      gpu().xferWrite().Release(*srcMemory);
    }
  }

  synchronize();

  return result;
}

bool KernelBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::BufferRect& srcRectIn,
                                       const amd::BufferRect& dstRectIn, const amd::Coord3D& sizeIn,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  bool rejected = false;

  // Fall into the PAL path for rejected transfers
  if (setup_.disableCopyBufferRect_ || gpuMem(srcMemory).isHostMemDirectAccess() ||
      gpuMem(dstMemory).isHostMemDirectAccess()) {
    if (!dev().settings().disableSdma_) {
      result = DmaBlitManager::copyBufferRect(srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn,
                                              entire, copyMetadata);
    }
    if (result) {
      synchronize();
      return result;
    }
  }

  uint blitType = BlitCopyBufferRect;
  size_t dim = 3;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  const static uint CopyRectAlignment[3] = {16, 4, 1};

  uint i;
  for (i = 0; i < sizeof(CopyRectAlignment) / sizeof(uint); i++) {
    // Check source alignments
    bool aligned = ((srcRectIn.rowPitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((srcRectIn.slicePitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((srcRectIn.start_ % CopyRectAlignment[i]) == 0);

    // Check destination alignments
    aligned &= ((dstRectIn.rowPitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((dstRectIn.slicePitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((dstRectIn.start_ % CopyRectAlignment[i]) == 0);

    // Check copy size alignment in the first dimension
    aligned &= ((sizeIn[0] % CopyRectAlignment[i]) == 0);

    if (aligned) {
      if (CopyRectAlignment[i] != 1) {
        blitType = BlitCopyBufferRectAligned;
      }
      break;
    }
  }

  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  amd::Coord3D size(sizeIn[0], sizeIn[1], sizeIn[2]);

  srcRect.rowPitch_ = srcRectIn.rowPitch_ / CopyRectAlignment[i];
  srcRect.slicePitch_ = srcRectIn.slicePitch_ / CopyRectAlignment[i];
  srcRect.start_ = srcRectIn.start_ / CopyRectAlignment[i];
  srcRect.end_ = srcRectIn.end_ / CopyRectAlignment[i];

  dstRect.rowPitch_ = dstRectIn.rowPitch_ / CopyRectAlignment[i];
  dstRect.slicePitch_ = dstRectIn.slicePitch_ / CopyRectAlignment[i];
  dstRect.start_ = dstRectIn.start_ / CopyRectAlignment[i];
  dstRect.end_ = dstRectIn.end_ / CopyRectAlignment[i];

  size.c[0] /= CopyRectAlignment[i];

  // Program the kernel's workload depending on the transfer dimensions
  if ((size[1] == 1) && (size[2] == 1)) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = 1;
    globalWorkSize[2] = 1;
    localWorkSize[0] = 256;
    localWorkSize[1] = 1;
    localWorkSize[2] = 1;
  } else if (size[2] == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = 1;
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }


  // Program kernels arguments for the blit operation
  Memory* mem = &gpuMem(srcMemory);
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = &gpuMem(dstMemory);
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);
  uint64_t src[4] = {srcRect.rowPitch_, srcRect.slicePitch_, srcRect.start_, 0};
  setArgument(kernels_[blitType], 2, sizeof(src), src);
  uint64_t dst[4] = {dstRect.rowPitch_, dstRect.slicePitch_, dstRect.start_, 0};
  setArgument(kernels_[blitType], 3, sizeof(dst), dst);
  uint64_t copySize[4] = {size[0], size[1], size[2], CopyRectAlignment[i]};
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);

  synchronize();

  return result;
}

bool KernelBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadBuffer_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable())) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = size[0];
    // Check if a pinned transfer can be executed with a single pin
    if ((pinSize <= dev().settings().pinnedXferSize_) && (pinSize > MinSizeForPinnedTransfer)) {
      size_t partial;
      amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

      if (amdMemory == NULL) {
        // Force SW copy
        result =
            HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
        synchronize();
        return result;
      }

      // Readjust host mem offset
      amd::Coord3D dstOrigin(partial);

      // Get device memory for this virtual device
      Memory* dstMemory = dev().getGpuMemory(amdMemory);

      // Copy image to buffer
      result = copyBuffer(srcMemory, *dstMemory, origin, dstOrigin, size, entire, copyMetadata);

      // Add pinned memory for a later release
      gpu().addPinnedMem(amdMemory);
    } else {
      result = DmaBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    }
  }

  synchronize();

  return result;
}

bool KernelBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                       const amd::BufferRect& bufRect,
                                       const amd::BufferRect& hostRect, const amd::Coord3D& size,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (gpuMem(srcMemory).isHostMemDirectAccess() && gpuMem(srcMemory).isCacheable())) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                             copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

    if (amdMemory == NULL) {
      // Force SW copy
      result = HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                               copyMetadata);
      synchronize();
      return result;
    }

    // Readjust host mem offset
    amd::BufferRect rect;
    rect.rowPitch_ = hostRect.rowPitch_;
    rect.slicePitch_ = hostRect.slicePitch_;
    rect.start_ = hostRect.start_ + partial;
    rect.end_ = hostRect.end_;

    // Get device memory for this virtual device
    Memory* dstMemory = dev().getGpuMemory(amdMemory);

    // Copy image to buffer
    result = copyBufferRect(srcMemory, *dstMemory, bufRect, rect, size, entire, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

bool KernelBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                    const amd::Coord3D& origin, const amd::Coord3D& size,
                                    bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableWriteBuffer_ ||
      (gpuMem(dstMemory).isHostMemDirectAccess() &&
       (gpuMem(dstMemory).memoryType() != Resource::ExternalPhysical)) ||
      (gpuMem(dstMemory).memoryType() == Resource::Persistent)) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = size[0];

    // Check if a pinned transfer can be executed with a single pin
    if ((pinSize <= dev().settings().pinnedXferSize_) && (pinSize > MinSizeForPinnedTransfer)) {
      size_t partial;
      amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

      if (amdMemory == NULL) {
        // Force SW copy
        result =
            DmaBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
        synchronize();
        return result;
      }

      // Readjust destination offset
      const amd::Coord3D srcOrigin(partial);

      // Get device memory for this virtual device
      Memory* srcMemory = dev().getGpuMemory(amdMemory);

      // Copy buffer rect
      result = copyBuffer(*srcMemory, dstMemory, srcOrigin, origin, size, entire, copyMetadata);

      // Add pinned memory for a later release
      gpu().addPinnedMem(amdMemory);
    } else {
      result = DmaBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    }
  }

  synchronize();

  return result;
}

bool KernelBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                        const amd::BufferRect& hostRect,
                                        const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                        bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access or it's persistent
  if (setup_.disableWriteBufferRect_ ||
      (gpuMem(dstMemory).isHostMemDirectAccess() &&
       (gpuMem(dstMemory).memoryType() != Resource::ExternalPhysical)) ||
      gpuMem(dstMemory).isPersistentDirectMap()) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                              copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

    if (amdMemory == NULL) {
      // Force SW copy
      result = HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                                copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Get device memory for this virtual device
    Memory* srcMemory = dev().getGpuMemory(amdMemory);

    // Readjust host mem offset
    amd::BufferRect rect;
    rect.rowPitch_ = hostRect.rowPitch_;
    rect.slicePitch_ = hostRect.slicePitch_;
    rect.start_ = hostRect.start_ + partial;
    rect.end_ = hostRect.end_;

    // Copy buffer rect
    result = copyBufferRect(*srcMemory, dstMemory, rect, bufRect, size, entire, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

bool KernelBlitManager::fillBuffer(device::Memory& memory, const void* pattern, size_t patternSize,
                                   const amd::Coord3D& surface, const amd::Coord3D& origin,
                                   const amd::Coord3D& size, bool entire, bool forceBlit) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host fill if memory has direct access
  if (setup_.disableFillBuffer_ || (!forceBlit && gpuMem(memory).isHostMemDirectAccess())) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::fillBuffer(memory, pattern, patternSize, size, origin, size, entire);
    synchronize();
    return result;
  } else {
    // Pack the fill buffer info, that handles unaligned memories.
    std::vector<FillBufferInfo> packed_vector{};
    FillBufferInfo::PackInfo(memory, size[0], origin[0], pattern, patternSize, packed_vector);

    size_t overall_offset = origin[0];
    for (auto& packed_obj : packed_vector) {
      constexpr uint32_t kFillType = FillBufferAligned;
      uint32_t kpattern_size = (packed_obj.pattern_expanded_)
                                   ? HostBlitManager::FillBufferInfo::kExtendedSize
                                   : patternSize;
      size_t kfill_size = packed_obj.fill_size_ / kpattern_size;
      uint64_t koffset = overall_offset;
      overall_offset += packed_obj.fill_size_;

      size_t globalWorkOffset[3] = {0, 0, 0};
      uint32_t alignment = (kpattern_size & 0xf) == 0   ? 2 * sizeof(uint64_t)
                           : (kpattern_size & 0x7) == 0 ? sizeof(uint64_t)
                           : (kpattern_size & 0x3) == 0 ? sizeof(uint32_t)
                           : (kpattern_size & 0x1) == 0 ? sizeof(uint16_t)
                                                        : sizeof(uint8_t);

      // Program kernels arguments for the fill operation
      Memory* mem = &gpuMem(memory);
      setArgument(kernels_[kFillType], 0, sizeof(cl_mem), &mem, koffset);
      const size_t localWorkSize = 256;
      size_t globalWorkSize = std::min(dev().settings().limit_blit_wg_ * localWorkSize, kfill_size);
      globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

      Memory& gpuCB = gpu().xferWrite().Acquire(patternSize);
      void* constBuf = gpuCB.map(&gpu(), Resource::NoWait);
      // If pattern has been expanded, use the expanded pattern, otherwise use the default pattern
      if (packed_obj.pattern_expanded_) {
        memcpy(constBuf, &packed_obj.expanded_pattern_, kpattern_size);
      } else {
        memcpy(constBuf, pattern, kpattern_size);
      }
      gpuCB.unmap(&gpu());
      Memory* pGpuCB = &gpuCB;
      setArgument(kernels_[kFillType], 1, sizeof(cl_mem), &pGpuCB);
      uint64_t offset = origin[0];

      // Adjust the pattern size in the copy type size
      kpattern_size /= alignment;
      setArgument(kernels_[kFillType], 2, sizeof(uint32_t), &kpattern_size);
      setArgument(kernels_[kFillType], 3, sizeof(alignment), &alignment);

      // Calculate max id
      uint64_t end_ptr = memory.virtualAddress() + koffset + kfill_size * kpattern_size * alignment;
      setArgument(kernels_[kFillType], 4, sizeof(end_ptr), &end_ptr);
      uint32_t next_chunk = globalWorkSize * kpattern_size;
      setArgument(kernels_[kFillType], 5, sizeof(uint32_t), &next_chunk);
      uint32_t lws = localWorkSize;
      setArgument(kernels_[kFillType], 6, sizeof(lws), &lws);

      // Create ND range object for the kernel's execution
      amd::NDRangeContainer ndrange(1, globalWorkOffset, &globalWorkSize, &localWorkSize);

      // Execute the blit
      address parameters = kernels_[kFillType]->parameters().values();
      result = gpu().submitKernelInternal(ndrange, *kernels_[kFillType], parameters);
      gpu().xferWrite().Release(gpuCB);
    }
  }

  synchronize();

  return result;
}

bool KernelBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                   const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                   const amd::Coord3D& sizeIn, bool entire,
                                   amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  if (!gpuMem(srcMemory).isHostMemDirectAccess() && !gpuMem(dstMemory).isHostMemDirectAccess()) {
    constexpr uint32_t kBlitType = BlitCopyBuffer;
    constexpr uint32_t kMaxAlignment = 2 * sizeof(uint64_t);
    amd::Coord3D size(sizeIn[0]);

    // Check alignments for source and destination
    bool aligned = ((srcOrigin[0] % kMaxAlignment) == 0) && ((dstOrigin[0] % kMaxAlignment) == 0);
    uint32_t aligned_size = (aligned) ? kMaxAlignment : sizeof(uint32_t);

    // Setup copy size accordingly to the alignment
    uint32_t remainder = size[0] % aligned_size;
    size.c[0] /= aligned_size;
    size.c[0] += (remainder != 0) ? 1 : 0;

    // Program the dispatch dimensions
    const size_t localWorkSize = (aligned) ? 512 : 1024;
    size_t globalWorkSize = std::min(dev().settings().limit_blit_wg_ * localWorkSize, size[0]);
    globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

    // Program kernels arguments for the blit operation
    Memory* mem = &gpuMem(srcMemory);
    // Program source origin
    uint64_t srcOffset = srcOrigin[0];
    setArgument(kernels_[kBlitType], 0, sizeof(cl_mem), &mem, srcOffset);
    mem = &gpuMem(dstMemory);
    // Program destinaiton origin
    uint64_t dstOffset = dstOrigin[0];
    setArgument(kernels_[kBlitType], 1, sizeof(cl_mem), &mem, dstOffset);

    uint64_t copySize = sizeIn[0];
    setArgument(kernels_[kBlitType], 2, sizeof(copySize), &copySize);

    setArgument(kernels_[kBlitType], 3, sizeof(remainder), &remainder);
    setArgument(kernels_[kBlitType], 4, sizeof(aligned_size), &aligned_size);

    // End pointer is the aligned copy size and destination offset
    uint64_t end_ptr = dstMemory.virtualAddress() + dstOffset + sizeIn[0] - remainder;
    setArgument(kernels_[kBlitType], 5, sizeof(end_ptr), &end_ptr);

    uint32_t next_chunk = globalWorkSize;
    setArgument(kernels_[kBlitType], 6, sizeof(next_chunk), &next_chunk);
    uint32_t lws = localWorkSize;
    setArgument(kernels_[kBlitType], 7, sizeof(lws), &lws);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(1, nullptr, &globalWorkSize, &localWorkSize);

    // Execute the blit
    address parameters = kernels_[kBlitType]->parameters().values();
    result = gpu().submitKernelInternal(ndrange, *kernels_[kBlitType], parameters);
  } else {
    result = DmaBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire,
                                        copyMetadata);
  }

  synchronize();

  return result;
}

bool KernelBlitManager::fillImage(device::Memory& memory, const void* pattern,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  bool entire) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  constexpr size_t kFillImageThreshold = 256 * 256;

  // Use host fill if memory has direct access and image is small
  if (setup_.disableFillImage_ || (gpuMem(memory).isHostMemDirectAccess() &&
                                   (size.c[0] * size.c[1] * size.c[2]) <= kFillImageThreshold)) {
    gpu().releaseGpuMemoryFence();

    result = HostBlitManager::fillImage(memory, pattern, origin, size, entire);
    synchronize();
    return result;
  }

  uint fillType;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];
  Memory* memView = &gpuMem(memory);
  amd::Image::Format newFormat(gpuMem(memory).owner()->asImage()->getImageFormat());
  bool swapLayer = memView->desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY;

  // Program the kernels workload depending on the fill dimensions
  fillType = FillImage;
  dim = 3;

  void* newpattern = const_cast<void*>(pattern);
  uint32_t iFillColor[4];

  bool rejected = false;
  bool releaseView = false;
  // For depth, we need to create a view
  if (memView->desc().format_.image_channel_order == CL_sRGBA) {
    // Find unsupported data type
    for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
      if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
        newFormat.image_channel_data_type = RejectedData[i].clNewType_;
        rejected = true;
        break;
      }
    }

    if (gpuMem(memory).desc().format_.image_channel_order == CL_sRGBA) {
      // Converting a linear RGB floating-point color value to a 8-bit unsigned integer sRGB value
      // because hw is not support write_imagef for sRGB.
      float* fColor = static_cast<float*>(newpattern);
      iFillColor[0] = sRGBmap(fColor[0]);
      iFillColor[1] = sRGBmap(fColor[1]);
      iFillColor[2] = sRGBmap(fColor[2]);
      iFillColor[3] = (uint32_t)(fColor[3] * 255.0f);
      newpattern = static_cast<void*>(&iFillColor[0]);
      for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
        if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
          newFormat.image_channel_order = RejectedOrder[i].clNewType_;
          rejected = true;
          break;
        }
      }
    }
  }
  // If the image format was rejected, then attempt to create a view
  if (rejected) {
    memView = createView(gpuMem(memory), newFormat);
    if (memView != NULL) {
      rejected = false;
      releaseView = true;
    }
  }

  // Perform workload split to allow multiple operations in a single thread
  globalWorkSize[0] = (size[0] + TransferSplitSize - 1) / TransferSplitSize;
  // Find the current blit type
  if (memView->desc().dimSize_ == 1) {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (memView->desc().dimSize_ == 2) {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  Memory* mem = memView;
  setArgument(kernels_[fillType], 0, sizeof(cl_mem), &mem);
  setArgument(kernels_[fillType], 1, sizeof(float[4]), newpattern);
  setArgument(kernels_[fillType], 2, sizeof(int32_t[4]), newpattern);
  setArgument(kernels_[fillType], 3, sizeof(uint32_t[4]), newpattern);

  int32_t fillOrigin[4] = {(int32_t)origin[0], (int32_t)origin[1], (int32_t)origin[2], 0};
  int32_t fillSize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  if (swapLayer) {
    fillOrigin[2] = fillOrigin[1];
    fillOrigin[1] = 0;
    fillSize[2] = fillSize[1];
    fillSize[1] = 1;
  }
  setArgument(kernels_[fillType], 4, sizeof(fillOrigin), fillOrigin);
  setArgument(kernels_[fillType], 5, sizeof(fillSize), fillSize);

  // Find the type of image
  uint32_t type = 0;
  switch (newFormat.image_channel_data_type) {
    case CL_SNORM_INT8:
    case CL_SNORM_INT16:
    case CL_UNORM_INT8:
    case CL_UNORM_INT16:
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
    case CL_UNORM_INT_101010:
    case CL_HALF_FLOAT:
    case CL_FLOAT:
      type = 0;
      break;
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      type = 1;
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      type = 2;
      break;
  }
  setArgument(kernels_[fillType], 6, sizeof(type), &type);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[fillType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[fillType], parameters);
  if (releaseView) {
    delete memView;
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::streamOpsWrite(device::Memory& memory, uint64_t value, size_t offset,
                                       size_t sizeBytes) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  uint blitType = StreamOpsWrite;
  size_t dim = 1;
  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};
  // Program kernels arguments for the write operation
  Memory* mem = &gpuMem(memory);
  bool is32BitWrite = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the write operation
  if (is32BitWrite) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
  }
  setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);
  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);
  synchronize();
  return result;
}

// ================================================================================================
bool KernelBlitManager::streamOpsWait(device::Memory& memory, uint64_t value, size_t offset,
                                      size_t sizeBytes, uint64_t flags, uint64_t mask) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  uint blitType = StreamOpsWait;
  size_t dim = 1;

  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};

  // Program kernels arguments for the wait operation
  Memory* mem = &gpuMem(memory);
  bool is32BitWait = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the wait operation
  if (is32BitWait) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
  }
  setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
  setArgument(kernels_[blitType], 3, sizeof(uint64_t), &flags);
  setArgument(kernels_[blitType], 4, sizeof(uint64_t), &mask);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);
  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                                 uint heap_size, uint number_of_initial_blocks) const {
  bool result;
  // Clear memory to 0 for device library logic and set
  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {256};
  size_t localWorkSize[1] = {256};

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);
  uint blitType = InitHeap;
  uint64_t management_heap_va = heap_to_initialize->virtualAddress();
  uint64_t initial_heap_va = 0;
  if (initial_blocks != nullptr) {
    initial_heap_va = initial_blocks->virtualAddress();
  }
  setArgument(kernels_[blitType], 0, sizeof(cl_ulong), &management_heap_va);
  setArgument(kernels_[blitType], 1, sizeof(cl_ulong), &initial_heap_va);
  setArgument(kernels_[blitType], 2, sizeof(uint), &heap_size);
  setArgument(kernels_[blitType], 3, sizeof(uint), &number_of_initial_blocks);
  address parameters = kernels_[blitType]->parameters().values();
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters);
  synchronize();

  return result;
}

bool KernelBlitManager::runScheduler(device::Memory& vqueue, device::Memory& params, uint paramIdx,
                                     uint threads) const {
  amd::ScopedLock k(lockXferOps_);

  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {threads};
  size_t localWorkSize[1] = {1};

  // Program kernels arguments
  Memory* q = &gpuMem(vqueue);
  Memory* p = &gpuMem(params);
  setArgument(kernels_[Scheduler], 0, sizeof(cl_mem), &q);
  setArgument(kernels_[Scheduler], 1, sizeof(cl_mem), &p);
  setArgument(kernels_[Scheduler], 2, sizeof(uint), &paramIdx);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[Scheduler]->parameters().values();
  bool result = gpu().submitKernelInternal(ndrange, *kernels_[Scheduler], parameters);

  synchronize();

  return result;
}

void KernelBlitManager::writeRawData(device::Memory& memory, size_t size, const void* data) const {
  amd::ScopedLock k(lockXferOps_);
  static_cast<pal::Memory&>(memory).writeRawData(gpu(), 0, size, data, false);

  synchronize();
}

bool KernelBlitManager::RunGwsInit(uint32_t value) const {
  amd::ScopedLock k(lockXferOps_);

  if (dev().settings().gwsInitSupported_ == false) {
    LogError("GWS Init is not supported on this target");
    return false;
  }

  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};

  // Program kernels arguments
  setArgument(kernels_[GwsInit], 0, sizeof(uint32_t), &value);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = kernels_[GwsInit]->parameters().values();
  bool result = gpu().submitKernelInternal(ndrange, *kernels_[GwsInit], parameters);

  synchronize();

  return result;
}

amd::Memory* DmaBlitManager::pinHostMemory(const void* hostMem, size_t pinSize,
                                           size_t& partial) const {
  size_t pinAllocSize;
  const static bool SysMem = true;
  amd::Memory* amdMemory;

  // Allign offset to 4K boundary (Vista/Win7 limitation)
  char* tmpHost = const_cast<char*>(
      amd::alignDown(reinterpret_cast<const char*>(hostMem), PinnedMemoryAlignment));

  // Find the partial size for unaligned copy
  partial = reinterpret_cast<const char*>(hostMem) - tmpHost;

  // Recalculate pin memory size
  pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);

  amdMemory = gpu().findPinnedMem(tmpHost, pinAllocSize);

  if (NULL != amdMemory) {
    return amdMemory;
  }

  amdMemory = new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, pinAllocSize);
  amdMemory->setVirtualDevice(&gpu());
  if ((amdMemory != NULL) && !amdMemory->create(tmpHost, SysMem)) {
    amdMemory->release();
    return NULL;
  }

  // Get device memory for this virtual device
  // @note: This will force real memory pinning
  Memory* srcMemory = dev().getGpuMemory(amdMemory);

  if (srcMemory == NULL) {
    // Release all pinned memory and attempt pinning again
    gpu().releasePinnedMem();
    srcMemory = dev().getGpuMemory(amdMemory);
    if (srcMemory == NULL) {
      // Release memory
      amdMemory->release();
      amdMemory = NULL;
    }
  }

  return amdMemory;
}

Memory* KernelBlitManager::createView(const Memory& parent, const cl_image_format format) const {
  assert(!parent.desc().buffer_ && "View supports images only");
  Memory* gpuImage = new Image(dev(), parent.size(), parent.desc().width_, parent.desc().height_,
                               parent.desc().depth_, format, parent.desc().topology_, 1);

  // Create resource
  if (NULL != gpuImage) {
    Resource::ImageViewParams params;
    const Memory& gpuMem = static_cast<const Memory&>(parent);

    params.owner_ = parent.owner();
    params.level_ = parent.desc().baseLevel_;
    params.layer_ = 0;
    params.resource_ = &gpuMem;
    params.memory_ = &gpuMem;
    params.gpu_ = &gpu();

    // Create memory object
    bool result = gpuImage->create(Resource::ImageView, &params);
    if (!result) {
      delete gpuImage;
      return NULL;
    }
  }

  return gpuImage;
}

}  // namespace amd::pal
