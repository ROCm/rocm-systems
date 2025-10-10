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

#include "platform/commandqueue.hpp"
#include "device/device.hpp"
#include "device/blit.hpp"
#include "utils/debug.hpp"

#include <cmath>

namespace amd::device {

HostBlitManager::HostBlitManager(VirtualDevice& vDev, Setup setup)
    : BlitManager(setup), vDev_(vDev), dev_(vDev.device()) {}

bool HostBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                 const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
  // Map the device memory to CPU visible
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly);
  if (NULL == src) {
    LogError("Couldn't map device memory for host read");
    return false;
  }
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "Using host memcpy D2H, src=%p, dst=%p, size=%zu",
          (reinterpret_cast<const_address>(src) + origin[0]), dstHost, size[0]);
  // Copy memory
  std::memcpy(dstHost, reinterpret_cast<const_address>(src) + origin[0], size[0]);

  // Unmap device memory
  srcMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                     const amd::BufferRect& bufRect,
                                     const amd::BufferRect& hostRect, const amd::Coord3D& size,
                                     bool entire, amd::CopyMetadata copyMetadata) const {
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }

  size_t srcOffset;
  size_t dstOffset;

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      srcOffset = bufRect.offset(0, y, z);
      dstOffset = hostRect.offset(0, y, z);

      // Copy memory line by line
      std::memcpy((reinterpret_cast<address>(dstHost) + dstOffset),
                  (reinterpret_cast<const_address>(src) + srcOffset), size[0]);
    }
  }

  // Unmap source memory
  srcMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                size_t rowPitch, size_t slicePitch, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (srcMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // rowPitch and slicePitch in bytes
  size_t srcRowPitch;
  size_t srcSlicePitch;

  // Get physical GPU memmory
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly, startLayer, numLayers, &srcRowPitch,
                               &srcSlicePitch);
  if (NULL == src) {
    LogError("Couldn't map GPU memory for host read");
    return false;
  }

  size_t elementSize = srcMemory.owner()->asImage()->getImageFormat().getElementSize();
  size_t srcOffsBase = origin[0] * elementSize;
  size_t copySize = size[0] * elementSize;
  size_t srcOffs;
  size_t dstOffs = 0;

  // Make sure we use the right pitch if it's not specified
  if (rowPitch == 0) {
    rowPitch = size[0] * elementSize;
  }

  // Make sure we use the right slice if it's not specified
  if (slicePitch == 0) {
    slicePitch = size[0] * size[1] * elementSize;
  }

  // Adjust destination offset with Y dimension
  srcOffsBase += srcRowPitch * origin[1];

  // Adjust the destination offset with Z dimension
  srcOffsBase += srcSlicePitch * origin[2];

  // Copy memory line by line
  for (size_t slice = 0; slice < size[2]; ++slice) {
    srcOffs = srcOffsBase + slice * srcSlicePitch;
    dstOffs = slice * slicePitch;

    // Copy memory line by line
    for (size_t row = 0; row < size[1]; ++row) {
      // Copy memory
      std::memcpy((reinterpret_cast<address>(dstHost) + dstOffs),
                  (reinterpret_cast<const_address>(src) + srcOffs), copySize);

      srcOffs += srcRowPitch;
      dstOffs += rowPitch;
    }
  }

  // Unmap the device memory
  srcMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                  const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                  amd::CopyMetadata copyMetadata) const {
  uint flags = 0;
  if (entire) {
    flags = Memory::CpuWriteOnly;
  }

  // Map the device memory to CPU visible
  void* dst = dstMemory.cpuMap(vDev_, flags);
  if (NULL == dst) {
    LogError("Couldn't map GPU memory for host write");
    return false;
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "Using host memcpy H2D, src=%p, dst=%p, size=%zu",
        srcHost, (reinterpret_cast<address>(dst) + origin[0]), size[0]);
  // Copy memory
  std::memcpy(reinterpret_cast<address>(dst) + origin[0], srcHost, size[0]);

  // Unmap the device memory
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                      const amd::BufferRect& hostRect,
                                      const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                      bool entire, amd::CopyMetadata copyMetadata) const {
  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  size_t srcOffset;
  size_t dstOffset;

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      srcOffset = hostRect.offset(0, y, z);
      dstOffset = bufRect.offset(0, y, z);

      // Copy memory line by line
      std::memcpy((reinterpret_cast<address>(dst) + dstOffset),
                  (reinterpret_cast<const_address>(srcHost) + srcOffset), size[0]);
    }
  }

  // Unmap destination memory
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin, const amd::Coord3D& size,
                                 size_t rowPitch, size_t slicePitch, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
  uint flags = 0;
  if (entire) {
    flags = Memory::CpuWriteOnly;
  }

  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (dstMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // rowPitch and slicePitch in bytes
  size_t dstRowPitch;
  size_t dstSlicePitch;
  // Map the device memory to CPU visible
  void* dst = dstMemory.cpuMap(vDev_, flags, startLayer, numLayers, &dstRowPitch, &dstSlicePitch);
  if (NULL == dst) {
    LogError("Couldn't map GPU memory for host write");
    return false;
  }

  size_t elementSize = dstMemory.owner()->asImage()->getImageFormat().getElementSize();
  size_t srcOffs = 0;
  size_t copySize = size[0] * elementSize;
  size_t dstOffsBase = origin[0] * elementSize;
  size_t dstOffs;

  // Make sure we use the right pitch if it's not specified
  if (rowPitch == 0) {
    rowPitch = size[0] * elementSize;
  }

  // Make sure we use the right slice if it's not specified
  if (slicePitch == 0) {
    slicePitch = size[0] * size[1] * elementSize;
  }

  // Adjust the destination offset with Y dimension
  dstOffsBase += dstRowPitch * origin[1];

  // Adjust the destination offset with Z dimension
  dstOffsBase += dstSlicePitch * origin[2];

  // Copy memory slice by slice
  for (size_t slice = 0; slice < size[2]; ++slice) {
    dstOffs = dstOffsBase + slice * dstSlicePitch;
    srcOffs = slice * slicePitch;

    // Copy memory line by line
    for (size_t row = 0; row < size[1]; ++row) {
      // Copy memory
      std::memcpy((reinterpret_cast<address>(dst) + dstOffs),
                  (reinterpret_cast<const_address>(srcHost) + srcOffs), copySize);

      dstOffs += dstRowPitch;
      srcOffs += rowPitch;
    }
  }

  // Unmap the device memory
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                 const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                 const amd::Coord3D& size, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_,
                               // Overlap detection
                               (&srcMemory == &dstMemory) ? 0 : Memory::CpuReadOnly);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }

  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY,
          "Using host memcpy for copyBuffer, src=%p, dst=%p, size=%zu",
          (reinterpret_cast<const_address>(src) + srcOrigin[0]),
          (reinterpret_cast<address>(dst) + dstOrigin[0]), size[0]);
  // Straight forward buffer copy
  std::memcpy((reinterpret_cast<address>(dst) + dstOrigin[0]),
              (reinterpret_cast<const_address>(src) + srcOrigin[0]), size[0]);

  // Unmap source and destination memory
  dstMemory.cpuUnmap(vDev_);
  srcMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                     const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                                     const amd::Coord3D& size, bool entire,
                                     amd::CopyMetadata copyMetadata) const {
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_,
                               // Overlap detection
                               (&srcMemory == &dstMemory) ? 0 : Memory::CpuReadOnly);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }

  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY,
          "Using host memcpy for copyBufferRect, src=%p, dst=%p, size=%zu",
          (reinterpret_cast<const_address>(src) + srcRect.offset(0, 0, 0)),
          (reinterpret_cast<address>(dst) + dstRect.offset(0, 0, 0)), size[0]);

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      size_t srcOffset = srcRect.offset(0, y, z);
      size_t dstOffset = dstRect.offset(0, y, z);

      // Copy memory line by line
      std::memcpy((reinterpret_cast<address>(dst) + dstOffset),
                  (reinterpret_cast<const_address>(src) + srcOffset), size[0]);
    }
  }

  // Unmap source and destination memory
  dstMemory.cpuUnmap(vDev_);
  srcMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                        const amd::Coord3D& srcOrigin,
                                        const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                        bool entire, size_t rowPitch, size_t slicePitch,
                                        amd::CopyMetadata copyMetadata) const {
  size_t startLayer = srcOrigin[2];
  size_t numLayers = size[2];
  if (srcMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = srcOrigin[1];
    numLayers = size[1];
  }
  // rowPitch and slicePitch in bytes
  size_t srcRowPitch;
  size_t srcSlicePitch;
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly, startLayer, numLayers, &srcRowPitch,
                               &srcSlicePitch);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }
  size_t elementSize = srcMemory.owner()->asImage()->getImageFormat().getElementSize();

  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  size_t srcOffs = srcOrigin[0];
  size_t dstOffs = dstOrigin[0];
  size_t srcOffsOrg;
  size_t copySize = size[0];

  // Calculate the offset in bytes
  srcOffs *= elementSize;
  copySize *= elementSize;

  // Adjust source offset with Y and Z dimensions
  srcOffs += srcRowPitch * srcOrigin[1];
  srcOffs += srcSlicePitch * srcOrigin[2];

  srcOffsOrg = srcOffs;

  // Copy memory slice by slice
  for (size_t slice = 0; slice < size[2]; ++slice) {
    srcOffs = srcOffsOrg + slice * srcSlicePitch;

    // Copy memory line by line
    for (size_t rows = 0; rows < size[1]; ++rows) {
      std::memcpy((reinterpret_cast<address>(dst) + dstOffs),
                  (reinterpret_cast<const_address>(src) + srcOffs), copySize);

      srcOffs += srcRowPitch;
      dstOffs += copySize;
    }
  }

  // Unmap source and destination memory
  srcMemory.cpuUnmap(vDev_);
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                        const amd::Coord3D& srcOrigin,
                                        const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                        bool entire, size_t rowPitch, size_t slicePitch,
                                        amd::CopyMetadata copyMetadata) const {
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }

  size_t startLayer = dstOrigin[2];
  size_t numLayers = size[2];
  if (dstMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = dstOrigin[1];
    numLayers = size[1];
  }
  // rowPitch and slicePitch in bytes
  size_t dstRowPitch;
  size_t dstSlicePitch;
  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0, startLayer, numLayers,
                               &dstRowPitch, &dstSlicePitch);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  size_t elementSize = dstMemory.owner()->asImage()->getImageFormat().getElementSize();
  size_t srcOffs = srcOrigin[0];
  size_t dstOffs = dstOrigin[0];
  size_t dstOffsOrg;
  size_t copySize = size[0];

  // Calculate the offset in bytes
  dstOffs *= elementSize;
  copySize *= elementSize;

  // Adjust destination offset with Y and Z dimension
  dstOffs += dstRowPitch * dstOrigin[1];
  dstOffs += dstSlicePitch * dstOrigin[2];

  dstOffsOrg = dstOffs;

  // Copy memory slice by slice
  for (size_t slice = 0; slice < size[2]; ++slice) {
    dstOffs = dstOffsOrg + slice * dstSlicePitch;

    // Copy memory line by line
    for (size_t rows = 0; rows < size[1]; ++rows) {
      std::memcpy((reinterpret_cast<address>(dst) + dstOffs),
                  (reinterpret_cast<const_address>(src) + srcOffs), copySize);

      srcOffs += copySize;
      dstOffs += dstRowPitch;
    }
  }

  // Unmap source and destination memory
  srcMemory.cpuUnmap(vDev_);
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  size_t startLayer = srcOrigin[2];
  size_t numLayers = size[2];
  if (srcMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = srcOrigin[1];
    numLayers = size[1];
  }
  // rowPitch and slicePitch in bytes
  size_t srcRowPitch;
  size_t srcSlicePitch;
  // Map source memory
  void* src = srcMemory.cpuMap(vDev_, Memory::CpuReadOnly, startLayer, numLayers, &srcRowPitch,
                               &srcSlicePitch);
  if (src == NULL) {
    LogError("Couldn't map source memory");
    return false;
  }
  if (dstMemory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = dstOrigin[1];
    numLayers = size[1];
  } else {
    startLayer = dstOrigin[2];
    numLayers = size[2];
  }

  // rowPitch and slicePitch in bytes
  size_t dstRowPitch;
  size_t dstSlicePitch;
  // Map destination memory
  void* dst = dstMemory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0, startLayer, numLayers,
                               &dstRowPitch, &dstSlicePitch);
  if (dst == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  size_t elementSize = dstMemory.owner()->asImage()->getImageFormat().getElementSize();
  assert(elementSize == srcMemory.owner()->asImage()->getImageFormat().getElementSize());

  size_t srcOffs = srcOrigin[0];
  size_t dstOffs = dstOrigin[0];
  size_t srcOffsOrg;
  size_t dstOffsOrg;
  size_t copySize = size[0];

  // Calculate the offsets in bytes
  srcOffs *= elementSize;
  dstOffs *= elementSize;
  copySize *= elementSize;

  // Adjust destination and sorce offsets with Y dimension
  srcOffs += srcRowPitch * srcOrigin[1];
  dstOffs += dstRowPitch * dstOrigin[1];

  // Adjust destination and sorce offsets with Z dimension
  srcOffs += srcSlicePitch * srcOrigin[2];
  dstOffs += dstSlicePitch * dstOrigin[2];

  srcOffsOrg = srcOffs;
  dstOffsOrg = dstOffs;

  // Copy memory slice by slice
  for (size_t slice = 0; slice < size[2]; ++slice) {
    srcOffs = srcOffsOrg + slice * srcSlicePitch;
    dstOffs = dstOffsOrg + slice * dstSlicePitch;

    // Copy memory line by line
    for (size_t rows = 0; rows < size[1]; ++rows) {
      std::memcpy((reinterpret_cast<address>(dst) + dstOffs),
                  (reinterpret_cast<const_address>(src) + srcOffs), copySize);

      srcOffs += srcRowPitch;
      dstOffs += dstRowPitch;
    }
  }

  // Unmap source and destination memory
  srcMemory.cpuUnmap(vDev_);
  dstMemory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::fillBuffer(device::Memory& memory, const void* pattern, size_t patternSize,
                                 const amd::Coord3D& surface, const amd::Coord3D& origin,
                                 const amd::Coord3D& size, bool entire, bool forceBlit) const {
  // Map memory
  void* fillMem = memory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0);
  if (fillMem == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  size_t offset = origin[0];
  size_t fillSize = size[0];

  if ((fillSize % patternSize) != 0) {
    LogError("Misaligned buffer size and pattern size!");
  }

  // Fill the buffer memory with a pattern
  for (size_t i = 0; i < (fillSize / patternSize); i++) {
    memcpy((reinterpret_cast<address>(fillMem) + offset),
           (reinterpret_cast<const_address>(pattern)), patternSize);
    offset += patternSize;
  }

  // Unmap source and destination memory
  memory.cpuUnmap(vDev_);

  return true;
}

bool HostBlitManager::fillImage(device::Memory& memory, const void* pattern,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                bool entire) const {
  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (memory.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }
  // rowPitch and slicePitch in bytes
  size_t devRowPitch;
  size_t devSlicePitch;

  void* newpattern = const_cast<void*>(pattern);
  float fFillColor[4];

  // Converting a linear RGB floating-point color value to a normalized 8-bit unsigned integer sRGB
  // value so that the cpu path can treat sRGB as RGB for host transfer.
  if (memory.owner()->asImage()->getImageFormat().image_channel_order == CL_sRGBA) {
    float* fColor = static_cast<float*>(newpattern);
    fFillColor[0] = sRGBmap(fColor[0]) / 255.0f;
    fFillColor[1] = sRGBmap(fColor[1]) / 255.0f;
    fFillColor[2] = sRGBmap(fColor[2]) / 255.0f;
    fFillColor[3] = fColor[3];
    newpattern = static_cast<void*>(&fFillColor[0]);
  }

  // Map memory
  void* fillMem = memory.cpuMap(vDev_, (entire) ? Memory::CpuWriteOnly : 0, startLayer, numLayers,
                                &devRowPitch, &devSlicePitch);
  if (fillMem == NULL) {
    LogError("Couldn't map destination memory");
    return false;
  }

  float fillValue[4];
  memset(fillValue, 0, sizeof(fillValue));
  memory.owner()->asImage()->getImageFormat().formatColor(newpattern, fillValue);

  size_t elementSize = memory.owner()->asImage()->getImageFormat().getElementSize();
  size_t offset = origin[0] * elementSize;
  size_t offsetOrg;

  // Adjust offset with Y dimension
  offset += devRowPitch * origin[1];

  // Adjust offset with Z dimension
  offset += devSlicePitch * origin[2];

  offsetOrg = offset;

  // Fill the image memory with a pattern
  for (size_t slice = 0; slice < size[2]; ++slice) {
    offset = offsetOrg + slice * devSlicePitch;

    for (size_t rows = 0; rows < size[1]; ++rows) {
      size_t pixOffset = offset;

      // Copy memory pixel by pixel
      for (size_t column = 0; column < size[0]; ++column) {
        memcpy((reinterpret_cast<address>(fillMem) + pixOffset),
               (reinterpret_cast<const_address>(fillValue)), elementSize);
        pixOffset += elementSize;
      }

      offset += devRowPitch;
    }
  }

  // Unmap memory
  memory.cpuUnmap(vDev_);

  return true;
}

uint32_t HostBlitManager::sRGBmap(float fc) const {
  double c = (double)fc;

#ifdef ATI_OS_LINUX
  if (std::isnan(c)) c = 0.0;
#else
  if (_isnan(c)) c = 0.0;
#endif

  if (c > 1.0)
    c = 1.0;
  else if (c < 0.0)
    c = 0.0;
  else if (c < 0.0031308)
    c = 12.92 * c;
  else
    c = (1055.0 / 1000.0) * pow(c, 5.0 / 12.0) - (55.0 / 1000.0);

  return (uint32_t)(c * 255.0 + 0.5);
}

// ================================================================================================
void HostBlitManager::FillBufferInfo::ExpandPattern(uint32_t pattern_size, const void* pattern) {
  // If pattern size exceeds extended, then runtime will select the normal path
  if (pattern_size >= kExtendedSize) {
    return;
  }

  pattern_expanded_ = true;
  if (pattern_size == sizeof(uint8_t)) {
    uint8_t pattern_byte = *reinterpret_cast<const uint8_t*>(pattern);
    for (uint32_t i = 0; i < kExtendedSize; ++i) {
      reinterpret_cast<uint8_t*>(expanded_pattern_)[i] = pattern_byte;
    }
  } else if (pattern_size == sizeof(uint16_t)) {
    uint16_t pattern_word = *reinterpret_cast<const uint16_t*>(pattern);
    for (uint32_t i = 0; i < kExtendedSize / sizeof(uint16_t); ++i) {
      reinterpret_cast<uint16_t*>(expanded_pattern_)[i] = pattern_word;
    }
  } else if (pattern_size == sizeof(uint32_t)) {
    uint32_t pattern_dword = *reinterpret_cast<const uint32_t*>(pattern);
    for (uint32_t i = 0; i < kExtendedSize / sizeof(uint32_t); ++i) {
      reinterpret_cast<uint32_t*>(expanded_pattern_)[i] = pattern_dword;
    }
  } else {
    uint64_t pattern_qword = *reinterpret_cast<const uint64_t*>(pattern);
    reinterpret_cast<uint64_t*>(expanded_pattern_)[0] = pattern_qword;
    reinterpret_cast<uint64_t*>(expanded_pattern_)[1] = pattern_qword;
  }
}

// ================================================================================================
void HostBlitManager::FillBufferInfo::PackInfo(const device::Memory& memory, size_t fill_size,
                                               size_t fill_origin, const void* pattern_ptr,
                                               size_t pattern_size,
                                               std::vector<FillBufferInfo>& packed_info) {
  // 1. Validate input arguments
  guarantee(fill_size >= pattern_size, "Pattern Size: %u cannot be greater than fill size: %u \n",
            pattern_size, fill_size);

  constexpr bool kDisablePackingOptimization = true;
  // Check if packing optimization is disabled
  if (kDisablePackingOptimization) {
    // Simple case: create a single FillBufferInfo without alignment optimization
    FillBufferInfo fill_info(fill_size);
    packed_info.push_back(fill_info);
    return;
  }

  // 2. Calculate the next closest dword aligned address for faster processing
  size_t dst_addr = memory.virtualAddress() + fill_origin;
  size_t aligned_dst_addr = amd::alignUp(dst_addr, kExtendedSize);
  guarantee(aligned_dst_addr >= dst_addr,
            "Aligned address: %u cannot be greater than destination"
            "address :%u \n",
            aligned_dst_addr, dst_addr);

  // 3. If given address is not aligned calculate head and tail size.
  size_t head_size = std::min(aligned_dst_addr - dst_addr, fill_size);
  size_t aligned_size = ((fill_size - head_size) / kExtendedSize) * kExtendedSize;
  size_t tail_size = (fill_size - head_size) % kExtendedSize;
  guarantee((head_size + aligned_size + tail_size) <= fill_size,
            "Head size, aligned size & tail"
            "size together cannot cross fill size");

  // 4. Fill the head, aligned, tail info if they exist.
  if (head_size > 0) {
    // Offsetted ptrs should align with pattern size. Runtime not responsible for rotating pattern.
    guarantee((head_size % pattern_size) == 0, "Offseted ptr should align with pattern_size");

    FillBufferInfo fill_info(head_size);
    packed_info.push_back(fill_info);
  }

  if (aligned_size > 0) {
    // Offsetted ptrs should align with pattern size. Runtime not responsible for rotating pattern.
    guarantee((aligned_size % pattern_size) == 0, "Offseted ptr should align with pattern_size");

    FillBufferInfo fill_info(aligned_size);
    fill_info.ExpandPattern(pattern_size, pattern_ptr);
    packed_info.push_back(fill_info);
  }

  if (tail_size > 0) {
    // Offsetted ptrs should align with pattern size. Runtime not responsible for rotating pattern.
    guarantee((tail_size % pattern_size) == 0, "Offseted ptr should align with pattern_size");

    FillBufferInfo fill_info(tail_size);
    packed_info.push_back(fill_info);
  }
}

}  // namespace amd::device
