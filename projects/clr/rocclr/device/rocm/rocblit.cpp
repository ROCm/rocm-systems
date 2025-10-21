/* Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc.

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
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocsched.hpp"
#include "utils/debug.hpp"
#include <algorithm>

namespace amd::roc {
DmaBlitManager::DmaBlitManager(VirtualGPU& gpu, Setup setup)
    : HostBlitManager(gpu, setup),
      MinSizeForPinnedXfer(dev().settings().pinnedMinXferSize_),
      PinXferSize(dev().settings().pinnedXferSize_),
      StagingXferSize(dev().settings().stagedXferSize_),
      completeOperation_(false),
      context_(nullptr) {
  dev().getSdmaRWMasks(&sdmaEngineReadMask_, &sdmaEngineWriteMask_);
}

inline void DmaBlitManager::synchronize() const {
  if (syncOperation_) {
    gpu().releaseGpuMemoryFence();
    gpu().releasePinnedMem();
  }
}

inline Memory& DmaBlitManager::gpuMem(device::Memory& mem) const {
  return static_cast<Memory&>(mem);
}

// ================================================================================================
bool DmaBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (dev().settings().blocking_blit_ &&
      (setup_.disableReadBuffer_ ||
       (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached()))) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
  }

  size_t copySize = size[0];
  if (copySize > 0) {
    const_address addrSrc = gpuMem(srcMemory).getDeviceMemory() + origin[0];
    address addrDst = reinterpret_cast<address>(dstHost);
    constexpr bool kHostToDev = false;
    constexpr bool kEnablePin = true;
    if (!hsaCopyStagedOrPinned(addrSrc, addrDst, copySize, kHostToDev, copyMetadata, kEnablePin)) {
      LogError("DmaBlitManager:: readBuffer copy failure!");
      return false;
    }
  }
  return true;
}

// ================================================================================================
bool DmaBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                    const amd::BufferRect& bufRect, const amd::BufferRect& hostRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                           copyMetadata);
  } else {
    const_address src = gpuMem(srcMemory).getDeviceMemory();

    size_t srcOffset;
    size_t dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        srcOffset = bufRect.offset(0, y, z);
        dstOffset = hostRect.offset(0, y, z);

        // Copy data from device to host - line by line
        address dst = reinterpret_cast<address>(dstHost) + dstOffset;
        bool retval = hsaCopyStagedOrPinned(src + srcOffset, dst, size[0], false, copyMetadata);
        if (!retval) {
          return retval;
        }
      }
    }
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::readImage(device::Memory& srcMemory, void* dstHost, const amd::Coord3D& origin,
                               const amd::Coord3D& size, size_t rowPitch, size_t slicePitch,
                               bool entire, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
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

// ================================================================================================
bool DmaBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (dev().settings().blocking_blit_ &&
      (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess() ||
       gpuMem(dstMemory).IsPersistentDirectMap())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
  }
  size_t copySize = size[0];
  if (copySize > 0) {
    address dstAddr = gpuMem(dstMemory).getDeviceMemory() + origin[0];
    const_address srcAddr = reinterpret_cast<const_address>(srcHost);
    constexpr bool kHostToDev = true;
    constexpr bool enablePin = true;
    if (!hsaCopyStagedOrPinned(srcAddr, dstAddr, copySize, kHostToDev, copyMetadata, enablePin)) {
      LogError("DmaBlitManager:: writeBuffer copy failure!");
      return false;
    }
  }
  return true;
}

// ================================================================================================
bool DmaBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                     const amd::BufferRect& hostRect,
                                     const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                     bool entire, amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                            copyMetadata);
  } else {
    address dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

    size_t srcOffset;
    size_t dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        srcOffset = hostRect.offset(0, y, z);
        dstOffset = bufRect.offset(0, y, z);

        // Copy data from host to device - line by line
        const_address src = reinterpret_cast<const_address>(srcHost) + srcOffset;
        constexpr bool kHostToDev = true;
        bool retval =
            hsaCopyStagedOrPinned(src, dst + dstOffset, size[0], kHostToDev, copyMetadata);
        if (!retval) {
          return retval;
        }
      }
    }
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                size_t rowPitch, size_t slicePitch, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
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

// ================================================================================================
bool DmaBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBuffer_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
       (dev().agent_profile() != HSA_PROFILE_FULL) && dstMemory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size, false,
                                       copyMetadata);
  } else {
    return hsaCopy(gpuMem(srcMemory), gpuMem(dstMemory), srcOrigin, dstOrigin, size, copyMetadata);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                    const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
       dstMemory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBufferRect(srcMemory, dstMemory, srcRect, dstRect, size, entire,
                                           copyMetadata);
  } else {
    gpu().releaseGpuMemoryFence(kSkipCpuWait);

    void* src = gpuMem(srcMemory).getDeviceMemory();
    void* dst = gpuMem(dstMemory).getDeviceMemory();

    // Detect the agents for memory allocations
    const hsa_agent_t srcAgent =
        (srcMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();
    const hsa_agent_t dstAgent =
        (dstMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();

    bool isSubwindowRectCopy = true;
    hsa_amd_copy_direction_t direction = hsaHostToHost;

    hsa_agent_t agent = dev().getBackendDevice();
    // Determine copy direction
    if (srcMemory.isHostMemDirectAccess() && !dstMemory.isHostMemDirectAccess()) {
      direction = hsaHostToDevice;
    } else if (!srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess()) {
      direction = hsaDeviceToHost;
    } else if (!srcMemory.isHostMemDirectAccess() && !dstMemory.isHostMemDirectAccess()) {
      direction = hsaDeviceToDevice;
    }

    hsa_pitched_ptr_t srcMem = {(reinterpret_cast<address>(src) + srcRect.offset(0, 0, 0)),
                                srcRect.rowPitch_, srcRect.slicePitch_};

    hsa_pitched_ptr_t dstMem = {(reinterpret_cast<address>(dst) + dstRect.offset(0, 0, 0)),
                                dstRect.rowPitch_, dstRect.slicePitch_};

    hsa_dim3_t dim = {static_cast<uint32_t>(size[0]), static_cast<uint32_t>(size[1]),
                      static_cast<uint32_t>(size[2])};
    hsa_dim3_t offset = {0, 0, 0};


    if ((srcRect.rowPitch_ % 4 != 0) || (srcRect.slicePitch_ % 4 != 0) ||
        (dstRect.rowPitch_ % 4 != 0) || (dstRect.slicePitch_ % 4 != 0)) {
      isSubwindowRectCopy = false;
    }

    HwQueueEngine engine = HwQueueEngine::Unknown;
    if (srcAgent.handle == dstAgent.handle) {
      // Same device transfer
      engine = HwQueueEngine::SdmaIntra;
    } else {
      // Different devices transfer
      if (srcAgent.handle == dev().getCpuAgent().handle) {
        // CPU to device
        engine = HwQueueEngine::SdmaWrite;
      } else if (dstAgent.handle == dev().getCpuAgent().handle) {
        // Device to CPU
        engine = HwQueueEngine::SdmaRead;
      } else {
        // Device to different device
        engine = HwQueueEngine::SdmaInter;
      }
    }

    auto wait_events = gpu().Barriers().WaitingSignal(engine);

    if (isSubwindowRectCopy) {
      hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

      // Copy memory line by line
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY2,
              "HSA Async Copy Rect dst=0x%zx, src=0x%zx, wait_event=0x%zx, "
              "completion_signal=0x%zx",
              dstMem.base, srcMem.base, (wait_events.size() != 0) ? wait_events[0].handle : 0,
              active.handle);

      hsa_status_t status =
          Hsa::memory_async_copy_rect(&dstMem, &offset, &srcMem, &offset, &dim, agent, direction,
                                      wait_events.size(), wait_events.data(), active);
      if (status != HSA_STATUS_SUCCESS) {
        gpu().Barriers().ResetCurrentSignal();
        LogPrintfError("DMA buffer failed with code %d", status);
        return false;
      }
    } else {
      // Fall to line by line copies
      const hsa_signal_value_t kInitVal = size[2] * size[1];
      hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitVal, gpu().timestamp());

      for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
          size_t srcOffset = srcRect.offset(0, y, z);
          size_t dstOffset = dstRect.offset(0, y, z);

          // Copy memory line by line
          ClPrint(amd::LOG_DEBUG, amd::LOG_COPY2,
                  "HSA Async Copy wait_event=0x%zx, completion_signal=0x%zx",
                  (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle);
          hsa_status_t status =
              Hsa::memory_async_copy((reinterpret_cast<address>(dst) + dstOffset), dstAgent,
                                     (reinterpret_cast<const_address>(src) + srcOffset), srcAgent,
                                     size[0], wait_events.size(), wait_events.data(), active);
          if (status != HSA_STATUS_SUCCESS) {
            gpu().Barriers().ResetCurrentSignal();
            LogPrintfError("DMA buffer failed with code %d", status);
            return false;
          }
        }
      }
    }
  }

  // The hsa copy api would result in a dirty cache state
  gpu().setFenceDirty(false);
  return true;
}

// ================================================================================================
bool DmaBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

  if (setup_.disableCopyImageToBuffer_) {
    result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    Image& srcImage = static_cast<roc::Image&>(srcMemory);
    Buffer& dstBuffer = static_cast<roc::Buffer&>(dstMemory);
    address dstHost = reinterpret_cast<address>(dstBuffer.getDeviceMemory()) + dstOrigin[0];

    // Use ROCm path for a transfer.
    // Note: it doesn't support SDMA
    hsa_ext_image_region_t image_region;
    image_region.offset.x = srcOrigin[0];
    image_region.offset.y = srcOrigin[1];
    image_region.offset.z = srcOrigin[2];
    image_region.range.x = size[0];
    image_region.range.y = size[1];
    image_region.range.z = size[2];

    hsa_status_t status = Hsa::image_export(gpu().gpu_device(), srcImage.getHsaImageObject(),
                                            dstHost, rowPitch, slicePitch, &image_region);
    result = (status == HSA_STATUS_SUCCESS) ? true : false;

    // hsa_ext_image_export need a system scope fence
    gpu().addSystemScope();

    // Check if a HostBlit transfer is required
    if (completeOperation_ && !result) {
      result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

// ================================================================================================
bool DmaBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

  if (setup_.disableCopyBufferToImage_) {
    result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    Buffer& srcBuffer = static_cast<roc::Buffer&>(srcMemory);
    Image& dstImage = static_cast<roc::Image&>(dstMemory);

    // Use ROC path for a transfer
    // Note: it doesn't support SDMA
    address srcHost = reinterpret_cast<address>(srcBuffer.getDeviceMemory()) + srcOrigin[0];

    hsa_ext_image_region_t image_region;
    image_region.offset.x = dstOrigin[0];
    image_region.offset.y = dstOrigin[1];
    image_region.offset.z = dstOrigin[2];
    image_region.range.x = size[0];
    image_region.range.y = size[1];
    image_region.range.z = size[2];

    hsa_status_t status = Hsa::image_import(gpu().gpu_device(), srcHost, rowPitch, slicePitch,
                                            dstImage.getHsaImageObject(), &image_region);
    result = (status == HSA_STATUS_SUCCESS) ? true : false;

    // hsa_ext_image_import need a system scope fence
    gpu().addSystemScope();

    // Check if a HostBlit tran sfer is required
    if (completeOperation_ && !result) {
      result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

// ================================================================================================
bool DmaBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                               const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                               const amd::Coord3D& size, bool entire,
                               amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

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

// ================================================================================================
inline bool DmaBlitManager::rocrCopyBuffer(address dst, hsa_agent_t& dstAgent, const_address src,
                                           hsa_agent_t& srcAgent, size_t size,
                                           amd::CopyMetadata& copyMetadata) const {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  uint32_t copyMask = 0;
  uint32_t freeEngineMask = 0;
  uint32_t recIdMask = 0;
  bool kUseRegularCopyApi = 0;
  constexpr size_t kRetainCountThreshold = 8;
  bool forceSDMA =
      (copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::SDMA);
  HwQueueEngine engine = HwQueueEngine::Unknown;

  // Determine engine based on source and destination agents
  if (srcAgent.handle == dstAgent.handle) {
    // Device to same device
    engine = HwQueueEngine::SdmaIntra;
  } else {
    // Different devices
    if (srcAgent.handle == dev().getCpuAgent().handle) {
      // CPU to device
      engine = HwQueueEngine::SdmaWrite;
    } else if (dstAgent.handle == dev().getCpuAgent().handle) {
      // Device to CPU
      engine = HwQueueEngine::SdmaRead;
    } else {
      // Device to different device
      engine = HwQueueEngine::SdmaInter;
    }
  }

  gpu().Barriers().SetActiveEngine(engine);
  auto wait_events = gpu().Barriers().WaitingSignal(engine);
  hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

  if (!kUseRegularCopyApi && engine != HwQueueEngine::Unknown) {
    copyMask = gpu().getLastUsedSdmaEngine();
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "Last copy mask 0x%x", copyMask);
    copyMask &= (engine == HwQueueEngine::SdmaRead ? sdmaEngineReadMask_ : sdmaEngineWriteMask_);
    if (copyMask == 0) {
      // Check SDMA engine status
      status = Hsa::memory_copy_engine_status(dstAgent, srcAgent, &freeEngineMask);

      if (status == HSA_STATUS_SUCCESS) {
        status = Hsa::memory_get_preferred_copy_engine(dstAgent, srcAgent, &recIdMask);
      }

      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
              "Query copy engine status %x, srcAgent %p, "
              "dstAgent %p, free_engine_mask 0x%x, rec_engine_mask 0x%x",
              status, srcAgent.handle, dstAgent.handle, freeEngineMask, recIdMask);

      // If requested engine is valid and available, use it
      if (recIdMask != 0 && (freeEngineMask & recIdMask) != 0) {
        copyMask = recIdMask - (recIdMask & (recIdMask - 1));
      } else {
        // Otherwise use first available engine
        copyMask = freeEngineMask - (freeEngineMask & (freeEngineMask - 1));
      }

      gpu().setLastUsedSdmaEngine(copyMask);
    }

    if (copyMask != 0 && status == HSA_STATUS_SUCCESS) {
      // Copy on the first available free engine if ROCr returns a valid mask
      hsa_amd_sdma_engine_id_t copyEngine = static_cast<hsa_amd_sdma_engine_id_t>(copyMask);

      // Check if engine type is SdmaInter and adjust agents accordingly
      // ROCr copy api would always choose SDMA engine of the srcAgent if its a GPU
      if (engine == HwQueueEngine::SdmaInter) {
        srcAgent = dev().getBackendDevice();
        forceSDMA = true;
      }

      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY2,
              "HSA Copy copy_engine=0x%x, dst=0x%zx, src=0x%zx, "
              "size=%ld, forceSDMA=%d, engineType=%d, wait_event=0x%zx, completion_signal=0x%zx",
              copyEngine, dst, src, size, forceSDMA, engine,
              (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle);

      status =
          Hsa::memory_async_copy_on_engine(dst, dstAgent, src, srcAgent, size, wait_events.size(),
                                           wait_events.data(), active, copyEngine, forceSDMA);
    } else {
      kUseRegularCopyApi = true;
    }
  }

  if (engine == HwQueueEngine::Unknown || kUseRegularCopyApi) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY2,
            "HSA Copy dst=0x%zx, src=0x%zx, size=%ld, wait_event=0x%zx, "
            "completion_signal=0x%zx, engineType=%d",
            dst, src, size, (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle,
            engine);

    status = Hsa::memory_async_copy(dst, dstAgent, src, srcAgent, size, wait_events.size(),
                                    wait_events.data(), active);
  }

  if (status == HSA_STATUS_SUCCESS) {
    gpu().addSystemScope();
    // The hsa copy api would result in a dirty cache state
    gpu().setFenceDirty(false);
  } else {
    gpu().Barriers().ResetCurrentSignal();
    LogPrintfError("HSA copy failed with code %d, falling to Blit copy", status);
  }

  return (status == HSA_STATUS_SUCCESS);
}


// ================================================================================================
bool DmaBlitManager::hsaCopy(const Memory& srcMemory, const Memory& dstMemory,
                             const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                             const amd::Coord3D& size, amd::CopyMetadata& copyMetadata) const {
  address src = reinterpret_cast<address>(srcMemory.getDeviceMemory());
  address dst = reinterpret_cast<address>(dstMemory.getDeviceMemory());

  bool skipCpuWait = true;

  src += srcOrigin[0];
  dst += dstOrigin[0];

  hsa_agent_t srcAgent;
  hsa_agent_t dstAgent;

  if (&srcMemory.dev() == &dstMemory.dev()) {
    // Detect the agents for memory allocations
    srcAgent = (srcMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();
    dstAgent = (dstMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();

    // When a memory is opened as IPCBuffer, the runtime is not aware of the agent that
    // owns the memory, thus query the pointer info here.
    if (static_cast<const amd::Memory*>(srcMemory.owner())->ipcShared()) {
      hsa_amd_pointer_info_t info = {sizeof(hsa_amd_pointer_info_t)};
      if (HSA_STATUS_SUCCESS ==
          Hsa::pointer_info(const_cast<address>(src), &info, nullptr, nullptr, nullptr)) {
        srcAgent = info.agentOwner;
      }
    }

    if (static_cast<const amd::Memory*>(dstMemory.owner())->ipcShared()) {
      hsa_amd_pointer_info_t info = {sizeof(hsa_amd_pointer_info_t)};
      if (HSA_STATUS_SUCCESS == Hsa::pointer_info(dst, &info, nullptr, nullptr, nullptr)) {
        dstAgent = info.agentOwner;
      }
    }
  } else {
    srcAgent = srcMemory.dev().getBackendDevice();
    dstAgent = dstMemory.dev().getBackendDevice();
  }

  // Blocking D2H copies need a wait anyways so better wait here
  // than having to wait on the device for dependent signals for SDMA which is slow
  if (!copyMetadata.isAsync_ && !srcMemory.isHostMemDirectAccess() &&
      dstMemory.isHostMemDirectAccess()) {
    skipCpuWait = false;
  }

  gpu().releaseGpuMemoryFence(skipCpuWait);

  return rocrCopyBuffer(dst, dstAgent, src, srcAgent, size[0], copyMetadata);
}


// ================================================================================================
// Get Staging or Pinned memory buffer
void DmaBlitManager::getBuffer(const_address hostMem, size_t size, bool enablePin, bool first_tx,
                               DmaBlitManager::BufferState& buffState) const {
  bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer);
  size_t copyChunkSize = doHostPinning ? PinXferSize : StagingXferSize;
  size_t xferSize = std::min(size, copyChunkSize);

  if (doHostPinning) {  // Pin host Memory
    char* alignedHost = reinterpret_cast<char*>(const_cast<unsigned char*>(hostMem));
    size_t partial1 = 0;
    size_t partial2 = 0;
    if (xferSize > PinXferSize && first_tx) {
      // Align to 4K boundary
      alignedHost = const_cast<char*>(
          amd::alignDown(reinterpret_cast<const char*>(hostMem), PinnedMemoryAlignment));
      // Find partial size of unaligned copy
      partial2 = reinterpret_cast<const char*>(hostMem) - alignedHost;
      size_t tmpSize = amd::alignUp(PinXferSize + partial2, PinnedMemoryAlignment);
      xferSize = std::min(tmpSize - partial2, size);
    }
    amd::Memory* pinnedMem = pinHostMemory(alignedHost, xferSize, partial1);
    if (pinnedMem != nullptr) {
      Memory* pinnedMemory = dev().getRocMemory(pinnedMem);
      address pinBuffer = pinnedMemory->getDeviceMemory();
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "HSA Copy Using Pinned resource size %d",
              xferSize);
      buffState.copySize_ = xferSize;
      buffState.buffer_ = pinBuffer + partial1 + partial2;
      buffState.pinnedMem_ = pinnedMem;
      return;
    }
    LogWarning("DmaBlitManager::getBuffer failed to pin a resource!");
  }
  // If Memory Pinning fails, failback to staging buffer
  xferSize = std::min(xferSize, StagingXferSize);
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "HSA Copy Using Staging resource size %d",
          xferSize);
  buffState.copySize_ = xferSize;
  buffState.buffer_ = gpu().Staging().Acquire(std::min(xferSize, StagingXferSize));
}

// ================================================================================================
void DmaBlitManager::releaseBuffer(BufferState& buffer) const {
  if (buffer.pinnedMem_) {
    gpu().addPinnedMem(buffer.pinnedMem_);
  }
}

// ================================================================================================
bool DmaBlitManager::hsaCopyStagedOrPinned(const_address hostSrc, address hostDst, size_t size,
                                           bool hostToDev, amd::CopyMetadata& copyMetadata,
                                           bool enablePin) const {
  // Do not skip wait here for D2H. Resolving dependent signals for SDMA engine is slow
  gpu().releaseGpuMemoryFence(hostToDev || !dev().settings().blocking_blit_);
  // If Pinning is enabled, Pin host Memory for copy size > MinSizeForPinnedTransfer
  // For 16KB < size <= MinSizeForPinnedTransfer Use staging buffer without pinning
  bool status = true;
  size_t copyOffset = 0;
  size_t totalSize = size;

  // Staging Buffer or Pinned Host Memory
  address stagingBuffer = 0;
  // src and dst agent for rocr
  hsa_agent_t srcAgent = hostToDev ? dev().getCpuAgent() : dev().getBackendDevice();
  hsa_agent_t dstAgent = hostToDev ? dev().getBackendDevice() : dev().getCpuAgent();
  bool firstTx = true;
  while (totalSize > 0) {
    const_address hostmem = hostToDev ? hostSrc : hostDst;
    // Get Pinned Host Memory or Staging buffer based on copy size
    BufferState outBuffer = {0};
    getBuffer(static_cast<const_address>(hostmem + copyOffset), totalSize, enablePin, firstTx,
              outBuffer);
    size_t copysize = outBuffer.copySize_;
    address stagingBuffer = outBuffer.buffer_;
    if (stagingBuffer == 0) {
      LogWarning("DmaBlitManager::hsaCopyStagedOrPinned Buffer creation failed!");
      status = false;
      break;
    }
    if (hostToDev) {                          // H2D Path
      if (outBuffer.pinnedMem_ == nullptr) {  // Copy to Staging Buffer
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "memcpy stg buf=%p, host src=%p, size=%zu",
                stagingBuffer, hostSrc + copyOffset, copysize);
        memcpy(stagingBuffer, hostSrc + copyOffset, copysize);
      }
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "HSA Async Copy staged H2D, Async=%d",
              copyMetadata.isAsync_);
      address dst = hostDst + copyOffset;
      status = rocrCopyBuffer(dst, dstAgent, stagingBuffer, srcAgent, copysize, copyMetadata);
      if (!status) {
        // Release Pinned Memory back to pool if any
        releaseBuffer(outBuffer);
        break;
      }
    } else {  // D2H Path
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "HSA Async Copy staged D2H, Async=%d",
              copyMetadata.isAsync_);
      const_address src = static_cast<const_address>(hostSrc) + copyOffset;
      status = rocrCopyBuffer(stagingBuffer, dstAgent, src, srcAgent, copysize, copyMetadata);
      if (status) {
        if (outBuffer.pinnedMem_ == nullptr) {
          // Wait for current signal of previous rocr copy if its not pinned mem
          gpu().Barriers().WaitCurrent();
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "memcpy host dst=%p, stg buf=%p, size=%zu",
                  hostDst + copyOffset, stagingBuffer, copysize);
          memcpy(hostDst + copyOffset, stagingBuffer, copysize);
        }
      } else {
        // Release Pinned Memory back to pool if any
        releaseBuffer(outBuffer);
        break;
      }
    }

    // Release Pinned Memory back to pool if any
    releaseBuffer(outBuffer);
    // Update Offset and Transfer Size
    copyOffset += copysize;
    totalSize -= copysize;
    firstTx = false;
  }

  // @note: HIP requires to unpin all memory after operation, due to an optimization with
  // direct HSA signal check HIP avoids the command completion wait
  if (amd::IS_HIP && (gpu().command() != nullptr) && gpu().command()->IsMemoryPinned()) {
    gpu().Barriers().WaitCurrent();
    gpu().command()->ReleasePinnedMemory();
  }

  if (!status) {
    return false;
  }

  return true;
}

// ================================================================================================
KernelBlitManager::KernelBlitManager(VirtualGPU& gpu, Setup setup)
    : DmaBlitManager(gpu, setup),
      program_(nullptr),
      xferBufferSize_(0),
      lockXferOps_(true) /* Transfer Ops Lock*/ {
  for (uint i = 0; i < BlitTotal; ++i) {
    kernels_[i] = nullptr;
  }

  completeOperation_ = false;
}

KernelBlitManager::~KernelBlitManager() {
  for (uint i = 0; i < NumBlitKernels(); ++i) {
    if (nullptr != kernels_[i]) {
      kernels_[i]->release();
    }
  }

  if (nullptr != program_) {
    program_->release();
  }

  if (nullptr != context_) {
    // Release a dummy context
    context_->release();
  }
}

bool KernelBlitManager::create(amd::Device& device) {
  if (!DmaBlitManager::create(device)) {
    return false;
  }

  if (!createProgram(static_cast<Device&>(device))) {
    return false;
  }
  return true;
}

// ================================================================================================
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
    for (uint i = 0; i < NumBlitKernels(); ++i) {
      const amd::Symbol* symbol = program_->findSymbol(BlitName[i]);
      if (symbol == nullptr) {
        // Not all blit kernels are needed in some setup, so continue with the rest
        continue;
      }
      kernels_[i] = new amd::Kernel(*program_, *symbol, BlitName[i]);
      if (kernels_[i] == nullptr) {
        break;
      }
      // Validate blit kernels for the scratch memory usage (pre SI)
      if (!device.validateKernel(*kernels_[i], &gpu())) {
        break;
      }
    }

    result = true;
  } while (!result);

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

// ================================================================================================
bool KernelBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  size_t imgRowPitch = size[0] * dstImage->getImageFormat().getElementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyBufferToImage_) {
    result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (srcMemory.isHostMemDirectAccess() &&
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
  }

  if (!result) {
    result = copyBufferToImageKernel(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                     rowPitch, slicePitch, copyMetadata);
  }

  synchronize();

  return result;
}

// ================================================================================================
void CalcRowSlicePitches(uint64_t* pitch, const int32_t* copySize, size_t rowPitch,
                         size_t slicePitch, const Memory& mem) {
  amd::Image* image = static_cast<amd::Image*>(mem.owner());
  uint32_t memFmtSize = image->getImageFormat().getElementSize();
  bool img1Darray = (mem.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? true : false;

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

// ================================================================================================
bool KernelBlitManager::copyBufferToImageKernel(
    device::Memory& srcMemory, device::Memory& dstMemory, const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin, const amd::Coord3D& size, bool entire, size_t rowPitch,
    size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  bool rejected = false;
  Memory* dstView = &gpuMem(dstMemory);
  bool result = false;
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image::Format newFormat(dstImage->getImageFormat());
  bool swapLayer =
      (dstImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

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
  if (rejected &&
      // todo ROC runtime has a problem with a view for this format
      (dstImage->getImageFormat().image_channel_data_type != CL_UNORM_INT_101010)) {
    dstView = createView(gpuMem(dstMemory), newFormat, CL_MEM_WRITE_ONLY);
    if (dstView != nullptr) {
      rejected = false;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return DmaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                             entire, rowPitch, slicePitch, copyMetadata);
  }

  // Use a common blit type with three dimensions by default
  uint blitType = BlitCopyBufferToImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  if (dstImage->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (dstImage->getDims() == 2) {
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
  cl_mem mem = as_cl<amd::Memory>(srcMemory.owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstView->owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);
  uint32_t memFmtSize = dstImage->getImageFormat().getElementSize();
  uint32_t components = dstImage->getImageFormat().getNumChannels();

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
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  size_t imgRowPitch = size[0] * srcImage->getImageFormat().getElementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyImageToBuffer_) {
    result = DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                               entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (dstMemory.isHostMemDirectAccess() &&
           (((rowPitch == 0) && (slicePitch == 0)) ||
            ((rowPitch == imgRowPitch) && ((slicePitch == 0) || (slicePitch == imgSlicePitch))))) {
    // First attempt to do this all with DMA,
    // but there are restriciton with older hardware
    // If the dest buffer is external physical(SDI), copy two step as
    // single step SDMA is causing corruption and the cause is under investigation
    if (dev().settings().imageDMA_) {
      result = DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                 entire, rowPitch, slicePitch, copyMetadata);
      if (result) {
        synchronize();
        return result;
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

// ================================================================================================
bool KernelBlitManager::copyImageToBufferKernel(
    device::Memory& srcMemory, device::Memory& dstMemory, const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin, const amd::Coord3D& size, bool entire, size_t rowPitch,
    size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  bool rejected = false;
  Memory* srcView = &gpuMem(srcMemory);
  bool result = false;
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image::Format newFormat(srcImage->getImageFormat());
  bool swapLayer =
      (srcImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

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
  if (rejected &&
      // todo ROC runtime has a problem with a view for this format
      (srcImage->getImageFormat().image_channel_data_type != CL_UNORM_INT_101010)) {
    srcView = createView(gpuMem(srcMemory), newFormat, CL_MEM_READ_ONLY);
    if (srcView != nullptr) {
      rejected = false;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                             entire, rowPitch, slicePitch, copyMetadata);
  }

  uint blitType = BlitCopyImageToBuffer;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if (srcImage->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (srcImage->getDims() == 2) {
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
  cl_mem mem = as_cl<amd::Memory>(srcView->owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstMemory.owner());
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
  uint32_t memFmtSize = srcImage->getImageFormat().getElementSize();
  uint32_t components = srcImage->getImageFormat().getNumChannels();

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
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                  const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                  const amd::Coord3D& size, bool entire,
                                  amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  Memory* srcView = &gpuMem(srcMemory);
  Memory* dstView = &gpuMem(dstMemory);
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  amd::Image::Format srcFormat(srcImage->getImageFormat());
  amd::Image::Format dstFormat(dstImage->getImageFormat());
  bool srcRejected = false, dstRejected = false;
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
        srcRejected = true;
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

  // Search for the rejected destionation channel's order only if the format was rejected
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
    srcView = createView(gpuMem(srcMemory), srcFormat, CL_MEM_READ_ONLY);
    if (srcView != nullptr) {
      srcRejected = false;
    }
  }

  if (dstRejected) {
    dstView = createView(gpuMem(dstMemory), dstFormat, CL_MEM_WRITE_ONLY);
    if (dstView != nullptr) {
      dstRejected = false;
    }
  }

  // Fall into the host path for the copy if the image format was rejected
  if (srcRejected || dstRejected) {
    result = DmaBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                       copyMetadata);
    synchronize();
  }

  uint blitType = BlitCopyImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if ((srcImage->getDims() == 1) || (dstImage->getDims() == 1)) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if ((srcImage->getDims() == 2) || (dstImage->getDims() == 2)) {
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
  if ((gpuMem(srcMemory).owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
      (gpuMem(dstMemory).owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY)) {
    blitType = BlitCopyImage1DA;
  }

  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(srcView->owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstView->owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);

  // Program source origin
  int32_t srcOrg[4] = {(int32_t)srcOrigin[0], (int32_t)srcOrigin[1], (int32_t)srcOrigin[2], 0};
  if ((srcImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10)) {
    srcOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 2, sizeof(srcOrg), srcOrg);

  // Program destinaiton origin
  int32_t dstOrg[4] = {(int32_t)dstOrigin[0], (int32_t)dstOrigin[1], (int32_t)dstOrigin[2], 0};
  if ((dstImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10)) {
    dstOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 3, sizeof(dstOrg), dstOrg);

  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  synchronize();

  return result;
}

// ================================================================================================
void FindPinSize(size_t& pinSize, const amd::Coord3D& size, size_t& rowPitch, size_t& slicePitch,
                 const Memory& mem) {
  amd::Image* image = static_cast<amd::Image*>(mem.owner());
  pinSize = size[0] * image->getImageFormat().getElementSize();
  if ((rowPitch == 0) || (rowPitch == pinSize)) {
    rowPitch = 0;
  } else {
    pinSize = rowPitch;
  }

  // Calculate the pin size, which should be equal to the copy size
  for (uint i = 1; i < image->getDims(); ++i) {
    pinSize *= size[i];
    if (i == 1) {
      if ((slicePitch == 0) || (slicePitch == pinSize)) {
        slicePitch = 0;
      } else {
        if (mem.owner()->getType() != CL_MEM_OBJECT_IMAGE1D_ARRAY) {
          pinSize = slicePitch;
        } else {
          pinSize = slicePitch * size[i];
        }
      }
    }
  }
}

// ================================================================================================
bool KernelBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  size_t rowPitch, size_t slicePitch, bool entire,
                                  amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadImage_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
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

    if (amdMemory == nullptr) {
      // Force SW copy
      result = DmaBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                         entire, copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D dstOrigin(partial);

    // Get device memory for this virtual device
    Memory* dstMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyImageToBuffer(srcMemory, *dstMemory, origin, dstOrigin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   size_t rowPitch, size_t slicePitch, bool entire,
                                   amd::CopyMetadata copyMetadata) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableWriteImage_ || dstMemory.isHostMemDirectAccess()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                         entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize;
    FindPinSize(pinSize, size, rowPitch, slicePitch, gpuMem(dstMemory));

    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force SW copy
      result = DmaBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                          entire, copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Get device memory for this virtual device
    Memory* srcMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyBufferToImage(*srcMemory, dstMemory, srcOrigin, origin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::BufferRect& srcRectIn,
                                       const amd::BufferRect& dstRectIn, const amd::Coord3D& sizeIn,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  bool rejected = false;

  // Fall into the ROC path for rejected transfers
  if (dev().info().pcie_atomics_ &&
      (setup_.disableCopyBufferRect_ || srcMemory.isHostMemDirectAccess() ||
       dstMemory.isHostMemDirectAccess())) {
    result = DmaBlitManager::copyBufferRect(srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn,
                                            entire, copyMetadata);

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
    bool aligned;
    // Check source alignments
    aligned = ((srcRectIn.rowPitch_ % CopyRectAlignment[i]) == 0);
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
  cl_mem mem = as_cl<amd::Memory>(srcMemory.owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstMemory.owner());
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
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  if (amd::IS_HIP) {
    // Update the command type for ROC profiler
    if (srcMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_WRITE_BUFFER_RECT);
    }
    if (dstMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_READ_BUFFER_RECT);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (dev().settings().blocking_blit_ &&
      (setup_.disableReadBuffer_ ||
       (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached()))) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t totalSize = size[0];
    // Do a staging copy
    bool useShaderCopyPath =
        setup_.disableHwlCopyBuffer_ || (totalSize <= dev().settings().sdmaCopyThreshold_) ||
        (copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::BLIT);

    if (!useShaderCopyPath) {
      // HSA copy using a staging resource
      result = DmaBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    }
    if (!result) {
      // Blit copy using a staging resource
      address srcAddr = gpuMem(srcMemory).getDeviceMemory();
      address dstAddr = reinterpret_cast<address>(dstHost);
      amd::Coord3D dstOrigin(0, 0, 0);
      size_t copySize = 0;
      size_t stagedCopyOffset = 0;
      constexpr bool kAttachSignal = true;

      while (totalSize > 0) {
        BufferState outBuffer = {0};
        constexpr bool kEnablePin = true;
        constexpr bool kFirstTx = false;
        getBuffer(static_cast<const_address>(dstAddr + stagedCopyOffset), totalSize, kEnablePin,
                  kFirstTx, outBuffer);
        copySize = outBuffer.copySize_;
        address stagingBuffer = outBuffer.buffer_;
        address currentSrcAddr = srcAddr + stagedCopyOffset;
        ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
                "Blit staging D2H copy stg buf=%p, src=%p, "
                "dstOrigin=0x%x, size=%zu, Async=%d",
                stagingBuffer, currentSrcAddr, dstOrigin[0], copySize, copyMetadata.isAsync_);
        // Flush caches for coherency after the copy as we need to std::memcpy
        // from staging buffer to unpinned dst. Also attach a signal to the dispatch packet
        // itself that we can wait on without extra barrier packet.
        gpu().addSystemScope();
        result =
            shaderCopyBuffer(stagingBuffer, currentSrcAddr, dstOrigin, origin, copySize, entire,
                             dev().settings().limit_blit_wg_, copyMetadata, kAttachSignal);
        if (!result) {
          break;
        }
        // Wait for current signal of previous blit copy if its not pinned mem
        if (outBuffer.pinnedMem_ == nullptr) {
          gpu().Barriers().WaitCurrent();
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "memcpy host dst=%p, stg buf=%p, size=%zu",
                  (void*)(dstAddr + stagedCopyOffset), stagingBuffer, copySize);
          memcpy(dstAddr + stagedCopyOffset, stagingBuffer, copySize);
        }
        totalSize -= copySize;
        stagedCopyOffset += copySize;
        // Release Pinned Memory back to pool
        releaseBuffer(outBuffer);
      }
    }
  }


  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                       const amd::BufferRect& bufRect,
                                       const amd::BufferRect& hostRect, const amd::Coord3D& size,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                             copyMetadata);
    synchronize();
    return result;
  } else {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Unpinned read rect path, Async = %d",
            copyMetadata.isAsync_);
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force SW copy
      result = DmaBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
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
    Memory* dstMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyBufferRect(srcMemory, *dstMemory, bufRect, rect, size, entire, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                    const amd::Coord3D& origin, const amd::Coord3D& size,
                                    bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (dev().settings().blocking_blit_ &&
      (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess() ||
       gpuMem(dstMemory).IsPersistentDirectMap())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t totalSize = size[0];
    // Do a staging copy
    bool useShaderCopyPath =
        setup_.disableHwlCopyBuffer_ || (totalSize <= dev().settings().sdmaCopyThreshold_) ||
        (copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::BLIT);

    if (!useShaderCopyPath) {
      // HSA copy using a staging resource
      result = DmaBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    }

    if (!result) {
      // Blit copy using a staging resource
      address dstAddr = gpuMem(dstMemory).getDeviceMemory();
      const_address srcAddr = reinterpret_cast<const_address>(srcHost);
      amd::Coord3D srcOrigin(0, 0, 0);
      size_t copySize = 0;
      size_t stagedCopyOffset = 0;

      while (totalSize > 0) {
        BufferState outBuffer = {0};
        // Disable pinned writes
        constexpr bool kEnablePin = false;
        constexpr bool kFirstTx = false;
        // Do not enable pinning for uploads. Always use staging buffer
        getBuffer(static_cast<const_address>(srcAddr + stagedCopyOffset), totalSize, kEnablePin,
                  kFirstTx, outBuffer);
        // Get an address from managed staging buffer
        address stagingBuffer = outBuffer.buffer_;
        copySize = outBuffer.copySize_;
        address currentDstAddr = dstAddr + stagedCopyOffset;
        if (outBuffer.pinnedMem_ == nullptr) {
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COPY, "memcpy stg buf=%p, host src=%p, size=%zu",
                  stagingBuffer, (void*)(srcAddr + stagedCopyOffset), copySize);
          memcpy(stagingBuffer, srcAddr + stagedCopyOffset, copySize);
        }
        ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
                "Blit staging H2D copy dst=%p, stg buf=%p, "
                "dstOrigin=0x%x, size=%zu, Async=%d",
                currentDstAddr, stagingBuffer, origin[0], copySize, copyMetadata.isAsync_);
        bool kAttachSignal = false;
        if (copyMetadata.isAsync_ == false) {
          // If its a blocking call, attach signal to the packet which we can track for
          // completion. Also flush caches as we may not need another packet to flush caches.
          kAttachSignal = true;
          gpu().addSystemScope();
        }
        result =
            shaderCopyBuffer(currentDstAddr, stagingBuffer, origin, srcOrigin, copySize, entire,
                             dev().settings().limit_blit_wg_, copyMetadata, kAttachSignal);
        if (!result) {
          break;
        }
        totalSize -= copySize;
        stagedCopyOffset += copySize;
        // Release pinned memory if any
        releaseBuffer(outBuffer);
      }
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                        const amd::BufferRect& hostRect,
                                        const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                        bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                              copyMetadata);
    synchronize();
    return result;
  } else {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Unpinned write rect path, Async = %d",
            copyMetadata.isAsync_);
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force DMA copy with staging
      result = DmaBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                               copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Get device memory for this virtual device
    Memory* srcMemory = dev().getRocMemory(amdMemory);

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

// ================================================================================================
bool KernelBlitManager::fillBuffer(device::Memory& memory, const void* pattern, size_t patternSize,
                                   const amd::Coord3D& surface, const amd::Coord3D& origin,
                                   const amd::Coord3D& size, bool entire, bool forceBlit) const {
  guarantee(size[0] > 0 && size[1] > 0 && size[2] > 0, "Dimension cannot be 0");

  if (size[1] == 1 && size[2] == 1) {
    return fillBuffer1D(memory, pattern, patternSize, surface, origin, size, entire, forceBlit);
  } else if (size[2] == 1) {
    return fillBuffer2D(memory, pattern, patternSize, surface, origin, size, entire, forceBlit);
  } else {
    bool ret_val = true;
    amd::Coord3D my_origin(origin);
    amd::Coord3D my_region{surface[1], surface[2], size[2]};
    amd::BufferRect rect;
    rect.create(static_cast<size_t*>(my_origin), static_cast<size_t*>(my_region), surface[0], 0);
    for (size_t slice = 0; slice < size[2]; ++slice) {
      const size_t row_offset = rect.offset(0, 0, slice);
      amd::Coord3D new_origin(row_offset, origin[1], origin[2]);
      ret_val |=
          fillBuffer2D(memory, pattern, patternSize, surface, new_origin, size, entire, forceBlit);
    }
    return ret_val;
  }
}

// ================================================================================================
bool KernelBlitManager::fillBuffer1D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host fill if memory has direct access
  if (setup_.disableFillBuffer_ || (!forceBlit && memory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
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
      size_t koffset = overall_offset;
      overall_offset += packed_obj.fill_size_;

      size_t globalWorkOffset[3] = {0, 0, 0};
      uint32_t alignment = (kpattern_size & 0xf) == 0   ? 2 * sizeof(uint64_t)
                           : (kpattern_size & 0x7) == 0 ? sizeof(uint64_t)
                           : (kpattern_size & 0x3) == 0 ? sizeof(uint32_t)
                           : (kpattern_size & 0x1) == 0 ? sizeof(uint16_t)
                                                        : sizeof(uint8_t);
      // Program kernels arguments for the fill operation
      cl_mem mem = as_cl<amd::Memory>(memory.owner());
      setArgument(kernels_[kFillType], 0, sizeof(cl_mem), &mem, koffset);
      const size_t localWorkSize = 256;
      size_t globalWorkSize = std::min(dev().settings().limit_blit_wg_ * localWorkSize, kfill_size);
      globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

      bool isGraphPktCapturing =
          gpu().command() != nullptr && gpu().command()->getPktCapturingState();
      auto constBuf = isGraphPktCapturing
                          ? gpu().command()->getGraphKernArg(kCBSize, kCBAlignment, dev().index())
                          : gpu().allocKernArg(kCBSize, kCBAlignment);

      // If pattern has been expanded, use the expanded pattern, otherwise use the default pattern.
      if (packed_obj.pattern_expanded_) {
        memcpy(constBuf, &packed_obj.expanded_pattern_, kpattern_size);
      } else {
        memcpy(constBuf, pattern, kpattern_size);
      }
      constexpr bool kDirectVa = true;
      setArgument(kernels_[kFillType], 1, sizeof(cl_mem), constBuf, 0, nullptr, kDirectVa);

      // Adjust the pattern size in the copy type size
      kpattern_size /= alignment;
      setArgument(kernels_[kFillType], 2, sizeof(uint32_t), &kpattern_size);
      setArgument(kernels_[kFillType], 3, sizeof(alignment), &alignment);

      // Calculate max id
      kfill_size = memory.virtualAddress() + koffset + kfill_size * kpattern_size * alignment;
      setArgument(kernels_[kFillType], 4, sizeof(kfill_size), &kfill_size);
      uint32_t next_chunk = globalWorkSize * kpattern_size;
      setArgument(kernels_[kFillType], 5, sizeof(uint32_t), &next_chunk);
      uint32_t lws = localWorkSize;
      setArgument(kernels_[kFillType], 6, sizeof(lws), &lws);

      // Create ND range object for the kernel's execution
      amd::NDRangeContainer ndrange(1, globalWorkOffset, &globalWorkSize, &localWorkSize);

      // Execute the blit
      address parameters = captureArguments(kernels_[kFillType]);
      result = gpu().submitKernelInternal(ndrange, *kernels_[kFillType], parameters, nullptr);
      releaseArguments(parameters);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillBuffer2D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host fill if memory has direct access
  if (setup_.disableFillBuffer_ || (!forceBlit && memory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::fillBuffer(memory, pattern, patternSize, size, origin, size, entire);
    synchronize();
    return result;
  } else {
    uint fillType = FillBufferAligned2D;
    uint64_t fillSizeX = (size[0] / patternSize) == 0 ? 1 : (size[0] / patternSize);
    uint64_t fillSizeY = size[1];

    size_t globalWorkOffset[3] = {0, 0, 0};
    size_t globalWorkSize[3] = {amd::alignUp(fillSizeX, 16), amd::alignUp(fillSizeY, 16), 1};
    size_t localWorkSize[3] = {16, 16, 1};

    uint32_t alignment = (patternSize & 0x7) == 0   ? sizeof(uint64_t)
                         : (patternSize & 0x3) == 0 ? sizeof(uint32_t)
                         : (patternSize & 0x1) == 0 ? sizeof(uint16_t)
                                                    : sizeof(uint8_t);

    cl_mem mem = as_cl<amd::Memory>(memory.owner());
    if (alignment == sizeof(uint64_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), &mem);
    } else if (alignment == sizeof(uint32_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    } else if (alignment == sizeof(uint16_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    } else {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    }

    // Get constant buffer to allow multipel fills
    bool isGraphPktCapturing =
        gpu().command() != nullptr && gpu().command()->getPktCapturingState();
    auto constBuf = isGraphPktCapturing
                        ? gpu().command()->getGraphKernArg(kCBSize, kCBAlignment, dev().index())
                        : gpu().allocKernArg(kCBSize, kCBAlignment);
    memcpy(constBuf, pattern, patternSize);

    constexpr bool kDirectVa = true;
    setArgument(kernels_[fillType], 4, sizeof(cl_mem), constBuf, 0, nullptr, kDirectVa);

    uint64_t mem_origin = static_cast<uint64_t>(origin[0]);
    uint64_t width = static_cast<uint64_t>(size[0]);
    uint64_t height = static_cast<uint64_t>(size[1]);
    uint64_t pitch = static_cast<uint64_t>(surface[0]);

    patternSize /= alignment;
    mem_origin /= alignment;
    pitch /= alignment;

    setArgument(kernels_[fillType], 5, sizeof(uint32_t), &patternSize);
    setArgument(kernels_[fillType], 6, sizeof(mem_origin), &mem_origin);
    setArgument(kernels_[fillType], 7, sizeof(width), &width);
    setArgument(kernels_[fillType], 8, sizeof(height), &height);
    setArgument(kernels_[fillType], 9, sizeof(pitch), &pitch);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(2, globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = captureArguments(kernels_[fillType]);
    result = gpu().submitKernelInternal(ndrange, *kernels_[fillType], parameters, nullptr);
    releaseArguments(parameters);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillBuffer3D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {
  ShouldNotReachHere();
  return false;
}

// ================================================================================================
bool KernelBlitManager::shaderCopyBuffer(address dst, address src, const amd::Coord3D& dstOrigin,
                                         const amd::Coord3D& srcOrigin, const amd::Coord3D& sizeIn,
                                         bool entire, const uint32_t blitWg,
                                         amd::CopyMetadata copyMetadata, bool attachSignal) const {
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
  size_t globalWorkSize = std::min(blitWg * localWorkSize, size[0]);
  globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

  // Program kernels arguments for the blit operation
  // Program source origin
  setArgument(kernels_[kBlitType], 0, sizeof(src), reinterpret_cast<void*>(src), srcOrigin[0],
              nullptr, true);

  // Program destinaiton origin
  setArgument(kernels_[kBlitType], 1, sizeof(dst), reinterpret_cast<void*>(dst), dstOrigin[0],
              nullptr, true);

  uint64_t copySize = sizeIn[0];
  setArgument(kernels_[kBlitType], 2, sizeof(copySize), &copySize);

  setArgument(kernels_[kBlitType], 3, sizeof(remainder), &remainder);
  setArgument(kernels_[kBlitType], 4, sizeof(aligned_size), &aligned_size);

  // End pointer is the aligned copy size and destination offset
  uint64_t end_ptr = reinterpret_cast<uint64_t>(dst) + dstOrigin[0] + sizeIn[0] - remainder;

  setArgument(kernels_[kBlitType], 5, sizeof(end_ptr), &end_ptr);

  uint32_t next_chunk = globalWorkSize;
  setArgument(kernels_[kBlitType], 6, sizeof(next_chunk), &next_chunk);
  uint32_t lws = localWorkSize;
  setArgument(kernels_[kBlitType], 7, sizeof(lws), &lws);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, nullptr, &globalWorkSize, &localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[kBlitType]);
  bool result = gpu().submitKernelInternal(ndrange, *kernels_[kBlitType], parameters, nullptr, 0,
                                           nullptr, nullptr, attachSignal);
  releaseArguments(parameters);

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                   const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                   const amd::Coord3D& sizeIn, bool entire,
                                   amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  bool p2p = false;
  uint32_t blitWg = dev().settings().limit_blit_wg_;

  if (&gpuMem(srcMemory).dev() != &gpuMem(dstMemory).dev()) {
    if (sizeIn[0] > dev().settings().sdma_p2p_threshold_) {
      p2p = true;
    } else {
      constexpr uint32_t kLimitWgForKernelP2p = 16;
      blitWg = kLimitWgForKernelP2p;
    }
  }

  // Determine if we should use shader copy path based on various conditions
  bool hwlCopyDisabled = setup_.disableHwlCopyBuffer_;

  // Check copy engine preferences
  bool isSdmaPreference =
      copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::SDMA;
  bool isBlitPreference =
      copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::BLIT;

  // Check memory access patterns
  bool isP2pOrIpc = p2p || srcMemory.owner()->ipcShared() || dstMemory.owner()->ipcShared();
  bool neitherMemoryIsHostDirectAccess =
      !srcMemory.isHostMemDirectAccess() && !dstMemory.isHostMemDirectAccess();

  // Determine shader copy path conditions
  bool smallSizeWithNonSdmaPreference =
      sizeIn[0] <= dev().settings().sdmaCopyThreshold_ && !isSdmaPreference;

  bool nonP2PIpcOrDirectAccess =
      !isP2pOrIpc && neitherMemoryIsHostDirectAccess && !isSdmaPreference;

  const bool useShaderCopyPath = hwlCopyDisabled || smallSizeWithNonSdmaPreference ||
                                 nonP2PIpcOrDirectAccess || isBlitPreference;

  if (!useShaderCopyPath) {
    if (amd::IS_HIP) {
      // Update the command type for ROC profiler
      if (srcMemory.isHostMemDirectAccess()) {
        gpu().SetCopyCommandType(CL_COMMAND_WRITE_BUFFER);
      }
      if (dstMemory.isHostMemDirectAccess()) {
        gpu().SetCopyCommandType(CL_COMMAND_READ_BUFFER);
      }
    }
    result = DmaBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire,
                                        copyMetadata);
  }

  if (!result) {
    // Check CL_MEM_SVM_ATOMICS flag to see if we used system_coarse_segment_
    auto memFlags = srcMemory.owner()->getMemFlags();
    bool srcSvmAtomics = (memFlags & CL_MEM_SVM_ATOMICS) != 0;
    if ((!srcSvmAtomics && srcMemory.isHostMemDirectAccess()) || (!copyMetadata.isAsync_)) {
      // Flush caches for coherency as the MTYPE of the src buffer is
      // non-coherent(ie read it again from memory).
      // For device to device copy(intra device), we dont need a flush.
      // If the source is host memory and the copy is blocking(aka memory need
      // to be coherent), then add system scope. For non blocking rely on the release
      // scope issued by synchronization packet.
      gpu().addSystemScope();
    }
    result =
        shaderCopyBuffer(reinterpret_cast<address>(dstMemory.virtualAddress()),
                         reinterpret_cast<address>(srcMemory.virtualAddress()), dstOrigin,
                         srcOrigin, sizeIn, entire, blitWg, copyMetadata, !copyMetadata.isAsync_);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillImage(device::Memory& memory, const void* pattern,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  bool entire) const {
  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  constexpr size_t kFillImageThreshold = 256 * 256;

  // Use host fill if memory has direct access and image is small
  if (setup_.disableFillImage_ || (gpuMem(memory).isHostMemDirectAccess() &&
                                   (size.c[0] * size.c[1] * size.c[2]) <= kFillImageThreshold)) {
    // Stall GPU before CPU access
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
  amd::Image* image = static_cast<amd::Image*>(memory.owner());
  amd::Image::Format newFormat(image->getImageFormat());
  bool swapLayer =
      (image->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

  // Program the kernels workload depending on the fill dimensions
  fillType = FillImage;
  dim = 3;

  void* newpattern = const_cast<void*>(pattern);
  uint32_t iFillColor[4];

  bool rejected = false;

  // For depth, we need to create a view
  if (newFormat.image_channel_order == CL_sRGBA) {
    // Find unsupported data type
    for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
      if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
        newFormat.image_channel_data_type = RejectedData[i].clNewType_;
        rejected = true;
        break;
      }
    }

    if (newFormat.image_channel_order == CL_sRGBA) {
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
    memView = createView(gpuMem(memory), newFormat, CL_MEM_WRITE_ONLY);
    if (memView != nullptr) {
      rejected = false;
    }
  }

  if (rejected) {
    return DmaBlitManager::fillImage(memory, pattern, origin, size, entire);
  }

  // Perform workload split to allow multiple operations in a single thread
  globalWorkSize[0] = (size[0] + TransferSplitSize - 1) / TransferSplitSize;
  // Find the current blit type
  if (image->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (image->getDims() == 2) {
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
  cl_mem mem = as_cl<amd::Memory>(memView->owner());
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
  address parameters = captureArguments(kernels_[fillType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[fillType], parameters, nullptr);
  releaseArguments(parameters);

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
  cl_mem mem = as_cl<amd::Memory>(memory.owner());
  bool is32BitWrite = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the write operation
  if (is32BitWrite) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 2, sizeof(uint32_t), &value);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
  }
  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);
  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
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
  cl_mem mem = as_cl<amd::Memory>(memory.owner());
  bool is32BitWait = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the wait operation
  if (is32BitWait) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 2, sizeof(uint32_t), &value);
    setArgument(kernels_[blitType], 3, sizeof(uint32_t), &flags);
    setArgument(kernels_[blitType], 4, sizeof(uint32_t), &mask);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
    setArgument(kernels_[blitType], 3, sizeof(uint64_t), &flags);
    setArgument(kernels_[blitType], 4, sizeof(uint64_t), &mask);
  }

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::batchMemOps(const void* paramArray, size_t paramSize,
                                    uint32_t count) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  uint blitType = BatchMemOp;
  size_t dim = 1;

  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {count};
  size_t localWorkSize[1] = {1};

  // Get constant buffer and copy the array of parameters
  constexpr bool kDirectVa = true;
  auto constBuf = gpu().allocKernArg((count * paramSize), kCBAlignment);
  memcpy(constBuf, paramArray, (count * paramSize));

  setArgument(kernels_[blitType], 0, sizeof(cl_mem), constBuf, 0, nullptr, kDirectVa);
  setArgument(kernels_[blitType], 1, sizeof(uint32_t), &count);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
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
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr, 0, nullptr,
                                      nullptr, true);
  releaseArguments(parameters);
  gpu().Barriers().WaitCurrent();
  return result;
}

// ================================================================================================

amd::Memory* DmaBlitManager::pinHostMemory(const void* hostMem, size_t pinSize,
                                           size_t& partial) const {
  size_t pinAllocSize;
  const static bool SysMem = true;
  amd::Memory* amdMemory;

  // Align offset to 4K boundary
  char* tmpHost = const_cast<char*>(
      amd::alignDown(reinterpret_cast<const char*>(hostMem), PinnedMemoryAlignment));

  // Find the partial size for unaligned copy
  partial = reinterpret_cast<const char*>(hostMem) - tmpHost;

  // Recalculate pin memory size
  pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);

  amdMemory = gpu().findPinnedMem(tmpHost, pinAllocSize);

  if (nullptr != amdMemory) {
    return amdMemory;
  }

  amdMemory = new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, pinAllocSize);
  amdMemory->setVirtualDevice(&gpu());
  if ((amdMemory != nullptr) && !amdMemory->create(tmpHost, SysMem)) {
    DevLogPrintfError("Buffer create failed, Buffer: 0x%x \n", amdMemory);
    amdMemory->release();
    return nullptr;
  }

  // Get device memory for this virtual device
  // @note: This will force real memory pinning
  Memory* srcMemory = dev().getRocMemory(amdMemory);

  if (srcMemory == nullptr) {
    // Release all pinned memory and attempt pinning again
    gpu().releasePinnedMem();
    srcMemory = dev().getRocMemory(amdMemory);
    if (srcMemory == nullptr) {
      // Release memory
      amdMemory->release();
      amdMemory = nullptr;
    }
  }

  return amdMemory;
}

Memory* KernelBlitManager::createView(const Memory& parent, cl_image_format format,
                                      cl_mem_flags flags) const {
  assert((parent.owner()->asBuffer() == nullptr) && "View supports images only");
  amd::Image* parentImage = static_cast<amd::Image*>(parent.owner());
  auto parent_dev_image = static_cast<Image*>(parentImage->getDeviceMemory(dev()));
  amd::Image* image = parent_dev_image->FindView(format);
  if (image == nullptr) {
    image = parentImage->createView(parent.owner()->getContext(), format, &gpu(), 0, flags, false,
                                    true);
    if (image == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }
    if (!parent_dev_image->AddView(image)) {
      // Another thread already added a view
      image->release();
      image = parent_dev_image->FindView(format);
    }
  }
  auto dev_image = static_cast<Image*>(image->getDeviceMemory(dev()));
  return dev_image;
}

address KernelBlitManager::captureArguments(const amd::Kernel* kernel) const {
  return kernel->parameters().values();
}

void KernelBlitManager::releaseArguments(address args) const {}

// ================================================================================================
bool KernelBlitManager::runScheduler(uint64_t vqVM, hsa_queue_t* schedulerQueue, uint threads,
                                     uint64_t aql_wrap) {
  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {threads};
  size_t localWorkSize[1] = {1};

  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);

  device::Kernel* devKernel =
      const_cast<device::Kernel*>(kernels_[Scheduler]->getDeviceKernel(dev()));

  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);

  auto* sp =
      reinterpret_cast<SchedulerParam*>(gpu().allocKernArg(sizeof(SchedulerParam), kCBAlignment));
  memset(sp, 0, sizeof(SchedulerParam));

  sp->kernarg_address = reinterpret_cast<uint64_t>(sp);
  sp->thread_counter = 0;
  sp->child_queue = reinterpret_cast<uint64_t>(schedulerQueue);
  sp->complete_signal = gpu().Barriers().ActiveSignal(kInitSignalValueOne, nullptr);
  sp->vqueue_header = vqVM;
  sp->parentAQL = aql_wrap;

  if (dev().info().maxEngineClockFrequency_ > 0) {
    sp->eng_clk = (1000 * 1024) / dev().info().maxEngineClockFrequency_;
  }

  if (!dev().info().pcie_atomics_) {
    // Use a device side global atomics to workaround the reliance of PCIe 3 atomics
    sp->write_index = Hsa::queue_load_write_index_relaxed(schedulerQueue);
  } else {
    sp->write_index = static_cast<uint64_t>(-1ULL);
  }

  constexpr bool kDirectVa = true;
  setArgument(kernels_[Scheduler], 0, sizeof(cl_mem), sp, 0, nullptr, kDirectVa);

  address parameters = captureArguments(kernels_[Scheduler]);

  if (!gpu().submitKernelInternal(ndrange, *kernels_[Scheduler], parameters, nullptr, 0, nullptr,
                                  &sp->scheduler_aql)) {
    return false;
  }
  releaseArguments(parameters);
  // Wait for the scheduler to finish all operations
  gpu().WaitCompleteSignal(sp->complete_signal);

  if (!dev().info().pcie_atomics_) {
    // @note: A wait shouldn't be really necessary, but the queue write_index may not get a proper
    // value without the wait for all previous commands (see the PCIE3 atomics workaround above).
    // The scheduler can enqueue extra commands, but the real queue write index didn't have any
    // progress. That leads to hangs and requires blocking. Then the wait causes problems
    // in DD mode with device enqueue and user events, because device enqueue is blocking below
    if (!WaitForSignal(sp->complete_signal)) {
      LogWarning("Failed schedulerSignal wait");
      return false;
    }
  }
  return true;
}

// ================================================================================================
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
  address parameters = captureArguments(kernels_[GwsInit]);

  bool result = gpu().submitKernelInternal(ndrange, *kernels_[GwsInit], parameters, nullptr);

  releaseArguments(parameters);

  return result;
}

}  // namespace amd::roc
