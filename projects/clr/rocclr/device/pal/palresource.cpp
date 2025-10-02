/* Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.

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

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "utils/flags.hpp"
#include "thread/monitor.hpp"
#include "device/pal/palresource.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/paltimestamp.hpp"
#include "hsa_ext_image.h"
#ifdef _WIN32
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#endif  // _WIN32
#include <GL/gl.h>
#include "GL/gl_interop.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace amd::pal {

// ================================================================================================
Pal::Result GpuMemoryReference::MakeResident() const {
  Pal::Result result = Pal::Result::Success;
  if (device_.settings().alwaysResident_) {
    Pal::GpuMemoryRef memRef = {};
    memRef.pGpuMemory = gpuMem_;
    result = device_.iDev()->AddGpuMemoryReferences(1, &memRef, nullptr, Pal::GpuMemoryRefCantTrim);
  }
  return result;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::GpuMemoryCreateInfo& createInfo) {
  Pal::Result result;
  size_t gpuMemSize = dev.iDev()->GetGpuMemorySize(createInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  if (memRef != nullptr) {
    result = dev.iDev()->CreateGpuMemory(createInfo, &memRef[1], &memRef->gpuMem_);
    if ((result != Pal::Result::Success) &&
        // Free cache if PAL failed allocation
        dev.resourceCache().free()) {
      // If cache was freed, then try to allocate again
      result = dev.iDev()->CreateGpuMemory(createInfo, &memRef[1], &memRef->gpuMem_);
    }
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  memRef->va_range_ = createInfo.flags.virtualAlloc;
  if (!createInfo.flags.sdiExternal && !createInfo.flags.virtualAlloc) {
    // Update free memory size counters
    dev.updateAllocedMemory(memRef->gpuMem_->Desc().heaps[0], memRef->gpuMem_->Desc().size, false);
  }
  return memRef;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::PinnedGpuMemoryCreateInfo& createInfo) {
  Pal::Result result;
  size_t gpuMemSize = dev.iDev()->GetPinnedGpuMemorySize(createInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  Pal::VaRange vaRange = Pal::VaRange::Default;
  if (memRef != nullptr) {
    result = dev.iDev()->CreatePinnedGpuMemory(createInfo, &memRef[1], &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  // Update free memory size counters
  dev.updateAllocedMemory(memRef->gpuMem_->Desc().heaps[0], memRef->gpuMem_->Desc().size, false);
  return memRef;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::SvmGpuMemoryCreateInfo& createInfo) {
  Pal::Result result;
  size_t gpuMemSize = dev.iDev()->GetSvmGpuMemorySize(createInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  if (memRef != nullptr) {
    result = dev.iDev()->CreateSvmGpuMemory(createInfo, &memRef[1], &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  // Update free memory size counters
  dev.updateAllocedMemory(memRef->gpuMem_->Desc().heaps[0], memRef->gpuMem_->Desc().size, false);
  return memRef;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::ExternalGpuMemoryOpenInfo& openInfo) {
  Pal::Result result;
  size_t gpuMemSize = dev.iDev()->GetExternalSharedGpuMemorySize(&result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  Pal::GpuMemoryCreateInfo createInfo = {};
  if (amd::IS_HIP) {  // creating svm buffer in case of interop with HIP
    createInfo.vaRange = Pal::VaRange::Svm;
  }
  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  if (memRef != nullptr) {
    result = dev.iDev()->OpenExternalSharedGpuMemory(openInfo, &memRef[1], &createInfo,
                                                     &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  return memRef;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::ExternalImageOpenInfo& openInfo,
                                               Pal::ImageCreateInfo* imgCreateInfo,
                                               Pal::IImage** image) {
  Pal::Result result;
  size_t gpuMemSize = 0;
  size_t imageSize = 0;
  if (Pal::Result::Success !=
      dev.iDev()->GetExternalSharedImageSizes(openInfo, &imageSize, &gpuMemSize, imgCreateInfo)) {
    return nullptr;
  }

  Pal::GpuMemoryCreateInfo createInfo = {};
  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  char* imgMem = new char[imageSize];
  if (memRef != nullptr) {
    result = dev.iDev()->OpenExternalSharedImage(openInfo, imgMem, &memRef[1], &createInfo, image,
                                                 &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  return memRef;
}

// ================================================================================================
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::PeerGpuMemoryOpenInfo& openInfo) {
  Pal::Result result;
  size_t gpuMemSize = dev.iDev()->GetPeerGpuMemorySize(openInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  if (memRef != nullptr) {
    result = dev.iDev()->OpenPeerGpuMemory(openInfo, &memRef[1], &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  return memRef;
}
GpuMemoryReference* GpuMemoryReference::Create(const Device& dev,
                                               const Pal::PeerImageOpenInfo& openInfo,
                                               Pal::IImage** image) {
  Pal::Result result;
  size_t imageSize = 0;
  size_t gpuMemSize = 0;
  dev.iDev()->GetPeerImageSizes(openInfo, &imageSize, &gpuMemSize, &result);
  if (Pal::Result::Success != result) {
    return nullptr;
  }

  char* imgMem = new char[imageSize];
  GpuMemoryReference* memRef = new (gpuMemSize) GpuMemoryReference(dev);
  if (memRef != nullptr) {
    result = dev.iDev()->OpenPeerImage(openInfo, imgMem, &memRef[1], image, &memRef->gpuMem_);
    if (result == Pal::Result::Success) {
      result = memRef->MakeResident();
    }
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }
  return memRef;
}
// ================================================================================================
GpuMemoryReference::GpuMemoryReference(const Device& dev)
    : gpuMem_(nullptr), cpuAddress_(nullptr), device_(dev), gpu_(nullptr) {}

// ================================================================================================
GpuMemoryReference::~GpuMemoryReference() {
  if (nullptr == iMem()) {
    return;
  }
  // Memory tracking per queue is disabled if alwaysResident is enabled. Thus, runtime can skip
  // updating residency state per every queue
  if (!device_.settings().alwaysResident_) {
    if (gpu_ == nullptr) {
      Device::ScopedLockVgpus lock(device_);
      // Release all memory objects on all virtual GPUs
      for (uint idx = 1; idx < device_.vgpus().size(); ++idx) {
        device_.vgpus()[idx]->releaseMemory(this);
      }
    } else {
      amd::ScopedLock l(gpu_->execution());
      gpu_->releaseMemory(this);
    }
    if (device_.vgpus().size() != 0) {
      assert(device_.vgpus()[0] == device_.xferQueue() && "Wrong transfer queue!");
      // Lock the transfer queue, since it's not handled by ScopedLockVgpus
      amd::ScopedLock k(device_.xferMgr().lockXfer());
      device_.vgpus()[0]->releaseMemory(this);
    }
  }
  // Destroy PAL object if it's not a suballocation
  if (cpuAddress_ != nullptr) {
    iMem()->Unmap();
  }
  if (!(iMem()->Desc().flags.isShared || iMem()->Desc().flags.isExternal ||
        iMem()->Desc().flags.isExternPhys || va_range_)) {
    // Update free memory size counters
    device_.updateAllocedMemory(iMem()->Desc().heaps[0], iMem()->Desc().size, true);
  }
  iMem()->Destroy();
  gpuMem_ = nullptr;
}

// ================================================================================================
Resource::Resource(const Device& gpuDev, size_t size)
    : elementSize_(0),
      gpuDevice_(gpuDev),
      mapCount_(0),
      address_(nullptr),
      offset_(0),
      memRef_(nullptr),
      subOffset_(0),
      viewOwner_(nullptr),
      image_(nullptr),
      hwSrd_(0),
      events_(gpuDev.numOfVgpus()) {
  // Fill resource descriptor fields
  desc_.state_ = 0;
  desc_.type_ = Empty;
  desc_.width_ = amd::alignUp(size, Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint)) /
                 Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint);
  desc_.height_ = 1;
  desc_.depth_ = 1;
  desc_.mipLevels_ = 1;
  desc_.format_.image_channel_order = CL_R;
  desc_.format_.image_channel_data_type = CL_FLOAT;
  desc_.flags_ = 0;
  desc_.pitch_ = 0;
  desc_.slice_ = 0;
  desc_.cardMemory_ = true;
  desc_.dimSize_ = 1;
  desc_.buffer_ = true;
  desc_.imageArray_ = false;
  desc_.topology_ = CL_MEM_OBJECT_BUFFER;
  desc_.SVMRes_ = false;
  desc_.scratch_ = false;
  desc_.isAllocExecute_ = false;
  desc_.baseLevel_ = 0;
  desc_.gl2CacheDisabled_ = false;
  gpuDev.addResource(this);
}

// ================================================================================================
Resource::Resource(const Device& gpuDev, size_t width, size_t height, size_t depth,
                   cl_image_format format, cl_mem_object_type imageType, uint mipLevels)
    : elementSize_(0),
      gpuDevice_(gpuDev),
      mapCount_(0),
      address_(nullptr),
      offset_(0),
      memRef_(nullptr),
      subOffset_(0),
      viewOwner_(nullptr),
      image_(nullptr),
      hwSrd_(0),
      events_(gpuDev.numOfVgpus()) {
  // Fill resource descriptor fields
  desc_.state_ = 0;
  desc_.type_ = Empty;
  desc_.width_ = width;
  desc_.height_ = height;
  desc_.depth_ = depth;
  desc_.mipLevels_ = mipLevels;
  desc_.format_ = format;
  desc_.flags_ = 0;
  desc_.pitch_ = 0;
  desc_.slice_ = 0;
  desc_.cardMemory_ = true;
  desc_.buffer_ = false;
  desc_.imageArray_ = false;
  desc_.topology_ = imageType;
  desc_.SVMRes_ = false;
  desc_.scratch_ = false;
  desc_.isAllocExecute_ = false;
  desc_.baseLevel_ = 0;
  desc_.gl2CacheDisabled_ = false;
  switch (imageType) {
    case CL_MEM_OBJECT_IMAGE2D:
      desc_.dimSize_ = 2;
      break;
    case CL_MEM_OBJECT_IMAGE3D:
      desc_.dimSize_ = 3;
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      desc_.dimSize_ = 3;
      desc_.imageArray_ = true;
      break;
    case CL_MEM_OBJECT_IMAGE1D:
      desc_.dimSize_ = 1;
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      desc_.dimSize_ = 2;
      desc_.imageArray_ = true;
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      desc_.dimSize_ = 1;
      break;
    default:
      desc_.dimSize_ = 1;
      LogError("Unknown image type!");
      break;
  }
  gpuDev.addResource(this);
}

// ================================================================================================
Resource::~Resource() {
  free();

  if ((nullptr != image_) &&
      ((memoryType() != ImageView) ||
       //! @todo PAL doesn't allow an SRD view creation with different pixel size
       (elementSize() != viewOwner_->elementSize()))) {
    image_->Destroy();
    delete[] reinterpret_cast<char*>(image_);
  }

  // Remove the current resource from the global resource list
  gpuDevice_.removeResource(this);
}

// ================================================================================================
static uint32_t GetHSAILImageFormatType(const cl_image_format& format) {
  static const uint32_t FormatType[] = {HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT,
                                        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24};

  uint idx = format.image_channel_data_type - CL_SNORM_INT8;
  assert((idx <= (CL_UNORM_INT24 - CL_SNORM_INT8)) && "Out of range format channel!");
  return FormatType[idx];
}

// ================================================================================================
static uint32_t GetHSAILImageOrderType(const cl_image_format& format) {
  static const uint32_t OrderType[] = {HSA_EXT_IMAGE_CHANNEL_ORDER_R,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_A,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RG,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RA,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RGB,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RX,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RGX,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA,
                                       HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR};

  uint idx = format.image_channel_order - CL_R;
  assert((idx <= (CL_ABGR - CL_R)) && "Out of range format order!");
  return OrderType[idx];
}

// ================================================================================================
void Resource::memTypeToHeap(Pal::GpuMemoryCreateInfo* createInfo) {
  createInfo->heapCount = 1;
  switch (memoryType()) {
    case Persistent:
      createInfo->heapCount = 2;
      createInfo->heaps[0] = Pal::GpuHeapLocal;
      createInfo->heaps[1] = Pal::GpuHeapGartUswc;
      createInfo->flags.peerWritable = dev().P2PAccessAllowed();
#ifdef ATI_OS_LINUX
      // Note: SSG in Linux requires DGMA heap
      if (dev().properties().gpuMemoryProperties.busAddressableMemSize > 0) {
        createInfo->flags.busAddressable = true;
      }
#endif
      break;
    case RemoteUSWC:
      createInfo->heaps[0] = Pal::GpuHeapGartUswc;
      desc_.cardMemory_ = false;
      break;
    case Remote:
      createInfo->heaps[0] = Pal::GpuHeapGartCacheable;
      desc_.cardMemory_ = false;
      break;
    case ExternalPhysical:
      desc_.cardMemory_ = false;
    case Shader:
    // Fall through to process the memory allocation ...
    case Local:
      createInfo->heapCount = 3;
      createInfo->heaps[0] = Pal::GpuHeapInvisible;
      createInfo->heaps[1] = Pal::GpuHeapLocal;
      createInfo->heaps[2] = Pal::GpuHeapGartUswc;
      createInfo->flags.peerWritable = dev().P2PAccessAllowed();
      break;
    case VaRange:
      createInfo->heapCount = 0;
      break;
    default:
      createInfo->heaps[0] = Pal::GpuHeapLocal;
      break;
  }

  // Pick the appropriate mall policy based on the mem type
  switch (memoryType()) {
    case Local:
    case Scratch:
    case Persistent:
      createInfo->mallPolicy = static_cast<Pal::GpuMemMallPolicy>(dev().settings().mallPolicy_);
      break;
    default:
      createInfo->mallPolicy = Pal::GpuMemMallPolicy::Never;
      break;
  }
}

// ================================================================================================
bool Resource::CreateImage(CreateParams* params, bool forceLinear) {
  Pal::Result result;
  Pal::SubresId ImgSubresId = {0, 0, 0};
  Pal::SubresRange ImgSubresRange = {ImgSubresId, 1, 1, 1};
  Pal::ChannelMapping channels;
  Pal::ChNumFormat format = dev().getPalFormat(desc().format_, &channels);

  if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    if (memoryType() == ImageBuffer) {
      ImageBufferParams* imageBuffer = reinterpret_cast<ImageBufferParams*>(params);
      viewOwner_ = imageBuffer->resource_;
      memRef_ = viewOwner_->memRef_;
      memRef_->retain();
      desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
      offset_ += viewOwner_->offset_;
    } else {
      Pal::GpuMemoryCreateInfo createInfo = {};
      createInfo.size = desc().width_ * elementSize();
      createInfo.size = amd::alignUp(createInfo.size, MaxGpuAlignment);
      createInfo.alignment = MaxGpuAlignment;
      createInfo.vaRange = Pal::VaRange::Default;
      createInfo.priority = Pal::GpuMemPriority::Normal;
      memTypeToHeap(&createInfo);
      // createInfo.priority;
      memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment,
                                                    nullptr, &subOffset_);
      if (nullptr == memRef_) {
        memRef_ = GpuMemoryReference::Create(dev(), createInfo);
        if (nullptr == memRef_) {
          LogError("Failed PAL memory allocation!");
          return false;
        }
      }
      offset_ += static_cast<size_t>(subOffset_);
    }
    // Check if memory is locked already and restore CPU pointer
    if (memRef_->cpuAddress_ != nullptr) {
      address_ = memRef_->cpuAddress_;
      memRef_->cpuAddress_ = nullptr;
      mapCount_++;
    }
    Pal::BufferViewInfo viewInfo = {};
    viewInfo.gpuAddr = vmAddress();
    viewInfo.range = memRef_->iMem()->Desc().size;
    viewInfo.stride = elementSize();
    viewInfo.swizzledFormat.format = format;
    viewInfo.swizzledFormat.swizzle = channels;
    // viewInfo.channels = channels;
    hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
    if ((0 == hwSrd_) && (memoryType() != ImageView)) {
      return false;
    }

    dev().iDev()->CreateTypedBufferViewSrds(1, &viewInfo, hwState_);
    hwState_[8] = GetHSAILImageFormatType(desc().format_);
    hwState_[9] = GetHSAILImageOrderType(desc().format_);
    hwState_[10] = static_cast<uint32_t>(desc().width_);
    hwState_[11] = 0;  // one extra reserved field in the argument
    return true;
  }

  Pal::ImageViewInfo viewInfo = {};
  Pal::ImageCreateInfo imgCreateInfo = {};
  Pal::GpuMemoryRequirements req = {};
  imgCreateInfo.imageType = Pal::ImageType::Tex2d;
  viewInfo.viewType = Pal::ImageViewType::Tex2d;
  viewInfo.possibleLayouts.engines = Pal::LayoutComputeEngine | Pal::LayoutDmaEngine;
  viewInfo.possibleLayouts.usages = Pal::LayoutShaderWrite;
  imgCreateInfo.extent.width = desc_.width_;
  imgCreateInfo.extent.height = desc_.height_;
  imgCreateInfo.extent.depth = desc_.depth_;
  imgCreateInfo.arraySize = 1;

  switch (desc_.topology_) {
    case CL_MEM_OBJECT_IMAGE3D:
      imgCreateInfo.imageType = Pal::ImageType::Tex3d;
      viewInfo.viewType = Pal::ImageViewType::Tex3d;
      break;
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      imgCreateInfo.imageType = Pal::ImageType::Tex1d;
      viewInfo.viewType = Pal::ImageViewType::Tex1d;
      break;
  }
  if (desc_.topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    ImgSubresRange.numSlices = imgCreateInfo.arraySize = desc_.height_;
    imgCreateInfo.extent.depth = desc_.height_;
    imgCreateInfo.extent.height = 1;
  }
  if (desc_.topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
    ImgSubresRange.numSlices = imgCreateInfo.arraySize = desc_.depth_;
  }

  if (memoryType() == ImageView) {
    ImageViewParams* imageView = reinterpret_cast<ImageViewParams*>(params);
    ImgSubresRange.startSubres.mipLevel = imageView->level_;
    desc_.baseLevel_ = imageView->level_;
    ImgSubresRange.startSubres.arraySlice = imageView->layer_;
    viewOwner_ = imageView->resource_;
    image_ = viewOwner_->image_;
  } else if (memoryType() == ImageBuffer || memoryType() == ImageExternalBuffer) {
    ImageBufferParams* imageBuffer = reinterpret_cast<ImageBufferParams*>(params);
    viewOwner_ = imageBuffer->resource_;
  }
  if (nullptr != viewOwner_) {
    offset_ = viewOwner_->offset();
  }
  ImgSubresRange.numMips = desc().mipLevels_;

  if ((memoryType() != ImageView) ||
      //! @todo PAL doesn't allow an SRD view creation with different pixel size
      (elementSize() != viewOwner_->elementSize())) {
    imgCreateInfo.usageFlags.shaderRead = true;
    imgCreateInfo.usageFlags.shaderWrite =
        (format == Pal::ChNumFormat::X8Y8Z8W8_Srgb) ? false : true;
    imgCreateInfo.swizzledFormat.format = format;
    imgCreateInfo.swizzledFormat.swizzle = channels;
    imgCreateInfo.mipLevels = (desc_.mipLevels_) ? desc_.mipLevels_ : 1;
    imgCreateInfo.samples = 1;
    imgCreateInfo.fragments = 1;
    Pal::ImageTiling tiling = forceLinear ? Pal::ImageTiling::Linear : Pal::ImageTiling::Optimal;
    uint32_t rowPitch = 0;

    if (memoryType() == ImageBuffer) {
      tiling = Pal::ImageTiling::Linear;
    } else if (memoryType() == ImageExternalBuffer) {
      // We cannot get tiling info from vulkan/d3d driver now. So assume it to be optimal.
      // When we get tiling info, we can easily update here
      tiling = Pal::ImageTiling::Optimal;
      // Pal will infer row pitch. If rowPitch != 0 and Pal inferred row picth isn't equal to
      // rowPitch, assert(false) will be called
      rowPitch = 0;
      offset_ += params->owner_->getOrigin();
    } else if (memoryType() == ImageView) {
      tiling = viewOwner_->image_->GetImageCreateInfo().tiling;
      // Find the new pitch in pixels for the new format
      rowPitch = viewOwner_->desc().pitch_ * viewOwner_->elementSize() / elementSize();
    }

    if (memoryType() == ImageBuffer) {
      if ((params->owner_ != NULL) && params->owner_->asImage() &&
          (params->owner_->asImage()->getRowPitch() != 0)) {
        rowPitch = params->owner_->asImage()->getRowPitch() / elementSize();
      } else {
        rowPitch = desc().width_;
      }
    }
    desc_.pitch_ = rowPitch;
    // Make sure the row pitch is aligned to pixels
    imgCreateInfo.rowPitch =
        amd::alignUp(elementSize() * rowPitch, dev().info().imagePitchAlignment_);
    imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
    imgCreateInfo.tiling = tiling;

    size_t imageSize = dev().iDev()->GetImageSize(imgCreateInfo, &result);
    if (result != Pal::Result::Success) {
      return false;
    }

    char* memImg = new char[imageSize];
    if (memImg != nullptr) {
      result = dev().iDev()->CreateImage(imgCreateInfo, memImg, &image_);
      if (result != Pal::Result::Success) {
        delete[] memImg;
        return false;
      }
    }
    image_->GetGpuMemoryRequirements(&req);
    // createInfo.priority;
  }

  if ((memoryType() != ImageView) && (memoryType() != ImageBuffer) &&
      (memoryType() != ImageExternalBuffer)) {
    Pal::GpuMemoryCreateInfo createInfo = {};
    createInfo.size = amd::alignUp(req.size, MaxGpuAlignment);
    createInfo.alignment = std::max(req.alignment, MaxGpuAlignment);
    createInfo.vaRange = Pal::VaRange::Default;
    createInfo.priority = Pal::GpuMemPriority::Normal;
    memTypeToHeap(&createInfo);

    memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment,
                                                  nullptr, &subOffset_);
    if (nullptr == memRef_) {
      memRef_ = GpuMemoryReference::Create(dev(), createInfo);
      if (nullptr == memRef_) {
        LogError("Failed PAL memory allocation!");
        return false;
      }
    }
    offset_ += static_cast<size_t>(subOffset_);
  } else {
    memRef_ = viewOwner_->memRef_;
    memRef_->retain();
    desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
    if (req.size > viewOwner_->iMem()->Desc().size) {
      LogWarning("Image is bigger than the original mem object!");
    }
  }
  // Check if memory is locked already and restore CPU pointer
  if (memRef_->cpuAddress_ != nullptr) {
    address_ = memRef_->cpuAddress_;
    memRef_->cpuAddress_ = nullptr;
    mapCount_++;
  }
  result = image_->BindGpuMemory(memRef_->gpuMem_, offset_);

  if (result != Pal::Result::Success) {
    LogPrintfError(
        "BindGpuMemory return %d, offset_=%zu, req.size=%zu,  "
        "viewOwner_->iMem()->Desc().size=%zu\n",
        result, offset_, req.size,
        viewOwner_ && viewOwner_->iMem() ? viewOwner_->iMem()->Desc().size : 0);
    return false;
  }

  hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
  if ((0 == hwSrd_) && (memoryType() != ImageView)) {
    return false;
  }
  viewInfo.pImage = image_;
  viewInfo.swizzledFormat.format = format;
  viewInfo.swizzledFormat.swizzle = channels;
  viewInfo.subresRange = ImgSubresRange;
  dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);

  hwState_[8] = GetHSAILImageFormatType(desc().format_);
  hwState_[9] = GetHSAILImageOrderType(desc().format_);
  hwState_[10] = static_cast<uint32_t>(desc().width_);
  hwState_[11] = 0;  // one extra reserved field in the argument

  desc_.tiled_ = (Pal::ImageTiling::Linear == image_->GetImageCreateInfo().tiling) ? false : true;
  return true;
}

// ================================================================================================
bool Resource::CreateInterop(CreateParams* params) {
  Pal::Result result;
  Pal::SubresId ImgSubresId = {0, 0, 0};
  Pal::SubresRange ImgSubresRange = {ImgSubresId, 1, 1, 1};
  Pal::ChannelMapping channels;
  Pal::ChNumFormat format = dev().getPalFormat(desc().format_, &channels);
  Pal::ExternalGpuMemoryOpenInfo gpuMemOpenInfo = {};
  Pal::ExternalResourceOpenInfo& openInfo = gpuMemOpenInfo.resourceInfo;
  uint misc = 0;
  uint layer = 0;
  uint mipLevel = 0;
  InteropType type = InteropTypeless;

  if (memoryType() == OGLInterop) {
    OGLInteropParams* oglRes = reinterpret_cast<OGLInteropParams*>(params);
    assert(oglRes->glPlatformContext_ && "We don't have OGL context!");
    switch (oglRes->type_) {
      case InteropVertexBuffer:
        glType_ = GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD;
        break;
      case InteropRenderBuffer:
        glType_ = GL_RESOURCE_ATTACH_RENDERBUFFER_AMD;
        break;
      case InteropTexture:
      case InteropTextureViewLevel:
      case InteropTextureViewCube:
        glType_ = GL_RESOURCE_ATTACH_TEXTURE_AMD;
        break;
      default:
        LogError("Unknown OGL interop type!");
        return false;
        break;
    }
    glPlatformContext_ = oglRes->glPlatformContext_;
    layer = oglRes->layer_;
    type = oglRes->type_;
    mipLevel = oglRes->mipLevel_;

    if (!dev().resGLAssociate(oglRes->glPlatformContext_, oglRes->handle_, glType_,
                              &openInfo.hExternalResource, &glInteropMbRes_, &offset_, desc_.format_
#if IS_WINDOWS
                              ,
                              openInfo.doppDesktopInfo
#endif
                              )) {
      return false;
    }
    desc_.isDoppTexture_ = (openInfo.doppDesktopInfo.gpuVirtAddr != 0);
    openInfo.flags.isDopp = desc_.isDoppTexture_;
    format = dev().getPalFormat(desc().format_, &channels);
  } else if (memoryType() == VkInterop) {
    VkInteropParams* vparams = reinterpret_cast<VkInteropParams*>(params);
    if (vparams->handle_) {
      openInfo.hExternalResource = vparams->handle_;
    } else if (vparams->name_) {
#if IS_WINDOWS
      Pal::ExternalHandleInfo eHandleInfo = {};
      eHandleInfo.objectType = Pal::ExternalObjectType::Allocation;
      eHandleInfo.pNtObjectName = reinterpret_cast<const wchar_t*>(vparams->name_);
      SECURITY_ATTRIBUTES securityAttributes = {};
      securityAttributes.bInheritHandle = TRUE;
      eHandleInfo.pSecurityAttributes = &securityAttributes;
      eHandleInfo.accessFlags = GENERIC_READ | GENERIC_WRITE;
      if (Pal::Result::Success !=
          dev().iDev()->OpenExternalHandleFromName(eHandleInfo, &openInfo.hExternalResource)) {
        return false;
      }
#else
      return false;
#endif
    }
    openInfo.flags.ntHandle = vparams->nt_handle_;
  }
#if IS_WINDOWS
  else {
    D3DInteropParams* d3dRes = reinterpret_cast<D3DInteropParams*>(params);
    openInfo.hExternalResource = d3dRes->handle_;
    misc = d3dRes->misc;
    layer = d3dRes->layer_;
    type = d3dRes->type_;
    mipLevel = d3dRes->mipLevel_;
  }
#endif
  //! @todo PAL query for image/buffer object doesn't work properly!
#if 0
  bool    isImage = false;
  if (Pal::Result::Success !=
    dev().iDev()->DetermineExternalSharedResourceType(openInfo, &isImage)) {
    return false;
  }
#endif  // 0
  if (desc().buffer_ || misc) {
    memRef_ = GpuMemoryReference::Create(dev(), gpuMemOpenInfo);
    if (nullptr == memRef_) {
      return false;
    }

    if (misc) {
      Pal::ImageCreateInfo imgCreateInfo = {};
      Pal::ExternalImageOpenInfo imgOpenInfo = {};
      imgOpenInfo.resourceInfo = openInfo;
      imgOpenInfo.swizzledFormat.format = format;
      imgOpenInfo.swizzledFormat.swizzle = channels;
      imgOpenInfo.usage.shaderRead = true;
      imgOpenInfo.usage.shaderWrite = true;
      size_t imageSize;
      size_t gpuMemSize;

      if (Pal::Result::Success != dev().iDev()->GetExternalSharedImageSizes(
                                      imgOpenInfo, &imageSize, &gpuMemSize, &imgCreateInfo)) {
        return false;
      }

      Pal::gpusize viewOffset = 0;
      imgCreateInfo.flags.shareable = false;
      imgCreateInfo.imageType = Pal::ImageType::Tex2d;
      imgCreateInfo.extent.width = desc().width_;
      imgCreateInfo.extent.height = desc().height_;
      imgCreateInfo.extent.depth = desc().depth_;
      imgCreateInfo.arraySize = 1;
      imgCreateInfo.usageFlags.shaderRead = true;
      imgCreateInfo.usageFlags.shaderWrite = true;
      imgCreateInfo.swizzledFormat.format = format;
      imgCreateInfo.swizzledFormat.swizzle = channels;
      imgCreateInfo.mipLevels = 1;
      imgCreateInfo.samples = 1;
      imgCreateInfo.fragments = 1;
      imgCreateInfo.tiling = Pal::ImageTiling::Linear;
      imgCreateInfo.depthPitch = desc().height_ * imgCreateInfo.rowPitch;

      switch (misc) {
        case 1:  // NV12 or P010 formats
          switch (layer) {
            case std::numeric_limits<uint>::max():
            case 0:
              break;
            case 1:
              // Y - plane size to the offset
              // NV12 format. UV is 2 times smaller plane Y
              viewOffset = 2 * imgCreateInfo.rowPitch * desc().height_;
              imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
              break;
            default:
              LogError("Unknown Interop View Type");
              return false;
          }
          break;
        case 2:  // YV12 format
          switch (layer) {
            case std::numeric_limits<uint>::max():
            case 0:
              break;
            case 1:
              // Y - plane size to the offset
              // YV12 format. U is 4 times smaller plane than Y
              viewOffset = 2 * imgCreateInfo.rowPitch * desc().height_;
              imgCreateInfo.rowPitch >>= 1;
              break;
            case 2:
              // Y + U plane sizes to the offest.
              // U plane is 4 times smaller than Y and U == V
              viewOffset = 5 * imgCreateInfo.rowPitch * desc().height_ / 2;
              imgCreateInfo.rowPitch >>= 1;
              break;
            default:
              LogError("Unknown Interop View Type");
              return false;
          }
          imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
          break;
        case 3:  // YUY2 format
          imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
          break;
        default:
          LogError("Unknown Interop View Type");
          return false;
      }

      imageSize = dev().iDev()->GetImageSize(imgCreateInfo, &result);
      if (result != Pal::Result::Success) {
        return false;
      }

      char* memImg = new char[imageSize];
      if (memImg != nullptr) {
        result = dev().iDev()->CreateImage(imgCreateInfo, memImg, &image_);
        if (result != Pal::Result::Success) {
          delete[] memImg;
          return false;
        }
      }
      offset_ += static_cast<size_t>(viewOffset);
      result = image_->BindGpuMemory(iMem(), offset_);
      if (result != Pal::Result::Success) {
        return false;
      }
      hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
      if ((0 == hwSrd_) && (memoryType() != ImageView)) {
        return false;
      }
      Pal::ImageViewInfo viewInfo = {};
      viewInfo.viewType = Pal::ImageViewType::Tex2d;
      viewInfo.pImage = image_;
      viewInfo.swizzledFormat.format = format;
      viewInfo.swizzledFormat.swizzle = channels;
      viewInfo.subresRange = ImgSubresRange;
      viewInfo.possibleLayouts.engines = Pal::LayoutComputeEngine | Pal::LayoutDmaEngine;
      viewInfo.possibleLayouts.usages = Pal::LayoutShaderWrite;
      dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);

      hwState_[8] = GetHSAILImageFormatType(desc().format_);
      hwState_[9] = GetHSAILImageOrderType(desc().format_);
      hwState_[10] = static_cast<uint32_t>(desc().width_);
      hwState_[11] = 0;  // one extra reserved field in the argument
    }
  } else if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    memRef_ = GpuMemoryReference::Create(dev(), gpuMemOpenInfo);
    if (nullptr == memRef_) {
      return false;
    }
    Pal::BufferViewInfo viewInfo = {};
    viewInfo.gpuAddr = vmAddress();
    viewInfo.range = memRef_->iMem()->Desc().size;
    viewInfo.stride = elementSize();
    viewInfo.swizzledFormat.format = format;
    viewInfo.swizzledFormat.swizzle = channels;
    hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
    if ((0 == hwSrd_) && (memoryType() != ImageView)) {
      return false;
    }

    dev().iDev()->CreateTypedBufferViewSrds(1, &viewInfo, hwState_);
    hwState_[8] = GetHSAILImageFormatType(desc().format_);
    hwState_[9] = GetHSAILImageOrderType(desc().format_);
    hwState_[10] = static_cast<uint32_t>(desc().width_);
    hwState_[11] = 0;  // one extra reserved field in the argument
  } else {
    Pal::ExternalImageOpenInfo imgOpenInfo = {};
    Pal::ImageCreateInfo imgCreateInfo = {};
    imgOpenInfo.resourceInfo = openInfo;
    imgOpenInfo.swizzledFormat.format = format;
    imgOpenInfo.swizzledFormat.swizzle = channels;
    imgOpenInfo.usage.shaderRead = true;
    imgOpenInfo.usage.shaderWrite = true;
#if defined(__unix__)
    imgOpenInfo.resourceInfo.handleType = Pal::HandleType::DmaBufFd;
#endif
    memRef_ = GpuMemoryReference::Create(dev(), imgOpenInfo, &imgCreateInfo, &image_);
    if (nullptr == memRef_) {
      return false;
    }

    hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
    if ((0 == hwSrd_) && (memoryType() != ImageView)) {
      return false;
    }
    Pal::ImageViewInfo viewInfo = {};
    viewInfo.possibleLayouts.engines = Pal::LayoutComputeEngine | Pal::LayoutDmaEngine;
    viewInfo.possibleLayouts.usages = Pal::LayoutShaderWrite;
    viewInfo.viewType = Pal::ImageViewType::Tex2d;
    switch (imgCreateInfo.imageType) {
      case Pal::ImageType::Tex3d:
        viewInfo.viewType = Pal::ImageViewType::Tex3d;
        break;
      case Pal::ImageType::Tex1d:
        viewInfo.viewType = Pal::ImageViewType::Tex1d;
        break;
      default:
        break;
    }
    viewInfo.pImage = image_;
    viewInfo.swizzledFormat.format = format;
    viewInfo.swizzledFormat.swizzle = channels;
    if ((type == InteropTextureViewLevel) || (type == InteropTextureViewCube)) {
      ImgSubresRange.startSubres.mipLevel = mipLevel;
      if (type == InteropTextureViewCube) {
        ImgSubresRange.startSubres.arraySlice = layer;
        viewInfo.viewType = Pal::ImageViewType::Tex2d;
      }
    }
    if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
      ImgSubresRange.numSlices = desc_.height_;
    }
    if (desc().topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
      ImgSubresRange.numSlices = desc_.depth_;
    }
    ImgSubresRange.numMips = desc().mipLevels_;
    viewInfo.subresRange = ImgSubresRange;

    dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);
    //! It's a workaround for D24S8 format, since PAL doesn't support this format
    //! and OGL decompresses 24bit DEPTH into D24S8 for OGL compatibility
    if ((desc().format_.image_channel_order == CL_DEPTH_STENCIL) &&
        (desc().format_.image_channel_data_type == CL_UNORM_INT24)) {
      hwState_[1] = (hwState_[1] & ~0x1ff00000) | 0x08d00000;
    }
    hwState_[8] = GetHSAILImageFormatType(desc().format_);
    hwState_[9] = GetHSAILImageOrderType(desc().format_);
    hwState_[10] = static_cast<uint32_t>(desc().width_);
    hwState_[11] = 0;  // one extra reserved field in the argument
  }
  return true;
}

// ================================================================================================
bool Resource::CreateIpc(CreateParams* params) {
  Pal::ExternalGpuMemoryOpenInfo gpuMemOpenInfo = {};
  Pal::ExternalResourceOpenInfo& openInfo = gpuMemOpenInfo.resourceInfo;

  openInfo.hExternalResource = *reinterpret_cast<const Pal::OsExternalHandle*>(
      reinterpret_cast<amd::IpcBuffer*>(params->owner_)->Handle());
  openInfo.flags.ntHandle = false;

  memRef_ = GpuMemoryReference::Create(dev(), gpuMemOpenInfo);
  if (nullptr == memRef_) {
    return false;
  }
  offset_ += params->owner_->getOffset();
  params->owner_->setSvmPtr(reinterpret_cast<void*>(memRef_->iMem()->Desc().gpuVirtAddr + offset_));
  return true;
}

// ================================================================================================
bool Resource::CreateP2PAccess(CreateParams* params) {
  if (params->owner_->asImage()) {
    Pal::PeerImageOpenInfo openInfo = {};
    openInfo.pOriginalImage = params->svmBase_->image();
    memRef_ = GpuMemoryReference::Create(dev(), openInfo, &image_);
  } else {
    Pal::PeerGpuMemoryOpenInfo openInfo = {};
    openInfo.pOriginalMem = params->svmBase_->iMem();
    memRef_ = GpuMemoryReference::Create(dev(), openInfo);
  }
  if (nullptr == memRef_) {
    return false;
  }
  desc_.cardMemory_ = false;
  offset_ = params->svmBase_->offset();
  return true;
}

// ================================================================================================
bool Resource::CreatePinned(CreateParams* params) {
  PinnedParams* pinned = reinterpret_cast<PinnedParams*>(params);
  size_t allocSize = pinned->size_;
  const amd::HostMemoryReference* hostMemRef = pinned->hostMemRef_;
  void* pinAddress = address_ = hostMemRef->hostMem();
  uint hostMemOffset = 0;
  // assert((allocSize == (desc().width_ * elementSize())) && "Sizes don't match");
  if (desc().topology_ == CL_MEM_OBJECT_BUFFER) {
    // Allign offset to 4K boundary (Vista/Win7 limitation)
    char* tmpHost = const_cast<char*>(
        amd::alignDown(reinterpret_cast<const char*>(address_), PinnedMemoryAlignment));

    // Find the partial size for unaligned copy
    hostMemOffset = static_cast<uint>(reinterpret_cast<const char*>(address_) - tmpHost);

    offset_ = hostMemOffset;

    pinAddress = tmpHost;

    if (hostMemOffset != 0) {
      allocSize += hostMemOffset;
    }
    allocSize = amd::alignUp(allocSize, PinnedMemoryAlignment);
    //            hostMemOffset &= ~(0xff);
  } else if (desc().topology_ == CL_MEM_OBJECT_IMAGE2D) {
    //! @todo: Width has to be aligned for 3D.
    //! Need to be replaced with a compute copy
    // Width aligned by 8 texels
    if (((desc().width_ % 0x8) != 0) ||
        // Pitch aligned by 64 bytes
        (((desc().width_ * elementSize()) % 0x40) != 0)) {
      return false;
    }
  } else {
    //! @todo GSL doesn't support pinning with resAlloc_
    return false;
  }

  if (dev().settings().svmFineGrainSystem_) {
    desc_.SVMRes_ = true;
  }

  // Ensure page alignment
  if ((uint64_t)(pinAddress) & (amd::Os::pageSize() - 1)) {
    return false;
  }
  Pal::PinnedGpuMemoryCreateInfo createInfo = {};
  createInfo.pSysMem = pinAddress;
  createInfo.size = allocSize;
  createInfo.vaRange = Pal::VaRange::Default;
  memRef_ = GpuMemoryReference::Create(dev(), createInfo);
  if (nullptr == memRef_) {
    LogError("Failed PAL memory allocation!");
    return false;
  }
  desc_.cardMemory_ = false;
  return true;
}

// ================================================================================================
bool Resource::CreateSvm(CreateParams* params, Pal::gpusize svmPtr) {
  const bool isFineGrain = (memoryType() == RemoteUSWC) || (memoryType() == Remote);
  size_t allocSize = amd::alignUp(desc().width_ * elementSize_, MaxGpuAlignment);
  if (isFineGrain) {
    Pal::SvmGpuMemoryCreateInfo createInfo = {};
    createInfo.isUsedForKernel = desc_.isAllocExecute_;
    createInfo.size = allocSize;
    createInfo.alignment = MaxGpuAlignment;
    createInfo.flags.gl2Uncached = desc_.gl2CacheDisabled_;
    if (svmPtr != 0) {
      createInfo.flags.useReservedGpuVa = true;
      createInfo.pReservedGpuVaOwner = params->svmBase_->iMem();
    } else {
      createInfo.flags.useReservedGpuVa = false;
      createInfo.pReservedGpuVaOwner = nullptr;
    }
    createInfo.flags.interprocess = desc_.interprocess_;
    if (!dev().settings().svmFineGrainSystem_) {
      memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment,
                                                    createInfo.pReservedGpuVaOwner, &subOffset_);
    }
    if (memRef_ == nullptr) {
      memRef_ = GpuMemoryReference::Create(dev(), createInfo);
    }
  } else {
    Pal::GpuMemoryCreateInfo createInfo = {};
    createInfo.size = allocSize;
    createInfo.alignment = MaxGpuAlignment;
    createInfo.vaRange = Pal::VaRange::Svm;
    createInfo.priority = Pal::GpuMemPriority::Normal;
    if (svmPtr != 0) {
      createInfo.flags.useReservedGpuVa = true;
      createInfo.pReservedGpuVaOwner = params->svmBase_->iMem();
    }
    memTypeToHeap(&createInfo);
    createInfo.flags.interprocess = desc_.interprocess_;

    memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment,
                                                  createInfo.pReservedGpuVaOwner, &subOffset_);
    if (memRef_ == nullptr) {
      createInfo.alignment = dev().properties().gpuMemoryProperties.fragmentSize;
      createInfo.size = amd::alignUp(createInfo.size, createInfo.alignment);
      memRef_ = GpuMemoryReference::Create(dev(), createInfo);
    }
  }
  if (nullptr == memRef_) {
    LogError("Failed PAL memory allocation!");
    return false;
  }
  desc_.cardMemory_ = false;
  if ((nullptr != params) && (nullptr != params->owner_) &&
      (nullptr != params->owner_->getSvmPtr())) {
    params->owner_->setSvmPtr(
        reinterpret_cast<void*>(memRef_->iMem()->Desc().gpuVirtAddr + subOffset_));
    offset_ += static_cast<size_t>(subOffset_);
    params->owner_->setOffset(offset_);
  }
  return true;
}

// ================================================================================================
bool Resource::create(MemoryType memType, CreateParams* params, bool forceLinear) {
  bool imageCreateView = false;
  bool foundCalRef = false;
  bool viewDefined = false;
  uint viewLayer = 0;
  uint viewLevel = 0;
  uint viewFlags = 0;
  Pal::ChannelMapping channels;
  Pal::ChNumFormat format = dev().getPalFormat(desc().format_, &channels);
  // Set the initial offset value for any resource to 0.
  // Note: Runtime can call create() more than once, if the initial memory type failed
  offset_ = 0;

  // This is a thread safe operation
  const_cast<Device&>(dev()).initializeHeapResources();

  if (memType == Shader) {
    if (dev().settings().svmFineGrainSystem_) {
      desc_.isAllocExecute_ = true;
      desc_.SVMRes_ = true;
      memType = RemoteUSWC;
    } else {
      memType = Local;
    }
  }

  // Get the element size
  elementSize_ = Pal::Formats::BytesPerPixel(format);
  desc_.type_ = memType;
  if (memType == Scratch) {
    // use local memory for scratch buffer
    desc_.type_ = Local;
    desc_.scratch_ = true;
  }

  // Force remote allocation if it was requested in the settings
  if (dev().settings().remoteAlloc_ && ((memoryType() == Local) || (memoryType() == Persistent))) {
    if (dev().settings().apuSystem_) {
      desc_.type_ = Remote;
    } else {
      desc_.type_ = RemoteUSWC;
    }
  }

  if (dev().settings().disablePersistent_ && (memoryType() == Persistent)) {
    desc_.type_ = RemoteUSWC;
  }
  desc_.interprocess_ = (nullptr != params) ? params->interprocess_ : false;

  switch (memoryType()) {
    case OGLInterop:
    case D3D9Interop:
    case D3D10Interop:
    case D3D11Interop:
    case VkInterop:
      return CreateInterop(params);
    case P2PAccess:
      return CreateP2PAccess(params);
    case Pinned:
      return CreatePinned(params);
    case View: {
      // Save the offset in the global heap
      ViewParams* view = reinterpret_cast<ViewParams*>(params);
      offset_ = view->offset_;

      // Make sure parent was provided
      if (nullptr != view->resource_) {
        viewOwner_ = view->resource_;
        offset_ += viewOwner_->offset();
        if (viewOwner_->data() != nullptr) {
          address_ = viewOwner_->data() + view->offset_;
          mapCount_++;
        }
        memRef_ = viewOwner_->memRef_;
        memRef_->retain();
        desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
      } else {
        desc_.type_ = Empty;
      }
      return true;
    }
    case IpcMemory:
      return CreateIpc(params);
    default:
      break;
  }

  if (!desc_.buffer_) {
    return CreateImage(params, forceLinear);
  }

  if (memoryType() != Resource::VaRange) {
    Pal::gpusize svmPtr = 0;
    if ((nullptr != params) && (nullptr != params->owner_) &&
        (nullptr != params->owner_->getSvmPtr())) {
      svmPtr = reinterpret_cast<Pal::gpusize>(params->owner_->getSvmPtr());
      desc_.SVMRes_ = true;
      desc_.reserved_va_ = (svmPtr == 1) ? false : true;
      svmPtr = (svmPtr == 1) ? 0 : svmPtr;
      if (params->owner_->getMemFlags() & CL_MEM_SVM_ATOMICS) {
        desc_.gl2CacheDisabled_ = true;
      }
    }
    if (desc_.SVMRes_) {
      return CreateSvm(params, svmPtr);
    }
  }

  Pal::GpuMemoryCreateInfo createInfo = {};
  createInfo.size = desc().width_ * elementSize_;
  createInfo.size = amd::alignUp(createInfo.size, MaxGpuAlignment);
  createInfo.alignment = (params && params->alignment_ != 0)
                             ? params->alignment_
                             : (desc().scratch_ ? 64 * Ki : MaxGpuAlignment);
  createInfo.vaRange = Pal::VaRange::Default;
  createInfo.priority = Pal::GpuMemPriority::Normal;

  if (memoryType() == ExternalPhysical) {
    cl_bus_address_amd bus_address = (reinterpret_cast<amd::Buffer*>(params->owner_))->busAddress();
    createInfo.surfaceBusAddr = bus_address.surface_bus_address;
    createInfo.markerBusAddr = bus_address.marker_bus_address;
    createInfo.flags.sdiExternal = true;
  } else if (memoryType() == BusAddressable) {
    createInfo.flags.busAddressable = true;
  } else if (memoryType() == VaRange) {
    createInfo.flags.virtualAlloc = true;
    createInfo.vaRange = Pal::VaRange::Svm;
    if (params->owner_->getSvmPtr() != nullptr) {
      createInfo.flags.useReservedGpuVa = true;
      createInfo.pReservedGpuVaOwner = params->svmBase_->iMem();
    }
  }

  memTypeToHeap(&createInfo);
  // createInfo.priority;
  memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment,
                                                nullptr, &subOffset_);
  if (nullptr == memRef_) {
    memRef_ = GpuMemoryReference::Create(dev(), createInfo);
    if (nullptr == memRef_) {
      LogError("Failed PAL memory allocation!");
      return false;
    }
  }
  offset_ += static_cast<size_t>(subOffset_);
  // Check if memory is locked already and restore CPU pointer
  if (memRef_->cpuAddress_ != nullptr) {
    address_ = memRef_->cpuAddress_;
    memRef_->cpuAddress_ = nullptr;
    mapCount_++;
  }
  if (memoryType() == VaRange) {
    void* gpu_virt_addr = reinterpret_cast<void*>(memRef_->iMem()->Desc().gpuVirtAddr);

    // if svm_ptr was not set, set the gpu_virt_addr
    if (params->owner_->getSvmPtr() == nullptr) {
      params->owner_->setSvmPtr(gpu_virt_addr);
    } else if (params->owner_->getSvmPtr() != gpu_virt_addr) {
      LogPrintfError("Cannot get different ptrs in va_range, svm_ptr: %p new_svm_ptr:%p \n",
                     params->owner_->getSvmPtr(), gpu_virt_addr);
      return false;
    }
  }
  return true;
}

// ================================================================================================
void Resource::free() {
  if (memRef_ == nullptr) {
    return;
  }

  const bool wait = (memoryType() != ImageView) && (memoryType() != ImageBuffer) &&
                    (memoryType() != ImageExternalBuffer) && (memoryType() != View);

  // OCL has to wait, even if resource is placed in the cache, since reallocation can occur
  // and resource can be reused on another async queue without a wait on a busy operation
  if (wait) {
    if (memRef_->gpu_ == nullptr) {
      amd::ScopedLock l(dev().vgpusAccess());
      // Release all memory objects on all virtual GPUs
      for (uint idx = 1; idx < dev().vgpus().size(); ++idx) {
        dev().vgpus()[idx]->waitForEvent(&events_[idx]);
      }
    } else {
      memRef_->gpu_->waitForEvent(&events_[memRef_->gpu_->index()]);
    }
  } else {
    // After a view destruction the original object is no longer can be associated with a vgpu
    memRef_->gpu_ = nullptr;
  }

  // Destroy PAL resource
  if (iMem() != 0) {
    if (mapCount_ != 0 && wait) {
      if ((memoryType() != Remote) && (memoryType() != RemoteUSWC)) {
        //! @note: This is a workaround for bad applications that don't unmap memory
        unmap(nullptr);
      } else {
        // Delay CPU address unmap until memRef_ destruction
        if (!desc_.SVMRes_) {
          assert(memRef_->cpuAddress_ == nullptr && "Memref shouldn't have a valid CPU address");
          memRef_->cpuAddress_ = address_;
        }
      }
    }

    // Add resource to the cache
    if (!dev().resourceCache().addGpuMemory(&desc_, memRef_, subOffset_)) {
      // Free PAL resource
      palFree();
    }
  }

  // Free SRD for images
  if (!desc().buffer_) {
    dev().srds().freeSrdSlot(hwSrd_);
  }

  memRef_ = nullptr;
}

// ================================================================================================
void Resource::writeRawData(VirtualGPU& gpu, size_t offset, size_t size, const void* data,
                            bool waitForEvent) const {
  GpuEvent event;

  // Write data size bytes to surface
  // size needs to be DWORD aligned
  assert((size & 3) == 0);
  gpu.eventBegin(MainEngine);
  gpu.queue(MainEngine).addCmdMemRef(memRef());
  gpu.iCmd()->CmdUpdateMemory(*iMem(), offset_ + offset, size,
                              reinterpret_cast<const uint32_t*>(data));
  gpu.eventEnd(MainEngine, event);

  if (waitForEvent) {
    //! @note: We don't really have to mark the allocations as busy
    //! if we are waiting for a transfer

    // Wait for event to complete
    gpu.waitForEvent(&event);
  } else {
    setBusy(gpu, event);
    // Update the global GPU event
    gpu.setGpuEvent(event, false);
  }
}

// ================================================================================================
static const Pal::ChNumFormat ChannelFmt(uint bytesPerElement) {
  if (bytesPerElement == 16) {
    return Pal::ChNumFormat::X32Y32Z32W32_Uint;
  } else if (bytesPerElement == 8) {
    return Pal::ChNumFormat::X32Y32_Uint;
  } else if (bytesPerElement == 4) {
    return Pal::ChNumFormat::X32_Uint;
  } else if (bytesPerElement == 2) {
    return Pal::ChNumFormat::X16_Uint;
  } else {
    return Pal::ChNumFormat::X8_Uint;
  }
}

// ================================================================================================
bool Resource::partialMemCopyTo(VirtualGPU& gpu, const amd::Coord3D& srcOrigin,
                                const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                Resource& dstResource, bool enableCopyRect, bool flushDMA,
                                uint bytesPerElement) const {
  GpuEvent event;
  EngineType activeEngineID = gpu.engineID_;
  static const bool waitOnBusyEngine = true;
  uint64_t gpuMemoryOffset = 0;
  uint64_t gpuMemoryRowPitch = 0;
  uint64_t imageOffsetx = 0;
  bool img1Darray = false;
  bool img2Darray = false;

  if (desc().buffer_ && !dstResource.desc().buffer_) {
    imageOffsetx = dstOrigin[0] % dstResource.elementSize();
    gpuMemoryOffset = srcOrigin[0] + offset();
    gpuMemoryRowPitch = (srcOrigin[1]) ? srcOrigin[1] : size[0] * dstResource.elementSize();
    img1Darray = (dstResource.desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY);
    img2Darray = (dstResource.desc().topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY);
  } else if (!desc().buffer_ && dstResource.desc().buffer_) {
    imageOffsetx = srcOrigin[0] % elementSize();
    gpuMemoryOffset = dstOrigin[0] + dstResource.offset();
    gpuMemoryRowPitch = (dstOrigin[1]) ? dstOrigin[1] : size[0] * elementSize();
    img1Darray = (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY);
    img2Darray = (desc().topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY);
  }

  if ((desc().buffer_ && !dstResource.desc().buffer_) ||
      (!desc().buffer_ && dstResource.desc().buffer_)) {
    // sDMA cannot be used for the below conditions
    // Make sure linear pitch in bytes is 4 bytes aligned
    if (((gpuMemoryRowPitch % 4) != 0) ||
        // another DRM restriciton... SI has 4 pixels
        (gpuMemoryOffset % 4 != 0) || (imageOffsetx != 0)) {
      return false;
    }
  }

  bool cp_dma = dev().settings().disableSdma_ ||
                (!enableCopyRect && desc().buffer_ && dstResource.desc().buffer_ &&
                 (size[0] < dev().settings().cpDmaCopySizeMax_));
  if (cp_dma) {
    // Make sure compute is done before CP DMA start
    gpu.addBarrier(RgpSqqtBarrierReason::MemDependency, BarrierType::KernelToCopy);
  } else {
    gpu.releaseGpuMemoryFence();
    gpu.engineID_ = SdmaEngine;

    if (gpu.validateSdmaOverlap(*this, dstResource)) {
      // Note: PAL should insert a NOP into the command buffer for synchronization
      gpu.addBarrier(RgpSqqtBarrierReason::MemDependency, BarrierType::CopyToCopy);
    }
  }

  // Wait for the resources, since runtime may use async transfers
  wait(gpu, waitOnBusyEngine);
  dstResource.wait(gpu, waitOnBusyEngine);

  Pal::ImageLayout imgLayout = {};
  gpu.eventBegin(gpu.engineID_);
  gpu.queue(gpu.engineID_).addCmdMemRef(memRef());
  gpu.queue(gpu.engineID_).addCmdMemRef(dstResource.memRef());
  if (desc().buffer_ && !dstResource.desc().buffer_) {
    uint32_t arraySliceIdx = img2Darray ? dstOrigin[2] : img1Darray ? dstOrigin[1] : 0;
    Pal::SubresId ImgSubresId = {0, dstResource.desc().baseLevel_, arraySliceIdx};
    Pal::MemoryImageCopyRegion copyRegion = {};
    copyRegion.imageSubres = ImgSubresId;
    copyRegion.imageOffset.x = dstOrigin[0];
    copyRegion.imageOffset.y = img1Darray ? 0 : dstOrigin[1];
    copyRegion.imageOffset.z = (img1Darray || img2Darray) ? 0 : dstOrigin[2];
    copyRegion.imageExtent.width = size[0];
    copyRegion.imageExtent.height = size[1];
    copyRegion.imageExtent.depth = size[2];
    copyRegion.numSlices = 1;
    if (img1Darray) {
      copyRegion.numSlices = copyRegion.imageExtent.height;
      copyRegion.imageExtent.height = 1;
    } else if (img2Darray) {
      copyRegion.numSlices = copyRegion.imageExtent.depth;
      copyRegion.imageExtent.depth = 1;
    }
    copyRegion.gpuMemoryOffset = gpuMemoryOffset;
    copyRegion.gpuMemoryRowPitch = gpuMemoryRowPitch;
    copyRegion.gpuMemoryDepthPitch =
        (srcOrigin[2]) ? srcOrigin[2]
                       : copyRegion.gpuMemoryRowPitch * copyRegion.imageExtent.height;
    gpu.iCmd()->CmdCopyMemoryToImage(*iMem(), *dstResource.image_, imgLayout, 1, &copyRegion);
  } else if (!desc().buffer_ && dstResource.desc().buffer_) {
    Pal::MemoryImageCopyRegion copyRegion = {};
    uint32_t arraySliceIdx = img2Darray ? srcOrigin[2] : img1Darray ? srcOrigin[1] : 0;
    Pal::SubresId ImgSubresId = {0, desc().baseLevel_, arraySliceIdx};
    copyRegion.imageSubres = ImgSubresId;
    copyRegion.imageOffset.x = srcOrigin[0];
    copyRegion.imageOffset.y = img1Darray ? 0 : srcOrigin[1];
    copyRegion.imageOffset.z = (img1Darray || img2Darray) ? 0 : srcOrigin[2];
    copyRegion.imageExtent.width = size[0];
    copyRegion.imageExtent.height = size[1];
    copyRegion.imageExtent.depth = size[2];
    copyRegion.numSlices = 1;
    if (img1Darray) {
      copyRegion.numSlices = copyRegion.imageExtent.height;
      copyRegion.imageExtent.height = 1;
    } else if (img2Darray) {
      copyRegion.numSlices = copyRegion.imageExtent.depth;
      copyRegion.imageExtent.depth = 1;
    }
    copyRegion.gpuMemoryOffset = gpuMemoryOffset;
    copyRegion.gpuMemoryRowPitch = gpuMemoryRowPitch;
    copyRegion.gpuMemoryDepthPitch =
        (dstOrigin[2]) ? dstOrigin[2]
                       : copyRegion.gpuMemoryRowPitch * copyRegion.imageExtent.height;
    gpu.iCmd()->CmdCopyImageToMemory(*image_, imgLayout, *dstResource.iMem(), 1, &copyRegion);
  } else {
    if (enableCopyRect) {
      Pal::TypedBufferCopyRegion copyRegion = {};
      Pal::ChannelMapping channels = {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
                                      Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W};
      copyRegion.srcBuffer.swizzledFormat.format = ChannelFmt(bytesPerElement);
      copyRegion.srcBuffer.swizzledFormat.swizzle = channels;
      copyRegion.srcBuffer.offset = srcOrigin[0] + offset();
      copyRegion.srcBuffer.rowPitch = srcOrigin[1];
      copyRegion.srcBuffer.depthPitch = srcOrigin[2];
      copyRegion.extent.width = size[0] / bytesPerElement;
      copyRegion.extent.height = size[1];
      copyRegion.extent.depth = size[2];
      copyRegion.dstBuffer.swizzledFormat.format = ChannelFmt(bytesPerElement);
      copyRegion.dstBuffer.swizzledFormat.swizzle = channels;
      copyRegion.dstBuffer.offset = dstOrigin[0] + dstResource.offset();
      copyRegion.dstBuffer.rowPitch = dstOrigin[1];
      copyRegion.dstBuffer.depthPitch = dstOrigin[2];
      gpu.iCmd()->CmdCopyTypedBuffer(*iMem(), *dstResource.iMem(), 1, &copyRegion);
    } else {
      Pal::MemoryCopyRegion copyRegion = {};
      copyRegion.srcOffset = srcOrigin[0] + offset();
      copyRegion.dstOffset = dstOrigin[0] + dstResource.offset();
      copyRegion.copySize = size[0];
      constexpr size_t CpCopySizeLimit = (1 << 26) - sizeof(uint64_t);
      if (cp_dma && (size[0] > CpCopySizeLimit)) {
        size_t orgSize = size[0];
        copyRegion.copySize = CpCopySizeLimit;
        do {
          gpu.iCmd()->CmdCopyMemory(*iMem(), *dstResource.iMem(), 1, &copyRegion);
          copyRegion.srcOffset += CpCopySizeLimit;
          copyRegion.dstOffset += CpCopySizeLimit;
          orgSize -= (orgSize > CpCopySizeLimit) ? CpCopySizeLimit : orgSize;
          if (orgSize < CpCopySizeLimit) {
            copyRegion.copySize = orgSize;
          }
        } while (orgSize > 0);
      } else {
        gpu.iCmd()->CmdCopyMemory(*iMem(), *dstResource.iMem(), 1, &copyRegion);
      }
    }
  }

  if (cp_dma) {
    // Make sure CP dma is done
    gpu.addBarrier(RgpSqqtBarrierReason::MemDependency, BarrierType::CopyToKernel);
  }

  gpu.eventEnd(gpu.engineID_, event);

  // Mark source and destination as busy
  setBusy(gpu, event);
  dstResource.setBusy(gpu, event);

  // Update the global GPU event
  gpu.setGpuEvent(event, flushDMA);

  // Restore the original engine
  gpu.engineID_ = activeEngineID;

  return true;
}

// ================================================================================================
void Resource::setBusy(VirtualGPU& gpu, GpuEvent gpuEvent) const {
  addGpuEvent(gpu, gpuEvent);

  // If current resource is a view, then update the parent event as well
  if (viewOwner_ != nullptr) {
    viewOwner_->setBusy(gpu, gpuEvent);
  }
}

// ================================================================================================
void Resource::wait(VirtualGPU& gpu, bool waitOnBusyEngine) const {
  GpuEvent* gpuEvent = getGpuEvent(gpu);

  // Check if we have to wait unconditionally
  if (!waitOnBusyEngine ||
      // or we have to wait only if another engine was used on this resource
      (gpuEvent->engineId_ != gpu.engineID_)) {
    gpu.waitForEvent(gpuEvent);
  }

  // If current resource is a view and not in the global heap,
  // then wait for the parent event as well
  if (viewOwner_ != nullptr) {
    viewOwner_->wait(gpu, waitOnBusyEngine);
  }
}

// ================================================================================================
bool Resource::hostWrite(VirtualGPU* gpu, const void* hostPtr, const amd::Coord3D& origin,
                         const amd::Coord3D& size, uint flags, size_t rowPitch, size_t slicePitch) {
  void* dst;

  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // Get physical GPU memmory
  dst = map(gpu, flags, startLayer, numLayers);
  if (nullptr == dst) {
    LogError("Couldn't map GPU memory for host write");
    return false;
  }

  if (1 == desc().dimSize_) {
    size_t copySize = (desc().buffer_) ? size[0] : size[0] * elementSize_;

    // Update the pointer
    dst = static_cast<void*>(static_cast<char*>(dst) + origin[0]);

    // Copy memory
    std::memcpy(dst, hostPtr, copySize);
  } else {
    size_t dstOffsBase = origin[0] * elementSize_;

    // Make sure we use the right pitch if it's not specified
    if (rowPitch == 0) {
      rowPitch = size[0] * elementSize_;
    }

    // Make sure we use the right slice if it's not specified
    if (slicePitch == 0) {
      slicePitch = size[0] * size[1] * elementSize_;
    }

    // Adjust the destination offset with Y dimension
    dstOffsBase += desc().pitch_ * origin[1] * elementSize_;

    // Adjust the destination offset with Z dimension
    dstOffsBase += desc().slice_ * origin[2] * elementSize_;

    // Copy memory slice by slice
    for (size_t slice = 0; slice < size[2]; ++slice) {
      size_t dstOffs = dstOffsBase + slice * desc().slice_ * elementSize_;
      size_t srcOffs = slice * slicePitch;

      // Copy memory line by line
      for (size_t row = 0; row < size[1]; ++row) {
        // Copy memory
        std::memcpy((reinterpret_cast<address>(dst) + dstOffs),
                    (reinterpret_cast<const_address>(hostPtr) + srcOffs), size[0] * elementSize_);

        dstOffs += desc().pitch_ * elementSize_;
        srcOffs += rowPitch;
      }
    }
  }

  // Unmap GPU memory
  unmap(gpu);

  return true;
}

// ================================================================================================
bool Resource::hostRead(VirtualGPU* gpu, void* hostPtr, const amd::Coord3D& origin,
                        const amd::Coord3D& size, size_t rowPitch, size_t slicePitch) {
  void* src;

  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // Get physical GPU memmory
  src = map(gpu, ReadOnly, startLayer, numLayers);
  if (nullptr == src) {
    LogError("Couldn't map GPU memory for host read");
    return false;
  }

  if (1 == desc().dimSize_) {
    size_t copySize = (desc().buffer_) ? size[0] : size[0] * elementSize_;

    // Update the pointer
    src = static_cast<void*>(static_cast<char*>(src) + origin[0]);

    // Copy memory
    std::memcpy(hostPtr, src, copySize);
  } else {
    size_t srcOffsBase = origin[0] * elementSize_;

    // Make sure we use the right pitch if it's not specified
    if (rowPitch == 0) {
      rowPitch = size[0] * elementSize_;
    }

    // Make sure we use the right slice if it's not specified
    if (slicePitch == 0) {
      slicePitch = size[0] * size[1] * elementSize_;
    }

    // Adjust destination offset with Y dimension
    srcOffsBase += desc().pitch_ * origin[1] * elementSize_;

    // Adjust the destination offset with Z dimension
    srcOffsBase += desc().slice_ * origin[2] * elementSize_;

    // Copy memory line by line
    for (size_t slice = 0; slice < size[2]; ++slice) {
      size_t srcOffs = srcOffsBase + slice * desc().slice_ * elementSize_;
      size_t dstOffs = slice * slicePitch;

      // Copy memory line by line
      for (size_t row = 0; row < size[1]; ++row) {
        // Copy memory
        std::memcpy((reinterpret_cast<address>(hostPtr) + dstOffs),
                    (reinterpret_cast<const_address>(src) + srcOffs), size[0] * elementSize_);

        srcOffs += desc().pitch_ * elementSize_;
        dstOffs += rowPitch;
      }
    }
  }

  // Unmap GPU memory
  unmap(gpu);

  return true;
}

// ================================================================================================
void* Resource::gpuMemoryMap(size_t* pitch, uint flags, Pal::IGpuMemory* resource) const {
  if (desc_.cardMemory_ && !isPersistentDirectMap()) {
    // @todo remove const cast
    Unimplemented();
    return nullptr;
    //        return const_cast<Device&>(dev()).resMapLocal(*pitch, resource, flags);
  } else {
    amd::ScopedLock lk(dev().lockPAL());
    void* address;
    if (image_ != nullptr) {
      constexpr Pal::SubresId ImgSubresId = {0, 0, 0};
      Pal::SubresLayout layout;
      image_->GetSubresourceLayout(ImgSubresId, &layout);
      *pitch = layout.rowPitch / elementSize();
    } else {
      *pitch = desc().width_;
    }
    if (Pal::Result::Success == resource->Map(&address)) {
      return address;
    } else {
      LogError("PAL GpuMemory->Map() failed!");
      return nullptr;
    }
  }
}

// ================================================================================================
void Resource::gpuMemoryUnmap(Pal::IGpuMemory* resource) const {
  if (desc_.cardMemory_ && !isPersistentDirectMap()) {
    Unimplemented();
  } else {
    Pal::Result result = resource->Unmap();
    if (Pal::Result::Success != result) {
      LogError("PAL GpuMemory->Unmap() failed!");
    }
  }
}

// ================================================================================================
bool Resource::glAcquire() {
  bool retVal = true;
  if (desc().type_ == OGLInterop) {
    retVal = dev().resGLAcquire(glPlatformContext_, glInteropMbRes_, glType_);
  }
  return retVal;
}

// ================================================================================================
bool Resource::glRelease() {
  bool retVal = true;
  if (desc().type_ == OGLInterop) {
    retVal = dev().resGLRelease(glPlatformContext_, glInteropMbRes_, glType_);
  }
  return retVal;
}

// ================================================================================================
void Resource::addGpuEvent(const VirtualGPU& gpu, GpuEvent event) const {
  uint idx = gpu.index();
  assert(idx < events_.size());
  events_[idx] = event;
}

// ================================================================================================
GpuEvent* Resource::getGpuEvent(const VirtualGPU& gpu) const {
  uint idx = gpu.index();
  assert((idx < events_.size()) && "Undeclared queue access!");
  return &events_[idx];
}

// ================================================================================================
void Resource::setModified(VirtualGPU& gpu, bool modified) const {
  uint idx = gpu.index();
  assert(idx < events_.size());
  events_[idx].modified_ = modified;

  // If current resource is a view, then update the parent as well
  if (viewOwner_ != nullptr) {
    viewOwner_->setModified(gpu, modified);
  }
}

// ================================================================================================
bool Resource::isModified(VirtualGPU& gpu) const {
  uint idx = gpu.index();
  assert(idx < events_.size());
  bool modified = events_[idx].modified_;

  // If current resource is a view, then get the parent state as well
  if (viewOwner_ != nullptr) {
    modified |= viewOwner_->isModified(gpu);
  }
  return modified;
}

// ================================================================================================
void Resource::palFree() const {
  if (desc().type_ == OGLInterop) {
    amd::ScopedLock lk(dev().lockPAL());
    dev().resGLFree(glPlatformContext_, glInteropMbRes_, glType_);
  }
  memRef_->release();
}

// ================================================================================================
bool Resource::isMemoryType(MemoryType memType) const {
  if (memoryType() == memType) {
    return true;
  } else if (memoryType() == View) {
    return viewOwner_->isMemoryType(memType);
  }

  return false;
}

// ================================================================================================
bool Resource::isPersistentDirectMap(bool writeMap) const {
  bool directMap = ((memoryType() == Resource::Persistent) && (desc().dimSize_ < 3) &&
                    !desc().imageArray_ && writeMap);

  // If direct map is possible, then validate it with the current tiling
  if (directMap && desc().tiled_) {
    // Latest HW does have tiling apertures
    directMap = false;
  }
  if (memoryType() == View) {
    directMap = viewOwner_->isPersistentDirectMap(writeMap);
  }

  return directMap;
}

// ================================================================================================
void* Resource::map(VirtualGPU* gpu, uint flags, uint startLayer, uint numLayers) {
  if (isMemoryType(Pinned)) {
    // Check if we have to wait
    if (!(flags & NoWait)) {
      if (gpu != nullptr) {
        wait(*gpu);
      }
    }
    return address_;
  }

  if (flags & ReadOnly) {
  }

  if (flags & WriteOnly) {
  }

  // Check if we have to wait
  if (!(flags & NoWait)) {
    if (gpu != nullptr) {
      wait(*gpu);
    }
  }

  // Check if memory wasn't mapped yet
  if (++mapCount_ == 1) {
    // Map current resource
    if (memRef_->cpuAddress_ != nullptr) {
      // Suballocations are mapped by the memory suballocator
      address_ = reinterpret_cast<uint8_t*>(memRef_->cpuAddress_) + subOffset_;
    } else {
      address_ = gpuMemoryMap(&desc_.pitch_, flags, iMem());
      address_ = reinterpret_cast<address>(address_) + offset_;
    }

    if (address_ == nullptr) {
      LogError("cal::ResMap failed!");
      --mapCount_;
      return nullptr;
    }
  }

  //! \note the atomic operation with counter doesn't
  // guarantee that the address will be valid,
  // since PAL could still process the first map
  if (address_ == nullptr) {
    for (uint i = 0; address_ == NULL && i < 10; ++i) {
      amd::Os::sleep(1);
    }
    assert((address_ != nullptr) && "Multiple maps failed!");
  }

  return address_;
}

// ================================================================================================
void Resource::unmap(VirtualGPU* gpu) {
  if (isMemoryType(Pinned)) {
    return;
  }

  // Decrement map counter
  int count = --mapCount_;

  // Check if it's the last unmap
  if (count == 0) {
    // Unmap current resource
    gpuMemoryUnmap(iMem());
    address_ = nullptr;
  } else if (count < 0) {
    LogError("dev().serialCalResUnmap failed!");
    ++mapCount_;
    return;
  }
}

// ================================================================================================
bool MemorySubAllocator::InitAllocator(GpuMemoryReference* mem_ref) {
  MemBuddyAllocator* allocator =
      new MemBuddyAllocator(device_, device_->settings().subAllocationChunkSize_,
                            device_->settings().subAllocationMinSize_);
  if (!((allocator != nullptr) && (allocator->Init() == Pal::Result::Success) &&
        heaps_.insert({mem_ref, allocator}).second)) {
    mem_ref->release();
    delete allocator;
    return false;
  }
  return true;
}

// ================================================================================================
void MemorySubAllocator::forceResident(GpuMemoryReference* mem_ref) {
  if (IS_WINDOWS) {
    // Write one DWORD using CPDMA to force resident
    GpuEvent event;
    auto gpu = device_->xferQueue();
    uint32_t data = 0;

    gpu->eventBegin(MainEngine);
    gpu->queue(MainEngine).addCmdMemRef(mem_ref);
    gpu->iCmd()->CmdUpdateMemory(*mem_ref->iMem(), 0, 4, &data);
    gpu->eventEnd(MainEngine, event);
    gpu->waitForEvent(&event);
  }
}

// ================================================================================================
bool MemorySubAllocator::CreateChunk(const Pal::IGpuMemory* reserved_va) {
  Pal::GpuMemoryCreateInfo createInfo = {};
  createInfo.size = device_->settings().subAllocationChunkSize_;
  createInfo.alignment = device_->properties().gpuMemoryProperties.fragmentSize;
  createInfo.vaRange = Pal::VaRange::Default;
  createInfo.priority = Pal::GpuMemPriority::Normal;
  createInfo.heapCount = 3;
  createInfo.heaps[0] = Pal::GpuHeapInvisible;
  createInfo.heaps[1] = Pal::GpuHeapLocal;
  createInfo.heaps[2] = Pal::GpuHeapGartUswc;
  createInfo.flags.peerWritable = device_->P2PAccessAllowed();
  createInfo.mallPolicy = static_cast<Pal::GpuMemMallPolicy>(device_->settings().mallPolicy_);
  if (amd::IS_HIP && PAL_HIP_IPC_FLAG) {
    // set interprocess for IPC memory support
    createInfo.flags.interprocess = 1;
  }
  GpuMemoryReference* mem_ref = GpuMemoryReference::Create(*device_, createInfo);
  if (mem_ref != nullptr) {
    // Workaround: some chunk memory are not guaranteed to be resident during initial allocation.
    forceResident(mem_ref);
    return InitAllocator(mem_ref);
  }
  return false;
}

// ================================================================================================
bool CoarseMemorySubAllocator::CreateChunk(const Pal::IGpuMemory* reserved_va) {
  Pal::GpuMemoryCreateInfo createInfo = {};
  createInfo.size = device_->settings().subAllocationChunkSize_;
  createInfo.alignment = device_->properties().gpuMemoryProperties.fragmentSize;
  createInfo.vaRange = Pal::VaRange::Svm;
  createInfo.priority = Pal::GpuMemPriority::Normal;
  createInfo.flags.useReservedGpuVa = (reserved_va != nullptr);
  createInfo.pReservedGpuVaOwner = reserved_va;
  createInfo.heapCount = 2;
  createInfo.heaps[0] = Pal::GpuHeapInvisible;
  createInfo.heaps[1] = Pal::GpuHeapLocal;
  createInfo.mallPolicy = static_cast<Pal::GpuMemMallPolicy>(device_->settings().mallPolicy_);
  if (amd::IS_HIP && PAL_HIP_IPC_FLAG) {
    // set interprocess for IPC memory support
    createInfo.flags.interprocess = 1;
  }
  GpuMemoryReference* mem_ref = GpuMemoryReference::Create(*device_, createInfo);
  if (mem_ref != nullptr) {
    // Workaround: some chunk memory are not guaranteed to be resident during initial allocation.
    forceResident(mem_ref);
    return InitAllocator(mem_ref);
  }
  return false;
}

// ================================================================================================
bool FineMemorySubAllocator::CreateChunk(const Pal::IGpuMemory* reserved_va) {
  Pal::SvmGpuMemoryCreateInfo createInfo = {};
  createInfo.isUsedForKernel = false;
  createInfo.size = device_->settings().subAllocationChunkSize_;
  createInfo.alignment = MaxGpuAlignment;
  createInfo.flags.useReservedGpuVa = (reserved_va != nullptr);
  createInfo.pReservedGpuVaOwner = reserved_va;
  createInfo.mallPolicy = Pal::GpuMemMallPolicy::Never;
  if (amd::IS_HIP && PAL_HIP_IPC_FLAG) {
    //set interprocess for IPC memory support
    createInfo.flags.interprocess = 1;
  }
  GpuMemoryReference* mem_ref = GpuMemoryReference::Create(*device_, createInfo);
  if ((mem_ref != nullptr) && InitAllocator(mem_ref)) {
    // Workaround: some chunk memory are not guaranteed to be resident during initial allocation.
    forceResident(mem_ref);
    mem_ref->iMem()->Map(&mem_ref->cpuAddress_);
    return mem_ref->cpuAddress_ != nullptr;
  }
  return false;
}

// ================================================================================================
bool FineUncachedMemorySubAllocator::CreateChunk(const Pal::IGpuMemory* reserved_va) {
  Pal::SvmGpuMemoryCreateInfo createInfo = {};
  createInfo.isUsedForKernel = false;
  createInfo.size = device_->settings().subAllocationChunkSize_;
  createInfo.alignment = MaxGpuAlignment;
  createInfo.flags.useReservedGpuVa = (reserved_va != nullptr);
  createInfo.pReservedGpuVaOwner = reserved_va;
  createInfo.flags.gl2Uncached = true;
  createInfo.mallPolicy = Pal::GpuMemMallPolicy::Never;
  if (amd::IS_HIP && PAL_HIP_IPC_FLAG) {
    //set interprocess for IPC memory support
    createInfo.flags.interprocess = 1;
  }
  GpuMemoryReference* mem_ref = GpuMemoryReference::Create(*device_, createInfo);
  if ((mem_ref != nullptr) && InitAllocator(mem_ref)) {
    // Workaround: some chunk memory are not guaranteed to be resident during initial allocation.
    forceResident(mem_ref);
    mem_ref->iMem()->Map(&mem_ref->cpuAddress_);
    return mem_ref->cpuAddress_ != nullptr;
  }
  return false;
}

// ================================================================================================
MemorySubAllocator::~MemorySubAllocator() {
  // Release memory heap for suballocations
  for (const auto& it : heaps_) {
    it.first->release();
    delete it.second;
  }
}

// ================================================================================================
GpuMemoryReference* MemorySubAllocator::Allocate(Pal::gpusize size, Pal::gpusize alignment,
                                                 const Pal::IGpuMemory* reserved_va,
                                                 Pal::gpusize* offset) {
  GpuMemoryReference* mem_ref = nullptr;
  MemBuddyAllocator* allocator = nullptr;
  // Check if the resource size and alignment are allowed for suballocation
  if ((size < device_->settings().subAllocationMaxSize_) &&
      (alignment <= device_->properties().gpuMemoryProperties.fragmentSize)) {
    uint i = 0;
    size = amd::alignUp(size, device_->settings().subAllocationMinSize_);
    do {
      // Find if current heap has enough empty space
      for (const auto& it : heaps_) {
        mem_ref = it.first;
        allocator = it.second;
        // SVM allocations may required a fixed VA, make sure we find the heap with the same VA
        if (reserved_va &&
            (reserved_va->Desc().gpuVirtAddr != mem_ref->iMem()->Desc().gpuVirtAddr)) {
          continue;
        }
        // If we have found a valid chunk, then suballocate memory
        if (Pal::Result::Success == allocator->Allocate(size, alignment, offset)) {
          return mem_ref;
        }
      }
      // We didn't find a valid chunk, so create a new one
      if (!CreateChunk(reserved_va)) {
        return nullptr;
      }
      i++;
    } while (i < 2);
  }
  return nullptr;
}

// ================================================================================================
bool MemorySubAllocator::Free(amd::Monitor* monitor, GpuMemoryReference* ref, Pal::gpusize offset) {
  bool release_mem = false;
  {
    amd::ScopedLock l(monitor);
    // Find if current memory reference is a chunk allocation
    auto it = heaps_.find(ref);
    if (it == heaps_.end()) {
      return false;
    }

    it->second->Free(offset);
    // If this suballocator empty, then release memory chunk
    // while keeping at least one chunk available, if retain_final_chunk is true
    if (it->second->IsEmpty() && (!(retain_final_chunk_ && heaps_.size() == 1) || !amd::IS_HIP)) {
      delete it->second;
      heaps_.erase(it);
      release_mem = true;
    }
  }
  if (release_mem) {
    ref->release();
  }
  return true;
}

// ================================================================================================
ResourceCache::~ResourceCache() { free(); }

// ================================================================================================
//! \note the cache works in FILO mode
bool ResourceCache::addGpuMemory(Resource::Descriptor* desc, GpuMemoryReference* ref,
                                 Pal::gpusize offset) {
  bool result = false;
  size_t size = ref->iMem()->Desc().size;

  // Check if runtime can free suballocation
  if (desc->type_ == Resource::VaRange) {
    // We do no sub allocate VA Range.
    result = false;
  } else if ((desc->type_ == Resource::Local) && !desc->SVMRes_) {
    result = mem_sub_alloc_local_.Free(&lockCacheOps_, ref, offset);
  } else if ((desc->type_ == Resource::Local) && desc->SVMRes_) {
    result = mem_sub_alloc_coarse_.Free(&lockCacheOps_, ref, offset);
  } else if (desc->SVMRes_) {
    if (desc->gl2CacheDisabled_) {
      result = mem_sub_alloc_fine_uncached_.Free(&lockCacheOps_, ref, offset);
    } else {
      result = mem_sub_alloc_fine_.Free(&lockCacheOps_, ref, offset);
    }
  }

  // If a resource was a suballocation, don't try to cache it
  if (result == true) {
    return result;
  }

  // Make sure current allocation isn't bigger than cache
  if (((desc->type_ == Resource::Local) || (desc->type_ == Resource::Persistent) ||
       (desc->type_ == Resource::Remote) || (desc->type_ == Resource::RemoteUSWC)) &&
      (size < cacheSizeLimit_ && !(desc->SVMRes_ && desc->reserved_va_))) {
    // Validate the cache size limit. Loop until we have enough space
    while ((cacheSize_ + size) > cacheSizeLimit_) {
      removeLast();
    }

    Resource::Descriptor* descCached = new Resource::Descriptor;
    if (descCached != nullptr) {
      // Copy the original desc to the cached version
      memcpy(descCached, desc, sizeof(Resource::Descriptor));

      amd::ScopedLock l(&lockCacheOps_);
      // Add the current resource to the cache
      resCache_.push_front({descCached, ref});
      ref->gpu_ = nullptr;
      cacheSize_ += size;
      if (desc->type_ == Resource::Local) {
        lclCacheSize_ += size;
      } else if (desc->type_ == Resource::Persistent) {
        persistentCacheSize_ += size;
      }
      result = true;
    }
  }

  return result;
}

// ================================================================================================
GpuMemoryReference* ResourceCache::findGpuMemory(Resource::Descriptor* desc, Pal::gpusize size,
                                                 Pal::gpusize alignment,
                                                 const Pal::IGpuMemory* reserved_va,
                                                 Pal::gpusize* offset) {
  amd::ScopedLock l(&lockCacheOps_);
  GpuMemoryReference* ref = nullptr;

  // Check if the runtime can suballocate memory
  if (desc->type_ == Resource::VaRange) {
    // Do not use suballocator for VA_Range.
    return nullptr;
  } else if ((desc->type_ == Resource::Local) && !desc->SVMRes_) {
    ref = mem_sub_alloc_local_.Allocate(size, alignment, reserved_va, offset);
  } else if ((desc->type_ == Resource::Local) && desc->SVMRes_) {
    ref = mem_sub_alloc_coarse_.Allocate(size, alignment, reserved_va, offset);
  } else if (desc->SVMRes_) {
    if (desc->gl2CacheDisabled_) {
      ref = mem_sub_alloc_fine_uncached_.Allocate(size, alignment, reserved_va, offset);
    } else {
      ref = mem_sub_alloc_fine_.Allocate(size, alignment, reserved_va, offset);
    }
  }

  if (ref != nullptr) {
    return ref;
  }

  // Early exit if resource is too big or it's SVM allocation with a reserved VA
  if (size >= cacheSizeLimit_ || (desc->SVMRes_ && desc->reserved_va_)) {
    //! \note we may need to free the cache here to reduce memory pressure
    return ref;
  }

  // Serach the right resource through the cache list
  for (const auto& it : resCache_) {
    Resource::Descriptor* entry = it.first;
    size_t sizeRes = it.second->iMem()->Desc().size;
    // Find if we can reuse this entry
    if ((entry->type_ == desc->type_) && (entry->flags_ == desc->flags_) && (size <= sizeRes) &&
        (size > (sizeRes >> 1)) && ((it.second->iMem()->Desc().gpuVirtAddr % alignment) == 0) &&
        (entry->isAllocExecute_ == desc->isAllocExecute_) && (entry->SVMRes_ == desc->SVMRes_) &&
        (entry->gl2CacheDisabled_ == desc->gl2CacheDisabled_) &&
        (entry->interprocess_ == desc->interprocess_)) {
      ref = it.second;
      cacheSize_ -= sizeRes;
      if (entry->type_ == Resource::Local) {
        lclCacheSize_ -= sizeRes;
      } else if (entry->type_ == Resource::Persistent) {
        persistentCacheSize_ -= sizeRes;
      }
      delete it.first;
      // Remove the found etry from the cache
      resCache_.remove(it);
      break;
    }
  }

  return ref;
}

// ================================================================================================
bool ResourceCache::free(size_t minCacheEntries) {
  bool result = false;
  if (minCacheEntries < resCache_.size()) {
    result = true;
    // Clear the cache
    while (static_cast<int64_t>(cacheSize_) > 0) {
      removeLast();
    }
    CondLog((cacheSize_ != 0), "Incorrect size for cache release!");
  }
  return result;
}

// ================================================================================================
void ResourceCache::removeLast() {
  std::pair<Resource::Descriptor*, GpuMemoryReference*> entry;
  {
    // Protect access to the global data
    amd::ScopedLock l(&lockCacheOps_);
    if (resCache_.size() > 0) {
      entry = resCache_.back();
      resCache_.pop_back();
      auto mem_size = entry.second->iMem()->Desc().size;
      cacheSize_ -= mem_size;
      if (entry.first->type_ == Resource::Local) {
        lclCacheSize_ -= mem_size;
      } else if (entry.first->type_ == Resource::Persistent) {
        persistentCacheSize_ -= mem_size;
      }
      // Delete Descriptor
      delete entry.first;
    }
  }

  // Destroy PAL resource
  entry.second->release();
}

}  // namespace amd::pal
