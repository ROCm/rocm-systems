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

//! Implementation of GPU device memory management

#include "top.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "device/device.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palblit.hpp"
#include "utils/flags.hpp"

#ifdef _WIN32
#include <d3d10_1.h>
#include "platform/interop_d3d9.hpp"
#include "platform/interop_d3d10.hpp"
#include "platform/interop_d3d11.hpp"
#endif  //_WIN32
#include "platform/interop_gl.hpp"
#include "platform/external_memory.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace amd::pal {

Memory::Memory(const Device& gpuDev, amd::Memory& owner, size_t size)
    : device::Memory(owner), Resource(gpuDev, size), pinnedMemory_(nullptr), parent_(nullptr) {
  if (owner.parent() != nullptr) {
    flags_ |= SubMemoryObject;
  }
}

Memory::Memory(const Device& gpuDev, size_t size)
    : device::Memory(size), Resource(gpuDev, size), pinnedMemory_(nullptr), parent_(nullptr) {}

Memory::Memory(const Device& gpuDev, amd::Memory& owner, size_t width, size_t height, size_t depth,
               cl_image_format format, cl_mem_object_type imageType, uint mipLevels)
    : device::Memory(owner),
      Resource(gpuDev, width, height, depth, format, imageType, mipLevels),
      pinnedMemory_(nullptr),
      parent_(nullptr) {
  if (owner.parent() != nullptr) {
    flags_ |= SubMemoryObject;
  }
}

Memory::Memory(const Device& gpuDev, size_t size, size_t width, size_t height, size_t depth,
               cl_image_format format, cl_mem_object_type imageType, uint mipLevels)
    : device::Memory(size),
      Resource(gpuDev, width, height, depth, format, imageType, mipLevels),
      pinnedMemory_(nullptr),
      parent_(nullptr) {}

#ifdef _WIN32
static HANDLE getSharedHandle(IUnknown* pIface) {
  // Sanity checks
  assert(pIface != nullptr);

  HRESULT hRes;
  HANDLE hShared;
  IDXGIResource* pDxgiRes = nullptr;
  if ((hRes = (const_cast<IUnknown*>(pIface))
                  ->QueryInterface(__uuidof(IDXGIResource), (void**)&pDxgiRes)) != S_OK) {
    return (HANDLE)0;
  }
  if (!pDxgiRes) {
    return (HANDLE)0;
  }
  hRes = pDxgiRes->GetSharedHandle(&hShared);
  pDxgiRes->Release();
  if (hRes != S_OK) {
    return (HANDLE)0;
  }
  return hShared;
}
#endif  //_WIN32

bool Memory::create(Resource::MemoryType memType, Resource::CreateParams* params,
                    bool forceLinear) {
  bool result;
  uint allocAttempt = 0;
  // Reset the flag in case we reallocate the heap in local/remote
  flags_ &= ~HostMemoryDirectAccess;

  if (!ValidateMemory(memType)) {
    return false;
  }

  do {
    // Assume that allocations will be placed into visible heap when ReBar is enabled
    // Only enable this assumption for small size local buffers
    constexpr size_t kLargeAlloc = (1ull << 27);
    if (!amd::IS_HIP && (memType == Local) && desc().buffer_ && (size() < kLargeAlloc) &&
        dev().info().largeBar_) {
      memType = Persistent;
    }

    if (amd::IS_HIP && dev().settings().apuSystem_) {
      Pal::gpusize totalAlloc = dev().TotalAlloc();
      if (memType == Local && totalAlloc > dev().GetMaxFrameBuffer()) {
        memType = RemoteUSWC;
      }
    }
    // Create a resource in PAL
    result = Resource::create(memType, params, forceLinear);
    if (!result) {
      size_t freeMemory[2];
      // if requested memory is greater than available then exit the loop
      dev().globalFreeMemory(freeMemory);
      // Local to Persistent
      if (memoryType() == Local) {
        // For dgpu freeMemory[0] reports a sum of visible+invisible fb
        if (size() > (freeMemory[0] * Ki)) {
          break;
        }
        memType = Persistent;
      }
      // Don't switch to USWC if persistent memory was explicitly asked
      else if ((allocAttempt > 0) && (memoryType() == Persistent)) {
        memType = RemoteUSWC;
      }
      // Remote cacheable to uncacheable
      else if (memoryType() == Remote) {
        memType = RemoteUSWC;
      } else if (dev().settings().apuSystem_ && memoryType() == RemoteUSWC) {
        if (size() > (freeMemory[0] * Ki) || allocAttempt >= 2) {
          break;
        }
      } else {
        break;
      }
      allocAttempt++;
    }
  } while (!result);

  // Check if CAL created a resource
  if (result) {
    switch (memoryType()) {
      case Resource::Pinned:
      case Resource::ExternalPhysical:
        // Marks memory object for direct GPU access to the host memory
        flags_ |= HostMemoryDirectAccess;
        break;
      case Resource::Remote:
      case Resource::RemoteUSWC:
        if ((!desc().tiled_) && (desc().dimSize_ != 3)) {
          // Marks memory object for direct GPU access to the host memory
          flags_ |= HostMemoryDirectAccess;
        }
        break;
      case Resource::View: {
        Resource::ViewParams* view = reinterpret_cast<Resource::ViewParams*>(params);
        // Check if parent was allocated in system memory
        if ((view->resource_->memoryType() == Resource::Pinned) ||
            (view->resource_->memoryType() == Resource::Remote) ||
            (view->resource_->memoryType() == Resource::RemoteUSWC)) {
          // Marks memory object for direct GPU access to the host memory
          flags_ |= HostMemoryDirectAccess;
        }
        if ((view->owner_ != nullptr) && (view->owner_->parent() != nullptr)) {
          parent_ = reinterpret_cast<const Memory*>(view->memory_);
          flags_ |= SubMemoryObject;
        }
        break;
      }
      case Resource::ImageView: {
        Resource::ImageViewParams* view = reinterpret_cast<Resource::ImageViewParams*>(params);
        parent_ = reinterpret_cast<const Memory*>(view->memory_);
        flags_ |= SubMemoryObject | (parent_->flags_ & HostMemoryDirectAccess);
        break;
      }
      case Resource::ImageExternalBuffer:
      case Resource::ImageBuffer: {
        Resource::ImageBufferParams* view = reinterpret_cast<Resource::ImageBufferParams*>(params);
        parent_ = reinterpret_cast<const Memory*>(view->memory_);
        flags_ |= SubMemoryObject | (parent_->flags_ & HostMemoryDirectAccess);
        break;
      }
      default:
        break;
    }
  }

  if (result) {
    if ((params != nullptr) && (memoryType() == Pinned)) {
      memRef()->gpu_ = params->gpu_;
    }
    if (memRef() != nullptr) {
      ClPrint(amd::LOG_DEBUG, amd::LOG_RESOURCE, "Alloc: %zx bytes, ptr[%llx-%llx], obj[%llx-%llx]",
              size(), vmAddress(), vmAddress() + size(), iMem()->Desc().gpuVirtAddr,
              iMem()->Desc().gpuVirtAddr + iMem()->Desc().size);
    }
  }

  return result;
}

bool Memory::processGLResource(GLResourceOP operation) {
  bool retVal = false;
  switch (operation) {
    case GLDecompressResource:
      retVal = glAcquire();
      break;
    case GLInvalidateFBO:
      retVal = glRelease();
      break;
    default:
      assert(false && "unknown GLResourceOP");
  }
  return retVal;
}

bool Memory::createInterop() {
  Resource::MemoryType memType = Resource::Empty;
  Resource::OGLInteropParams oglRes;
  Resource::VkInteropParams vkRes;
#ifdef _WIN32
  Resource::D3DInteropParams d3dRes;
#endif  //_WIN32

  // Only external objects support interop
  assert(owner() != nullptr);

  Resource::CreateParams* createParams = nullptr;

  amd::InteropObject* interop = owner()->getInteropObj();
  assert((interop != nullptr) && "An invalid interop object is impossible!");

  auto ext_memory = interop->asExternalMemory();
  amd::GLObject* glObject = interop->asGLObject();
#ifdef _WIN32
  amd::D3D10Object* d3d10Object = interop->asD3D10Object();
  amd::D3D11Object* d3d11Object = interop->asD3D11Object();
  amd::D3D9Object* d3d9Object = interop->asD3D9Object();

  if (d3d10Object != nullptr) {
    createParams = &d3dRes;

    d3dRes.owner_ = owner();

    const amd::D3D10ObjDesc_t* objDesc = d3d10Object->getObjDesc();

    memType = Resource::D3D10Interop;

    // Get shared handle
    if ((d3dRes.handle_ = getSharedHandle(d3d10Object->getD3D10Resource()))) {
      d3dRes.iDirect3D_ = static_cast<void*>(d3d10Object->getD3D10Resource());
      d3dRes.type_ = Resource::InteropTypeless;
    }

    d3dRes.misc = 0;
    // Find D3D10 object type
    switch (objDesc->objDim_) {
      case D3D10_RESOURCE_DIMENSION_BUFFER:
        d3dRes.type_ = Resource::InteropVertexBuffer;
        break;
      case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
      case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
      case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
        d3dRes.type_ = Resource::InteropTexture;
        if (objDesc->mipLevels_ > 1) {
          d3dRes.type_ = Resource::InteropTextureViewLevel;

          if (objDesc->arraySize_ > 1) {
            d3dRes.layer_ = d3d10Object->getSubresource() / objDesc->mipLevels_;
            d3dRes.mipLevel_ = d3d10Object->getSubresource() % objDesc->mipLevels_;
          } else {
            d3dRes.layer_ = 0;
            d3dRes.mipLevel_ = d3d10Object->getSubresource();
          }
        }
        break;
      default:
        return false;
        break;
    }
  } else if (d3d11Object != nullptr) {
    createParams = &d3dRes;

    d3dRes.owner_ = owner();

    const amd::D3D11ObjDesc_t* objDesc = d3d11Object->getObjDesc();

    memType = Resource::D3D11Interop;

    // Get shared handle
    if ((d3dRes.handle_ = getSharedHandle(d3d11Object->getD3D11Resource()))) {
      d3dRes.iDirect3D_ = static_cast<void*>(d3d11Object->getD3D11Resource());
      d3dRes.type_ = Resource::InteropTypeless;
    }

    d3dRes.misc = 0;
    // Find D3D11 object type
    switch (objDesc->objDim_) {
      case D3D11_RESOURCE_DIMENSION_BUFFER:
        d3dRes.type_ = Resource::InteropVertexBuffer;
        break;
      case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
      case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
      case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
        d3dRes.type_ = Resource::InteropTexture;
        d3dRes.layer_ = d3d11Object->getPlane();
        d3dRes.misc = d3d11Object->getMiscFlag();
        if (objDesc->mipLevels_ > 1) {
          d3dRes.type_ = Resource::InteropTextureViewLevel;

          if (objDesc->arraySize_ > 1) {
            d3dRes.layer_ = d3d11Object->getSubresource() / objDesc->mipLevels_;
            d3dRes.mipLevel_ = d3d11Object->getSubresource() % objDesc->mipLevels_;
          } else {
            d3dRes.layer_ = 0;
            d3dRes.mipLevel_ = d3d11Object->getSubresource();
          }
        }
        break;
      default:
        return false;
        break;
    }
  } else if (d3d9Object != nullptr) {
    createParams = &d3dRes;

    d3dRes.owner_ = owner();

    const amd::D3D9ObjDesc_t* objDesc = d3d9Object->getObjDesc();

    memType = Resource::D3D9Interop;

    // Get shared handle
    if ((d3dRes.handle_ = d3d9Object->getD3D9SharedHandle())) {
      d3dRes.iDirect3D_ = static_cast<void*>(d3d9Object->getD3D9Resource());
      d3dRes.type_ = Resource::InteropSurface;
      d3dRes.mipLevel_ = 0;
      d3dRes.layer_ = d3d9Object->getPlane();
      d3dRes.misc = d3d9Object->getMiscFlag();
    }
  } else
#endif  //_WIN32
    if (ext_memory != nullptr) {
      createParams = &vkRes;
      vkRes.owner_ = owner();
      memType = Resource::VkInterop;
      vkRes.handle_ = ext_memory->Handle();
      vkRes.name_ = ext_memory->Name();
      vkRes.type_ = Resource::InteropTypeless;
      vkRes.nt_handle_ = ((ext_memory->Type() != amd::ExternalMemory::HandleType::OpaqueFd) &&
                          (ext_memory->Type() != amd::ExternalMemory::HandleType::OpaqueWin32Kmt) &&
                          (ext_memory->Type() != amd::ExternalMemory::HandleType::D3D11ResourceKmt))
                             ? true
                             : false;
    }

    else if (glObject != nullptr) {
      createParams = &oglRes;

      oglRes.owner_ = owner();

      memType = Resource::OGLInterop;

      // Fill the interop creation parameters
      oglRes.handle_ = static_cast<uint>(glObject->getGLName());

      // Find OGL object type
      switch (glObject->getCLGLObjectType()) {
        case CL_GL_OBJECT_BUFFER:
          oglRes.type_ = Resource::InteropVertexBuffer;
          break;
        case CL_GL_OBJECT_TEXTURE_BUFFER:
        case CL_GL_OBJECT_TEXTURE1D:
        case CL_GL_OBJECT_TEXTURE1D_ARRAY:
        case CL_GL_OBJECT_TEXTURE2D:
        case CL_GL_OBJECT_TEXTURE2D_ARRAY:
        case CL_GL_OBJECT_TEXTURE3D:
          oglRes.type_ = Resource::InteropTexture;
          if (GL_TEXTURE_CUBE_MAP == glObject->getGLTarget()) {
            switch (glObject->getCubemapFace()) {
              case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
              case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
              case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
              case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
              case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
              case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
                oglRes.type_ = Resource::InteropTextureViewCube;
                oglRes.layer_ = glObject->getCubemapFace() - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
                oglRes.mipLevel_ = glObject->getGLMipLevel();
                break;
              default:
                break;
            }
          } else if (glObject->getGLMipLevel() != 0) {
            oglRes.type_ = Resource::InteropTextureViewLevel;
            oglRes.layer_ = 0;
            oglRes.mipLevel_ = glObject->getGLMipLevel();
          }
          break;
        case CL_GL_OBJECT_RENDERBUFFER:
          oglRes.type_ = Resource::InteropRenderBuffer;
          break;
        default:
          return false;
          break;
      }

      oglRes.glPlatformContext_ = owner()->getContext().info().hCtx_;
    } else {
      return false;
    }

  // Create memory object
  if (!create(memType, createParams)) {
    return false;
  }

  return true;
}

Memory::~Memory() {
  if (memRef() != nullptr) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_RESOURCE, "Free-: %8llx bytes, VM[%10llx, %10llx]",
            iMem()->Desc().size, iMem()->Desc().gpuVirtAddr,
            iMem()->Desc().gpuVirtAddr + iMem()->Desc().size);
  }
  // Clean VA cache
  dev().removeVACache(this);

  // Release associated map target, if any
  if (nullptr != mapMemory_) {
    if (owner()->getSvmPtr() != nullptr) {
      owner()->uncommitSvmMemory();
    }

    mapMemory()->unmap(nullptr);
    mapMemory_->release();
  }

  // Destory pinned memory
  if (flags_ & PinnedMemoryAlloced) {
    delete pinnedMemory_;
  }

  if ((owner() != nullptr) && isHostMemDirectAccess() && !(flags_ & SubMemoryObject) &&
      (memoryType() != Resource::ExternalPhysical)) {
    // Unmap memory if direct access was requested
    // Note: runtime will perform unmap on the actual resource destruction
    // unmap(nullptr);
  }
}

void Memory::syncCacheFromHost(VirtualGPU& gpu, device::Memory::SyncFlags syncFlags) {
  amd::ScopedLock lock(owner()->lockMemoryOps());
  // If the last writer was another GPU, then make a writeback
  if (isChacheCoherencySync() && (owner()->getLastWriter() != nullptr) &&
      (&dev() != owner()->getLastWriter())) {
    mgpuCacheWriteBack(gpu);
  }

  // If host memory doesn't have direct access, then we have to synchronize
  if (isChacheCoherencySync() && (nullptr != owner()->getHostMem())) {
    bool hasUpdates = true;

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (flags_ & SubMemoryObject)) {
      pal::Memory* gpuMemory = dev().getGpuMemory(owner()->parent());

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      gpuMemory->syncCacheFromHost(gpu, syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((version_ == owner()->getVersion()) || (&dev() == owner()->getLastWriter())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          pal::Memory* gpuSub = reinterpret_cast<pal::Memory*>(devSub);
          gpuSub->syncCacheFromHost(gpu, syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because this GPU device was the last writer
    if (&dev() != owner()->getLastWriter()) {
      // Update the latest version
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);

    // If host memory was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      if (desc().buffer_) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().copyBuffer(*pinnedMemory_, *this, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = gpu.blitMgr().copyBufferToImage(*pinnedMemory_, *this, origin, origin,
                                                 image.getRegion(), Entire, image.getRowPitch(),
                                                 image.getSlicePitch());
      }
    }

    if (!result) {
      if (desc().buffer_) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().writeBuffer(owner()->getHostMem(), *this, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = gpu.blitMgr().writeImage(owner()->getHostMem(), *this, origin, image.getRegion(),
                                          image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    //!@todo A wait isn't really necessary. However processMemObjects()
    // may lose the track of dependencies with a compute transfer(if sdma failed).
    wait(gpu);

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}

void Memory::syncHostFromCache(device::VirtualDevice* vDev, device::Memory::SyncFlags syncFlags) {
  VirtualGPU* gpu = (vDev != nullptr) ? reinterpret_cast<VirtualGPU*>(vDev) : dev().xferQueue();
  // Sanity checks
  assert(owner() != nullptr);

  // If host memory doesn't have direct access, then we have to synchronize
  if (isChacheCoherencySync()) {
    bool hasUpdates = true;

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (flags_ & SubMemoryObject)) {
      device::Memory* m = owner()->parent()->getDeviceMemory(dev());

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      m->syncHostFromCache(gpu, syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((nullptr == owner()->getLastWriter()) || (version_ == owner()->getVersion())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      amd::ScopedLock lock(owner()->lockMemoryOps());
      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          pal::Memory* gpuSub = reinterpret_cast<pal::Memory*>(devSub);
          gpuSub->syncHostFromCache(gpu, syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because CPU was the last writer
    if (nullptr != owner()->getLastWriter()) {
      // Mark parent as up to date, set our version accordingly
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);
    // If device on the provided queue doesn't match the device memory was allocated,
    // then use blit manager on device
    const auto& bltMgr = (&gpu->dev() != &dev()) ? dev().xferMgr() : gpu->blitMgr();

    // If backing store was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      if (desc().buffer_) {
        amd::Coord3D region(owner()->getSize());
        result = bltMgr.copyBuffer(*this, *pinnedMemory_, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = bltMgr.copyImageToBuffer(*this, *pinnedMemory_, origin, origin, image.getRegion(),
                                          Entire, image.getRowPitch(), image.getSlicePitch());
      }
    }

    // Just do a basic host read
    if (!result) {
      if (desc().buffer_) {
        amd::Coord3D region(owner()->getSize());
        result = bltMgr.readBuffer(*this, owner()->getHostMem(), origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = bltMgr.readImage(*this, owner()->getHostMem(), origin, image.getRegion(),
                                  image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}

pal::Memory* Memory::createBufferView(amd::Memory& subBufferOwner) {
  pal::Memory* viewMemory;
  Resource::ViewParams params;

  size_t offset = subBufferOwner.getOrigin();
  size_t size = subBufferOwner.getSize();

  // Create a memory object
  viewMemory = new pal::Memory(dev(), subBufferOwner, size);
  if (nullptr == viewMemory) {
    return nullptr;
  }

  params.owner_ = &subBufferOwner;
  params.gpu_ = static_cast<VirtualGPU*>(subBufferOwner.getVirtualDevice());
  params.offset_ = offset;
  params.size_ = size;
  params.resource_ = this;
  params.memory_ = this;
  if (!viewMemory->create(Resource::View, &params)) {
    delete viewMemory;
    return nullptr;
  }

  // Explicitly set the host memory location,
  // because the parent location could change after reallocation
  if (nullptr != owner()->getHostMem()) {
    subBufferOwner.setHostMem(reinterpret_cast<char*>(owner()->getHostMem()) + offset);
  } else {
    subBufferOwner.setHostMem(nullptr);
  }

  return viewMemory;
}

void Memory::decIndMapCount() {
  // Map/unmap must be serialized
  amd::ScopedLock lock(owner()->lockMemoryOps());

  if (indirectMapCount_ == 0) {
    if (!mipMapped()) {
      LogError("decIndMapCount() called when indirectMapCount_ already zero");
    }
    return;
  }

  // Decrement the counter and release indirect map if it's the last op
  if (--indirectMapCount_ == 0) {
    if (nullptr != mapMemory_) {
      amd::Memory* memory = mapMemory_;
      amd::Memory* empty = nullptr;

      // Get GPU memory
      Memory* gpuMemory = mapMemory();
      gpuMemory->unmap(nullptr);

      if (!dev().addMapTarget(memory)) {
        memory->release();
      }

      // Map/unamp is serialized for the same memory object,
      // so it's safe to clear the pointer
      assert((mapMemory_ != nullptr) && "Mapped buffer should be valid");
      mapMemory_ = nullptr;
    }
  }
}

// Note - must be called by the device under the async lock, so no spinning
// or long pauses allowed in this function.
void* Memory::allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region, uint mapFlags,
                             size_t* rowPitch, size_t* slicePitch) {
  // Sanity checks
  assert(owner() != nullptr);

  // Map/unmap must be serialized
  amd::ScopedLock lock(owner()->lockMemoryOps());

  address mapAddress = nullptr;
  size_t offset = origin[0];

  // For SVM implementation, we cannot use cached map. if svm space, use the svm host pointer
  void* initHostPtr = owner()->getSvmPtr();
  if (nullptr != initHostPtr) {
    owner()->commitSvmMemory();
  }

  constexpr size_t largeAlloc = (1ull << 31);
  if ((owner()->numDevices() > 1) || (owner()->getSize() > largeAlloc)) {
    if ((nullptr == initHostPtr) && (owner()->getHostMem() == nullptr)) {
      static const bool forceAllocHostMem = true;
      if (!owner()->allocHostMemory(nullptr, forceAllocHostMem)) {
        return nullptr;
      }
      //! \note Ignore pinning result
      bool ok = pinSystemMemory(owner()->getHostMem(), owner()->getSize());
    }
  }

  incIndMapCount();
  // If host memory exists, use it
  if ((owner()->getHostMem() != nullptr) && isDirectMap()) {
    mapAddress = reinterpret_cast<address>(owner()->getHostMem());
  }
  // If resource is a persistent allocation, we can use it directly
  else if (((isPersistentDirectMap(mapFlags & CL_MAP_WRITE) && (getMapCount() == 0)) ||
            isPersistentMapped()) &&
           (owner()->getSvmPtr() == nullptr)) {
    if (nullptr == map(nullptr)) {
      LogError("Could not map target persistent resource");
      decIndMapCount();
      return nullptr;
    }
    if (getMapCount() == 1) {
      setPersistentMapFlag(true);
    }
    mapAddress = data();
  }
  // Otherwise we can use a remote resource:
  else {
    // Are we in range?
    size_t elementCount = desc().width_;
    size_t rSize = elementCount * elementSize();
    if (offset >= rSize || offset + region[0] > rSize) {
      LogWarning("Memory::allocMapTarget() - offset/size out of bounds");
      return nullptr;
    }

    // Allocate a map resource if there isn't any yet
    if (indirectMapCount_ == 1) {
      const static bool SysMem = true;
      bool failed = false;
      amd::Memory* memory = nullptr;
      // Search for a possible indirect resource
      cl_mem_flags flag = 0;
      bool canBeCached = true;
      if (nullptr != initHostPtr) {
        // make sure the host memory is committed already, or we have a big problem.
        assert(owner()->isSvmPtrCommited() && "The host svm memory not committed yet!");
        flag = CL_MEM_USE_HOST_PTR;
        canBeCached = false;
      } else {
        memory = dev().findMapTarget(owner()->getSize());
      }

      if (memory == nullptr) {
        // for map target of svm buffer , we need use svm host ptr
        memory = new (dev().context()) amd::Buffer(dev().context(), flag, owner()->getSize());

        do {
          if ((memory == nullptr) || !memory->create(initHostPtr, SysMem)) {
            failed = true;
            break;
          }
          memory->setCacheStatus(canBeCached);

          Memory* gpuMemory = reinterpret_cast<Memory*>(memory->getDeviceMemory(dev()));

          // Create, Map and get the base pointer for the resource
          if ((gpuMemory == nullptr) || (nullptr == gpuMemory->map(nullptr))) {
            failed = true;
            break;
          }
        } while (false);
      }

      if (failed) {
        if (memory != nullptr) {
          memory->release();
        }
        decIndMapCount();
        LogError("Could not map target resource");
        return nullptr;
      }

      // Map/unamp is serialized for the same memory object,
      // so it's safe to assign the new pointer
      assert((mapMemory_ == nullptr) && "Mapped buffer can't be valid");
      mapMemory_ = memory;
    } else {
      // Did the map resource allocation fail?
      if (mapMemory_ == nullptr) {
        LogError("Could not map target resource");
        return nullptr;
      }
    }
    mapAddress = mapMemory()->data();
  }

  return mapAddress + offset;
}

bool Memory::pinSystemMemory(void* hostPtr, size_t size) {
  bool result = false;

  // If memory has a direct access already, then skip the host memory pinning
  if (isHostMemDirectAccess()) {
    return true;
  }

  // Memory was pinned already
  if (flags_ & PinnedMemoryAlloced) {
    return true;
  }

  // Allocate memory for the pinned object
  pinnedMemory_ = new Memory(dev(), size);

  if (pinnedMemory_ == nullptr) {
    return false;
  }

  // Check if it's a view
  if (flags_ & SubMemoryObject) {
    const pal::Memory* gpuMemory;
    if (owner() != nullptr) {
      gpuMemory = dev().getGpuMemory(owner()->parent());
    } else {
      gpuMemory = parent();
    }

    if (gpuMemory->flags_ & PinnedMemoryAlloced) {
      Resource::ViewParams params;
      params.owner_ = owner();
      params.offset_ = owner()->getOrigin();
      params.size_ = owner()->getSize();
      params.resource_ = gpuMemory->pinnedMemory_;
      params.memory_ = nullptr;
      result = pinnedMemory_->create(Resource::View, &params);
    }
  } else {
    Resource::PinnedParams params;
    // Fill resource creation parameters
    params.owner_ = owner();
    params.hostMemRef_ = owner()->getHostMemRef();
    params.size_ = size;

    // Create resource
    result = pinnedMemory_->create(Resource::Pinned, &params);
  }

  if (!result) {
    delete pinnedMemory_;
    pinnedMemory_ = nullptr;
    return false;
  }

  flags_ |= PinnedMemoryAlloced;
  return true;
}

void* Memory::cpuMap(device::VirtualDevice& vDev, uint flags, uint startLayer, uint numLayers,
                     size_t* rowPitch, size_t* slicePitch) {
  uint resFlags = 0;
  if (flags == Memory::CpuReadOnly) {
    resFlags = Resource::ReadOnly;
  } else if (flags == Memory::CpuWriteOnly) {
    resFlags = Resource::WriteOnly;
  }

  void* ptr = map(&static_cast<VirtualGPU&>(vDev), resFlags, startLayer, numLayers);
  if (!desc().buffer_) {
    *rowPitch = desc().pitch_ * elementSize();
    *slicePitch = desc().slice_ * elementSize();
  }
  return ptr;
}

void Memory::cpuUnmap(device::VirtualDevice& vDev) { unmap(&static_cast<VirtualGPU&>(vDev)); }

Memory* Memory::mapMemory() const {
  Memory* map = nullptr;
  if (nullptr != mapMemory_) {
    map = reinterpret_cast<Memory*>(mapMemory_->getDeviceMemory(dev()));
  }
  return map;
}

void Memory::mgpuCacheWriteBack(VirtualGPU& gpu) {
  // Lock memory object, so only one write back can occur
  amd::ScopedLock lock(owner()->lockMemoryOps());

  // Attempt to allocate a staging buffer if don't have any
  if (!owner()->P2PAccess() && (owner()->getHostMem() == nullptr)) {
    if (nullptr != owner()->getSvmPtr()) {
      owner()->commitSvmMemory();
      owner()->setHostMem(owner()->getSvmPtr());
    } else {
      static const bool forceAllocHostMem = true;
      owner()->allocHostMemory(nullptr, forceAllocHostMem);
    }
  }

  // Make synchronization
  if (owner()->getHostMem() != nullptr) {
    //! \note Ignore pinning result
    bool ok = pinSystemMemory(owner()->getHostMem(), owner()->getSize());
    owner()->cacheWriteBack(&gpu);
  }
}

// ================================================================================================
Memory* Buffer::createBufferView(amd::Memory& subBufferOwner) const {
  pal::Memory* subBuffer;
  Resource::ViewParams params;

  size_t offset = subBufferOwner.getOrigin();
  size_t size = subBufferOwner.getSize();

  // Create a memory object
  subBuffer = new pal::Buffer(dev(), subBufferOwner, size);
  if (nullptr == subBuffer) {
    return nullptr;
  }

  // Allocate a view for this buffer object
  params.owner_ = &subBufferOwner;
  params.offset_ = offset;
  params.size_ = size;
  params.resource_ = this;
  params.memory_ = this;

  if (!subBuffer->create(Resource::View, &params)) {
    delete subBuffer;
    return nullptr;
  }

  return subBuffer;
}

// ================================================================================================
bool Buffer::ExportHandle(void* handle) const {
  Pal::GpuMemoryExportInfo exportInfo = {};
#if IS_WINDOWS
  // Set default flags in case they are not provided by application
  exportInfo.accessFlags = GENERIC_READ | GENERIC_WRITE;
#endif
  *reinterpret_cast<Pal::OsExternalHandle*>(handle) = iMem()->ExportExternalHandle(exportInfo);
  return true;
}

// ================================================================================================
void* Image::allocMapTarget(const amd::Coord3D& origin, const amd::Coord3D& region, uint mapFlags,
                            size_t* rowPitch, size_t* slicePitch) {
  // Sanity checks
  assert(owner() != nullptr);
  bool useRemoteResource = true;
  size_t slicePitchTmp = 0;
  size_t height = desc().height_;
  size_t depth = desc().depth_;

  // Map/unmap must be serialized
  amd::ScopedLock lock(owner()->lockMemoryOps());

  address mapAddress = nullptr;
  size_t offset = origin[0];

  incIndMapCount();

  // If host memory exists, use it
  if ((owner()->getHostMem() != nullptr) && isDirectMap()) {
    useRemoteResource = false;
    mapAddress = reinterpret_cast<address>(owner()->getHostMem());
    amd::Image* amdImage = owner()->asImage();

    // Calculate the offset in bytes
    offset *= elementSize();

    // Update the row and slice pitches value
    *rowPitch =
        (amdImage->getRowPitch() == 0) ? (desc().width_ * elementSize()) : amdImage->getRowPitch();
    slicePitchTmp =
        (amdImage->getSlicePitch() == 0) ? (height * (*rowPitch)) : amdImage->getSlicePitch();

    // Adjust the offset in Y and Z dimensions
    offset += origin[1] * (*rowPitch);
    offset += origin[2] * slicePitchTmp;
  }
  // If resource is a persistent allocation, we can use it directly
  //! @note Even if resource is a persistent allocation,
  //! runtime can't use it directly,
  //! because CAL volume map doesn't work properly.
  //! @todo arrays can be added for persistent lock with some CAL changes
  else if ((isPersistentDirectMap(mapFlags & CL_MAP_WRITE) && (getMapCount() == 0)) ||
           isPersistentMapped()) {
    if (nullptr == map(nullptr)) {
      useRemoteResource = true;
      LogError("Could not map target persistent resource, try remote resource");
    } else {
      useRemoteResource = false;
      mapAddress = data();
      if (getMapCount() == 1) {
        setPersistentMapFlag(true);
      }
      // Calculate the offset in bytes
      offset *= elementSize();

      // Update the row pitch value
      *rowPitch = desc().pitch_ * elementSize();

      // Adjust the offset in Y dimension
      offset += origin[1] * (*rowPitch);
    }
  }

  // Otherwise we can use a remote resource:
  if (useRemoteResource) {
    // Calculate X offset in bytes
    offset *= elementSize();

    // Allocate a map resource if there isn't any yet
    if (indirectMapCount_ == 1) {
      const static bool SysMem = true;
      bool failed = false;
      amd::Memory* memory;
      // Search for a possible indirect resource
      memory = dev().findMapTarget(owner()->getSize());

      if (memory == nullptr) {
        // Allocate a new buffer to use as the map target
        //! @note Allocate a 1D buffer, since CAL issues with 3D
        //! Also HW doesn't support untiled images
        memory = new (dev().context())
            amd::Buffer(dev().context(), 0, desc().width_ * height * depth * elementSize());
        memory->setVirtualDevice(owner()->getVirtualDevice());

        do {
          if ((memory == nullptr) || !memory->create(nullptr, SysMem)) {
            failed = true;
            break;
          }

          Memory* gpuMemory = reinterpret_cast<Memory*>(memory->getDeviceMemory(dev()));

          // Create, Map and get the base pointer for the resource
          if ((gpuMemory == nullptr) || (nullptr == gpuMemory->map(nullptr))) {
            failed = true;
            break;
          }
        } while (false);
      }

      if (failed) {
        if (memory != nullptr) {
          memory->release();
        }
        decIndMapCount();
        LogError("Could not map target resource");
        return nullptr;
      }

      // Map/unamp is serialized for the same memory object,
      // so it's safe to assign the new pointer
      assert((mapMemory_ == nullptr) && "Mapped buffer can't be valid");
      mapMemory_ = memory;
    } else {
      // Did the map resource allocation fail?
      if (mapMemory_ == nullptr) {
        LogError("Could not map target resource");
        return nullptr;
      }
    }

    mapAddress = mapMemory()->data();

    // Update the row and slice pitches value
    *rowPitch = region[0] * elementSize();
    if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
      slicePitchTmp = *rowPitch;
    } else {
      slicePitchTmp = *rowPitch * region[1];
    }
    // Use start of the indirect buffer
    offset = 0;
  }

  if (slicePitch != nullptr) {
    *slicePitch = slicePitchTmp;
  }

  return mapAddress + offset;
}

bool Image::ValidateMemory(Resource::MemoryType memType) {
  if (dev().settings().imageBufferWar_ && (memType == ImageBuffer) && (owner() != nullptr) &&
      ((owner()->asImage()->getWidth() * owner()->asImage()->getImageFormat().getElementSize()) <
       owner()->asImage()->getRowPitch())) {
    constexpr bool ForceLinear = true;
    // Create a native image without pitch for validation
    copyImageBuffer_ = new pal::Image(dev(), size(), desc().width_, desc().height_, desc().depth_,
                                      desc().format_, desc().topology_, 0);
    if ((copyImageBuffer_ == nullptr) ||
        !copyImageBuffer_->create(Resource::Local, nullptr, ForceLinear)) {
      return false;
    }
    constexpr Pal::SubresId ImgSubresId = {0, 0, 0};
    Pal::SubresLayout layout;
    copyImageBuffer_->image()->GetSubresourceLayout(ImgSubresId, &layout);
    // Destroy temporary linear image, since it was allocated for the pitch validation only
    delete copyImageBuffer_;
    copyImageBuffer_ = nullptr;
    // If pitch doesn't match HW expectation, then create a backing store
    if (owner()->asImage()->getRowPitch() != layout.rowPitch) {
      // Create a native image without pitch as a backing store
      copyImageBuffer_ = new pal::Image(dev(), size(), desc().width_, desc().height_, desc().depth_,
                                        desc().format_, desc().topology_, 0);
      if ((copyImageBuffer_ == nullptr) || !copyImageBuffer_->create(Resource::Local)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace amd::pal
