/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/context.hpp"
#include "platform/interop_gl.hpp"
#include "vdi_common.hpp"
#include "platform/commandqueue.hpp"

#include <algorithm>
#include <functional>

// Context property constants (replaces CL header includes for these integer-only values).
// Values verified against cl.h, cl_gl.h, cl_ext.h, cl_d3d10.h, cl_d3d11.h, cl_dx9_media_sharing.h
#define ROCCLR_CL_CONTEXT_PLATFORM               0x1084  // CL_CONTEXT_PLATFORM
#define ROCCLR_CL_CONTEXT_INTEROP_USER_SYNC      0x1085  // CL_CONTEXT_INTEROP_USER_SYNC
#define ROCCLR_CL_GL_CONTEXT_KHR                 0x2008  // CL_GL_CONTEXT_KHR
#define ROCCLR_CL_EGL_DISPLAY_KHR                0x2009  // CL_EGL_DISPLAY_KHR
#define ROCCLR_CL_GLX_DISPLAY_KHR                0x200A  // CL_GLX_DISPLAY_KHR
#define ROCCLR_CL_WGL_HDC_KHR                    0x200B  // CL_WGL_HDC_KHR
#define ROCCLR_CL_CGL_SHAREGROUP_KHR             0x200C  // CL_CGL_SHAREGROUP_KHR
#define ROCCLR_CL_CONTEXT_OFFLINE_DEVICES_AMD    0x403F  // CL_CONTEXT_OFFLINE_DEVICES_AMD (AMD ext)
#ifdef _WIN32
#include <d3d10_1.h>
#include <dxgi.h>
// D3D interop context property constants (from CL/cl_d3d10.h, cl_d3d11.h, cl_dx9_media_sharing.h)
// These are integer constants only; no types from those headers are needed here.
#define ROCCLR_CL_CONTEXT_D3D10_DEVICE_KHR       0x4014  // CL_CONTEXT_D3D10_DEVICE_KHR
#define ROCCLR_CL_CONTEXT_D3D11_DEVICE_KHR       0x401D  // CL_CONTEXT_D3D11_DEVICE_KHR
#define ROCCLR_CL_CONTEXT_ADAPTER_D3D9_KHR       0x2025  // CL_CONTEXT_ADAPTER_D3D9_KHR
#define ROCCLR_CL_CONTEXT_ADAPTER_D3D9EX_KHR     0x2026  // CL_CONTEXT_ADAPTER_D3D9EX_KHR
#define ROCCLR_CL_CONTEXT_ADAPTER_DXVA_KHR       0x2027  // CL_CONTEXT_ADAPTER_DXVA_KHR
#endif  //_WIN32

namespace amd {

Context::Context(const std::vector<Device*>& devices, const Info& info)
    : devices_(devices),
      info_(info),
      properties_(NULL),
      glenv_(NULL),
      customHostAllocDevice_(NULL) {
  for (const auto& device : devices) {
    device->retain();
    if (customHostAllocDevice_ == NULL && device->customHostAllocator()) {
      customHostAllocDevice_ = device;
    }
    if (device->svmSupport()) {
      svmAllocDevice_.push_back(device);
    }
  }

  if (svmAllocDevice_.size() > 1) {
    uint isFirstDeviceFGSEnabled = svmAllocDevice_.front()->isFineGrainedSystem(true);
    for (auto& dev : svmAllocDevice_) {
      // allocation on fine - grained system incapable device first
      if (isFirstDeviceFGSEnabled && !dev->isFineGrainedSystem(true)) {
        std::swap(svmAllocDevice_.front(), dev);
        break;
      }
    }
  }
}

Context::~Context() {
  static const bool VALIDATE_ONLY = false;

  // Loop through all devices
  for (const auto& it : devices_) {
    // Dissociate OCL context with any external device
    if (info_.flags_ & (GLDeviceKhr | D3D10DeviceKhr | D3D11DeviceKhr)) {
      it->unbindExternalDevice(info_.flags_, info_.hDev_, info_.hCtx_, VALIDATE_ONLY);
    }

    // Notify device about context destroy
    it->ContextDestroy();

    // Release device
    it->release();
  }

  delete[] properties_;
  delete glenv_;
}

int Context::checkProperties(const intptr_t* properties, Context::Info* info) {
  void* pfmId = nullptr;
  uint count = 0;

  const struct Element {
    intptr_t name;
    void* ptr;
  }* p = reinterpret_cast<const Element*>(properties);

  // Clear the context infor structure
  ::memset(info, 0, sizeof(Context::Info));

  if (properties == nullptr) {
    return static_cast<int>(amd::Status::Success);
  }

  // Process all properties
  while (p->name != 0) {
    switch (p->name) {
      case ROCCLR_CL_CONTEXT_INTEROP_USER_SYNC:
        if (p->ptr == reinterpret_cast<void*>(true)) {
          info->flags_ |= InteropUserSync;
        }
        break;
#ifdef _WIN32
      case ROCCLR_CL_CONTEXT_D3D10_DEVICE_KHR:
        if (p->ptr == NULL) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        info->hDev_[D3D10DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D10DeviceKhr;
        break;
      case ROCCLR_CL_CONTEXT_D3D11_DEVICE_KHR:
        if (p->ptr == NULL) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        info->hDev_[D3D11DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D11DeviceKhr;
        break;
      case ROCCLR_CL_CONTEXT_ADAPTER_D3D9_KHR:
        if (p->ptr == NULL) {  // not supported for xp
          return static_cast<int>(amd::Status::InvalidValue);
        }
        info->hDev_[D3D9DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceKhr;
        break;
      case ROCCLR_CL_CONTEXT_ADAPTER_D3D9EX_KHR:
        if (p->ptr == NULL) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        info->hDev_[D3D9DeviceEXKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceEXKhr;
        break;
      case ROCCLR_CL_CONTEXT_ADAPTER_DXVA_KHR:
        if (p->ptr == NULL) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        info->hDev_[D3D9DeviceVAKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceVAKhr;
        break;
#endif  //_WIN32

      case ROCCLR_CL_EGL_DISPLAY_KHR:
        info->flags_ |= EGLDeviceKhr;

#ifdef _WIN32
      case ROCCLR_CL_WGL_HDC_KHR:
#endif  //_WIN32

#if defined(__linux__)
      case ROCCLR_CL_GLX_DISPLAY_KHR:
#endif  // linux
        if (p->ptr == NULL) {
          return -1000 /* CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR */;
        }
      case ROCCLR_HIP_GLX_DISPLAY_KHR:
      case ROCCLR_HIP_WGL_HDC_KHR:
        info->hDev_[GLDeviceKhrIdx] = p->ptr;
        break;
#if defined(__APPLE__) || defined(__MACOSX)
      case ROCCLR_CL_CGL_SHAREGROUP_KHR:
        Unimplemented();
        break;
#endif  //__APPLE__ || MACOS
      case ROCCLR_CL_GL_CONTEXT_KHR:
        if (p->ptr == NULL) {
          return -1000 /* CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR */;
        }
        // skip the null case in the case of hip-gl, it will be initialized in create
      case ROCCLR_HIP_GL_CONTEXT_KHR:
        info->hCtx_ = p->ptr;
        info->flags_ |= GLDeviceKhr;
        break;
      case ROCCLR_CL_CONTEXT_PLATFORM:
        pfmId = p->ptr;
        if ((nullptr != pfmId) && (AMD_PLATFORM != pfmId)) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        break;
      case ROCCLR_CL_CONTEXT_OFFLINE_DEVICES_AMD:
        if (p->ptr != reinterpret_cast<void*>(1)) {
          return static_cast<int>(amd::Status::InvalidValue);
        }
        // Set the offline device flag
        info->flags_ |= OfflineDevices;
        break;
      default:
        return static_cast<int>(amd::Status::InvalidValue);
    }
    p++;
    count++;
  }

  info->propertiesSize_ = count * sizeof(Element) + sizeof(intptr_t);
  return static_cast<int>(amd::Status::Success);
}

int Context::create(const intptr_t* properties) {
  static const bool VALIDATE_ONLY = false;
  int result = static_cast<int>(amd::Status::Success);

  if (properties != NULL) {
    properties_ = new intptr_t[info().propertiesSize_ / sizeof(intptr_t)];
    if (properties_ == NULL) {
      return static_cast<int>(amd::Status::OutOfHostMemory);
    }

    ::memcpy(properties_, properties, info().propertiesSize_);
  }

  // if the context passed in is null, it's the GL interop case and we need to get the current
  // context
  if (info_.hCtx_ == nullptr) {
    if (info_.flags_ & GLDeviceKhr) {
      // Init context for GL interop
      if (glenv_ == NULL) {
        HMODULE h = (HMODULE)Os::loadLibrary(
#ifdef _WIN32
            "OpenGL32.dll"
#else   //!_WIN32
            "libGL.so.1"
#endif  //!_WIN32
        );
        if (h && (glenv_ = new GLFunctions(h, (info_.flags_ & Flags::EGLDeviceKhr) != 0))) {
#ifdef _WIN32
          info_.hCtx_ = (void*)glenv_->wglGetCurrentContext_();
          info_.hDev_[GLDeviceKhrIdx] = (void*)glenv_->wglGetCurrentDC_();

#else
          info_.hCtx_ = (void*)glenv_->glXGetCurrentContext_();
          info_.hDev_[GLDeviceKhrIdx] = (void*)glenv_->glXGetCurrentDisplay_();
#endif
        }
      }

      struct Element {
        intptr_t name;
        void* ptr;
      }* p = reinterpret_cast<Element*>(properties_);
      while (p->name != 0) {
        switch (p->name) {
          case ROCCLR_HIP_GLX_DISPLAY_KHR:
          case ROCCLR_HIP_WGL_HDC_KHR:
            p->ptr = info_.hDev_[GLDeviceKhrIdx];
            break;
          case ROCCLR_HIP_GL_CONTEXT_KHR:
            p->ptr = info_.hCtx_;
            break;
        }
        p++;
      }
    }
  }

  // Check if OCL context can be associated with any external device
  if (info_.flags_ & (D3D10DeviceKhr | D3D11DeviceKhr | GLDeviceKhr | D3D9DeviceKhr |
                      D3D9DeviceEXKhr | D3D9DeviceVAKhr)) {
    // Loop through all devices
    for (const auto& it : devices_) {
      if (!it->bindExternalDevice(info_.flags_, info_.hDev_, info_.hCtx_, VALIDATE_ONLY)) {
        result = static_cast<int>(amd::Status::InvalidValue);
      }
    }
  }

  // Check if the device binding wasn't successful
  if (result != static_cast<int>(amd::Status::Success)) {
    if (info_.flags_ & GLDeviceKhr) {
      result = -1000 /* CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR */;
    } else if (info_.flags_ & D3D10DeviceKhr) {
      // return static_cast<int>(amd::Status::InvalidValue); // FIXME_odintsov: CL_INVALID_D3D_INTEROP;
    } else if (info_.flags_ & D3D11DeviceKhr) {
      // return static_cast<int>(amd::Status::InvalidValue); // FIXME_odintsov: CL_INVALID_D3D_INTEROP;
    } else if (info_.flags_ & (D3D9DeviceKhr | D3D9DeviceEXKhr | D3D9DeviceVAKhr)) {
      // return CL_INVALID_DX9_MEDIA_ADAPTER_KHR;
    }
  } else {
    if (info_.flags_ & GLDeviceKhr) {
      if (glenv_ == NULL) {
        HMODULE h = (HMODULE)Os::loadLibrary(
#ifdef _WIN32
            "OpenGL32.dll"
#else   //!_WIN32
            "libGL.so.1"
#endif  //!_WIN32
        );
        if (!h || !(glenv_ = new GLFunctions(h, (info_.flags_ & Flags::EGLDeviceKhr) != 0))) {
          return -1000 /* CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR */;
        }
      }
      if (!glenv_->init(reinterpret_cast<intptr_t>(info_.hDev_[GLDeviceKhrIdx]),
                        reinterpret_cast<intptr_t>(info_.hCtx_))) {
        delete glenv_;
        glenv_ = NULL;
        result = -1000 /* CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR */;
      }
    }
  }

  return result;
}

void* Context::hostAlloc(size_t size, size_t alignment, bool atomics) const {
  if (customHostAllocDevice_ != NULL) {
    return customHostAllocDevice_->hostAlloc(
        size, alignment,
        atomics ? Device::MemorySegment::kAtomics : Device::MemorySegment::kNoAtomics);
  }
  return AlignedMemory::allocate(size, alignment);
}

void Context::hostFree(void* ptr) const {
  if (customHostAllocDevice_ != NULL) {
    customHostAllocDevice_->hostFree(ptr);
    return;
  }
  AlignedMemory::deallocate(ptr);
}

void* Context::svmAlloc(size_t size, size_t alignment, amd::MemFlags flags,
                        const amd::Device* curDev, void* svmPtr) {
  unsigned int numSVMDev = svmAllocDevice_.size();
  if (numSVMDev < 1) {
    return nullptr;
  }

  void* svmPtrAlloced = svmPtr;

  amd::ScopedLock lock(&ctxLock_);

  if (curDev != nullptr) {
    if (!static_cast<uint64_t>(flags & amd::MemFlags::SvmAtomics) ||
        static_cast<uint64_t>(curDev->info().svmCapabilities_ & amd::SvmCapabilities::Atomics)) {
      svmPtrAlloced = curDev->svmAlloc(*this, size, alignment, flags, svmPtrAlloced);
      if (svmPtrAlloced == nullptr) {
        return nullptr;
      }
    }
  }

  for (const auto& dev : svmAllocDevice_) {
    if (dev == curDev) {
      continue;
    }
    // check if the device support svm platform atomics,
    // skipped allocation for platform atomics if not supported by this device
    if (static_cast<uint64_t>(flags & amd::MemFlags::SvmAtomics) &&
        !static_cast<uint64_t>(dev->info().svmCapabilities_ & amd::SvmCapabilities::Atomics)) {
      continue;
    }
    svmPtrAlloced = dev->svmAlloc(*this, size, alignment, flags, svmPtrAlloced);
    if (svmPtrAlloced == nullptr) {
      return nullptr;
    }
  }
  return svmPtrAlloced;
}

void Context::svmFree(void* ptr) const {
  amd::ScopedLock lock(&ctxLock_);
  // Atomically remove from map before any device frees the GPU VA.
  // This prevents a concurrent allocation from reusing the same VA and being
  // wrongly freed by a subsequent device iteration (MGPU race).
  // The actual HSA free (release) is deferred until after all devices have
  // iterated, so KFD cannot reuse the VA during the loop.
  amd::Memory* svmMem = amd::MemObjMap::FindAndRemoveMemObj(ptr);
  for (const auto& dev : svmAllocDevice_) {
    dev->svmFree(ptr);  // FindMemObj returns nullptr → no-op for GPU path
  }
  if (svmMem != nullptr) {
    svmMem->release();
  }
}

bool Context::containsDevice(const Device* device) const {
  for (const auto& it : devices_) {
    if (device == it) {
      return true;
    }
  }
  return false;
}

DeviceQueue* Context::defDeviceQueue(const Device& dev) const {
  const auto it = deviceQueues_.find(&dev);
  if (it != deviceQueues_.cend()) {
    return it->second.defDeviceQueue_;
  } else {
    return NULL;
  }
}

bool Context::isDevQueuePossible(const Device& dev) {
  return (deviceQueues_[&dev].deviceQueueCnt_ < dev.info().maxOnDeviceQueues_) ? true : false;
}

void Context::addDeviceQueue(const Device& dev, DeviceQueue* queue, bool defDevQueue) {
  DeviceQueueInfo& info = deviceQueues_[&dev];
  info.deviceQueueCnt_++;
  if (defDevQueue) {
    info.defDeviceQueue_ = queue;
  }
}

void Context::removeDeviceQueue(const Device& dev, DeviceQueue* queue) {
  DeviceQueueInfo& info = deviceQueues_[&dev];
  assert((info.deviceQueueCnt_ != 0) && "The device queue map is empty!");
  info.deviceQueueCnt_--;
  if (info.defDeviceQueue_ == queue) {
    info.defDeviceQueue_ = NULL;
  }
}

}  // namespace amd
