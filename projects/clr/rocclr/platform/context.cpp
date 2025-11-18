/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#include "platform/context.hpp"
#include "platform/interop_gl.hpp"
#include "vdi_common.hpp"
#include "platform/commandqueue.hpp"

#include <algorithm>
#include <functional>

#ifdef _WIN32
#include <d3d10_1.h>
#include <dxgi.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
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

int Context::checkProperties(const cl_context_properties* properties, Context::Info* info) {
  cl_platform_id pfmId = 0;
  uint count = 0;

  const struct Element {
    intptr_t name;
    void* ptr;
  }* p = reinterpret_cast<const Element*>(properties);

  // Clear the context infor structure
  ::memset(info, 0, sizeof(Context::Info));

  if (properties == NULL) {
    return CL_SUCCESS;
  }

  // Process all properties
  while (p->name != 0) {
    switch (p->name) {
      case CL_CONTEXT_INTEROP_USER_SYNC:
        if (p->ptr == reinterpret_cast<void*>(true)) {
          info->flags_ |= InteropUserSync;
        }
        break;
#ifdef _WIN32
      case CL_CONTEXT_D3D10_DEVICE_KHR:
        if (p->ptr == NULL) {
          return CL_INVALID_VALUE;
        }
        info->hDev_[D3D10DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D10DeviceKhr;
        break;
      case CL_CONTEXT_D3D11_DEVICE_KHR:
        if (p->ptr == NULL) {
          return CL_INVALID_VALUE;
        }
        info->hDev_[D3D11DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D11DeviceKhr;
        break;
      case CL_CONTEXT_ADAPTER_D3D9_KHR:
        if (p->ptr == NULL) {  // not supported for xp
          return CL_INVALID_VALUE;
        }
        info->hDev_[D3D9DeviceKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceKhr;
        break;
      case CL_CONTEXT_ADAPTER_D3D9EX_KHR:
        if (p->ptr == NULL) {
          return CL_INVALID_VALUE;
        }
        info->hDev_[D3D9DeviceEXKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceEXKhr;
        break;
      case CL_CONTEXT_ADAPTER_DXVA_KHR:
        if (p->ptr == NULL) {
          return CL_INVALID_VALUE;
        }
        info->hDev_[D3D9DeviceVAKhrIdx] = p->ptr;
        info->flags_ |= D3D9DeviceVAKhr;
        break;
#endif  //_WIN32

      case CL_EGL_DISPLAY_KHR:
        info->flags_ |= EGLDeviceKhr;

#ifdef _WIN32
      case CL_WGL_HDC_KHR:
#endif  //_WIN32

#if defined(__linux__)
      case CL_GLX_DISPLAY_KHR:
#endif  // linux
        if (p->ptr == NULL) {
          return CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
        }
      case ROCCLR_HIP_GLX_DISPLAY_KHR:
      case ROCCLR_HIP_WGL_HDC_KHR:
        info->hDev_[GLDeviceKhrIdx] = p->ptr;
        break;
#if defined(__APPLE__) || defined(__MACOSX)
      case CL_CGL_SHAREGROUP_KHR:
        Unimplemented();
        break;
#endif  //__APPLE__ || MACOS
      case CL_GL_CONTEXT_KHR:
        if (p->ptr == NULL) {
          return CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
        }
        // skip the null case in the case of hip-gl, it will be initialized in create
      case ROCCLR_HIP_GL_CONTEXT_KHR:
        info->hCtx_ = p->ptr;
        info->flags_ |= GLDeviceKhr;
        break;
      case CL_CONTEXT_PLATFORM:
        pfmId = reinterpret_cast<cl_platform_id>(p->ptr);
        if ((NULL != pfmId) && (AMD_PLATFORM != pfmId)) {
          return CL_INVALID_VALUE;
        }
        break;
      case CL_CONTEXT_OFFLINE_DEVICES_AMD:
        if (p->ptr != reinterpret_cast<void*>(1)) {
          return CL_INVALID_VALUE;
        }
        // Set the offline device flag
        info->flags_ |= OfflineDevices;
        break;
      default:
        return CL_INVALID_VALUE;
    }
    p++;
    count++;
  }

  info->propertiesSize_ = count * sizeof(Element) + sizeof(intptr_t);
  return CL_SUCCESS;
}

int Context::create(const intptr_t* properties) {
  static const bool VALIDATE_ONLY = false;
  int result = CL_SUCCESS;

  if (properties != NULL) {
    properties_ = new cl_context_properties[info().propertiesSize_ / sizeof(cl_context_properties)];
    if (properties_ == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
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
        result = CL_INVALID_VALUE;
      }
    }
  }

  // Check if the device binding wasn't successful
  if (result != CL_SUCCESS) {
    if (info_.flags_ & GLDeviceKhr) {
      result = CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
    } else if (info_.flags_ & D3D10DeviceKhr) {
      // return CL_INVALID_VALUE; // FIXME_odintsov: CL_INVALID_D3D_INTEROP;
    } else if (info_.flags_ & D3D11DeviceKhr) {
      // return CL_INVALID_VALUE; // FIXME_odintsov: CL_INVALID_D3D_INTEROP;
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
          return CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
        }
      }
      if (!glenv_->init(reinterpret_cast<intptr_t>(info_.hDev_[GLDeviceKhrIdx]),
                        reinterpret_cast<intptr_t>(info_.hCtx_))) {
        delete glenv_;
        glenv_ = NULL;
        result = CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
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

void* Context::svmAlloc(size_t size, size_t alignment, cl_svm_mem_flags flags,
                        const amd::Device* curDev, void* svmPtr) {
  unsigned int numSVMDev = svmAllocDevice_.size();
  if (numSVMDev < 1) {
    return nullptr;
  }

  void* svmPtrAlloced = svmPtr;

  amd::ScopedLock lock(&ctxLock_);

  if (curDev != nullptr) {
    if (!(flags & CL_MEM_SVM_ATOMICS) ||
        (curDev->info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS)) {
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
    if ((flags & CL_MEM_SVM_ATOMICS) && !(dev->info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS)) {
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
  for (const auto& dev : svmAllocDevice_) {
    dev->svmFree(ptr);
  }
  return;
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
