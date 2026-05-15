/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "hip/hip_runtime.h"
#include "hip/hip_gl_interop.h"
#include "hip_internal.hpp"
#include "platform/interop_gl.hpp"
#include "cl_common.hpp"
#include <GL/gl.h>
#include <GL/glext.h>
#include "hip_conversions.hpp"
#include <mutex>
#include <shared_mutex>

namespace amd {
// Track the currently associated GL context for interop.
// Using shared_mutex to allow parallel operations when context is stable,
// while serializing only during context switches.
// When the current GL context differs from the associated one, re-setup is required.
static std::shared_mutex glInteropRWMutex;
static void* associatedGLContext = nullptr;
}

namespace hip {

// Helper to get current GL context using existing glenv, returns nullptr if glenv doesn't exist.
static void* getCurrentGLContext(amd::GLFunctions* glenv) {
  if (glenv == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return glenv->wglGetCurrentContext_();
#else
  return glenv->glXGetCurrentContext_();
#endif
}

// Helper to get current GL display/DC using existing glenv, returns nullptr if glenv doesn't exist.
static void* getCurrentGLDisplay(amd::GLFunctions* glenv) {
  if (glenv == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return glenv->wglGetCurrentDC_();
#else
  return glenv->glXGetCurrentDisplay_();
#endif
}

// Sets up GL context association with amd context.
// Handles both initial setup and GL context switches.
// Returns true on success, false on failure.
// NOTE: Refer to Context setup code in OCLTestImp.cpp
static bool setupGLInterop() {
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  amd::GLFunctions* glenv = amdContext->glenv();

  // Get the current GL context and display using existing glenv if available.
  // For first-time setup (glenv == nullptr), pass nullptr and let lower level read from current thread.
  // For context switch (glenv exists), explicitly pass the new context values.
  void* currentGLContext = getCurrentGLContext(glenv);
  void* currentGLDisplay = getCurrentGLDisplay(glenv);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)AMD_PLATFORM,
                                        ROCCLR_HIP_GL_CONTEXT_KHR,
                                        (cl_context_properties)currentGLContext,
#ifdef _WIN32
                                        ROCCLR_HIP_WGL_HDC_KHR,
                                        (cl_context_properties)currentGLDisplay,
#else
                                        ROCCLR_HIP_GLX_DISPLAY_KHR,
                                        (cl_context_properties)currentGLDisplay,
#endif
                                        0};

  amd::Context::Info info;
  if (CL_SUCCESS != amd::Context::checkProperties(properties, &info)) {
    LogError("Context setup failed: checkProperties");
    return false;
  }

  amdContext->setInfo(info);
  if (CL_SUCCESS != amdContext->create(properties)) {
    LogError("Context setup failed: create");
    return false;
  }
  return true;
}

// RAII guard ensuring GL interop validity for the duration of an operation.
// Uses read-write lock pattern: shared lock for fast path (parallel access when
// context is stable), exclusive lock for slow path (context setup/switch).
// Thread-safety: protects concurrent HIP GL interop operations. Application must
// not call wglMakeCurrent/glXMakeCurrent during HIP GL interop operations.
class GLInteropGuard {
 public:
  GLInteropGuard() : valid_(false) {
    amd::Context* amdContext = hip::getCurrentDevice()->asContext();
    amd::GLFunctions* glenv = amdContext->glenv();

    // Fast path: shared lock for parallel access when context is stable
    std::shared_lock<std::shared_mutex> readLock(amd::glInteropRWMutex);

    // If glenv doesn't exist, we need first-time setup
    if (glenv == nullptr) {
      readLock.unlock();
      std::unique_lock<std::shared_mutex> writeLock(amd::glInteropRWMutex);

      // Double-check after acquiring write lock
      glenv = amdContext->glenv();
      if (glenv == nullptr) {
        if (!setupGLInterop()) {
          return;
        }
        glenv = amdContext->glenv();
        if (glenv == nullptr) {
          return;
        }
        amd::associatedGLContext = getCurrentGLContext(glenv);
      }

      exclusiveLock_ = std::move(writeLock);
      valid_ = true;
      return;
    }

    // glenv exists, check if context has changed
    void* currentGLContext = getCurrentGLContext(glenv);
    if (currentGLContext == nullptr) {
      return;
    }

    if (amd::associatedGLContext == currentGLContext) {
      sharedLock_ = std::move(readLock);
      valid_ = true;
      return;
    }

    // Slow path: context switch detected, need exclusive lock for re-association
    readLock.unlock();
    std::unique_lock<std::shared_mutex> writeLock(amd::glInteropRWMutex);

    // Re-read context under exclusive lock (another thread may have completed setup)
    glenv = amdContext->glenv();
    void* verifiedContext = getCurrentGLContext(glenv);
    if (verifiedContext == nullptr) {
      return;
    }

    // Double-check pattern: context may have been set up during lock gap
    if (amd::associatedGLContext != verifiedContext) {
      if (!setupGLInterop()) {
        return;
      }
      amd::associatedGLContext = verifiedContext;
    }

    exclusiveLock_ = std::move(writeLock);
    valid_ = true;
  }

  ~GLInteropGuard() = default;

  GLInteropGuard(const GLInteropGuard&) = delete;
  GLInteropGuard& operator=(const GLInteropGuard&) = delete;
  GLInteropGuard(GLInteropGuard&&) = delete;
  GLInteropGuard& operator=(GLInteropGuard&&) = delete;

  bool isValid() const { return valid_; }

 private:
  std::shared_lock<std::shared_mutex> sharedLock_;
  std::unique_lock<std::shared_mutex> exclusiveLock_;
  bool valid_;
};

static inline hipError_t validateResources(hipGraphicsResource_t* resources, int count = 1) {
  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    return hipErrorNoDevice;
  }

  if (count <= 0) {
    LogError("invalid count");
    return hipErrorInvalidValue;
  }

  if (resources == nullptr) {
    return hipErrorInvalidValue;
  }

  for (int i = 0; i < count; i++) {
    if (resources[i] == nullptr) {
      return hipErrorInvalidValue;
    }
    if (!device->registeredGraphics().isValid(resources[i])) {
      return hipErrorInvalidHandle;
    }
    if (!device->mappedGraphics().isValid(resources[i])) {
      return hipErrorNotMapped;
    }
  }
  return hipSuccess;
}

static inline hipError_t hipSetInteropObjects(int num_objects, void** mem_objects,
                                              std::vector<amd::Memory*>& interopObjects) {
  if ((num_objects == 0 && mem_objects != nullptr) ||
      (num_objects != 0 && mem_objects == nullptr)) {
    return hipErrorUnknown;
  }

  while (num_objects-- > 0) {
    void* obj = *mem_objects++;
    if (obj == nullptr) {
      return hipErrorInvalidHandle;
    }

    amd::Memory* mem = reinterpret_cast<amd::Memory*>(obj);

    if (mem->getInteropObj() == nullptr) {
      return hipErrorInvalidHandle;
    }

    interopObjects.push_back(mem);
  }
  return hipSuccess;
}

// NOTE: This method cooresponds to OpenCL functionality in clGetGLContextInfoKHR()
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  HIP_INIT_API(hipGLGetDevices, pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList);

  // Guard holds the lock for entire function scope, preventing TOCTOU race
  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  constexpr bool VALIDATE_ONLY = true;
  if (deviceList == hipGLDeviceListNextFrame) {
    LogError("hipGLDeviceListNextFrame not supported yet");
    HIP_RETURN(hipErrorNotSupported);
  }
  if (pHipDeviceCount == nullptr || pHipDevices == nullptr || hipDeviceCount == 0) {
    LogError("Invalid Argument");
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceCount = std::min(hipDeviceCount, static_cast<unsigned int>(g_devices.size()));

  amd::Context::Info info = hip::getCurrentDevice()->asContext()->info();
  if (!(info.flags_ & amd::Context::GLDeviceKhr)) {
    LogError("Failed : Invalid Shared Group Reference");
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::GLFunctions* glenv = hip::getCurrentDevice()->asContext()->glenv();
  if (glenv != nullptr) {
#ifdef _WIN32
    info.hCtx_ = glenv->wglGetCurrentContext_();
#else
    info.hCtx_ = glenv->glXGetCurrentContext_();
#endif
    hip::getCurrentDevice()->asContext()->setInfo(info);
    glenv->update(reinterpret_cast<intptr_t>(info.hCtx_));
  }
  *pHipDeviceCount = 0;
  if (deviceList != hipGLDeviceListCurrentFrame && deviceList != hipGLDeviceListAll) {
    LogWarning("Invalid deviceList value");
    HIP_RETURN(hipErrorInvalidValue);
  }

  const bool findOnlyFirst = (deviceList == hipGLDeviceListCurrentFrame);
  unsigned int foundDeviceCount = 0;

  for (unsigned int i = 0; i < hipDeviceCount; ++i) {
    const std::vector<amd::Device*>& devices = g_devices[i]->devices();
    if (!devices.empty() &&
        devices[0]->bindExternalDevice(info.flags_, info.hDev_, info.hCtx_, VALIDATE_ONLY)) {
      pHipDevices[foundDeviceCount++] = i;
      if (findOnlyFirst) {
        break;
      }
    }
  }
  *pHipDeviceCount = foundDeviceCount;
  HIP_RETURN(*pHipDeviceCount > 0 ? hipSuccess : hipErrorNoDevice);
}

static inline void clearGLErrors(const amd::Context& amdContext) {
  GLenum glErr, glLastErr = GL_NO_ERROR;
  while (true) {
    glErr = amdContext.glenv()->glGetError_();
    if (glErr == GL_NO_ERROR || glErr == glLastErr) {
      break;
    }
    glLastErr = glErr;
    LogWarning("GL error");
  }
}

static inline GLenum checkForGLError(const amd::Context& amdContext) {
  GLenum glRetErr = GL_NO_ERROR;
  GLenum glErr;
  while (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
    glRetErr = glErr;  // Just return the last GL error
    LogWarning("Check GL error");
  }
  return glRetErr;
}

hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel) {
  HIP_INIT_API(hipGraphicsSubResourceGetMappedArray, array, resource, arrayIndex, mipLevel);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Context& amdContext = *(hip::getCurrentDevice()->asContext());
  if (array == nullptr) {
    LogError("invalid array");
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status = validateResources(&resource);
  if (status != hipSuccess) {
    LogError("invalid resource");
    HIP_RETURN(status);
  }

  amd::Image* image = (reinterpret_cast<amd::Memory*>(resource))->asImage();
  if (image == nullptr) {
    LogError("invalid resource/image");
    HIP_RETURN(hipErrorNotMappedAsArray);
  }
  // arrayIndex higher than zero not implemented
  if (arrayIndex > 0) {
    LogError("invalid arrayIndex, arrayIndex higher than zero not implemented");
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (mipLevel >= image->getMipLevels()) {
    LogError("invalid mipLevel");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Image* view = image->createView(amdContext, image->getImageFormat(), nullptr, mipLevel, 0);

  hipArray* myarray = new hipArray();

  myarray->data = as_cl<amd::Memory>(view);

  myarray->width = view->getWidth();
  myarray->height = view->getHeight();
  myarray->depth = view->getDepth();

  const cl_mem_object_type image_type =
      hip::getCLMemObjectType(myarray->width, myarray->height, myarray->depth, hipArrayDefault);
  myarray->type = image_type;
  amd::Image::Format f = image->getImageFormat();
  myarray->Format = hip::getCL2hipArrayFormat(f.image_channel_data_type);
  myarray->desc = hip::getChannelFormatDesc(f.getNumChannels(), myarray->Format);
  myarray->NumChannels = hip::getNumChannels(myarray->desc);
  myarray->isDrv = 0;
  myarray->textureType = 0;
  *array = myarray;
  {
    amd::ScopedLock lock(hip::hipArraySetLock);
    hip::hipArraySet.insert(*array);
  }
  HIP_RETURN(hipSuccess);
}

// Helper function to convert from OpenGL Flags to HIP Memory Flags
hipError_t HipToClMemoryFlags(uint32_t gl_flags, cl_mem_flags* cl_flags) {
  if (cl_flags == nullptr) {
    return hipErrorInvalidValue;
  }
  switch (gl_flags) {
      case hipGraphicsRegisterFlagsNone:
        *cl_flags = 0;
        break;
      case hipGraphicsRegisterFlagsReadOnly:
        *cl_flags = CL_MEM_READ_ONLY;
        break;
      case hipGraphicsRegisterFlagsWriteDiscard:
        *cl_flags = CL_MEM_WRITE_ONLY;
        break;
      case hipGraphicsRegisterFlagsSurfaceLoadStore:
        *cl_flags = CL_MEM_READ_WRITE;
        break;
      case hipGraphicsRegisterFlagsTextureGather:
        *cl_flags = CL_MEM_READ_WRITE | CL_MEM_READ_ONLY;
        break;
      default:
        return hipErrorInvalidValue;
        break;
  }
  return hipSuccess;
}

hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags) {
  HIP_INIT_API(hipGraphicsGLRegisterImage, resource, image, target, flags);

  // Guard holds the lock for entire function scope, detecting context switches
  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  cl_mem_flags cl_flags = 0;
  hipError_t status = HipToClMemoryFlags(flags, &cl_flags);
  if (status != hipSuccess) {
    LogPrintfError("invalid parameter \"flags\" %u, gl interop can not convert", flags);
    HIP_RETURN(status);
  }

  if (resource == nullptr) {
    LogError("invalid resource");
    HIP_RETURN(hipErrorInvalidValue);
  }

  GLint miplevel = 0;
  amd::Context& amdContext = *(hip::getCurrentDevice()->asContext());

  amd::GLFunctions::SetIntEnv ie(amdContext.glenv());
  if (!ie.isValid()) {
    LogWarning("\"amdContext\" is not created from GL context or share list");
    HIP_RETURN(hipErrorUnknown);
  }

  amd::ImageGL* pImageGL = nullptr;
  GLenum glErr;
  GLenum glTarget = 0;
  GLenum glInternalFormat;
  cl_image_format clImageFormat;
  uint dim = 1;
  cl_mem_object_type clType;
  cl_gl_object_type clGLType;
  GLsizei numSamples = 1;

  GLint gliTexWidth = 1;
  GLint gliTexHeight = 1;
  GLint gliTexDepth = 1;

  clearGLErrors(amdContext);
  if ((GL_FALSE == amdContext.glenv()->glIsTexture_(image)) ||
      (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
    LogWarning("\"texture\" is not a GL texture object");
    HIP_RETURN(hipErrorInvalidValue);
  }

  bool isImage = true;

  switch (target) {
    case GL_TEXTURE_BUFFER:
      glTarget = GL_TEXTURE_BUFFER;
      dim = 1;
      clType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
      clGLType = CL_GL_OBJECT_TEXTURE_BUFFER;
      isImage = false;
      break;

    case GL_TEXTURE_1D:
      glTarget = GL_TEXTURE_1D;
      dim = 1;
      clType = CL_MEM_OBJECT_IMAGE1D;
      clGLType = CL_GL_OBJECT_TEXTURE1D;
      break;

    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
      glTarget = GL_TEXTURE_CUBE_MAP;
      dim = 2;
      clType = CL_MEM_OBJECT_IMAGE2D;
      clGLType = CL_GL_OBJECT_TEXTURE2D;
      break;

    case GL_TEXTURE_1D_ARRAY:
      glTarget = GL_TEXTURE_1D_ARRAY;
      dim = 2;
      clType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
      clGLType = CL_GL_OBJECT_TEXTURE1D_ARRAY;
      break;

    case GL_TEXTURE_2D:
      glTarget = GL_TEXTURE_2D;
      dim = 2;
      clType = CL_MEM_OBJECT_IMAGE2D;
      clGLType = CL_GL_OBJECT_TEXTURE2D;
      break;

    case GL_TEXTURE_2D_MULTISAMPLE:
      glTarget = GL_TEXTURE_2D_MULTISAMPLE;
      dim = 2;
      clType = CL_MEM_OBJECT_IMAGE2D;
      clGLType = CL_GL_OBJECT_TEXTURE2D;
      break;

    case GL_TEXTURE_RECTANGLE_ARB:
      glTarget = GL_TEXTURE_RECTANGLE_ARB;
      dim = 2;
      clType = CL_MEM_OBJECT_IMAGE2D;
      clGLType = CL_GL_OBJECT_TEXTURE2D;
      break;

    case GL_TEXTURE_2D_ARRAY:
      glTarget = GL_TEXTURE_2D_ARRAY;
      dim = 3;
      clType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      clGLType = CL_GL_OBJECT_TEXTURE2D_ARRAY;
      break;

    case GL_TEXTURE_3D:
      glTarget = GL_TEXTURE_3D;
      dim = 3;
      clType = CL_MEM_OBJECT_IMAGE3D;
      clGLType = CL_GL_OBJECT_TEXTURE3D;
      break;

    default:
      LogWarning("invalid \"target\" value");
      HIP_RETURN(hipErrorInvalidValue);
  }
  amdContext.glenv()->glBindTexture_(glTarget, image);

  if (isImage) {
    GLint gliTexBaseLevel;
    GLint gliTexMaxLevel;

    clearGLErrors(amdContext);
    amdContext.glenv()->glGetTexParameteriv_(glTarget, GL_TEXTURE_BASE_LEVEL, &gliTexBaseLevel);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get base mipmap level of a GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetTexParameteriv_(glTarget, GL_TEXTURE_MAX_LEVEL, &gliTexMaxLevel);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get max mipmap level of a GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    if ((gliTexBaseLevel > miplevel) || (miplevel > gliTexMaxLevel)) {
      LogWarning("\"miplevel\" is not a valid mipmap level of the GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    clearGLErrors(amdContext);
    amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_INTERNAL_FORMAT,
                                                  (GLint*)&glInternalFormat);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_SAMPLES,
                                                  (GLint*)&numSamples);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get number of samples of GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }
    if (numSamples > 1) {
      LogWarning("MSAA \"texture\" object is not supported for the device");
      HIP_RETURN(hipErrorInvalidValue);
    }

    int iBytesPerPixel = 0;
    if (!amd::getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel, 0)) {
      LogWarning("\"texture\" format does not map to an appropriate CL image format");
      HIP_RETURN(hipErrorInvalidValue);
    }

    switch (dim) {
      case 3:
        clearGLErrors(amdContext);
        amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_DEPTH,
                                                      &gliTexDepth);
        if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
          LogWarning("Cannot get the depth of \"miplevel\" of GL \"texture\"");
          HIP_RETURN(hipErrorInvalidValue);
        }
        [[fallthrough]];
      case 2:
        clearGLErrors(amdContext);
        amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_HEIGHT,
                                                      &gliTexHeight);
        if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
          LogWarning("Cannot get the height of \"miplevel\" of GL \"texture\"");
          HIP_RETURN(hipErrorInvalidValue);
        }
        [[fallthrough]];
      case 1:
        clearGLErrors(amdContext);
        amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_WIDTH,
                                                      &gliTexWidth);
        if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
          LogWarning("Cannot get the width of \"miplevel\" of GL \"texture\"");
          HIP_RETURN(hipErrorInvalidValue);
        }
        break;
      default:
        LogWarning("invalid \"target\" value");
        HIP_RETURN(hipErrorInvalidValue);
    }

  } else {
    GLint size;
    GLint backingBuffer;
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetTexLevelParameteriv_(glTarget, 0, GL_TEXTURE_BUFFER_DATA_STORE_BINDING,
                                                  &backingBuffer);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get backing buffer for GL \"texture buffer\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }
    amdContext.glenv()->glBindBuffer_(glTarget, backingBuffer);

    clearGLErrors(amdContext);
    amdContext.glenv()->glGetIntegerv_(GL_TEXTURE_BUFFER_FORMAT_EXT,
                                       reinterpret_cast<GLint*>(&glInternalFormat));
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    int iBytesPerPixel = 0;
    if (!amd::getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel,
                                cl_flags)) {
      LogWarning("\"texture\" format does not map to an appropriate CL image format");
      HIP_RETURN(hipErrorInvalidValue);
    }

    clearGLErrors(amdContext);
    amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &size);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    gliTexWidth = size / iBytesPerPixel;
  }
  size_t imageSize = (clType == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? static_cast<size_t>(gliTexHeight)
                                                             : static_cast<size_t>(gliTexDepth);

  if (!amd::Image::validateDimensions(
          amdContext.devices(), clType, static_cast<size_t>(gliTexWidth),
          static_cast<size_t>(gliTexHeight), static_cast<size_t>(gliTexDepth), imageSize)) {
    LogWarning("The GL \"texture\" data store is not created or out of supported dimensions");
    HIP_RETURN(hipErrorInvalidValue);
  }
  target = (glTarget == GL_TEXTURE_CUBE_MAP) ? target : 0;

  pImageGL = new (amdContext)
      amd::ImageGL(amdContext, clType, cl_flags, clImageFormat, static_cast<size_t>(gliTexWidth),
                   static_cast<size_t>(gliTexHeight), static_cast<size_t>(gliTexDepth), glTarget,
                   image, 0, glInternalFormat, clGLType, numSamples, target);
  if (!pImageGL->create()) {
    pImageGL->release();
    HIP_RETURN(hipErrorUnknown);
  }
  // Create interop object
  if (pImageGL->getInteropObj() == nullptr) {
    LogWarning("cannot create interop object for ImageGL");
    pImageGL->release();
    HIP_RETURN(hipErrorUnknown);
  }
  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext.devices().size() == 1);

  const amd::Device& dev = *(amdContext.devices()[0]);

  device::Memory* mem = pImageGL->getDeviceMemory(dev);
  if (nullptr == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", pImageGL->getSize());
    pImageGL->release();
    HIP_RETURN(hipErrorUnknown);
  }
  mem->processGLResource(device::Memory::GLDecompressResource);

  *resource = reinterpret_cast<hipGraphicsResource*>(pImageGL);

  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    return hipErrorNoDevice;
  }

  if (!device->registeredGraphics().add(*resource)) {
    LogError("duplicate resource");
    HIP_RETURN(hipErrorUnknown);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  HIP_INIT_API(hipGraphicsGLRegisterBuffer, resource, buffer, flags);

  // Guard holds the lock for entire function scope, detecting context switches
  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Validate flags: SurfaceLoadStore and TextureGather are image-specific, not valid for buffers
  if ((flags & hipGraphicsRegisterFlagsSurfaceLoadStore) ||
      (flags & hipGraphicsRegisterFlagsTextureGather)) {
    LogError("invalid flags for buffer registration: SurfaceLoadStore and TextureGather are image-specific");
    HIP_RETURN(hipErrorInvalidValue);
  }

  cl_mem_flags cl_flags = 0;
  hipError_t status = HipToClMemoryFlags(flags, &cl_flags);
  if (status != hipSuccess) {
    LogPrintfError("invalid parameter \"flags\" %u, gl interop can not convert", flags);
    HIP_RETURN(status);
  }

  if (resource == nullptr) {
    LogError("invalid resource");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::BufferGL* pBufferGL = nullptr;
  GLenum glErr;
  GLenum glTarget = GL_ARRAY_BUFFER;
  GLint gliSize = 0;

  amd::Context& amdContext = *(hip::getCurrentDevice()->asContext());

  // Add this scope to bound the scoped lock
  {
    amd::GLFunctions::SetIntEnv ie(amdContext.glenv());
    if (!ie.isValid()) {
      LogWarning("\"amdContext\" is not created from GL context or share list");
      HIP_RETURN(hipErrorUnknown);
    }

    // Verify GL buffer object
    clearGLErrors(amdContext);
    if ((GL_FALSE == amdContext.glenv()->glIsBuffer_(buffer)) ||
        (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
      LogWarning("\"buffer\" is not a GL buffer object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    // Check if size is available - data store is created
    amdContext.glenv()->glBindBuffer_(glTarget, buffer);
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &gliSize);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("cannot get the GL buffer size");
      HIP_RETURN(hipErrorInvalidValue);
    }
    if (gliSize == 0) {
      LogWarning("the GL buffer's data store is not created");
      HIP_RETURN(hipErrorInvalidValue);
    }

  }  // Release scoped lock

  // Now create BufferGL object
  pBufferGL = new (amdContext) amd::BufferGL(amdContext, cl_flags, gliSize, 0, buffer);
  if (!pBufferGL->create()) {
    pBufferGL->release();
    HIP_RETURN(hipErrorUnknown);
  }

  // Create interop object
  if (pBufferGL->getInteropObj() == nullptr) {
    LogWarning("cannot create interop object for BufferGL");
    pBufferGL->release();
    HIP_RETURN(hipErrorUnknown);
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext.devices().size() == 1);

  const auto it = amdContext.devices().cbegin();
  const amd::Device& dev = *(*it);

  device::Memory* mem = pBufferGL->getDeviceMemory(dev);
  if (nullptr == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", pBufferGL->getSize());
    HIP_RETURN(hipErrorUnknown);
  }
  mem->processGLResource(device::Memory::GLDecompressResource);

  *resource = reinterpret_cast<hipGraphicsResource*>(pBufferGL);

  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    return hipErrorNoDevice;
  }

  if (!device->registeredGraphics().add(*resource)) {
    LogError("duplicate resource");
    HIP_RETURN(hipErrorUnknown);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  HIP_INIT_API(hipGraphicsMapResources, count, resources, stream);

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  hipError_t status = validateResources(resources, count);
  if (status != hipErrorNotMapped) {
    LogError("invalid resource(s)");
    if (status == hipSuccess) {
      status = hipErrorAlreadyMapped;
    }
    HIP_RETURN(status);
  }

  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  if (!amdContext || !amdContext->glenv()) {
    HIP_RETURN(hipErrorUnknown);
  }
  clearGLErrors(*amdContext);
  amdContext->glenv()->glFinish_();
  if (checkForGLError(*amdContext) != GL_NO_ERROR) {
    HIP_RETURN(hipErrorUnknown);
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  if (nullptr == hip_stream) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  if (!hip_stream->context().glenv() || !hip_stream->context().glenv()->isAssociated()) {
    LogWarning("\"amdContext\" is not created from GL context or share list");
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  std::vector<amd::Memory*> memObjects;
  hipError_t err = hipSetInteropObjects(count, reinterpret_cast<void**>(resources), memObjects);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

  amd::Command::EventWaitList nullWaitList;

  //! Now create command and enqueue
  amd::AcquireExtObjectsCommand* command = new amd::AcquireExtObjectsCommand(
      *hip_stream, nullWaitList, count, memObjects, CL_COMMAND_ACQUIRE_GL_OBJECTS);
  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorUnknown);
  }

  command->enqueue();

  if (as_cl(&command->event()) == nullptr) {
    command->release();
  }

  const auto it = amdContext->devices().cbegin();
  amd::Device* curDev = *it;
  for (auto& mobj : memObjects) {
    device::Memory* mem = reinterpret_cast<device::Memory*>(mobj->getDeviceMemory(*curDev));
    amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(mem->virtualAddress()), mobj);
    mobj->retain();
  }
  // Track mapping status
  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    return hipErrorNoDevice;
  }
  for (int i = 0; i < count; i++) {
    if (!device->mappedGraphics().add(resources[i])) {
      HIP_RETURN(hipErrorMapFailed);
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsResourceGetMappedPointer, devPtr, size, resource);

  if (devPtr == nullptr) {
    LogError("invalid device pointer");
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (size == nullptr) {
    LogError("invalid size");
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status = validateResources(&resource);
  if (status != hipSuccess) {
    LogError("invalid resource");
    HIP_RETURN(status);
  }

  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  if (!amdContext || !amdContext->glenv()) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext->devices().size() == 1);

  const auto it = amdContext->devices().cbegin();

  amd::Device* curDev = *it;
  amd::Memory* amdMem = reinterpret_cast<amd::Memory*>(resource);

  // Check if not a buffer
  amd::Buffer* buffer = amdMem->asBuffer();
  if (buffer == nullptr) {
    LogError("resource not mapped as pointer");
    HIP_RETURN(hipErrorNotMappedAsPointer);
  }

  *size = amdMem->getSize();

  // Interop resources don't have svm allocations they are added to
  // amd::MemObjMap using device virtual address during creation.
  device::Memory* devMem = reinterpret_cast<device::Memory*>(amdMem->getDeviceMemory(*curDev));

  // Not mapped
  if (devMem == nullptr) {
    LogError("resource not mapped");
    HIP_RETURN(hipErrorNotMapped);
  }

  *devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(devMem->virtualAddress()));
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  HIP_INIT_API(hipGraphicsUnmapResources, count, resources, stream);
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  // Wait for the current host queue
  hip::getStream(stream)->finish();

  hipError_t status = validateResources(resources, count);
  if (status != hipSuccess) {
    LogError("resource(s) not mapped");
    HIP_RETURN(status);
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  if (nullptr == hip_stream) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  std::vector<amd::Memory*> memObjects;
  hipError_t err = hipSetInteropObjects(count, reinterpret_cast<void**>(resources), memObjects);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

  amd::Command::EventWaitList nullWaitList;

  // Now create command and enqueue
  amd::ReleaseExtObjectsCommand* command = new amd::ReleaseExtObjectsCommand(
      *hip_stream, nullWaitList, count, memObjects, CL_COMMAND_RELEASE_GL_OBJECTS);
  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorUnknown);
  }

  command->enqueue();

  if (as_cl(&command->event()) == nullptr) {
    command->release();
  }

  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    HIP_RETURN(hipErrorNoDevice);
  }

  const amd::Device* curDev = device->devices()[0];
  for (auto& mobj : memObjects) {
    device::Memory* mem = reinterpret_cast<device::Memory*>(mobj->getDeviceMemory(*curDev));
    if (mem) {
      amd::MemObjMap::RemoveMemObj(reinterpret_cast<void*>(mem->virtualAddress()));
    }
    mobj->release();
  }

  // Remove mapping from registry
  for (uint8_t i = 0; i < count; i++) {
    if (!device->mappedGraphics().remove(resources[i])) {
      LogError("failed to unmap resource");
      HIP_RETURN(hipErrorUnknown);
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsUnregisterResource, resource);

  if (resource == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    LogError("no device");
    HIP_RETURN(hipErrorNoDevice);
  }

  if (device->mappedGraphics().isValid(resource)) {
    LogError("resource already mapped");
    HIP_RETURN(hipErrorAlreadyMapped);
  }

  if (!device->registeredGraphics().isValid(resource)) {
    LogError("resource not registered");
    HIP_RETURN(hipErrorInvalidHandle);
  }

  // Safe to remove from registered list
  if (!device->registeredGraphics().remove(resource)) {
    LogError("failed to unregister resource");
    HIP_RETURN(hipErrorUnknown);
  }

  reinterpret_cast<amd::BufferGL*>(resource)->release();

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
