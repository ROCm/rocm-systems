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

// Thread-safe one-time initialization for GL function pointers
static std::once_flag glFuncInitFlag;
#ifdef _WIN32
typedef void* (WINAPI* PFN_wglGetCurrentContext)(void);
typedef void* (WINAPI* PFN_wglGetCurrentDC)(void);
static PFN_wglGetCurrentContext wglGetCurrentContext_ptr = nullptr;
static PFN_wglGetCurrentDC wglGetCurrentDC_ptr = nullptr;
#else
typedef void* (*PFN_glXGetCurrentContext)(void);
typedef void* (*PFN_glXGetCurrentDisplay)(void);
static PFN_glXGetCurrentContext glXGetCurrentContext_ptr = nullptr;
static PFN_glXGetCurrentDisplay glXGetCurrentDisplay_ptr = nullptr;
#endif
}

namespace hip {

// Thread-safe one-time initialization of GL function pointers
static void initGLFunctionPointers() {
#ifdef _WIN32
  HMODULE hOpenGL = GetModuleHandleA("opengl32.dll");
  if (hOpenGL == nullptr) {
    hOpenGL = LoadLibraryA("opengl32.dll");
  }
  if (hOpenGL != nullptr) {
    amd::wglGetCurrentContext_ptr =
        (amd::PFN_wglGetCurrentContext)GetProcAddress(hOpenGL, "wglGetCurrentContext");
    amd::wglGetCurrentDC_ptr =
        (amd::PFN_wglGetCurrentDC)GetProcAddress(hOpenGL, "wglGetCurrentDC");
  }
#else
  void* libGL = dlopen("libGL.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (libGL == nullptr) {
    libGL = dlopen("libGL.so.1", RTLD_NOW);
  }
  if (libGL != nullptr) {
    amd::glXGetCurrentContext_ptr =
        (amd::PFN_glXGetCurrentContext)dlsym(libGL, "glXGetCurrentContext");
    amd::glXGetCurrentDisplay_ptr =
        (amd::PFN_glXGetCurrentDisplay)dlsym(libGL, "glXGetCurrentDisplay");
  }
#endif
}

// Helper function to get the current GL context without requiring glenv to be initialized
// Uses std::call_once for thread-safe initialization of function pointers
static void* getCurrentGLContextRaw() {
  std::call_once(amd::glFuncInitFlag, initGLFunctionPointers);

#ifdef _WIN32
  return amd::wglGetCurrentContext_ptr ? amd::wglGetCurrentContext_ptr() : nullptr;
#else
  return amd::glXGetCurrentContext_ptr ? amd::glXGetCurrentContext_ptr() : nullptr;
#endif
}

// Helper function to get the current GL display/DC
// Uses the same cached function pointers initialized by initGLFunctionPointers()
static void* getCurrentGLDisplayRaw() {
  std::call_once(amd::glFuncInitFlag, initGLFunctionPointers);

#ifdef _WIN32
  return amd::wglGetCurrentDC_ptr ? amd::wglGetCurrentDC_ptr() : nullptr;
#else
  return amd::glXGetCurrentDisplay_ptr ? amd::glXGetCurrentDisplay_ptr() : nullptr;
#endif
}

// Sets up GL context association with amd context.
// Handles both initial setup and GL context switches.
// Returns true on success, false on failure.
// NOTE: Refer to Context setup code in OCLTestImp.cpp
static bool setupGLInterop() {
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();

  // Get the current GL context and display BEFORE setting up properties.
  // This ensures we use the same function pointers that getCurrentGLContextRaw() uses,
  // avoiding library handle mismatches between dlopen and Os::loadLibrary.
  void* currentGLContext = getCurrentGLContextRaw();
  if (currentGLContext == nullptr) {
    LogError("GL interop setup failed: no current GL context");
    return false;
  }

  // Get current display using the cached function pointers
  void* currentGLDisplay = getCurrentGLDisplayRaw();

  // Pass the actual GL context and display in the properties instead of nullptr.
  // This avoids the issue where Context::create() skips context retrieval
  // when glenv_ is already initialized from a previous setup.
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

// Convert HIP graphics register flags to OpenCL memory flags.
// HIP flags indicate access patterns, CL flags indicate memory properties.
// The mapping preserves the semantic meaning (read-only, write-only, read-write).
static cl_mem_flags convertHipGraphicsFlagsToCL(unsigned int hipFlags) {
  // Check for read-only flags
  if (hipFlags & hipGraphicsRegisterFlagsReadOnly ||
      hipFlags & hipGraphicsRegisterFlagsTextureGather) {
    return CL_MEM_READ_ONLY;
  }
  // Check for write-only flag
  if (hipFlags & hipGraphicsRegisterFlagsWriteDiscard) {
    return CL_MEM_WRITE_ONLY;
  }
  // Default to read-write (covers FlagsNone and FlagsSurfaceLoadStore)
  return CL_MEM_READ_WRITE;
}

// RAII guard ensuring GL interop validity for the duration of an operation.
// Uses read-write lock pattern: shared lock for fast path (parallel access when
// context is stable), exclusive lock for slow path (context setup/switch).
// Thread-safety: protects concurrent HIP GL interop operations. Application must
// not call wglMakeCurrent/glXMakeCurrent during HIP GL interop operations.
class GLInteropGuard {
public:
  GLInteropGuard() : valid_(false) {
    // Fast path: shared lock for parallel access when context is stable
    std::shared_lock<std::shared_mutex> readLock(amd::glInteropRWMutex);

    void* currentGLContext = getCurrentGLContextRaw();
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
    void* verifiedContext = getCurrentGLContextRaw();
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

static inline hipError_t hipSetInteropObjects(int num_objects, void** mem_objects,
                                              std::vector<amd::Memory*>& interopObjects) {
  if ((num_objects == 0 && mem_objects != nullptr) ||
      (num_objects != 0 && mem_objects == nullptr)) {
    return hipErrorUnknown;
  }

  while (num_objects-- > 0) {
    void* obj = *mem_objects++;
    if (obj == nullptr) {
      return hipErrorInvalidResourceHandle;
    }

    amd::Memory* mem = reinterpret_cast<amd::Memory*>(obj);

    if (mem->getInteropObj() == nullptr) {
      return hipErrorInvalidResourceHandle;
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
  if (array == nullptr || resource == nullptr) {
    LogError("invalid array/resource");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Image* image = (reinterpret_cast<amd::Memory*>(resource))->asImage();
  if (image == nullptr) {
    LogError("invalid resource/image");
    HIP_RETURN(hipErrorInvalidValue);
  }
  // arrayIndex higher than zero not implemented
  if (arrayIndex > 0) {
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

hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags) {
  HIP_INIT_API(hipGraphicsGLRegisterImage, resource, image, target, flags);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Valid flags for image registration (can be combined or None)
  constexpr unsigned int kValidImageFlags =
      hipGraphicsRegisterFlagsReadOnly | hipGraphicsRegisterFlagsWriteDiscard |
      hipGraphicsRegisterFlagsSurfaceLoadStore | hipGraphicsRegisterFlagsTextureGather;
  // Reject if any bits outside the valid mask are set
  if (flags & ~kValidImageFlags) {
    LogError("invalid parameter \"flags\"");
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (resource == nullptr) {
    LogError("invalid resource");
    HIP_RETURN(hipErrorInvalidValue);
  }

  GLint miplevel = 0;
  amd::Context& amdContext = *(hip::getCurrentDevice()->asContext());

  if (amdContext.glenv() == nullptr) {
    LogError("invalid context, gl interop not initialized");
    HIP_RETURN(hipErrorInvalidValue);
  }

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

    // Get GL texture format and check if it's compatible with CL format
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetIntegerv_(GL_TEXTURE_BUFFER_FORMAT_EXT,
                                       reinterpret_cast<GLint*>(&glInternalFormat));
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
      HIP_RETURN(hipErrorInvalidValue);
    }

    // Now get CL format from GL format and bytes per pixel
    int iBytesPerPixel = 0;
    if (!amd::getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel, flags)) {
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

  cl_mem_flags clFlags = convertHipGraphicsFlagsToCL(flags);
  pImageGL = new (amdContext)
      amd::ImageGL(amdContext, clType, clFlags, clImageFormat, static_cast<size_t>(gliTexWidth),
                   static_cast<size_t>(gliTexHeight), static_cast<size_t>(gliTexDepth), glTarget,
                   image, 0, glInternalFormat, clGLType, numSamples, target);

  if (!pImageGL) {
    LogWarning("Cannot create class ImageGL - out of memory?");
    HIP_RETURN(hipErrorUnknown);
  }

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
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  HIP_INIT_API(hipGraphicsGLRegisterBuffer, resource, buffer, flags);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Valid flags for buffer registration (can be combined or None)
  constexpr unsigned int kValidBufferFlags =
      hipGraphicsRegisterFlagsReadOnly | hipGraphicsRegisterFlagsWriteDiscard;
  // Reject if any bits outside the valid mask are set
  if (flags & ~kValidBufferFlags) {
    LogError("invalid parameter \"flags\"");
    HIP_RETURN(hipErrorInvalidValue);
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

  if (amdContext.glenv() == nullptr) {
    LogError("invalid context, gl interop not initialized");
    HIP_RETURN(hipErrorInvalidValue);
  }

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
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }

    // Check if size is available - data store is created
    amdContext.glenv()->glBindBuffer_(glTarget, buffer);
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &gliSize);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("cannot get the GL buffer size");
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }
    if (gliSize == 0) {
      LogWarning("the GL buffer's data store is not created");
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }

  }  // Release scoped lock

  // Now create BufferGL object
  cl_mem_flags clFlags = convertHipGraphicsFlagsToCL(flags);
  pBufferGL = new (amdContext) amd::BufferGL(amdContext, clFlags, gliSize, 0, buffer);

  if (!pBufferGL) {
    LogWarning("cannot create object of class BufferGL");
    HIP_RETURN(hipErrorUnknown);
  }

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

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  HIP_INIT_API(hipGraphicsMapResources, count, resources, stream);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
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
    HIP_RETURN(hipErrorUnknown);
  }

  if (!hip_stream->context().glenv() || !hip_stream->context().glenv()->isAssociated()) {
    LogWarning("\"amdContext\" is not created from GL context or share list");
    HIP_RETURN(hipErrorUnknown);
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
  if (command == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

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
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsResourceGetMappedPointer, devPtr, size, resource);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  if (!amdContext || !amdContext->glenv()) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext->devices().size() == 1);

  amd::Device* curDev = *(amdContext->devices().cbegin());
  amd::Memory* amdMem = reinterpret_cast<amd::Memory*>(resource);
  *size = amdMem->getSize();

  // Interop resources don't have svm allocations they are added to
  // amd::MemObjMap using device virtual address during creation.
  device::Memory* mem = reinterpret_cast<device::Memory*>(amdMem->getDeviceMemory(*curDev));
  *devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(mem->virtualAddress()));
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  HIP_INIT_API(hipGraphicsUnmapResources, count, resources, stream);

  GLInteropGuard glGuard;
  if (!glGuard.isValid()) {
    LogError("No GL context is current");
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  if (nullptr == hip_stream) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Wait for the current host queue
  hip_stream->finish();

  std::vector<amd::Memory*> memObjects;
  hipError_t err = hipSetInteropObjects(count, reinterpret_cast<void**>(resources), memObjects);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

  amd::Command::EventWaitList nullWaitList;

  // Now create command and enqueue
  amd::ReleaseExtObjectsCommand* command = new amd::ReleaseExtObjectsCommand(
      *hip_stream, nullWaitList, count, memObjects, CL_COMMAND_RELEASE_GL_OBJECTS);
  if (command == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorUnknown);
  }

  command->enqueue();

  if (as_cl(&command->event()) == nullptr) {
    command->release();
  }
  for (auto& mobj : memObjects) {
    mobj->release();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsUnregisterResource, resource);

  if (resource == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Cast to amd::Memory* (base class) since resource can be either BufferGL or ImageGL
  reinterpret_cast<amd::Memory*>(resource)->release();

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
