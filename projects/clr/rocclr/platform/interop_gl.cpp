/* Copyright (c) 2010 - 2023 Advanced Micro Devices, Inc.

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

#ifdef _WIN32
#include <d3d10_1.h>
#include <d3d9.h>
#include <dxgi.h>
#endif  //_WIN32

#include <GL/gl.h>
#include <GL/glext.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>

#include "cl_common.hpp"

#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/interop_gl.hpp"

/* The pixel internal format for DOPP texture defined in gl_enum.h */
#define GL_BGR8_ATI 0x8083
#define GL_BGRA8_ATI 0x8088

#include <cstring>
#include <vector>

// Placed here as opposed to command.cpp, as glext.h and cl_gl_amd.hpp will have
// to be included because of the GL calls
bool amd::ClGlEvent::waitForFence() {
  GLenum ret;
  // get fence id associated with fence event
  GLsync gs =
      !command().data().empty() ? reinterpret_cast<GLsync>(command().data().back()) : nullptr;
  if (!gs) return false;

  // Try to use DC and GLRC of current thread, if it doesn't exist
  // create a new GL context on this thread, which is shared with the original context

#ifdef _WIN32
  HDC tempDC_ = context().glenv()->wglGetCurrentDC_();
  HGLRC tempGLRC_ = context().glenv()->wglGetCurrentContext_();
  // Set DC and GLRC
  if (tempDC_ && tempGLRC_) {
    amd::GLFunctions::Lock lock(context().glenv());
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  } else {
    tempDC_ = context().glenv()->getDC();
    tempGLRC_ = context().glenv()->getIntGLRC();
    if (!context().glenv()->init(reinterpret_cast<intptr_t>(tempDC_),
                                 reinterpret_cast<intptr_t>(tempGLRC_)))
      return false;

    // Make the newly created GL context current to this thread
    amd::GLFunctions::SetIntEnv ie(context().glenv());

    // If fence has not yet executed, wait till it finishes
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  }
#else  // Lnx
  Display* tempDpy_ = context().glenv()->glXGetCurrentDisplay_();
  GLXDrawable tempDrawable_ = context().glenv()->glXGetCurrentDrawable_();
  GLXContext tempCtx_ = context().glenv()->glXGetCurrentContext_();
  // Set internal Display and GLXContext
  if (tempDpy_ && tempCtx_) {
    amd::GLFunctions::Lock lock(context().glenv());
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  } else {
    if (!context().glenv()->init(reinterpret_cast<intptr_t>(context().glenv()->getIntDpy()),
                                 reinterpret_cast<intptr_t>(context().glenv()->getIntCtx())))
      return false;

    // Make the newly created GL context current to this thread
    amd::GLFunctions::SetIntEnv ie(context().glenv());

    // If fence has not yet executed, wait till it finishes
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  }
#endif
  // If we reach this point, fence should have completed
  setStatus(CL_COMPLETE);
  return true;
}


void amd::BufferGL::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(BufferGL));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

void amd::ImageGL::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(ImageGL));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

#ifdef _WIN32
#define CONVERT_CHAR_GLUBYTE
#else  //!_WIN32
#define CONVERT_CHAR_GLUBYTE (GLubyte*)
#endif  //!_WIN32

#define GLPREFIX(rtype, fcn, dclargs)                                                              \
  if (!(fcn##_ = (PFN_##fcn)GETPROCADDRESS(libHandle_, #fcn))) {                                   \
    if (!(fcn##_ = (PFN_##fcn)GetProcAddress_(reinterpret_cast<FCN_STR_TYPE>(#fcn)))) ++missed_;   \
  }

amd::GLFunctions::SetIntEnv::SetIntEnv(GLFunctions* env) : env_(env) {
  env_->getLock().lock();

  // Set environment (DC and GLRC)
  isValid_ = env_->setIntEnv();
}

amd::GLFunctions::SetIntEnv::~SetIntEnv() {
  // Restore environment (CL DC and CL GLRC)
  env_->restoreEnv();

  env_->getLock().unlock();
}

amd::GLFunctions::Lock::Lock(GLFunctions* env) : env_(env) { env_->getLock().lock(); }

amd::GLFunctions::Lock::~Lock() { env_->getLock().unlock(); }

amd::GLFunctions::GLFunctions(HMODULE h, bool isEGL)
    : libHandle_(h),
      missed_(0),
      eglDisplay_(EGL_NO_DISPLAY),
      eglOriginalContext_(EGL_NO_CONTEXT),
      eglInternalContext_(EGL_NO_CONTEXT),
      eglTempContext_(EGL_NO_CONTEXT),
      isEGL_(isEGL),
#ifdef _WIN32
      hOrigGLRC_(0),
      hDC_(0),
      hIntGLRC_(0)
#else   //!_WIN32
      Dpy_(0),
      Drawable_(0),
      origCtx_(0),
      intDpy_(0),
      intDrawable_(0),
      intCtx_(0),
      XOpenDisplay_(nullptr),
      XCloseDisplay_(nullptr),
      glXGetCurrentDrawable_(nullptr),
      glXGetCurrentDisplay_(nullptr),
      glXGetCurrentContext_(nullptr),
      glXChooseVisual_(nullptr),
      glXCreateContext_(nullptr),
      glXDestroyContext_(nullptr),
      glXMakeCurrent_(nullptr)
#endif  //!_WIN32
{
#define VERIFY_POINTER(p)                                                                          \
  if (nullptr == p) {                                                                              \
    missed_++;                                                                                     \
  }

  if (isEGL_) {
    GetProcAddress_ = (PFN_xxxGetProcAddress)GETPROCADDRESS(h, "eglGetProcAddress");
  } else {
    GetProcAddress_ = (PFN_xxxGetProcAddress)GETPROCADDRESS(h, API_GETPROCADDR);
  }
#ifndef _WIN32
  // Initialize pointers to X11/GLX functions
  // We can not link with these functions on compile time since we need to support
  // console mode. In console mode X server and X server components may be absent.
  // Hence linking with X11 or libGL will fail module image loading in console mode.-tzachi cohen

  if (!isEGL_) {
    glXGetCurrentDrawable_ = (PFNglXGetCurrentDrawable)GETPROCADDRESS(h, "glXGetCurrentDrawable");
    VERIFY_POINTER(glXGetCurrentDrawable_)
    glXGetCurrentDisplay_ = (PFNglXGetCurrentDisplay)GETPROCADDRESS(h, "glXGetCurrentDisplay");
    VERIFY_POINTER(glXGetCurrentDisplay_)
    glXGetCurrentContext_ = (PFNglXGetCurrentContext)GETPROCADDRESS(h, "glXGetCurrentContext");
    VERIFY_POINTER(glXGetCurrentContext_)
    glXChooseVisual_ = (PFNglXChooseVisual)GETPROCADDRESS(h, "glXChooseVisual");
    VERIFY_POINTER(glXChooseVisual_)
    glXCreateContext_ = (PFNglXCreateContext)GETPROCADDRESS(h, "glXCreateContext");
    VERIFY_POINTER(glXCreateContext_)
    glXDestroyContext_ = (PFNglXDestroyContext)GETPROCADDRESS(h, "glXDestroyContext");
    VERIFY_POINTER(glXDestroyContext_)
    glXMakeCurrent_ = (PFNglXMakeCurrent)GETPROCADDRESS(h, "glXMakeCurrent");
    VERIFY_POINTER(glXMakeCurrent_)

    HMODULE hXModule = (HMODULE)Os::loadLibrary("libX11.so.6");
    if (nullptr != hXModule) {
      XOpenDisplay_ = (PFNXOpenDisplay)GETPROCADDRESS(hXModule, "XOpenDisplay");
      VERIFY_POINTER(XOpenDisplay_)
      XCloseDisplay_ = (PFNXCloseDisplay)GETPROCADDRESS(hXModule, "XCloseDisplay");
      VERIFY_POINTER(XCloseDisplay_)
      XFree_ = (PFNXFree)GETPROCADDRESS(hXModule, "XFree");
      VERIFY_POINTER(XFree_)
    } else {
      missed_ += 2;
    }
  }
// Initialize pointers to GL functions
#include "gl_functions.hpp"
#else
  if (!isEGL_) {
    wglCreateContext_ = (PFN_wglCreateContext)GETPROCADDRESS(h, "wglCreateContext");
    VERIFY_POINTER(wglCreateContext_)
    wglGetCurrentContext_ = (PFN_wglGetCurrentContext)GETPROCADDRESS(h, "wglGetCurrentContext");
    VERIFY_POINTER(wglGetCurrentContext_)
    wglGetCurrentDC_ = (PFN_wglGetCurrentDC)GETPROCADDRESS(h, "wglGetCurrentDC");
    VERIFY_POINTER(wglGetCurrentDC_)
    wglDeleteContext_ = (PFN_wglDeleteContext)GETPROCADDRESS(h, "wglDeleteContext");
    VERIFY_POINTER(wglDeleteContext_)
    wglMakeCurrent_ = (PFN_wglMakeCurrent)GETPROCADDRESS(h, "wglMakeCurrent");
    VERIFY_POINTER(wglMakeCurrent_)
    wglShareLists_ = (PFN_wglShareLists)GETPROCADDRESS(h, "wglShareLists");
    VERIFY_POINTER(wglShareLists_)
  }
#endif
}

amd::GLFunctions::~GLFunctions() {
#ifdef _WIN32
  if (hIntGLRC_) {
    if (!wglDeleteContext_(hIntGLRC_)) {
      DWORD dwErr = GetLastError();
      LogWarning("Cannot delete GLRC");
    }
  }
#else   //!_WIN32
  if (intDpy_) {
    if (intCtx_) {
      glXDestroyContext_(intDpy_, intCtx_);
      intCtx_ = nullptr;
    }
    XCloseDisplay_(intDpy_);
    intDpy_ = nullptr;
  }
#endif  //!_WIN32
}
// in case of HIP GL interop we want to make sure we have the updated context
bool amd::GLFunctions::update(intptr_t hglrc) {
#ifdef _WIN32
  DWORD err;
  if (hOrigGLRC_ == (HGLRC)hglrc) {
    return true;
  }
  hOrigGLRC_ = (HGLRC)hglrc;
  if (hIntGLRC_ != nullptr) {
    wglDeleteContext_(hIntGLRC_);
  }
  if (!(hIntGLRC_ = wglCreateContext_(wglGetCurrentDC_()))) {
    err = GetLastError();
    return false;
  }
  if (!wglShareLists_(hOrigGLRC_, hIntGLRC_)) {
    err = GetLastError();
    return false;
  }
#else  //!_WIN32
  Dpy_ = glXGetCurrentDisplay_();
  Drawable_ = glXGetCurrentDrawable_();
  if (origCtx_ == (GLXContext)hglrc) {
    return true;
  }

  origCtx_ = (GLXContext)hglrc;
  if (intCtx_ != nullptr) {
    glXDestroyContext_(Dpy_, intCtx_);
  }

  int attribList[] = {GLX_RGBA, None};
  XVisualInfo* vis;
  int defaultScreen = DefaultScreen(intDpy_);
  if (!(vis = glXChooseVisual_(intDpy_, defaultScreen, attribList))) {
    return false;
  }
  intCtx_ = glXCreateContext_(intDpy_, vis, origCtx_, true);
  XFree_(vis);

  if (!intCtx_) {
    return false;
  }
#endif
  return true;
}

void amd::GLFunctions::WaitCurrentGlContext(const amd::Context::Info& info) const {
  if (IsCurrentGlContext(info)) {
    glFinish_();
  }
}


bool amd::GLFunctions::init(intptr_t hdc, intptr_t hglrc) {
  if (isEGL_) {
    eglDisplay_ = (EGLDisplay)hdc;
    eglOriginalContext_ = (EGLContext)hglrc;
    return true;
  }

#ifdef _WIN32
  DWORD err;

  if (missed_) {
    return false;
  }

  if (!hdc) {
    hDC_ = wglGetCurrentDC_();
  } else {
    hDC_ = (HDC)hdc;
  }
  hOrigGLRC_ = (HGLRC)hglrc;
  if (!(hIntGLRC_ = wglCreateContext_(hDC_))) {
    err = GetLastError();
    return false;
  }
  if (!wglShareLists_(hOrigGLRC_, hIntGLRC_)) {
    err = GetLastError();
    return false;
  }

  bool makeCurrentNull = false;

  if (wglGetCurrentContext_() == nullptr) {
    wglMakeCurrent_(hDC_, hIntGLRC_);

    makeCurrentNull = true;
  }

// Initialize pointers to GL functions
#include "gl_functions.hpp"

  if (makeCurrentNull) {
    wglMakeCurrent_(nullptr, nullptr);
  }

  if (missed_ == 0) {
    return true;
  }
#else  //!_WIN32
  if (!missed_) {
    if (!hdc) {
      Dpy_ = glXGetCurrentDisplay_();
    } else {
      Dpy_ = (Display*)hdc;
    }
    Drawable_ = glXGetCurrentDrawable_();
    origCtx_ = (GLXContext)hglrc;

    int attribList[] = {GLX_RGBA, None};
    if (!(intDpy_ = XOpenDisplay_(DisplayString(Dpy_)))) {
#if defined(ATI_ARCH_X86)
      asm("int $3");
#endif
    }
    intDrawable_ = DefaultRootWindow(intDpy_);

    XVisualInfo* vis;
    int defaultScreen = DefaultScreen(intDpy_);
    if (!(vis = glXChooseVisual_(intDpy_, defaultScreen, attribList))) {
      return false;
    }
    intCtx_ = glXCreateContext_(intDpy_, vis, origCtx_, true);
    XFree_(vis);

    if (!intCtx_) {
      return false;
    }
    return true;
  }
#endif  //!_WIN32
  return false;
}

bool amd::GLFunctions::setIntEnv() {
  if (isEGL_) {
    return true;
  }
#ifdef _WIN32
  // Save current DC and GLRC
  tempDC_ = wglGetCurrentDC_();
  tempGLRC_ = wglGetCurrentContext_();
  // Set internal DC and GLRC
  if (tempDC_ != getDC() || tempGLRC_ != getIntGLRC()) {
    if (!wglMakeCurrent_(getDC(), getIntGLRC())) {
      DWORD err = GetLastError();
      LogWarning("cannot set internal GL environment");
      return false;
    }
  }
#else   //!_WIN32
  tempDpy_ = glXGetCurrentDisplay_();
  tempDrawable_ = glXGetCurrentDrawable_();
  tempCtx_ = glXGetCurrentContext_();
  // Set internal Display and GLXContext
  if (tempDpy_ != getDpy() || tempCtx_ != getIntCtx()) {
    if (!glXMakeCurrent_(getIntDpy(), getIntDrawable(), getIntCtx())) {
      LogWarning("cannot set internal GL environment");
      return false;
    }
  }
#endif  //!_WIN32

  return true;
}

bool amd::GLFunctions::restoreEnv() {
  if (isEGL_) {
    // eglMakeCurrent( );
    return true;
  }
#ifdef _WIN32
  // Restore original DC and GLRC
  if (!wglMakeCurrent_(tempDC_, tempGLRC_)) {
    DWORD err = GetLastError();
    LogWarning("cannot restore original GL environment");
    return false;
  }
#else   //!_WIN32
  // Restore Display and GLXContext
  if (tempDpy_) {
    if (!glXMakeCurrent_(tempDpy_, tempDrawable_, tempCtx_)) {
      LogWarning("cannot restore original GL environment");
      return false;
    }
  } else {
    // Just release internal context
    if (!glXMakeCurrent_(getIntDpy(), None, nullptr)) {
      LogWarning("cannot reelase internal GL environment");
      return false;
    }
  }
#endif  //!_WIN32

  return true;
}


//! Function getCLFormatFromGL returns "true" if GL format
//! is compatible with CL format, "false" otherwise.
bool amd::getCLFormatFromGL(const amd::Context& amdContext, GLint gliInternalFormat,
                            cl_image_format* pclImageFormat, int* piBytesPerPixel,
                            cl_mem_flags flags) {
  bool bRetVal = false;

  /*
  Available values for "image_channel_order"
  ==========================================
  CL_R
  CL_A
  CL_INTENSITY
  CL_LUMINANCE
  CL_RG
  CL_RA
  CL_RGB
  CL_RGBA
  CL_ARGB
  CL_BGRA

  Available values for "image_channel_data_type"
  ==============================================
  CL_SNORM_INT8
  CL_SNORM_INT16
  CL_UNORM_INT8
  CL_UNORM_INT16
  CL_UNORM_SHORT_565
  CL_UNORM_SHORT_555
  CL_UNORM_INT_101010
  CL_SIGNED_INT8
  CL_SIGNED_INT16
  CL_SIGNED_INT32
  CL_UNSIGNED_INT8
  CL_UNSIGNED_INT16
  CL_UNSIGNED_INT32
  CL_HALF_FLOAT
  CL_FLOAT
  */

  switch (gliInternalFormat) {
    case GL_RGB10_EXT:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT_101010;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGB10_A2:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT_101010;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_BGR8_ATI:
    case GL_BGRA8_ATI:
      pclImageFormat->image_channel_order = CL_BGRA;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT8;  // CL_UNSIGNED_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_ALPHA8:
      pclImageFormat->image_channel_order = CL_A;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT8;  // CL_UNSIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_R8:
    case GL_R8UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_R8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_R8I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_RG8:
    case GL_RG8UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RG8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RG8I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RGB8:
    case GL_RGB8UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGB8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 3;
      bRetVal = true;
      break;

    case GL_RGB8I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 3;
      bRetVal = true;
      break;

    case GL_RGBA:
    case GL_RGBA8:
    case GL_RGBA8UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGBA8UI) ? CL_UNSIGNED_INT8 : CL_UNORM_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGBA8I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R16:
    case GL_R16UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_R16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      bRetVal = true;
      *piBytesPerPixel = 2;
      break;

    case GL_R16I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_R16F:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RG16:
    case GL_RG16UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RG16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG16I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG16F:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGB16:
    case GL_RGB16UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGB16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGB16I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGB16F:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGBA16:
    case GL_RGBA16UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGBA16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGBA16I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGBA16F:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_R32I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R32UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R32F:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG32I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RG32UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RG32F:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGB32I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGB32UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGB32F:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGBA32I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;

    case GL_RGBA32UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;

    case GL_RGBA32F:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;
    case GL_DEPTH_COMPONENT32F:
      pclImageFormat->image_channel_order = CL_DEPTH;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;
    case GL_DEPTH_COMPONENT16:
      pclImageFormat->image_channel_order = CL_DEPTH;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT16;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;
    case GL_DEPTH24_STENCIL8:
      pclImageFormat->image_channel_order = CL_DEPTH_STENCIL;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT24;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;
    case GL_DEPTH32F_STENCIL8:
      pclImageFormat->image_channel_order = CL_DEPTH_STENCIL;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 5;
      bRetVal = true;
      break;
    default:
      LogWarning("unsupported GL internal format");
      break;
  }
  amd::Image::Format imageFormat(*pclImageFormat);
  if (bRetVal && !imageFormat.isSupported(amdContext, 0, flags)) {
    bRetVal = false;
  }
  return bRetVal;
}
