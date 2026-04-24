/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef _WIN32
#include <windows.h>
#else  //!_WIN32
#include <dlfcn.h>
#endif  //!_WIN32

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#include "CL/cl_gl.h"
#if !defined(_WIN32) && !defined(__APPLE__)
#include <GL/glx.h>
#endif  //!_WIN32

#if defined(__APPLE__)
struct _XDisplay;
struct XVisualInfo;
struct __GLXcontextRec;
typedef _XDisplay Display;
typedef unsigned long Window;
typedef unsigned long GLXDrawable;
typedef __GLXcontextRec* GLXContext;
typedef int Bool;
#ifndef _Xconst
#define _Xconst const
#endif
#ifndef None
#define None 0L
#endif
#ifndef GLX_RGBA
#define GLX_RGBA 4
#endif
typedef void* EGLDisplay;
typedef void* EGLContext;
#ifndef EGL_NO_DISPLAY
#define EGL_NO_DISPLAY nullptr
#endif
#ifndef EGL_NO_CONTEXT
#define EGL_NO_CONTEXT nullptr
#endif
#ifndef GL_RGB10_EXT
#define GL_RGB10_EXT 0x8052
#endif
#ifndef GL_RGB8UI
#define GL_RGB8UI 0x8D7D
#endif
#ifndef GL_RGB8I
#define GL_RGB8I 0x8D8F
#endif
#ifndef GL_RGBA8UI
#define GL_RGBA8UI 0x8D7C
#endif
#ifndef GL_RGBA8I
#define GL_RGBA8I 0x8D8E
#endif
#ifndef GL_RGB16UI
#define GL_RGB16UI 0x8D77
#endif
#ifndef GL_RGB16I
#define GL_RGB16I 0x8D89
#endif
#ifndef GL_RGB16F
#define GL_RGB16F 0x881B
#endif
#ifndef GL_RGBA16UI
#define GL_RGBA16UI 0x8D76
#endif
#ifndef GL_RGBA16I
#define GL_RGBA16I 0x8D88
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_RGB32I
#define GL_RGB32I 0x8D83
#endif
#ifndef GL_RGB32UI
#define GL_RGB32UI 0x8D71
#endif
#ifndef GL_RGB32F
#define GL_RGB32F 0x8815
#endif
#ifndef GL_RGBA32I
#define GL_RGBA32I 0x8D82
#endif
#ifndef GL_RGBA32UI
#define GL_RGBA32UI 0x8D70
#endif
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif
#ifndef GL_TEXTURE_BUFFER
#define GL_TEXTURE_BUFFER 0x8C2A
#endif
#ifndef GL_TEXTURE_1D_ARRAY
#define GL_TEXTURE_1D_ARRAY 0x8C18
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_TEXTURE_2D_MULTISAMPLE
#define GL_TEXTURE_2D_MULTISAMPLE 0x9100
#endif
#ifndef GL_TEXTURE_SAMPLES
#define GL_TEXTURE_SAMPLES 0x9106
#endif
#ifndef GL_TEXTURE_BUFFER_DATA_STORE_BINDING
#define GL_TEXTURE_BUFFER_DATA_STORE_BINDING 0x8C2D
#endif
#ifndef GL_TEXTURE_BUFFER_FORMAT_EXT
#define GL_TEXTURE_BUFFER_FORMAT_EXT 0x8C2E
#endif
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#endif

#include "platform/context.hpp"
#include "platform/command.hpp"

namespace amd {

//! Class GLObject keeps all the info about the GL object
//! from which the CL object is created
class GLObject : public InteropObject {
 protected:
  cl_gl_object_type clGLType_;  //!< CL GL object type
  GLenum glTarget_;
  GLuint gluiName_;
  GLint gliMipLevel_;
  GLenum glInternalFormat_;
  GLint gliWidth_;
  GLint gliHeight_;
  GLint gliDepth_;
  GLenum glCubemapFace_;
  GLsizei glNumSamples_;

 public:
  //! GLObject constructor initializes member variables
  GLObject(GLenum glTarget, GLuint gluiName, GLint gliMipLevel, GLenum glInternalFormat,
           GLint gliWidth, GLint gliHeight, GLint gliDepth, cl_gl_object_type clGLType,
           GLenum glCubemapFace,
           GLsizei glNumSamples)
      :  // Initialization of member variables
        clGLType_(clGLType),
        glTarget_(glTarget),
        gluiName_(gluiName),
        gliMipLevel_(gliMipLevel),
        glInternalFormat_(glInternalFormat),
        gliWidth_(gliWidth),
        gliHeight_(gliHeight),
        gliDepth_(gliDepth),
        glCubemapFace_(glCubemapFace),
        glNumSamples_(glNumSamples) {}

  virtual ~GLObject() {}
  virtual GLObject* asGLObject() { return this; }

  //! GLObject query functions to get GL info from member variables
  GLenum getGLTarget() const { return glTarget_; }
  GLuint getGLName() const { return gluiName_; }
  GLint getGLMipLevel() const { return gliMipLevel_; }
  GLenum getGLInternalFormat() const { return glInternalFormat_; }
  GLint getGLSize() const { return gliWidth_; }
  GLint getGLWidth() const { return gliWidth_; }
  GLint getGLHeight() const { return gliHeight_; }
  GLint getGLDepth() const { return gliDepth_; }
  cl_gl_object_type getCLGLObjectType() const { return clGLType_; }
  GLenum getCubemapFace() const { return glCubemapFace_; }
  GLsizei getNumSamples() const { return glNumSamples_; }
};


//! Class BufferGL is drived from classes Buffer and GLObject
//! where the former keeps all data for CL object and
//! the latter keeps all data for GL object
class BufferGL : public Buffer, public GLObject {
 protected:
  //! Initializes the device memory array which is nested
  // after'BufferGL' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! BufferGL constructor just calls constructors of base classes
  //! to pass down the parameters
  BufferGL(Context& amdContext, cl_mem_flags clFlags, size_t uiSizeInBytes, GLenum glTarget,
           GLuint gluiName)
      :  // Call base classes constructors
        Buffer(amdContext, clFlags, uiSizeInBytes),
        GLObject(glTarget, gluiName,
                 0,                // Mipmap level default
                 GL_ARRAY_BUFFER,  // Just init to some value
                 (GLint)uiSizeInBytes, 1, 1, CL_GL_OBJECT_BUFFER, 0, 0) {
    setInteropObj(this);
  }
  virtual ~BufferGL() {}

  virtual BufferGL* asBufferGL() { return this; }
};


//! Class ImageGL is derived from classes Image and GLObject
//! where the former keeps all data for CL object and
//! the latter keeps all data for GL object
class ImageGL : public Image, public GLObject {
 public:
  //! ImageGL constructor just calls constructors of base classes
  //! to pass down the parameters
  ImageGL(Context& amdContext, cl_mem_object_type clType, cl_mem_flags clFlags,
          const Format& format, size_t width, size_t height, size_t depth, GLenum glTarget,
          GLuint gluiName, GLint gliMipLevel, GLenum glInternalFormat, cl_gl_object_type clGLType,
          GLsizei numSamples, GLenum glCubemapFace = 0)
      : Image(amdContext, clType, clFlags, format, width, height, depth,
              Format(format).getElementSize() * width,
              Format(format).getElementSize() * width * height),
        GLObject(glTarget, gluiName, gliMipLevel, glInternalFormat, static_cast<GLint>(width),
                 static_cast<GLint>(height), static_cast<GLint>(depth), clGLType, glCubemapFace,
                 numSamples) {
    setInteropObj(this);
  }

  //! ImageGL constructor for mipmapped image,
  //! just calls constructors of base classes to pass down the parameters
  ImageGL(Context& amdContext, cl_mem_object_type clType, cl_mem_flags clFlags,
          const Format& format, size_t width, size_t height, size_t depth, GLenum glTarget,
          GLuint gluiName, GLint gliMipLevel, GLenum glInternalFormat, cl_gl_object_type clGLType,
          GLsizei numSamples, cl_int num_mip_levels, GLenum glCubemapFace = 0)
      : Image(amdContext, clType, clFlags, format, width, height, depth,
              Format(format).getElementSize() * width,
              Format(format).getElementSize() * width * depth, num_mip_levels),
        GLObject(glTarget, gluiName, gliMipLevel, glInternalFormat, static_cast<GLint>(width),
                 static_cast<GLint>(height), static_cast<GLint>(depth), clGLType, glCubemapFace,
                 numSamples) {
    setInteropObj(this);
  }

  virtual ~ImageGL() {}

 protected:
  //! Initializes the device memory array which is nested
  // after'BufferGL' object in memory layout.
  virtual void initDeviceMemory();
};

typedef EGLContext (*PFN_eglGetCurrentContext)();
#ifdef _WIN32
#define APICALL WINAPI
#define GETPROCADDRESS GetProcAddress
#define API_GETPROCADDR "wglGetProcAddress"
#define FCN_STR_TYPE LPCSTR
typedef PROC(WINAPI* PFN_xxxGetProcAddress)(LPCSTR fcnName);
typedef HGLRC(APICALL* PFN_wglCreateContext)(HDC hdc);
typedef HGLRC(APICALL* PFN_wglGetCurrentContext)(void);
typedef HDC(APICALL* PFN_wglGetCurrentDC)(void);
typedef BOOL(APICALL* PFN_wglDeleteContext)(HGLRC hglrc);
typedef BOOL(APICALL* PFN_wglMakeCurrent)(HDC hdc, HGLRC hglrc);
typedef BOOL(APICALL* PFN_wglShareLists)(HGLRC hglrc1, HGLRC hglrc2);
#else            //!_WIN32
#define APICALL  // __stdcall   //??? todo odintsov
#define API_GETPROCADDR "glXGetProcAddress"
#define GETPROCADDRESS dlsym
#define FCN_STR_TYPE const GLubyte*
#define WINAPI
#define PROC void*
typedef void* (*PFN_xxxGetProcAddress)(const GLubyte* procName);
// X11 typedef
typedef Display* (*PFNXOpenDisplay)(_Xconst char* display_name);
typedef int (*PFNXCloseDisplay)(Display* display);
typedef int (*PFNXFree)(void* data);

// glx typedefs
typedef GLXDrawable (*PFNglXGetCurrentDrawable)();
typedef Display* (*PFNglXGetCurrentDisplay)();
typedef GLXContext (*PFNglXGetCurrentContext)(void);
typedef XVisualInfo* (*PFNglXChooseVisual)(Display* dpy, int screen, int* attribList);
typedef GLXContext (*PFNglXCreateContext)(Display* dpy, XVisualInfo* vis, GLXContext shareList,
                                          Bool direct);
typedef void (*PFNglXDestroyContext)(Display* dpy, GLXContext ctx);
typedef Bool (*PFNglXMakeCurrent)(Display* dpy, GLXDrawable drawable, GLXContext ctx);
typedef void* HMODULE;
#endif  //!_WIN32

#define GLPREFIX(rtype, fcn, dclargs) typedef rtype(APICALL* PFN_##fcn) dclargs;

// Declare prototypes for GL functions
#include "gl_functions.hpp"

class GLFunctions {
 public:
  //! Locks any access to the virtual GPUs
  class SetIntEnv : public amd::StackObject {
   public:
    //! Default constructor
    SetIntEnv(GLFunctions* env);

    //! Destructor
    ~SetIntEnv();

    //! Checks if the environment setup was successful
    bool isValid() const { return isValid_; }

   private:
    GLFunctions* env_;  //!< GL environment
    bool isValid_;      //!< If TRUE, then it's a valid setup
  };

  //! Locks any access to the virtual GPUs
  class Lock : public amd::StackObject {
   public:
    //! Default constructor
    Lock(GLFunctions* env);

    //! Destructor
    ~Lock();

   private:
    GLFunctions* env_;  //!< GL environment
  };

 private:
  HMODULE libHandle_;
  int missed_;  // Indicates how many GL functions not init'ed, if any

  amd::Monitor lock_;

  EGLDisplay eglDisplay_;
  EGLContext eglOriginalContext_;
  EGLContext eglInternalContext_;
  EGLContext eglTempContext_;
  bool isEGL_;
  PFN_eglGetCurrentContext eglGetCurrentContext_;

#ifdef _WIN32
  HGLRC hOrigGLRC_;
  HDC hDC_;
  HGLRC hIntGLRC_;  // handle for internal GLRC to access shared context
  HDC tempDC_;
  HGLRC tempGLRC_;

 public:
  PFN_wglCreateContext wglCreateContext_;
  PFN_wglGetCurrentContext wglGetCurrentContext_;
  PFN_wglGetCurrentDC wglGetCurrentDC_;
  PFN_wglDeleteContext wglDeleteContext_;
  PFN_wglMakeCurrent wglMakeCurrent_;
  PFN_wglShareLists wglShareLists_;
#else
 public:
  Display* Dpy_;
  GLXDrawable Drawable_;
  GLXContext origCtx_;
  Display* intDpy_;
  Window intDrawable_;
  GLXContext intCtx_;
  Display* tempDpy_;
  GLXDrawable tempDrawable_;
  GLXContext tempCtx_;

  // pointers to X11 functions
  PFNXOpenDisplay XOpenDisplay_;
  PFNXCloseDisplay XCloseDisplay_;
  PFNXFree XFree_;

  // pointers to GLX functions
  PFNglXGetCurrentDrawable glXGetCurrentDrawable_;
  PFNglXGetCurrentDisplay glXGetCurrentDisplay_;
  PFNglXGetCurrentContext glXGetCurrentContext_;
  PFNglXChooseVisual glXChooseVisual_;
  PFNglXCreateContext glXCreateContext_;
  PFNglXDestroyContext glXDestroyContext_;
  PFNglXMakeCurrent glXMakeCurrent_;
#endif
 public:
  GLFunctions(HMODULE h, bool isEGL);
  ~GLFunctions();

  bool update(intptr_t hglrc);
  bool IsCurrentGlContext(const amd::Context::Info& info) const {
    if (isEGL_) {
      return ((info.hCtx_ != nullptr) && (eglGetCurrentContext_ != nullptr) &&
              (info.hCtx_ == eglGetCurrentContext_()));
    } else {
#ifdef _WIN32
      return ((info.hCtx_ != nullptr) && (info.hCtx_ == wglGetCurrentContext_()));
#else
      return ((info.hCtx_ != nullptr) && (info.hCtx_ == glXGetCurrentContext_()));
#endif  // _WIN32
    }
  }

  void WaitCurrentGlContext(const amd::Context::Info& info) const;

  // Query CL-GL context association
  bool isAssociated() const {
    if (isEGL_ && eglDisplay_ && eglOriginalContext_) return true;
#ifdef _WIN32
    if (hDC_ && hOrigGLRC_) return true;
#else   //!_WIN32
    if (Dpy_ && origCtx_) return true;
#endif  //!_WIN32
    return false;
  }
  bool isEGL() const { return isEGL_; }
  // Accessor methods
#ifdef _WIN32
  HGLRC getOrigGLRC() const { return hOrigGLRC_; }
  HDC getDC() const { return hDC_; }
  HGLRC getIntGLRC() const { return hIntGLRC_; }
#else   //!_WIN32
  Display* getDpy() const { return Dpy_; }
  GLXDrawable getDrawable() const { return Drawable_; }
  GLXContext getOrigCtx() const { return origCtx_; }

  Display* getIntDpy() const { return intDpy_; }
  GLXDrawable getIntDrawable() const { return intDrawable_; }
  GLXContext getIntCtx() const { return intCtx_; }

  EGLDisplay getEglDpy() const { return eglDisplay_; }
  EGLContext getEglOrigCtx() const { return eglOriginalContext_; }
#endif  //!_WIN32

  // Initialize GL dynamic library and function pointers
  bool init(intptr_t hdc, intptr_t hglrc);

  // Return true if successful, false - if error occurred
  bool setIntEnv();
  bool restoreEnv();

  amd::Monitor& getLock() { return lock_; }

  PFN_xxxGetProcAddress GetProcAddress_;

#define GLPREFIX(rtype, fcn, dclargs) PFN_##fcn fcn##_;
// Declare pointers to GL functions
#include "gl_functions.hpp"
};

//! Functions for executing the GL related stuff
cl_mem clCreateFromGLBufferAMD(Context& amdContext, cl_mem_flags flags, GLuint bufobj,
                               cl_int* errcode_ret);
cl_mem clCreateFromGLTextureAMD(Context& amdContext, cl_mem_flags flags, GLenum target,
                                GLint miplevel, GLuint texture, int* errcode_ret);
cl_mem clCreateFromGLRenderbufferAMD(Context& amdContext, cl_mem_flags flags, GLuint renderbuffer,
                                     int* errcode_ret);

bool getCLFormatFromGL(const Context& amdContext, GLint gliInternalFormat,
                       cl_image_format* pclImageFormat, int* piBytesPerPixel, cl_mem_flags flags);

}  // namespace amd
