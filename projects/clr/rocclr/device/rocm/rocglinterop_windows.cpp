/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "device/rocm/rocglinterop.hpp"
#include "GL/gl_interop.h"
#include "platform/interop_gl.hpp"

namespace amd::roc {
namespace GlInterop {

typedef PROC(WINAPI* PFNWGLGETPROCADDRESS)(LPCSTR name);
typedef HGLRC(WINAPI* PFNWGLGETCURRENTCONTEXT)(void);
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXT)(HDC hdc);
typedef BOOL(WINAPI* PFNWGLDELETECONTEXT)(HGLRC hglrc);
typedef BOOL(WINAPI* PFNWGLMAKECURRENT)(HDC hdc, HGLRC hglrc);
static PFNWGLGETPROCADDRESS pfnWglGetProcAddress = nullptr;
static PFNWGLGETCURRENTCONTEXT pfnWglGetCurrentContext = nullptr;
static PFNWGLCREATECONTEXT pfnWglCreateContext = nullptr;
static PFNWGLDELETECONTEXT pfnWglDeleteContext = nullptr;
static PFNWGLMAKECURRENT pfnWglMakeCurrent = nullptr;
static PFNWGLBEGINCLINTEROPAMD wglBeginCLInteropAMD = nullptr;
static PFNWGLENDCLINTEROPAMD wglEndCLInteropAMD = nullptr;
static PFNWGLRESOURCEATTACHAMD wglResourceAttachAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceAcquireAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceReleaseAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceDetachAMD = nullptr;
static PFNWGLGETCONTEXTGPUINFOAMD wglGetContextGPUInfoAMD = nullptr;

// ================================================================================================
bool initGLInteropPrivateExt(void* GLdeviceContext) {
  static std::once_flag gGlFuncInit;
  static bool gGlFuncLoaded = false;


  std::call_once(gGlFuncInit, [GLdeviceContext]() {
    if (!GLdeviceContext) {
      LogError("GLdeviceContext is null");
      return;
    }

    HMODULE h = static_cast<HMODULE>(amd::Os::loadLibrary("opengl32.dll"));

    if (!h) {
      LogError("Couldn't load opengl32.dll");
      return;
    }

    pfnWglGetProcAddress =
        reinterpret_cast<PFNWGLGETPROCADDRESS>(GetProcAddress(h, "wglGetProcAddress"));
    pfnWglGetCurrentContext =
        reinterpret_cast<PFNWGLGETCURRENTCONTEXT>(GetProcAddress(h, "wglGetCurrentContext"));
    pfnWglCreateContext =
        reinterpret_cast<PFNWGLCREATECONTEXT>(GetProcAddress(h, "wglCreateContext"));
    pfnWglDeleteContext =
        reinterpret_cast<PFNWGLDELETECONTEXT>(GetProcAddress(h, "wglDeleteContext"));
    pfnWglMakeCurrent = reinterpret_cast<PFNWGLMAKECURRENT>(GetProcAddress(h, "wglMakeCurrent"));

    if (!pfnWglGetProcAddress || !pfnWglGetCurrentContext || !pfnWglCreateContext ||
        !pfnWglDeleteContext || !pfnWglMakeCurrent) {
      LogError("Couldn't obtain WGL context API");
      return;
    }

    // Create a temporary GL context if none is current; WGL extension
    // functions require some current GL context when loaded.
    HGLRC fakeRC = nullptr;
    if (!pfnWglGetCurrentContext()) {
      fakeRC = pfnWglCreateContext(static_cast<HDC>(GLdeviceContext));
      if (fakeRC == nullptr) {
        LogError("Couldn't create temporary WGL context");
        return;
      }
      pfnWglMakeCurrent(static_cast<HDC>(GLdeviceContext), fakeRC);
    }

    wglBeginCLInteropAMD =
        reinterpret_cast<PFNWGLBEGINCLINTEROPAMD>(pfnWglGetProcAddress("wglBeginCLInteroperabilityAMD"));
    wglEndCLInteropAMD = reinterpret_cast<PFNWGLENDCLINTEROPAMD>(pfnWglGetProcAddress("wglEndCLInteroperabilityAMD"));
    wglResourceAttachAMD = reinterpret_cast<PFNWGLRESOURCEATTACHAMD>(pfnWglGetProcAddress("wglResourceAttachAMD"));
    wglResourceAcquireAMD = reinterpret_cast<PFNWGLRESOURCEDETACHAMD>(pfnWglGetProcAddress("wglResourceAcquireAMD"));
    wglResourceReleaseAMD = reinterpret_cast<PFNWGLRESOURCEDETACHAMD>(pfnWglGetProcAddress("wglResourceReleaseAMD"));
    wglResourceDetachAMD = reinterpret_cast<PFNWGLRESOURCEDETACHAMD>(pfnWglGetProcAddress("wglResourceDetachAMD"));
    wglGetContextGPUInfoAMD =
        reinterpret_cast<PFNWGLGETCONTEXTGPUINFOAMD>(pfnWglGetProcAddress("wglGetContextGPUInfoAMD"));

    if (fakeRC) {
      pfnWglMakeCurrent(nullptr, nullptr);
      pfnWglDeleteContext(fakeRC);
    }

    gGlFuncLoaded = wglBeginCLInteropAMD && wglEndCLInteropAMD && wglResourceAttachAMD &&
                    wglResourceAcquireAMD && wglResourceReleaseAMD &&
                    wglResourceDetachAMD && wglGetContextGPUInfoAMD;
  });

  return gGlFuncLoaded;
}

// ================================================================================================
bool glCanInterop(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  bool canInteroperate = false;

  LUID glAdapterLuid = {0, 0};
  UINT glChainBitMask = 0;
  HGLRC hRC = static_cast<HGLRC>(GLplatformContext);

  // get GL context's LUID and chainBitMask from UGL
  if (wglGetContextGPUInfoAMD(hRC, &glAdapterLuid, &glChainBitMask)) {
    // match the adapter
    canInteroperate = device->info().luidLowPart_ == glAdapterLuid.LowPart &&
                      device->info().luidHighPart_ == glAdapterLuid.HighPart &&
                      (((1 << device->index()) & glChainBitMask) != 0);
  }

  return canInteroperate;
}

// ================================================================================================
bool glAssociate(Device* device, uint flags, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(flags); // unused

  if (!initGLInteropPrivateExt(GLdeviceContext)) return false;

  if (!glCanInterop(device, GLplatformContext, GLdeviceContext)) {
    return false;
  }

  return wglBeginCLInteropAMD(static_cast<HGLRC>(GLplatformContext), 0) != FALSE;
}

// ================================================================================================
bool glDissociate(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(device); // unused

  if (!initGLInteropPrivateExt(GLdeviceContext)) return false;

  return wglEndCLInteropAMD(static_cast<HGLRC>(GLplatformContext), 0) != FALSE;
}

// ================================================================================================
bool Export(amd::Memory* mem, GLenum targetType, int miplevel, hsa_handle_t* handle, int* offset) {
  assert(mem->getInteropObj() != nullptr);
  assert(mem->getInteropObj()->asGLObject() != nullptr);

  const auto* obj = mem->getInteropObj()->asGLObject();
  const auto GLContext = mem->getContext().info().hCtx_;
  const auto name = static_cast<uint>(obj->getGLName());

  GLenum type;
  switch (obj->getCLGLObjectType()) {
    case CL_GL_OBJECT_BUFFER:
      type = GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD;
      break;
    case CL_GL_OBJECT_RENDERBUFFER:
      type = GL_RESOURCE_ATTACH_RENDERBUFFER_AMD;
      break;
    case CL_GL_OBJECT_TEXTURE_BUFFER:
    case CL_GL_OBJECT_TEXTURE1D:
    case CL_GL_OBJECT_TEXTURE1D_ARRAY:
    case CL_GL_OBJECT_TEXTURE2D:
    case CL_GL_OBJECT_TEXTURE2D_ARRAY:
    case CL_GL_OBJECT_TEXTURE3D:
      type = GL_RESOURCE_ATTACH_TEXTURE_AMD;
      break;
    default:
      LogError("Unknown OpenGL interop type: 0x%x", obj->getCLGLObjectType());
      return false;
  }

  const auto glRenderContext = reinterpret_cast<HGLRC>(GLContext);
  GLResource glResource = {.type = type, .name = name};
  GLResourceData glResourceData = {.version = GL_RESOURCE_DATA_VERSION};

  if (!wglResourceAttachAMD(glRenderContext, static_cast<GLvoid*>(&glResource), &glResourceData))
    return false;
  *handle = reinterpret_cast<hsa_handle_t>(glResourceData.handle);
  *offset = static_cast<int>(glResourceData.offset);

  return true;
}

} // namespace GlInterop
} // namespace amd::roc
