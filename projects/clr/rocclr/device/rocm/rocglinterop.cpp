/* Copyright (c) 2016 - 2021 Advanced Micro Devices, Inc.

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

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace amd::roc {

namespace MesaInterop {

#if !defined(_WIN32)
static PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC* GlxInfo = nullptr;
static PFNMESAGLINTEROPGLXEXPORTOBJECTPROC* GlxExport = nullptr;
static PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC* EglInfo = nullptr;
static PFNMESAGLINTEROPEGLEXPORTOBJECTPROC* EglExport = nullptr;
static MESA_INTEROP_KIND loadedGLAPITypes(MESA_INTEROP_NONE);

using PFNGLXGETPROCADDRESSPROC = void* (*)(const GLubyte* procname);

using PFNEGLGETPROCADDRESSPROC = void* (*)(const char* procName);
#endif

static constexpr const char* errorStrings[] = {"MESA_GLINTEROP_SUCCESS",
                                               "MESA_GLINTEROP_OUT_OF_RESOURCES",
                                               "MESA_GLINTEROP_OUT_OF_HOST_MEMORY",
                                               "MESA_GLINTEROP_INVALID_OPERATION",
                                               "MESA_GLINTEROP_INVALID_VERSION",
                                               "MESA_GLINTEROP_INVALID_DISPLAY",
                                               "MESA_GLINTEROP_INVALID_CONTEXT",
                                               "MESA_GLINTEROP_INVALID_TARGET",
                                               "MESA_GLINTEROP_INVALID_OBJECT",
                                               "MESA_GLINTEROP_INVALID_MIP_LEVEL",
                                               "MESA_GLINTEROP_UNSUPPORTED"};

bool Supported() {
#ifdef _WIN32
  return false;
#else
  return true;
#endif
}

#if !defined(_WIN32)
// Fallback for older OS' and Mesa versions
static void LegacyInitGLX() {
  if (!GlxInfo) {
    GlxInfo = (PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*)dlsym(RTLD_DEFAULT,
                                                             "MesaGLInteropGLXQueryDeviceInfo");
  }

  if (!GlxExport) {
    GlxExport =
        (PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*)dlsym(RTLD_DEFAULT, "MesaGLInteropGLXExportObject");
  }
}

static void LegacyInitEGL() {
  if (!EglInfo) {
    EglInfo = (PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*)dlsym(RTLD_DEFAULT,
                                                             "MesaGLInteropEGLQueryDeviceInfo");
  }

  if (!EglExport) {
    EglExport =
        (PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*)dlsym(RTLD_DEFAULT, "MesaGLInteropEGLExportObject");
  }
}
#endif

// Returns true if the required subsystem is supported on the GL device.
// Must be called at least once, may be called multiple times.
bool Init(MESA_INTEROP_KIND Kind) {
#if defined(_WIN32)
  return false;
#else
  static std::once_flag gGlFuncInit;

  std::call_once(gGlFuncInit, [&]() {
    if (loadedGLAPITypes == MESA_INTEROP_NONE) {
      auto glx_procaddr_fn = (PFNGLXGETPROCADDRESSPROC)dlsym(RTLD_DEFAULT, "glXGetProcAddress");
      auto egl_procaddr_fn = (PFNEGLGETPROCADDRESSPROC)dlsym(RTLD_DEFAULT, "eglGetProcAddress");

      if (glx_procaddr_fn) {
        GlxInfo = (PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*)glx_procaddr_fn(
            (const GLubyte*)"glXGLInteropQueryDeviceInfoMESA");
        GlxExport = (PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*)glx_procaddr_fn(
            (const GLubyte*)"glXGLInteropExportObjectMESA");
      }

      if (egl_procaddr_fn) {
        EglInfo = (PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*)egl_procaddr_fn(
            "eglGLInteropQueryDeviceInfoMESA");
        EglExport =
            (PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*)egl_procaddr_fn("eglGLInteropExportObjectMESA");
      }

      if (!GlxInfo || !GlxExport) {
        LegacyInitGLX();
      }

      if (!EglInfo || !EglExport) {
        LegacyInitEGL();
      }

      uint32_t ret = MESA_INTEROP_NONE;
      if (GlxInfo && GlxExport) {
        ret |= MESA_INTEROP_GLX;
      }

      if (EglInfo && EglExport) {
        ret |= MESA_INTEROP_EGL;
      }

      loadedGLAPITypes = MESA_INTEROP_KIND(ret);
    }
  });

  return ((loadedGLAPITypes & Kind) == Kind);
#endif
}

bool GetInfo(mesa_glinterop_device_info& info, MESA_INTEROP_KIND Kind, const DisplayHandle display,
             const ContextHandle context) {
#ifdef _WIN32
  return false;
#else
  assert((loadedGLAPITypes & Kind) == Kind && "Requested interop API is not currently loaded.");
  int ret;
  switch (Kind) {
    case MESA_INTEROP_GLX:
      ret = GlxInfo(display.glxDisplay, context.glxContext, &info);
      break;
    case MESA_INTEROP_EGL:
      ret = EglInfo(display.eglDisplay, context.eglContext, &info);
      break;
    default:
      assert(false && "Invalid interop kind.");
      return false;
  }
  if (ret == MESA_GLINTEROP_SUCCESS) return true;
  if (ret < int(sizeof(errorStrings) / sizeof(errorStrings[0])))
    LogPrintfError("Mesa interop: GetInfo failed with \"%s\".\n", errorStrings[ret]);
  else
    LogError("Mesa interop: GetInfo failed with invalid error code.\n");
  return false;
#endif
}

bool Export(mesa_glinterop_export_in& in, mesa_glinterop_export_out& out, MESA_INTEROP_KIND Kind,
            const DisplayHandle display, const ContextHandle context) {
#ifdef _WIN32
  return false;
#else
  assert((loadedGLAPITypes & Kind) == Kind && "Requested interop API is not currently loaded.");
  int ret;
  switch (Kind) {
    case MESA_INTEROP_GLX:
      ret = GlxExport(display.glxDisplay, context.glxContext, &in, &out);
      break;
    case MESA_INTEROP_EGL:
      ret = EglExport(display.eglDisplay, context.eglContext, &in, &out);
      break;
    default:
      assert(false && "Invalid interop kind.");
      return false;
  }
  if (ret == MESA_GLINTEROP_SUCCESS) return true;
  if (ret < int(sizeof(errorStrings) / sizeof(errorStrings[0])))
    LogPrintfError("Mesa interop: Export failed with \"%s\".\n", errorStrings[ret]);
  else
    LogError("Mesa interop: Export failed with invalid error code.\n");
  return false;
#endif
}

bool glAssociate(Device* device, uint flags, void* gfxContext, void* glDevice) {
#ifdef _WIN32
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;

  MesaInterop::MESA_INTEROP_KIND kind = MesaInterop::MESA_INTEROP_NONE;
  MesaInterop::DisplayHandle display;
  MesaInterop::ContextHandle context;

  if ((flags & amd::Context::EGLDeviceKhr) != 0) {
    kind = MesaInterop::MESA_INTEROP_EGL;
    display.eglDisplay = reinterpret_cast<EGLDisplay>(glDevice);
    context.eglContext = reinterpret_cast<EGLContext>(gfxContext);
  } else {
    kind = MesaInterop::MESA_INTEROP_GLX;
    display.glxDisplay = reinterpret_cast<Display*>(glDevice);
    context.glxContext = reinterpret_cast<GLXContext>(gfxContext);
  }

  mesa_glinterop_device_info info;
  info.version = MESA_GLINTEROP_DEVICE_INFO_VERSION;

  // TODO: Lazy init with race condition issue
  if (!MesaInterop::Init(kind)) {
    return false;
  }

  if (!MesaInterop::GetInfo(info, kind, display, context)) {
    return false;
  }

  return device->info().deviceTopology_.pcie.bus == info.pci_bus &&
         device->info().deviceTopology_.pcie.device == info.pci_device &&
         device->info().deviceTopology_.pcie.function == info.pci_function &&
         device->info().vendorId_ == info.vendor_id && device->info().pcieDeviceId_ == info.device_id;
#endif
}

bool glDissociate(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(device);
  static_cast<void>(GLplatformContext);
  static_cast<void>(GLdeviceContext);

#ifdef _WIN32
  return false;
#else
  return true;
#endif
}

}  // namespace MesaInterop

#ifdef _WIN32
namespace WGLInterop {
#include "GL/gl_interop.h"

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
      fakeRC = pfnWglCreateContext((HDC)GLdeviceContext);
      if (fakeRC == nullptr) {
        LogError("Couldn't create temporary WGL context");
        return;
      }
      pfnWglMakeCurrent((HDC)GLdeviceContext, fakeRC);
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
                      ((1 << device->index()) == glChainBitMask);
  }

  return canInteroperate;
}

bool glAssociate(Device* device, uint flags, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(flags); // unused

  if (!initGLInteropPrivateExt(GLdeviceContext)) return false;

  if (!glCanInterop(device, GLplatformContext, GLdeviceContext)) {
    return false;
  }

  return wglBeginCLInteropAMD(static_cast<HGLRC>(GLplatformContext), 0) != FALSE;
}

bool glDissociate(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(device); // unused

  if (!initGLInteropPrivateExt(GLdeviceContext)) return false;

  return wglEndCLInteropAMD(static_cast<HGLRC>(GLplatformContext), 0) != FALSE;
}
}  // namespace WGLInterop

#endif

#if defined(_WIN32)
glinterop_funcs_t gGlInteropFuncs = {WGLInterop::glAssociate, WGLInterop::glDissociate};
#else
glinterop_funcs_t gGlInteropFuncs = {MesaInterop::glAssociate, MesaInterop::glDissociate};
#endif

}  // namespace amd::roc
