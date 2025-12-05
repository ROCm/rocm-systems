/* Copyright (c) 2016 - 2025 Advanced Micro Devices, Inc.

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

#if defined(_WIN32)
#error "This file is not expected to be compiled on Windows."
#endif

#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "device/rocm/rocglinterop.hpp"

#include <dlfcn.h>

namespace amd::roc {
namespace GlInterop {

static PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC* GlxInfo = nullptr;
static PFNMESAGLINTEROPGLXEXPORTOBJECTPROC* GlxExport = nullptr;
static PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC* EglInfo = nullptr;
static PFNMESAGLINTEROPEGLEXPORTOBJECTPROC* EglExport = nullptr;
static MESA_INTEROP_KIND loadedGLAPITypes(MESA_INTEROP_NONE);

using PFNGLXGETPROCADDRESSPROC = void* (*)(const GLubyte* procname);
using PFNEGLGETPROCADDRESSPROC = void* (*)(const char* procName);

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

// ================================================================================================
// Fallback for older OS' and Mesa versions
static void LegacyInitGLX() {
  if (!GlxInfo) {
    GlxInfo = reinterpret_cast<PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*>(
        dlsym(RTLD_DEFAULT, "MesaGLInteropGLXQueryDeviceInfo"));
  }

  if (!GlxExport) {
    GlxExport = reinterpret_cast<PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*>(
        dlsym(RTLD_DEFAULT, "MesaGLInteropGLXExportObject"));
  }
}

// ================================================================================================
static void LegacyInitEGL() {
  if (!EglInfo) {
    EglInfo = reinterpret_cast<PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*>(
        dlsym(RTLD_DEFAULT, "MesaGLInteropEGLQueryDeviceInfo"));
  }

  if (!EglExport) {
    EglExport = reinterpret_cast<PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*>(
        dlsym(RTLD_DEFAULT, "MesaGLInteropEGLExportObject"));
  }
}

// ================================================================================================
// Returns true if the required subsystem is supported on the GL device.
bool Init(MESA_INTEROP_KIND Kind) {
  static std::once_flag gGlFuncInit;
  std::call_once(gGlFuncInit, [&]() {
    auto glx_procaddr_fn =
        reinterpret_cast<PFNGLXGETPROCADDRESSPROC>(dlsym(RTLD_DEFAULT, "glXGetProcAddress"));
    auto egl_procaddr_fn =
        reinterpret_cast<PFNEGLGETPROCADDRESSPROC>(dlsym(RTLD_DEFAULT, "eglGetProcAddress"));

    if (glx_procaddr_fn) {
      GlxInfo = reinterpret_cast<PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*>(
          glx_procaddr_fn(reinterpret_cast<const GLubyte*>("glXGLInteropQueryDeviceInfoMESA")));
      GlxExport = reinterpret_cast<PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*>(
          glx_procaddr_fn(reinterpret_cast<const GLubyte*>("glXGLInteropExportObjectMESA")));
    }

    if (egl_procaddr_fn) {
      EglInfo = reinterpret_cast<PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*>(
          egl_procaddr_fn("eglGLInteropQueryDeviceInfoMESA"));
      EglExport = reinterpret_cast<PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*>(
          egl_procaddr_fn("eglGLInteropExportObjectMESA"));
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
  });

  return ((loadedGLAPITypes & Kind) == Kind);
}

// ================================================================================================
bool GetInfo(mesa_glinterop_device_info& info, MESA_INTEROP_KIND Kind, const DisplayHandle display,
             const ContextHandle context) {
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
  if (ret < static_cast<int>(sizeof(errorStrings) / sizeof(errorStrings[0])))
    LogPrintfError("Mesa interop: GetInfo failed with \"%s\".\n", errorStrings[ret]);
  else
    LogError("Mesa interop: GetInfo failed with invalid error code.\n");
  return false;
}

// ================================================================================================
bool Export(mesa_glinterop_export_in& in, mesa_glinterop_export_out& out, MESA_INTEROP_KIND Kind,
            const DisplayHandle display, const ContextHandle context) {
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
  if (ret < static_cast<int>(sizeof(errorStrings) / sizeof(errorStrings[0])))
    LogPrintfError("Mesa interop: Export failed with \"%s\".\n", errorStrings[ret]);
  else
    LogError("Mesa interop: Export failed with invalid error code.\n");
  return false;
}

// ================================================================================================
bool glAssociate(Device* device, uint flags, void* gfxContext, void* glDevice) {
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;

  MESA_INTEROP_KIND kind;
  DisplayHandle display;
  ContextHandle context;

  if ((flags & amd::Context::EGLDeviceKhr) != 0) {
    kind = MESA_INTEROP_EGL;
    display.eglDisplay = reinterpret_cast<EGLDisplay>(glDevice);
    context.eglContext = reinterpret_cast<EGLContext>(gfxContext);
  } else {
    kind = MESA_INTEROP_GLX;
    display.glxDisplay = reinterpret_cast<Display*>(glDevice);
    context.glxContext = reinterpret_cast<GLXContext>(gfxContext);
  }

  mesa_glinterop_device_info info = {.version = MESA_GLINTEROP_DEVICE_INFO_VERSION};

  if (!Init(kind) || !GetInfo(info, kind, display, context))
    return false;

  const auto& pcie = device->info().deviceTopology_.pcie;
  const auto& dev_info = device->info();
  return pcie.bus == info.pci_bus &&
         pcie.device == info.pci_device &&
         pcie.function == info.pci_function &&
         dev_info.vendorId_ == info.vendor_id &&
         dev_info.pcieDeviceId_ == info.device_id;
}

// ================================================================================================
bool glDissociate(Device*, void*, void*) {
  return true;
}

}  // namespace GlInterop
}  // namespace amd::roc
