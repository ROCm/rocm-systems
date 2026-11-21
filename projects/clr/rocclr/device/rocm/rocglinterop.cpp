/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#if defined(_WIN32)
#error "This file is not expected to be compiled on Windows."
#endif

#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "device/rocm/rocglinterop.hpp"

#include <dlfcn.h>
#include <atomic>
#include <mutex>

namespace amd::roc {
namespace GlInterop {

static PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC* GlxInfo = nullptr;
static PFNMESAGLINTEROPGLXEXPORTOBJECTPROC* GlxExport = nullptr;
static PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC* EglInfo = nullptr;
static PFNMESAGLINTEROPEGLEXPORTOBJECTPROC* EglExport = nullptr;
// Keep an existing loader reference for locally loaded libraries as long as cached
// entry points can be called. Missing libraries are retried on the next Init call.
static void* GlxLibrary = nullptr;
static void* EglLibrary = nullptr;
// Function pointers for each backend are immutable after its bit is published.
// The release/acquire pair also protects callers while the other backend is loaded.
static std::atomic<uint32_t> loadedGLAPITypes{MESA_INTEROP_NONE};

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
static void* FindGlSymbol(void* library, const char* name) {
  void* symbol = dlsym(RTLD_DEFAULT, name);
  return symbol != nullptr || library == nullptr ? symbol : dlsym(library, name);
}

// ================================================================================================
// Fallback for older OS' and Mesa versions
static void LegacyInitGLX() {
  if (!GlxInfo) {
    GlxInfo = reinterpret_cast<PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*>(
        FindGlSymbol(GlxLibrary, "MesaGLInteropGLXQueryDeviceInfo"));
  }

  if (!GlxExport) {
    GlxExport = reinterpret_cast<PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*>(
        FindGlSymbol(GlxLibrary, "MesaGLInteropGLXExportObject"));
  }
}

// ================================================================================================
static void LegacyInitEGL() {
  if (!EglInfo) {
    EglInfo = reinterpret_cast<PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*>(
        FindGlSymbol(EglLibrary, "MesaGLInteropEGLQueryDeviceInfo"));
  }

  if (!EglExport) {
    EglExport = reinterpret_cast<PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*>(
        FindGlSymbol(EglLibrary, "MesaGLInteropEGLExportObject"));
  }
}

// ================================================================================================
// Returns true if the required subsystem is supported on the GL device.
bool Init(MESA_INTEROP_KIND Kind) {
  if ((loadedGLAPITypes.load(std::memory_order_acquire) & Kind) == Kind) {
    return true;
  }

  static std::mutex gGlFuncInitMutex;
  std::lock_guard<std::mutex> lock(gGlFuncInitMutex);
  uint32_t loaded = loadedGLAPITypes.load(std::memory_order_relaxed);

  // Retry only unavailable requested backends: an application may load GLX or EGL
  // after the first interop call. Never overwrite pointers already in use.
  if ((Kind & MESA_INTEROP_GLX) != 0 && (loaded & MESA_INTEROP_GLX) == 0) {
    if (GlxLibrary == nullptr) {
      GlxLibrary = dlopen("libGL.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    auto glx_procaddr_fn =
        reinterpret_cast<PFNGLXGETPROCADDRESSPROC>(FindGlSymbol(GlxLibrary, "glXGetProcAddress"));
    if (glx_procaddr_fn) {
      GlxInfo = reinterpret_cast<PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC*>(
          glx_procaddr_fn(reinterpret_cast<const GLubyte*>("glXGLInteropQueryDeviceInfoMESA")));
      GlxExport = reinterpret_cast<PFNMESAGLINTEROPGLXEXPORTOBJECTPROC*>(
          glx_procaddr_fn(reinterpret_cast<const GLubyte*>("glXGLInteropExportObjectMESA")));
    }
    LegacyInitGLX();
    if (GlxInfo && GlxExport) {
      loaded |= MESA_INTEROP_GLX;
    }
  }

  if ((Kind & MESA_INTEROP_EGL) != 0 && (loaded & MESA_INTEROP_EGL) == 0) {
    if (EglLibrary == nullptr) {
      EglLibrary = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    auto egl_procaddr_fn =
        reinterpret_cast<PFNEGLGETPROCADDRESSPROC>(FindGlSymbol(EglLibrary, "eglGetProcAddress"));
    if (egl_procaddr_fn) {
      EglInfo = reinterpret_cast<PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC*>(
          egl_procaddr_fn("eglGLInteropQueryDeviceInfoMESA"));
      EglExport = reinterpret_cast<PFNMESAGLINTEROPEGLEXPORTOBJECTPROC*>(
          egl_procaddr_fn("eglGLInteropExportObjectMESA"));
    }
    LegacyInitEGL();
    if (EglInfo && EglExport) {
      loaded |= MESA_INTEROP_EGL;
    }
  }

  loadedGLAPITypes.store(loaded, std::memory_order_release);
  return ((loaded & Kind) == Kind);
}

// ================================================================================================
bool GetInfo(mesa_glinterop_device_info& info, MESA_INTEROP_KIND Kind, const DisplayHandle display,
             const ContextHandle context) {
  if (!Init(Kind)) {
    return false;
  }
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
  if (!Init(Kind)) {
    return false;
  }
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
bool glAssociate(Device* device, uint flags, void* gfxContext, void* glDevice,
                 bool validateOnly) {
  static_cast<void>(validateOnly); // no session begin/attach step to skip on Linux
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
