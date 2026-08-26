/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef ROCDECODE_USE_DLOPEN_VA

#include "vaapi_loader.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Path detection
// ---------------------------------------------------------------------------

std::string VaapiLoader::FindVaDrmLibPath() {
    namespace fs = std::filesystem;

    // Strategy 1: locate librocdecode.so via dladdr on a symbol in this
    // translation unit, then look for rocm_sysdeps/lib/ next to it.
    // e.g. /opt/rocm/lib/librocdecode.so  ->  /opt/rocm/lib/rocm_sysdeps/lib/
    Dl_info dl_info{};
    if (dladdr(reinterpret_cast<void *>(&VaapiLoader::FindVaDrmLibPath), &dl_info) &&
        dl_info.dli_fname) {
        fs::path sysdeps_lib = fs::path(dl_info.dli_fname).parent_path() / "rocm_sysdeps" / "lib";
        for (const char *candidate :
             {"librocm_sysdeps_va-drm.so.2", "librocm_sysdeps_va-drm.so"}) {
            fs::path full = sysdeps_lib / candidate;
            if (fs::exists(full)) {
                return full.string();
            }
        }
    }

    // Strategy 2: fall back to $ROCM_PATH.
    const char *rocm_path = std::getenv("ROCM_PATH");
    if (rocm_path) {
        fs::path sysdeps_lib = fs::path(rocm_path) / "lib" / "rocm_sysdeps" / "lib";
        for (const char *candidate :
             {"librocm_sysdeps_va-drm.so.2", "librocm_sysdeps_va-drm.so"}) {
            fs::path full = sysdeps_lib / candidate;
            if (fs::exists(full)) {
                return full.string();
            }
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// Symbol loading helper
// ---------------------------------------------------------------------------

template <typename T>
void VaapiLoader::LoadSym(const char *name, T *&fn_ptr) {
    dlerror(); // clear any prior error
    fn_ptr = reinterpret_cast<T *>(dlsym(va_drm_handle_, name));
    const char *err = dlerror();
    if (err || !fn_ptr) {
        throw std::runtime_error(std::string("VaapiLoader: dlsym('") + name + "'): " +
                                 (err ? err : "symbol not found"));
    }
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

VaapiLoader::VaapiLoader() {
    std::string va_drm_path = FindVaDrmLibPath();
    if (va_drm_path.empty()) {
        throw std::runtime_error(
            "VaapiLoader: cannot locate librocm_sysdeps_va-drm.so.2; "
            "set ROCM_PATH to the ROCm installation prefix");
    }

    // dlmopen(LM_ID_NEWLM) cannot be used here due to a glibc limitation:
    //   - RTLD_LOCAL with LM_ID_NEWLM leaves the private namespace's global-scope
    //     array uninitialised; libva's vaInitialize calls dlopen(radeonsi_drv_video.so,
    //     RTLD_GLOBAL) internally, which crashes in add_to_global_resize trying to
    //     expand that uninitialised array.
    //   - RTLD_GLOBAL with LM_ID_NEWLM is rejected by glibc with EINVAL.
    //
    // Instead, use dlopen with RTLD_LOCAL | RTLD_DEEPBIND:
    //   RTLD_LOCAL  — librocm_sysdeps_va.so.2's symbols are not added to the
    //                 process-wide global symbol scope, so they cannot collide
    //                 with system libva.so.2 even if libavcodec loads it into the
    //                 same process.
    //   RTLD_DEEPBIND — libva and its backend driver prefer their own symbol
    //                 closure when resolving symbols, before looking in the global
    //                 scope.  This prevents system libva symbols (if any are global)
    //                 from being picked up by librocm_sysdeps_va's internal calls.
    //
    // The inner dlopen(radeonsi_drv_video.so, RTLD_GLOBAL) from vaInitialize works
    // normally.  radeonsi_drv_video.so has DT_NEEDED: librocm_sysdeps_va.so.2, so
    // with RTLD_DEEPBIND it resolves its va* symbols through its own local scope
    // (ROCm libva) rather than picking up system libva from the global scope.
    // This is required even when libavcodec is loaded in the same process (e.g.
    // FFmpeg-based extended tests), because libavcodec pulls system libva into
    // the global scope and radeonsi would otherwise bind to it.
    //
    // Rocdecode's own va* calls are isolated via the function-pointer macros in
    // vaapi_videodecoder.cpp and never go through the global symbol scope.
    //
    // Note: RTLD_DEEPBIND causes libva-internal allocations to bypass ASan's
    // interposed malloc/free, so they are not tracked by ASan. This is
    // acceptable because libva manages its own memory and rocdecode only holds
    // opaque VA handles — there are no cross-boundary frees for ASan to catch.
    va_drm_handle_ = dlopen(va_drm_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!va_drm_handle_) {
        throw std::runtime_error(std::string("VaapiLoader: dlopen('") + va_drm_path +
                                 "'): " + dlerror());
    }

    // Resolve all VA-API symbols.  dlsym searches the handle's DSO and all
    // transitive DT_NEEDED dependencies, so va-core symbols reachable through
    // librocm_sysdeps_va.so.2 are found from the va_drm_handle_ alone.
    LoadSym("vaGetDisplayDRM",          fn.vaGetDisplayDRM);
    LoadSym("vaInitialize",             fn.vaInitialize);
    LoadSym("vaTerminate",              fn.vaTerminate);
    LoadSym("vaSetInfoCallback",        fn.vaSetInfoCallback);
    LoadSym("vaQueryVendorString",      fn.vaQueryVendorString);
    LoadSym("vaErrorStr",               fn.vaErrorStr);
    LoadSym("vaMaxNumProfiles",         fn.vaMaxNumProfiles);
    LoadSym("vaQueryConfigProfiles",    fn.vaQueryConfigProfiles);
    LoadSym("vaGetConfigAttributes",    fn.vaGetConfigAttributes);
    LoadSym("vaCreateConfig",           fn.vaCreateConfig);
    LoadSym("vaDestroyConfig",          fn.vaDestroyConfig);
    LoadSym("vaQuerySurfaceAttributes", fn.vaQuerySurfaceAttributes);
    LoadSym("vaCreateSurfaces",         fn.vaCreateSurfaces);
    LoadSym("vaDestroySurfaces",        fn.vaDestroySurfaces);
    LoadSym("vaCreateContext",          fn.vaCreateContext);
    LoadSym("vaDestroyContext",         fn.vaDestroyContext);
    LoadSym("vaCreateBuffer",           fn.vaCreateBuffer);
    LoadSym("vaDestroyBuffer",          fn.vaDestroyBuffer);
    LoadSym("vaBeginPicture",           fn.vaBeginPicture);
    LoadSym("vaRenderPicture",          fn.vaRenderPicture);
    LoadSym("vaEndPicture",             fn.vaEndPicture);
    LoadSym("vaQuerySurfaceStatus",     fn.vaQuerySurfaceStatus);
    LoadSym("vaSyncSurface",            fn.vaSyncSurface);
    LoadSym("vaExportSurfaceHandle",    fn.vaExportSurfaceHandle);
}

VaapiLoader::~VaapiLoader() {
    if (va_drm_handle_) {
        dlclose(va_drm_handle_);
        va_drm_handle_ = nullptr;
    }
}

#endif // ROCDECODE_USE_DLOPEN_VA
