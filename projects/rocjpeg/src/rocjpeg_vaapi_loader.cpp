/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef ROCJPEG_USE_DLOPEN_VA

#include "rocjpeg_vaapi_loader.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Path detection
// ---------------------------------------------------------------------------

std::string RocJpegVaapiLoader::FindVaDrmLibPath() {
    namespace fs = std::filesystem;

    // Strategy 1: locate librocjpeg.so via dladdr on a symbol in this
    // translation unit, then look for rocm_sysdeps/lib/ next to it.
    Dl_info dl_info{};
    if (dladdr(reinterpret_cast<void *>(&RocJpegVaapiLoader::FindVaDrmLibPath), &dl_info) &&
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
void RocJpegVaapiLoader::LoadSym(const char *name, T *&fn_ptr) {
    dlerror();
    fn_ptr = reinterpret_cast<T *>(dlsym(va_drm_handle_, name));
    const char *err = dlerror();
    if (err || !fn_ptr) {
        throw std::runtime_error(std::string("RocJpegVaapiLoader: dlsym('") + name + "'): " +
                                 (err ? err : "symbol not found"));
    }
}

// ---------------------------------------------------------------------------
// Shared-instance accessor
// ---------------------------------------------------------------------------

std::shared_ptr<RocJpegVaapiLoader> RocJpegVaapiLoader::GetShared() {
    static std::mutex mtx;
    static std::weak_ptr<RocJpegVaapiLoader> weak;

    std::lock_guard<std::mutex> lock(mtx);
    if (auto shared = weak.lock()) {
        return shared;
    }
    // Allocate manually because the constructor is private.
    auto loader = std::shared_ptr<RocJpegVaapiLoader>(new RocJpegVaapiLoader());
    weak = loader;
    return loader;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

RocJpegVaapiLoader::RocJpegVaapiLoader() {
    std::string va_drm_path = FindVaDrmLibPath();
    if (va_drm_path.empty()) {
        throw std::runtime_error(
            "RocJpegVaapiLoader: cannot locate librocm_sysdeps_va-drm.so.2; "
            "set ROCM_PATH to the ROCm installation prefix");
    }

    // RTLD_LOCAL keeps librocm_sysdeps_va symbols out of the process-wide
    // global symbol scope, preventing collision with system libva.so.2.
    // RTLD_DEEPBIND makes libva and its backend driver prefer their own
    // symbol closure over the global scope. This is required even when
    // libavcodec (and hence system libva) is loaded — e.g. in extended tests
    // that use FFmpeg for demuxing — to ensure radeonsi_drv_video.so resolves
    // its va* symbols to ROCm libva rather than system libva.
    // Note: RTLD_DEEPBIND causes libva-internal allocations to bypass ASan's
    // interposed malloc/free, so they are not tracked by ASan. This is
    // acceptable because libva manages its own memory and rocjpeg only holds
    // opaque VA handles — there are no cross-boundary frees for ASan to catch.
    // librocm_sysdeps_va.so.2 is pulled in automatically as a DT_NEEDED
    // dependency of librocm_sysdeps_va-drm.so.2, so all va-core symbols
    // are reachable from va_drm_handle_ via transitive dlsym search.
    va_drm_handle_ = dlopen(va_drm_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!va_drm_handle_) {
        throw std::runtime_error(std::string("RocJpegVaapiLoader: dlopen('") + va_drm_path +
                                 "'): " + dlerror());
    }

    LoadSym("vaGetDisplayDRM",          fn.vaGetDisplayDRM);
    LoadSym("vaInitialize",             fn.vaInitialize);
    LoadSym("vaTerminate",              fn.vaTerminate);
    LoadSym("vaSetInfoCallback",        fn.vaSetInfoCallback);
    LoadSym("vaQueryVendorString",      fn.vaQueryVendorString);
    LoadSym("vaErrorStr",               fn.vaErrorStr);
    LoadSym("vaMaxNumEntrypoints",      fn.vaMaxNumEntrypoints);
    LoadSym("vaQueryConfigEntrypoints", fn.vaQueryConfigEntrypoints);
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
    LoadSym("vaSyncSurface",            fn.vaSyncSurface);
    LoadSym("vaExportSurfaceHandle",    fn.vaExportSurfaceHandle);
}

RocJpegVaapiLoader::~RocJpegVaapiLoader() {
    if (va_drm_handle_) {
        dlclose(va_drm_handle_);
        va_drm_handle_ = nullptr;
    }
}

#endif // ROCJPEG_USE_DLOPEN_VA
