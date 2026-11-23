/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "vaapi_loader.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

std::filesystem::path VaapiLoader::FindVaDisplayLibPath() {
    namespace fs = std::filesystem;

    // Strategy 1: locate librocencode.so via dladdr on a symbol in this
    // translation unit, then look for rocm_sysdeps/lib/ next to it.
    // e.g. /opt/rocm/lib/librocencode.so -> /opt/rocm/lib/rocm_sysdeps/lib/
    Dl_info dl_info{};
    if (dladdr(reinterpret_cast<void *>(&VaapiLoader::FindVaDisplayLibPath), &dl_info) &&
        dl_info.dli_fname) {
        fs::path sysdeps_lib = fs::path(dl_info.dli_fname).parent_path() / "rocm_sysdeps" / "lib";
        for (const char *candidate :
             {"librocm_sysdeps_va-drm.so.2", "librocm_sysdeps_va-drm.so"}) {
            fs::path full = sysdeps_lib / candidate;
            if (fs::exists(full)) {
                return full;
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
                return full;
            }
        }
    }

    return {};
}

template <typename T>
void VaapiLoader::LoadSym(LibHandle handle, const char *name, T *&fn_ptr) {
    dlerror();  // clear any prior error
    fn_ptr = reinterpret_cast<T *>(dlsym(handle, name));
    const char *err = dlerror();
    if (err || !fn_ptr) {
        throw std::runtime_error(std::string("VaapiLoader: dlsym('") + name + "'): " +
                                 (err ? err : "symbol not found"));
    }
}

VaapiLoader::VaapiLoader() {
    // The destructor does not run when a constructor throws, so release
    // anything loaded before the failure here.
    try {
        Load();
    } catch (...) {
        Unload();
        throw;
    }
}

VaapiLoader::~VaapiLoader() {
    Unload();
}

void VaapiLoader::Load() {
    std::filesystem::path va_display_path = FindVaDisplayLibPath();
    if (va_display_path.empty()) {
        throw std::runtime_error(
            "VaapiLoader: cannot locate librocm_sysdeps_va-drm.so.2; "
            "set ROCM_PATH to the ROCm installation prefix");
    }

    va_drm_handle_ = dlopen(va_display_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!va_drm_handle_) {
        throw std::runtime_error(std::string("VaapiLoader: dlopen('") + va_display_path.string() +
                                 "'): " + dlerror());
    }

    // dlsym searches the handle's DSO and all transitive DT_NEEDED
    // dependencies, so the va core symbols reachable through
    // librocm_sysdeps_va.so.2 are found from va_drm_handle_ alone.
    LoadSym(va_drm_handle_, "vaGetDisplayDRM", fn.vaGetDisplayDRM);

    LoadSym(va_drm_handle_, "vaInitialize",             fn.vaInitialize);
    LoadSym(va_drm_handle_, "vaTerminate",              fn.vaTerminate);
    LoadSym(va_drm_handle_, "vaSetInfoCallback",        fn.vaSetInfoCallback);
    LoadSym(va_drm_handle_, "vaQueryVendorString",      fn.vaQueryVendorString);
    LoadSym(va_drm_handle_, "vaErrorStr",               fn.vaErrorStr);
    LoadSym(va_drm_handle_, "vaMaxNumProfiles",         fn.vaMaxNumProfiles);
    LoadSym(va_drm_handle_, "vaMaxNumEntrypoints",      fn.vaMaxNumEntrypoints);
    LoadSym(va_drm_handle_, "vaQueryConfigProfiles",    fn.vaQueryConfigProfiles);
    LoadSym(va_drm_handle_, "vaQueryConfigEntrypoints", fn.vaQueryConfigEntrypoints);
    LoadSym(va_drm_handle_, "vaGetConfigAttributes",    fn.vaGetConfigAttributes);
    LoadSym(va_drm_handle_, "vaCreateConfig",           fn.vaCreateConfig);
    LoadSym(va_drm_handle_, "vaDestroyConfig",          fn.vaDestroyConfig);
    LoadSym(va_drm_handle_, "vaQuerySurfaceAttributes", fn.vaQuerySurfaceAttributes);
}

void VaapiLoader::Unload() noexcept {
    fn = {};
    if (va_drm_handle_) {
        dlclose(va_drm_handle_);
        va_drm_handle_ = nullptr;
    }
}
