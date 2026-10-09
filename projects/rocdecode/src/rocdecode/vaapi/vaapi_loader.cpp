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

#ifdef _WIN32
namespace {

// Formats a Win32 error code as "<system message> (error <code>)".
std::string Win32ErrorString(DWORD error) {
    char *msg = nullptr;
    DWORD len = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                   FORMAT_MESSAGE_IGNORE_INSERTS,
                               nullptr, error, 0, reinterpret_cast<LPSTR>(&msg), 0, nullptr);
    std::string result;
    if (len && msg) {
        result.assign(msg, len);
        // FormatMessage terminates the message with "\r\n".
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
            result.pop_back();
        }
    }
    if (msg) {
        LocalFree(msg);
    }
    return result + " (error " + std::to_string(error) + ")";
}

} // namespace
#endif // _WIN32

// ---------------------------------------------------------------------------
// Path detection
// ---------------------------------------------------------------------------

#ifdef _WIN32

std::filesystem::path VaapiLoader::FindVaDisplayLibPath() {
    namespace fs = std::filesystem;
    const wchar_t *candidate = L"rocm_sysdeps_va_win32.dll";

    // Strategy 1: locate rocdecode.dll via GetModuleHandleExW on a symbol in
    // this translation unit. DLLs are installed in <prefix>/bin while the
    // sysdeps DLLs live in <prefix>/lib/rocm_sysdeps/bin.
    // e.g. C:/opt/rocm/bin/rocdecode.dll  ->  C:/opt/rocm/lib/rocm_sysdeps/bin/
    HMODULE self = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&VaapiLoader::FindVaDisplayLibPath), &self)) {
        std::wstring module_path(MAX_PATH, L'\0');
        for (;;) {
            DWORD len = GetModuleFileNameW(self, module_path.data(), static_cast<DWORD>(module_path.size()));
            if (len == 0) {
                module_path.clear();
                break;
            }
            if (len < module_path.size()) {
                module_path.resize(len);
                break;
            }
            // Truncated; grow the buffer and retry.
            module_path.resize(module_path.size() * 2);
        }
        if (!module_path.empty()) {
            fs::path full = fs::path(module_path).parent_path().parent_path() / "lib" / "rocm_sysdeps" / "bin" / candidate;
            if (fs::exists(full)) {
                return full;
            }
        }
    }

    // Strategy 2: fall back to %ROCM_PATH%.
    const wchar_t *rocm_path = _wgetenv(L"ROCM_PATH");
    if (rocm_path) {
        fs::path full = fs::path(rocm_path) / "lib" / "rocm_sysdeps" / "bin" / candidate;
        if (fs::exists(full)) {
            return full;
        }
    }

    // Strategy 3: search PATH, which is how the DLL was found when rocdecode
    // linked against it directly (e.g. %ROCM_PATH%\lib\rocm_sysdeps\bin on PATH).
    std::wstring found(MAX_PATH, L'\0');
    DWORD len = SearchPathW(nullptr, candidate, nullptr, static_cast<DWORD>(found.size()), found.data(), nullptr);
    if (len > found.size()) {
        // Buffer too small; len is the required size including the terminator.
        found.resize(len);
        len = SearchPathW(nullptr, candidate, nullptr, static_cast<DWORD>(found.size()), found.data(), nullptr);
    }
    if (len > 0 && len < found.size()) {
        found.resize(len);
        return fs::path(found);
    }

    return {};
}

#else

std::filesystem::path VaapiLoader::FindVaDisplayLibPath() {
    namespace fs = std::filesystem;

    // Strategy 1: locate librocdecode.so via dladdr on a symbol in this
    // translation unit, then look for rocm_sysdeps/lib/ next to it.
    // e.g. /opt/rocm/lib/librocdecode.so  ->  /opt/rocm/lib/rocm_sysdeps/lib/
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

#endif // _WIN32

// ---------------------------------------------------------------------------
// Symbol loading helper
// ---------------------------------------------------------------------------

template <typename T>
void VaapiLoader::LoadSym(LibHandle handle, const char *name, T *&fn_ptr) {
#ifdef _WIN32
    fn_ptr = reinterpret_cast<T *>(GetProcAddress(handle, name));
    if (!fn_ptr) {
        DWORD error = GetLastError();
        throw std::runtime_error(std::string("VaapiLoader: GetProcAddress('") + name + "'): " +
                                 Win32ErrorString(error));
    }
#else
    dlerror(); // clear any prior error
    fn_ptr = reinterpret_cast<T *>(dlsym(handle, name));
    const char *err = dlerror();
    if (err || !fn_ptr) {
        throw std::runtime_error(std::string("VaapiLoader: dlsym('") + name + "'): " +
                                 (err ? err : "symbol not found"));
    }
#endif
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

VaapiLoader::VaapiLoader() {
    // The destructor does not run when a constructor throws, so release any
    // library loaded before the failure here.
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
#ifdef _WIN32
    if (va_display_path.empty()) {
        throw std::runtime_error(
            "VaapiLoader: cannot locate rocm_sysdeps_va_win32.dll; "
            "set ROCM_PATH to the ROCm installation prefix");
    }

    // Windows has no process-wide symbol scope: each DLL's imports bind to a
    // specific module by name, so there is no RTLD_LOCAL/RTLD_DEEPBIND
    // equivalent to worry about. Loading at runtime removes the link-time
    // dependency on the sysdeps import libraries and the need to have
    // %ROCM_PATH%\lib\rocm_sysdeps\bin on PATH.
    //
    //   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR — resolve the DLL's own dependencies
    //                 (rocm_sysdeps_va.dll) from the directory it is loaded from.
    //   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS — keep the application directory,
    //                 System32 (d3d12.dll, dxgi.dll) and AddDllDirectory paths
    //                 searchable.
    constexpr DWORD load_flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    va_win32_handle_ = LoadLibraryExW(va_display_path.c_str(), nullptr, load_flags);
    if (!va_win32_handle_) {
        DWORD error = GetLastError();
        throw std::runtime_error(std::string("VaapiLoader: LoadLibraryExW('") + va_display_path.u8string() +
                                 "'): " + Win32ErrorString(error));
    }

    // GetProcAddress, unlike dlsym, only searches the given module and not its
    // dependencies, so the va core symbols need their own handle. The module is
    // already mapped as a dependency of rocm_sysdeps_va_win32.dll; this only
    // takes a reference to it.
    std::filesystem::path va_core_path = va_display_path.parent_path() / L"rocm_sysdeps_va.dll";
    va_handle_ = LoadLibraryExW(va_core_path.c_str(), nullptr, load_flags);
    if (!va_handle_) {
        DWORD error = GetLastError();
        throw std::runtime_error(std::string("VaapiLoader: LoadLibraryExW('") + va_core_path.u8string() +
                                 "'): " + Win32ErrorString(error));
    }

    LoadSym(va_win32_handle_, "vaGetDisplayWin32", fn.vaGetDisplayWin32);
    LibHandle va_core_handle = va_handle_;
#else
    if (va_display_path.empty()) {
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
    va_drm_handle_ = dlopen(va_display_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!va_drm_handle_) {
        throw std::runtime_error(std::string("VaapiLoader: dlopen('") + va_display_path.string() +
                                 "'): " + dlerror());
    }

    // dlsym searches the handle's DSO and all transitive DT_NEEDED
    // dependencies, so va-core symbols reachable through
    // librocm_sysdeps_va.so.2 are found from the va_drm_handle_ alone.
    LoadSym(va_drm_handle_, "vaGetDisplayDRM", fn.vaGetDisplayDRM);
    LibHandle va_core_handle = va_drm_handle_;
#endif

    // Resolve the va core symbols.
    LoadSym(va_core_handle, "vaInitialize",             fn.vaInitialize);
    LoadSym(va_core_handle, "vaTerminate",              fn.vaTerminate);
    LoadSym(va_core_handle, "vaSetInfoCallback",        fn.vaSetInfoCallback);
    LoadSym(va_core_handle, "vaQueryVendorString",      fn.vaQueryVendorString);
    LoadSym(va_core_handle, "vaErrorStr",               fn.vaErrorStr);
    LoadSym(va_core_handle, "vaMaxNumProfiles",         fn.vaMaxNumProfiles);
    LoadSym(va_core_handle, "vaMaxNumEntrypoints",      fn.vaMaxNumEntrypoints);
    LoadSym(va_core_handle, "vaQueryConfigProfiles",    fn.vaQueryConfigProfiles);
    LoadSym(va_core_handle, "vaQueryConfigEntrypoints", fn.vaQueryConfigEntrypoints);
    LoadSym(va_core_handle, "vaGetConfigAttributes",    fn.vaGetConfigAttributes);
    LoadSym(va_core_handle, "vaCreateConfig",           fn.vaCreateConfig);
    LoadSym(va_core_handle, "vaDestroyConfig",          fn.vaDestroyConfig);
    LoadSym(va_core_handle, "vaQuerySurfaceAttributes", fn.vaQuerySurfaceAttributes);
    LoadSym(va_core_handle, "vaCreateSurfaces",         fn.vaCreateSurfaces);
    LoadSym(va_core_handle, "vaDestroySurfaces",        fn.vaDestroySurfaces);
    LoadSym(va_core_handle, "vaCreateContext",          fn.vaCreateContext);
    LoadSym(va_core_handle, "vaDestroyContext",         fn.vaDestroyContext);
    LoadSym(va_core_handle, "vaCreateBuffer",           fn.vaCreateBuffer);
    LoadSym(va_core_handle, "vaDestroyBuffer",          fn.vaDestroyBuffer);
    LoadSym(va_core_handle, "vaBeginPicture",           fn.vaBeginPicture);
    LoadSym(va_core_handle, "vaRenderPicture",          fn.vaRenderPicture);
    LoadSym(va_core_handle, "vaEndPicture",             fn.vaEndPicture);
    LoadSym(va_core_handle, "vaQuerySurfaceStatus",     fn.vaQuerySurfaceStatus);
    LoadSym(va_core_handle, "vaSyncSurface",            fn.vaSyncSurface);
    LoadSym(va_core_handle, "vaExportSurfaceHandle",    fn.vaExportSurfaceHandle);
}

void VaapiLoader::Unload() noexcept {
    fn = {};
#ifdef _WIN32
    if (va_handle_) {
        FreeLibrary(va_handle_);
        va_handle_ = nullptr;
    }
    if (va_win32_handle_) {
        FreeLibrary(va_win32_handle_);
        va_win32_handle_ = nullptr;
    }
#else
    if (va_drm_handle_) {
        dlclose(va_drm_handle_);
        va_drm_handle_ = nullptr;
    }
#endif
}

#endif // ROCDECODE_USE_DLOPEN_VA
