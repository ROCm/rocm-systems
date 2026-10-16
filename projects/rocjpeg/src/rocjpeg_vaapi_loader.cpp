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
#ifdef _WIN32
#include <string_view>
#endif

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

// Returns an owned copy of an environment variable, or an empty string if it
// is not set. GetEnvironmentVariableW reads the live process environment (the
// one the Windows loader uses), so changes made with SetEnvironmentVariableW
// after CRT startup are seen, unlike _wgetenv's CRT copy.
std::wstring GetEnvVar(const wchar_t *name) {
    std::wstring value;
    DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    // The variable can change between calls; retry while the buffer is too small.
    while (size > value.size()) {
        value.resize(size);
        size = GetEnvironmentVariableW(name, value.data(), static_cast<DWORD>(value.size()));
    }
    value.resize(size);
    return value;
}

} // namespace

// ---------------------------------------------------------------------------
// Path detection
// ---------------------------------------------------------------------------

std::filesystem::path RocJpegVaapiLoader::FindVaDisplayLibPath() {
    namespace fs = std::filesystem;
    const wchar_t *candidate = L"rocm_sysdeps_va_win32.dll";

    // Strategy 1: locate rocjpeg.dll via GetModuleHandleExW on a symbol in
    // this translation unit. DLLs are installed in <prefix>/bin while the
    // sysdeps DLLs live in <prefix>/lib/rocm_sysdeps/bin.
    // e.g. C:/opt/rocm/bin/rocjpeg.dll  ->  C:/opt/rocm/lib/rocm_sysdeps/bin/
    HMODULE self = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&RocJpegVaapiLoader::FindVaDisplayLibPath), &self)) {
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

    // Strategy 2: fall back to %ROCM_PATH%. A relative value is ignored so the
    // lookup never depends on the current directory.
    fs::path rocm_path(GetEnvVar(L"ROCM_PATH"));
    if (rocm_path.is_absolute()) {
        fs::path full = rocm_path / "lib" / "rocm_sysdeps" / "bin" / candidate;
        if (fs::exists(full)) {
            return full;
        }
    }

    // Strategy 3: search the directories listed in PATH. PATH is walked
    // explicitly rather than with SearchPathW, which also searches the current
    // directory and would let a DLL planted there be loaded. Relative entries
    // (e.g. ".") are skipped for the same reason.
    const std::wstring path_env = GetEnvVar(L"PATH");
    std::wstring_view entries(path_env);
    while (!entries.empty()) {
        size_t sep = entries.find(L';');
        std::wstring_view entry = entries.substr(0, sep);
        entries = (sep == std::wstring_view::npos) ? std::wstring_view() : entries.substr(sep + 1);
        if (entry.size() >= 2 && entry.front() == L'"' && entry.back() == L'"') {
            entry = entry.substr(1, entry.size() - 2);
        }
        fs::path dir(entry);
        if (!dir.is_absolute()) {
            continue;
        }
        std::error_code ec;
        fs::path full = dir / candidate;
        if (fs::is_regular_file(full, ec)) {
            return full;
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// Symbol loading helper
// ---------------------------------------------------------------------------

template <typename T>
void RocJpegVaapiLoader::LoadSym(HMODULE handle, const char *name, T *&fn_ptr) {
    fn_ptr = reinterpret_cast<T *>(GetProcAddress(handle, name));
    if (!fn_ptr) {
        DWORD error = GetLastError();
        throw std::runtime_error(std::string("RocJpegVaapiLoader: GetProcAddress('") + name + "'): " +
                                 Win32ErrorString(error));
    }
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

RocJpegVaapiLoader::RocJpegVaapiLoader() {
    // The destructor does not run when a constructor throws, so release any
    // library loaded before the failure here.
    try {
        Load();
    } catch (...) {
        Unload();
        throw;
    }
}

RocJpegVaapiLoader::~RocJpegVaapiLoader() {
    Unload();
}

void RocJpegVaapiLoader::Load() {
    std::filesystem::path va_display_path = FindVaDisplayLibPath();
    if (va_display_path.empty()) {
        throw std::runtime_error(
            "RocJpegVaapiLoader: cannot locate rocm_sysdeps_va_win32.dll; "
            "set ROCM_PATH to the ROCm installation prefix");
    }

    // Windows has no process-wide symbol scope: each DLL's imports bind to a
    // specific module by name, so there is no RTLD_LOCAL/RTLD_DEEPBIND
    // equivalent to worry about.
    //   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR — resolve the DLL's own dependencies
    //                 (rocm_sysdeps_va.dll) from the directory it is loaded from.
    //   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS — keep the application directory,
    //                 System32 (d3d12.dll, dxgi.dll) and AddDllDirectory paths
    //                 searchable.
    constexpr DWORD load_flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    va_win32_handle_ = LoadLibraryExW(va_display_path.c_str(), nullptr, load_flags);
    if (!va_win32_handle_) {
        DWORD error = GetLastError();
        throw std::runtime_error(std::string("RocJpegVaapiLoader: LoadLibraryExW('") + va_display_path.u8string() +
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
        throw std::runtime_error(std::string("RocJpegVaapiLoader: LoadLibraryExW('") + va_core_path.u8string() +
                                 "'): " + Win32ErrorString(error));
    }

    LoadSym(va_win32_handle_, "vaGetDisplayWin32",        fn.vaGetDisplayWin32);
    LoadSym(va_handle_,       "vaInitialize",             fn.vaInitialize);
    LoadSym(va_handle_,       "vaTerminate",              fn.vaTerminate);
    LoadSym(va_handle_,       "vaSetInfoCallback",        fn.vaSetInfoCallback);
    LoadSym(va_handle_,       "vaQueryVendorString",      fn.vaQueryVendorString);
    LoadSym(va_handle_,       "vaErrorStr",               fn.vaErrorStr);
    LoadSym(va_handle_,       "vaMaxNumEntrypoints",      fn.vaMaxNumEntrypoints);
    LoadSym(va_handle_,       "vaQueryConfigEntrypoints", fn.vaQueryConfigEntrypoints);
    LoadSym(va_handle_,       "vaGetConfigAttributes",    fn.vaGetConfigAttributes);
    LoadSym(va_handle_,       "vaCreateConfig",           fn.vaCreateConfig);
    LoadSym(va_handle_,       "vaDestroyConfig",          fn.vaDestroyConfig);
    LoadSym(va_handle_,       "vaQuerySurfaceAttributes", fn.vaQuerySurfaceAttributes);
    LoadSym(va_handle_,       "vaCreateSurfaces",         fn.vaCreateSurfaces);
    LoadSym(va_handle_,       "vaDestroySurfaces",        fn.vaDestroySurfaces);
    LoadSym(va_handle_,       "vaCreateContext",          fn.vaCreateContext);
    LoadSym(va_handle_,       "vaDestroyContext",         fn.vaDestroyContext);
    LoadSym(va_handle_,       "vaCreateBuffer",           fn.vaCreateBuffer);
    LoadSym(va_handle_,       "vaDestroyBuffer",          fn.vaDestroyBuffer);
    LoadSym(va_handle_,       "vaBeginPicture",           fn.vaBeginPicture);
    LoadSym(va_handle_,       "vaRenderPicture",          fn.vaRenderPicture);
    LoadSym(va_handle_,       "vaEndPicture",             fn.vaEndPicture);
    LoadSym(va_handle_,       "vaSyncSurface",            fn.vaSyncSurface);
    LoadSym(va_handle_,       "vaExportSurfaceHandle",    fn.vaExportSurfaceHandle);
}

void RocJpegVaapiLoader::Unload() noexcept {
    fn = {};
    if (va_handle_) {
        FreeLibrary(va_handle_);
        va_handle_ = nullptr;
    }
    if (va_win32_handle_) {
        FreeLibrary(va_win32_handle_);
        va_win32_handle_ = nullptr;
    }
}

#else // !_WIN32

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

#endif // _WIN32

#endif // ROCJPEG_USE_DLOPEN_VA
