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

#pragma once

#ifdef ROCJPEG_USE_DLOPEN_VA

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <va/va.h>
#ifdef _WIN32
#include <va/va_win32.h>
#else
#include <va/va_drm.h>
#endif
#include <va/va_drmcommon.h>

// Function pointer table for all VA-API entry points used by rocjpeg.
// Populated by RocJpegVaapiLoader via dlsym after dlopen (Linux) or
// GetProcAddress after LoadLibraryExW (Windows).
struct RocJpegVaapiVtable {
#ifdef _WIN32
    // va_win32
    VADisplay    (*vaGetDisplayWin32)(const LUID *adapter_luid);
#else
    // va-drm
    VADisplay    (*vaGetDisplayDRM)(int fd);
#endif
    // va core
    VAStatus     (*vaInitialize)(VADisplay dpy, int *major_version, int *minor_version);
    VAStatus     (*vaTerminate)(VADisplay dpy);
    void         (*vaSetInfoCallback)(VADisplay dpy, VAMessageCallback callback, void *user_context);
    const char * (*vaQueryVendorString)(VADisplay dpy);
    const char * (*vaErrorStr)(VAStatus error_status);
    int          (*vaMaxNumEntrypoints)(VADisplay dpy);
    VAStatus     (*vaQueryConfigEntrypoints)(VADisplay dpy, VAProfile profile,
                                            VAEntrypoint *entrypoint_list, int *num_entrypoints);
    VAStatus     (*vaGetConfigAttributes)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                         VAConfigAttrib *attrib_list, int num_attribs);
    VAStatus     (*vaCreateConfig)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                  VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id);
    VAStatus     (*vaDestroyConfig)(VADisplay dpy, VAConfigID config_id);
    VAStatus     (*vaQuerySurfaceAttributes)(VADisplay dpy, VAConfigID config,
                                            VASurfaceAttrib *attrib_list, unsigned int *num_attribs);
    VAStatus     (*vaCreateSurfaces)(VADisplay dpy, unsigned int format, unsigned int width,
                                     unsigned int height, VASurfaceID *surfaces,
                                     unsigned int num_surfaces, VASurfaceAttrib *attrib_list,
                                     unsigned int num_attribs);
    VAStatus     (*vaDestroySurfaces)(VADisplay dpy, VASurfaceID *surfaces, int num_surfaces);
    VAStatus     (*vaCreateContext)(VADisplay dpy, VAConfigID config_id, int picture_width,
                                   int picture_height, int flag, VASurfaceID *render_targets,
                                   int num_render_targets, VAContextID *context);
    VAStatus     (*vaDestroyContext)(VADisplay dpy, VAContextID context);
    VAStatus     (*vaCreateBuffer)(VADisplay dpy, VAContextID context, VABufferType type,
                                  unsigned int size, unsigned int num_elements, void *data,
                                  VABufferID *buf_id);
    VAStatus     (*vaDestroyBuffer)(VADisplay dpy, VABufferID buffer_id);
    VAStatus     (*vaBeginPicture)(VADisplay dpy, VAContextID context, VASurfaceID render_target);
    VAStatus     (*vaRenderPicture)(VADisplay dpy, VAContextID context,
                                   VABufferID *buffers, int num_buffers);
    VAStatus     (*vaEndPicture)(VADisplay dpy, VAContextID context);
    VAStatus     (*vaSyncSurface)(VADisplay dpy, VASurfaceID render_target);
    VAStatus     (*vaExportSurfaceHandle)(VADisplay dpy, VASurfaceID surface_id,
                                         uint32_t mem_type, uint32_t flags, void *descriptor);
};

// Loads librocm_sysdeps_va-drm.so.2 (and its transitive dependency
// librocm_sysdeps_va.so.2) via dlopen with RTLD_LOCAL | RTLD_DEEPBIND,
// then resolves all VA-API symbols via dlsym into RocJpegVaapiVtable.
//
// RTLD_LOCAL keeps sysdeps libva symbols out of the process-wide global
// symbol scope so they cannot collide with system libva.so.2 loaded by
// other libraries (e.g. libavcodec) in the same process.
// RTLD_DEEPBIND makes libva and its backend driver prefer their own symbol
// closure when resolving symbols, before consulting the global scope.
//
// On Windows, loads rocm_sysdeps_va_win32.dll (and its dependency
// rocm_sysdeps_va.dll) via LoadLibraryExW instead. vaGetDisplayWin32 replaces
// vaGetDisplayDRM; there is no DRM on Windows. Unlike dlsym, GetProcAddress
// does not search a module's dependencies, so the va core symbols are resolved
// from a separate handle to rocm_sysdeps_va.dll.
class RocJpegVaapiLoader {
public:
    RocJpegVaapiVtable fn{};

    // Returns the process-wide shared loader, constructing it on first call.
    // Thread-safe. Throws std::runtime_error if the library cannot be found
    // or loaded.
    static std::shared_ptr<RocJpegVaapiLoader> GetShared();

    ~RocJpegVaapiLoader();

    RocJpegVaapiLoader(const RocJpegVaapiLoader &) = delete;
    RocJpegVaapiLoader &operator=(const RocJpegVaapiLoader &) = delete;

private:
    RocJpegVaapiLoader();

#ifdef _WIN32
    HMODULE va_win32_handle_ = nullptr;  // rocm_sysdeps_va_win32.dll
    HMODULE va_handle_ = nullptr;        // rocm_sysdeps_va.dll (va core)

    // Finds rocm_sysdeps_va_win32.dll: next to the installed rocjpeg.dll
    // (<prefix>/bin -> <prefix>/lib/rocm_sysdeps/bin), then under %ROCM_PATH%,
    // then in the absolute directories listed in PATH (never the current directory).
    static std::filesystem::path FindVaDisplayLibPath();

    // Loads the libraries and resolves all symbols; throws on failure.
    void Load();
    // Releases every library handle and clears the function table.
    void Unload() noexcept;

    template <typename T>
    void LoadSym(HMODULE handle, const char *name, T *&fn_ptr);
#else
    void *va_drm_handle_ = nullptr;

    static std::string FindVaDrmLibPath();

    template <typename T>
    void LoadSym(const char *name, T *&fn_ptr);
#endif
};

#endif // ROCJPEG_USE_DLOPEN_VA
