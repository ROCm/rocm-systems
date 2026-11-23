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

#pragma once

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>

// Function pointer table for the VA-API entry points used by rocEncode.
// Populated by VaapiLoader via dlsym after dlopen.
//
// Phase 1 resolves only what the capability probe needs plus the handful of
// lifecycle calls.  Encode submission entry points (vaBeginPicture,
// vaRenderPicture, vaEndPicture, vaMapBuffer, vaExportSurfaceHandle, ...) are
// deliberately absent until the phase that uses them; a symbol in this table
// that nothing calls is a symbol whose absence would fail the load for no
// reason.
struct VaapiVtable {
    // va-drm
    VADisplay       (*vaGetDisplayDRM)(int fd);
    // va core
    VAStatus        (*vaInitialize)(VADisplay dpy, int *major_version, int *minor_version);
    VAStatus        (*vaTerminate)(VADisplay dpy);
    void            (*vaSetInfoCallback)(VADisplay dpy, VAMessageCallback callback, void *user_context);
    const char *    (*vaQueryVendorString)(VADisplay dpy);
    const char *    (*vaErrorStr)(VAStatus error_status);
    int             (*vaMaxNumProfiles)(VADisplay dpy);
    int             (*vaMaxNumEntrypoints)(VADisplay dpy);
    VAStatus        (*vaQueryConfigProfiles)(VADisplay dpy, VAProfile *profile_list, int *num_profiles);
    VAStatus        (*vaQueryConfigEntrypoints)(VADisplay dpy, VAProfile profile,
                                                VAEntrypoint *entrypoint_list, int *num_entrypoints);
    VAStatus        (*vaGetConfigAttributes)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                             VAConfigAttrib *attrib_list, int num_attribs);
    VAStatus        (*vaCreateConfig)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                      VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id);
    VAStatus        (*vaDestroyConfig)(VADisplay dpy, VAConfigID config_id);
    VAStatus        (*vaQuerySurfaceAttributes)(VADisplay dpy, VAConfigID config,
                                                VASurfaceAttrib *attrib_list, unsigned int *num_attribs);
};

// Loads the ROCm sysdeps libva display backend at runtime and resolves all
// VA-API symbols into the VaapiVtable.
//
// dlopens librocm_sysdeps_va-drm.so.2 (and its transitive dependency
// librocm_sysdeps_va.so.2) with RTLD_LOCAL | RTLD_DEEPBIND, then resolves
// symbols via dlsym.  RTLD_LOCAL keeps the sysdeps va* symbols out of the
// global scope, isolating them from any system libva.so.2 loaded by another
// library in the same process (e.g. libavcodec).  RTLD_DEEPBIND makes libva
// and the radeonsi backend prefer their own symbol closure, so radeonsi's
// internal va* calls bind to the ROCm libva rather than a system one.
//
// dlmopen(LM_ID_NEWLM) cannot be used: glibc leaves the private namespace's
// global-scope array uninitialised with RTLD_LOCAL (vaInitialize's internal
// dlopen of radeonsi_drv_video.so with RTLD_GLOBAL then crashes), and rejects
// RTLD_GLOBAL with LM_ID_NEWLM outright.
class VaapiLoader {
public:
    VaapiVtable fn{};

    // Detects the location of the sysdeps libva display backend at runtime
    // (relative to the rocencode library's own location) and loads it.
    // Throws std::runtime_error on failure.
    VaapiLoader();
    ~VaapiLoader();

    VaapiLoader(const VaapiLoader &) = delete;
    VaapiLoader &operator=(const VaapiLoader &) = delete;

private:
    using LibHandle = void *;
    void *va_drm_handle_ = nullptr;

    // Finds librocm_sysdeps_va-drm.so.*:
    //   Primary: dladdr on a symbol in this translation unit to locate
    //   librocencode.so, then rocm_sysdeps/lib/ as a sibling directory.
    //   Fallback: $ROCM_PATH/lib/rocm_sysdeps/lib/.
    static std::filesystem::path FindVaDisplayLibPath();

    // Loads the library and resolves all symbols; throws on failure.
    void Load();
    // Releases the library handle and clears the function table.
    void Unload() noexcept;

    template <typename T>
    void LoadSym(LibHandle handle, const char *name, T *&fn_ptr);
};
