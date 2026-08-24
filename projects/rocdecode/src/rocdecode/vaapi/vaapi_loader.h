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

#pragma once

#ifdef ROCDECODE_USE_DLOPEN_VA

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>

// Function pointer table for all VA-API entry points used by rocdecode.
// Populated by VaapiLoader via dlsym after dlopen.
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
    VAStatus        (*vaQueryConfigProfiles)(VADisplay dpy, VAProfile *profile_list, int *num_profiles);
    VAStatus        (*vaGetConfigAttributes)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                            VAConfigAttrib *attrib_list, int num_attribs);
    VAStatus        (*vaCreateConfig)(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                     VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id);
    VAStatus        (*vaDestroyConfig)(VADisplay dpy, VAConfigID config_id);
    VAStatus        (*vaQuerySurfaceAttributes)(VADisplay dpy, VAConfigID config,
                                               VASurfaceAttrib *attrib_list, unsigned int *num_attribs);
    VAStatus        (*vaCreateSurfaces)(VADisplay dpy, unsigned int format, unsigned int width,
                                        unsigned int height, VASurfaceID *surfaces,
                                        unsigned int num_surfaces, VASurfaceAttrib *attrib_list,
                                        unsigned int num_attribs);
    VAStatus        (*vaDestroySurfaces)(VADisplay dpy, VASurfaceID *surfaces, int num_surfaces);
    VAStatus        (*vaCreateContext)(VADisplay dpy, VAConfigID config_id, int picture_width,
                                      int picture_height, int flag, VASurfaceID *render_targets,
                                      int num_render_targets, VAContextID *context);
    VAStatus        (*vaDestroyContext)(VADisplay dpy, VAContextID context);
    VAStatus        (*vaCreateBuffer)(VADisplay dpy, VAContextID context, VABufferType type,
                                     unsigned int size, unsigned int num_elements, void *data,
                                     VABufferID *buf_id);
    VAStatus        (*vaDestroyBuffer)(VADisplay dpy, VABufferID buffer_id);
    VAStatus        (*vaBeginPicture)(VADisplay dpy, VAContextID context, VASurfaceID render_target);
    VAStatus        (*vaRenderPicture)(VADisplay dpy, VAContextID context,
                                      VABufferID *buffers, int num_buffers);
    VAStatus        (*vaEndPicture)(VADisplay dpy, VAContextID context);
    VAStatus        (*vaQuerySurfaceStatus)(VADisplay dpy, VASurfaceID render_target,
                                           VASurfaceStatus *status);
    VAStatus        (*vaSyncSurface)(VADisplay dpy, VASurfaceID render_target);
    VAStatus        (*vaExportSurfaceHandle)(VADisplay dpy, VASurfaceID surface_id,
                                            uint32_t mem_type, uint32_t flags, void *descriptor);
};

// Loads librocm_sysdeps_va-drm.so.2 (and its transitive dependency
// librocm_sysdeps_va.so.2) via dlopen(RTLD_LOCAL | RTLD_DEEPBIND), then resolves
// all VA-API symbols via dlsym into the VaapiVtable.
//
// RTLD_LOCAL keeps sysdeps va* symbols out of the global scope, isolating them
// from any system libva.so.2 loaded by other libraries (e.g. libavcodec).
class VaapiLoader {
public:
    VaapiVtable fn{};

    // Detects the path of librocm_sysdeps_va-drm.so.2 at runtime (relative
    // to librocdecode.so's own location) and dlopens it.
    VaapiLoader();
    ~VaapiLoader();

    VaapiLoader(const VaapiLoader &) = delete;
    VaapiLoader &operator=(const VaapiLoader &) = delete;

private:
    void *va_drm_handle_ = nullptr;

    // Finds the path of librocm_sysdeps_va-drm.so.* at runtime.
    // Primary strategy: dladdr on a symbol in this translation unit to locate
    // librocdecode.so, then look for rocm_sysdeps/lib/ as a sibling directory.
    // Fallback: $ROCM_PATH/lib/rocm_sysdeps/lib/.
    static std::string FindVaDrmLibPath();

    template <typename T>
    void LoadSym(const char *name, T *&fn_ptr);
};

#endif // ROCDECODE_USE_DLOPEN_VA
