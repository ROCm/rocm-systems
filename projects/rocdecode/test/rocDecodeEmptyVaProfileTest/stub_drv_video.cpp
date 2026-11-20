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


// Minimal VA-API driver for rocDecodeEmptyVaProfileTest. It loads and initializes like a real driver but advertises
// only VAProfileNone (the video post-processing profile) and no decode profile -- the shape of a VA stack that responds
// yet cannot decode. Only vaTerminate and vaQueryConfigProfiles are implemented; libva requires the other core vtable
// entries to be non-null, so they point at a stub that reports VA_STATUS_ERROR_UNIMPLEMENTED and is never expected to
// be called.

#include <va/va_backend.h>

#define STUB_CONCAT_(a, b, c) a##b##_##c
#define STUB_CONCAT(a, b, c) STUB_CONCAT_(a, b, c)
#define STUB_DRIVER_INIT STUB_CONCAT(__vaDriverInit_, VA_MAJOR_VERSION, VA_MINOR_VERSION)

namespace {

VAStatus StubTerminate(VADriverContextP) {
    return VA_STATUS_SUCCESS;
}

VAStatus StubQueryConfigProfiles(VADriverContextP, VAProfile *profile_list, int *num_profiles) {
    profile_list[0] = VAProfileNone;
    *num_profiles = 1;
    return VA_STATUS_SUCCESS;
}

VAStatus StubUnimplemented() {
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) VAStatus STUB_DRIVER_INIT(VADriverContextP ctx) {
    VADriverVTable *vtable = ctx->vtable;
#define STUB_UNIMPLEMENTED(entry) vtable->entry = reinterpret_cast<decltype(vtable->entry)>(&StubUnimplemented)
    vtable->vaTerminate = StubTerminate;
    vtable->vaQueryConfigProfiles = StubQueryConfigProfiles;
    STUB_UNIMPLEMENTED(vaQueryConfigEntrypoints);
    STUB_UNIMPLEMENTED(vaGetConfigAttributes);
    STUB_UNIMPLEMENTED(vaCreateConfig);
    STUB_UNIMPLEMENTED(vaDestroyConfig);
    STUB_UNIMPLEMENTED(vaQueryConfigAttributes);
    STUB_UNIMPLEMENTED(vaCreateSurfaces);
    STUB_UNIMPLEMENTED(vaDestroySurfaces);
    STUB_UNIMPLEMENTED(vaCreateContext);
    STUB_UNIMPLEMENTED(vaDestroyContext);
    STUB_UNIMPLEMENTED(vaCreateBuffer);
    STUB_UNIMPLEMENTED(vaBufferSetNumElements);
    STUB_UNIMPLEMENTED(vaMapBuffer);
    STUB_UNIMPLEMENTED(vaUnmapBuffer);
    STUB_UNIMPLEMENTED(vaDestroyBuffer);
    STUB_UNIMPLEMENTED(vaBeginPicture);
    STUB_UNIMPLEMENTED(vaRenderPicture);
    STUB_UNIMPLEMENTED(vaEndPicture);
    STUB_UNIMPLEMENTED(vaSyncSurface);
    STUB_UNIMPLEMENTED(vaQuerySurfaceStatus);
    STUB_UNIMPLEMENTED(vaPutSurface);
    STUB_UNIMPLEMENTED(vaQueryImageFormats);
    STUB_UNIMPLEMENTED(vaCreateImage);
    STUB_UNIMPLEMENTED(vaDeriveImage);
    STUB_UNIMPLEMENTED(vaDestroyImage);
    STUB_UNIMPLEMENTED(vaSetImagePalette);
    STUB_UNIMPLEMENTED(vaGetImage);
    STUB_UNIMPLEMENTED(vaPutImage);
    STUB_UNIMPLEMENTED(vaQuerySubpictureFormats);
    STUB_UNIMPLEMENTED(vaCreateSubpicture);
    STUB_UNIMPLEMENTED(vaDestroySubpicture);
    STUB_UNIMPLEMENTED(vaSetSubpictureImage);
    STUB_UNIMPLEMENTED(vaSetSubpictureChromakey);
    STUB_UNIMPLEMENTED(vaSetSubpictureGlobalAlpha);
    STUB_UNIMPLEMENTED(vaAssociateSubpicture);
    STUB_UNIMPLEMENTED(vaDeassociateSubpicture);
    STUB_UNIMPLEMENTED(vaQueryDisplayAttributes);
    STUB_UNIMPLEMENTED(vaGetDisplayAttributes);
    STUB_UNIMPLEMENTED(vaSetDisplayAttributes);
#undef STUB_UNIMPLEMENTED

    ctx->max_profiles = 1;
    ctx->max_entrypoints = 1;
    ctx->max_attributes = 1;
    ctx->max_image_formats = 1;
    ctx->max_subpic_formats = 1;
    ctx->max_display_attributes = 1;
    ctx->str_vendor = "rocDecode empty-profile stub driver";
    return VA_STATUS_SUCCESS;
}
