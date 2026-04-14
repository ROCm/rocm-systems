// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#if defined(_MSC_VER)
#    pragma warning(disable : 4200)
#endif
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
// Errors due to anonymous struct/union and flexible array member
#endif
#include "va/va.h"
#include "va/va_drm.h"
#include "va/va_drmcommon.h"
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <cstdint>
#include <cstdlib>

namespace rocprofsys
{
namespace component
{
struct vaapi_gotcha : tim::component::base<vaapi_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 30;

    using gotcha_data  = tim::component::gotcha_data;
    using exit_func_t  = void (*)(int);
    using abort_func_t = void (*)();

    ROCPROFSYS_DEFAULT_OBJECT(vaapi_gotcha)

    // string id for component
    static std::string label() { return "vaapi_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VAContextID context, VASurfaceID render_target);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VAContextID context, VABufferType type, unsigned int size,
                      unsigned int num_elements, void* data, VABufferID* buf_id);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VAProfile profile, VAEntrypoint entrypoint,
                      VAConfigAttrib* attrib_list, int num_attribs,
                      VAConfigID* config_id);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VAConfigID config_id, int picture_width, int picture_height,
                      int flag, VASurfaceID* render_targets, int num_render_targets,
                      VAContextID* context);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      unsigned int format, unsigned int width, unsigned int height,
                      VASurfaceID* surfaces, unsigned int num_surfaces,
                      VASurfaceAttrib* attrib_list, unsigned int num_attribs);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAImageFormat* format, int width, int height, VAImage* image);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VASurfaceID surface, VAImageID image, int src_x, int src_y,
                      unsigned int src_width, unsigned int src_height, int dest_x,
                      int dest_y, unsigned int dest_width, unsigned int dest_height);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VASurfaceID* surfaces, int num_surfaces);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VAContextID context);
    static void audit(const gotcha_data&, audit::incoming, VADisplay dpy,
                      VASurfaceID surface_id, uint32_t mem_type, uint32_t flags,
                      void* descriptor);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VASurfaceID surface, int x, int y, unsigned int width,
                      unsigned int height, VAImageID image);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAProfile profile, VAEntrypoint entrypoint,
                      VAConfigAttrib* attrib_list, int num_attribs);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      int* major_version, int* minor_version);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAImageFormat* format_list, int* num_formats);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAProfile profile, VAEntrypoint* entrypoint_list,
                      int* num_entrypoints);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAConfigID config, VASurfaceAttrib* attrib_list,
                      unsigned int* num_attribs);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VASurfaceID render_target, VASurfaceStatus* status);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VASurfaceID surface, VAImage* image);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VABufferID buf_id, void** pbuf);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VABufferID buf_id, VABufferInfo* buf_info);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                      VAContextID context, VABufferID* buffers, int num_buffers);
    static void audit(const gotcha_data& _data, audit::incoming, VADisplay dpy);

    static void audit(const gotcha_data&, audit::outgoing, VAStatus);
};
}  // namespace component

using vaapi_bundle_t = tim::component_bundle<category::vaapi, component::vaapi_gotcha>;
using vaapi_gotcha_t = tim::component::gotcha<component::vaapi_gotcha::gotcha_capacity,
                                              vaapi_bundle_t, category::vaapi>;
}  // namespace rocprofsys
