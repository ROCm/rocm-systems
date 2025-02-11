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

#include "library/components/vaapi_gotcha.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdlib>

namespace rocprofsys
{
namespace component
{
namespace
{
auto&
get_vaapi_gotcha()
{
    static auto _v = tim::lightweight_tuple<vaapi_gotcha_t>{};
    return _v;
}
}  // namespace

void
vaapi_gotcha::configure()
{
    // don't emit warnings for missing functions unless debug or verbosity >= 3
    if(get_verbose_env() < 3 && !get_debug_env())
    {
        for(size_t i = 0; i < vaapi_gotcha_t::capacity(); ++i)
        {
            auto* itr = vaapi_gotcha_t::at(i);
            if(itr) itr->verbose = -1;
        }
    }

    vaapi_gotcha_t::get_initializer() = []() {
        vaapi_gotcha_t::configure<0, VAStatus, VADisplay, VAContextID, VASurfaceID>(
            "vaBeginPicture");
        vaapi_gotcha_t::configure<1, VAStatus, VADisplay, VAContextID, VABufferType,
                                  unsigned int, unsigned int, void*, VABufferID*>(
            "vaCreateBuffer");
        vaapi_gotcha_t::configure<2, VAStatus, VADisplay, VAProfile, VAEntrypoint,
                                  VAConfigAttrib*, int, VAConfigID*>("vaCreateConfig");
        vaapi_gotcha_t::configure<3, VAStatus, VADisplay, VAConfigID, int, int, int,
                                  VASurfaceID*, int, VAContextID*>("vaCreateContext");
        vaapi_gotcha_t::configure<4, VAStatus, VADisplay, unsigned int, unsigned int,
                                  unsigned int, VASurfaceID*, unsigned int,
                                  VASurfaceAttrib*, unsigned int>("vaCreateSurfaces");
        vaapi_gotcha_t::configure<5, VAStatus, VADisplay, VASurfaceID*, int>(
            "vaDestroySurfaces");
        vaapi_gotcha_t::configure<6, VAStatus, VADisplay, VASurfaceID>("vaSyncSurface");
        vaapi_gotcha_t::configure<7, VAStatus, VADisplay, VABufferID>("vaDestroyBuffer");
        vaapi_gotcha_t::configure<8, VAStatus, VADisplay, VAConfigID>("vaDestroyConfig");
        vaapi_gotcha_t::configure<9, VAStatus, VADisplay, VAContextID>(
            "vaDestroyContext");
        vaapi_gotcha_t::configure<10, VAStatus, VADisplay, VAContextID>("vaEndPicture");
        vaapi_gotcha_t::configure<11, VAStatus, VADisplay, VASurfaceID, uint32_t,
                                  uint32_t, void*>("vaExportSurfaceHandle");
        vaapi_gotcha_t::configure<12, VAStatus, VADisplay, VAProfile, VAEntrypoint,
                                  VAConfigAttrib*, int>("vaGetConfigAttributes");
        vaapi_gotcha_t::configure<13, VAStatus, VADisplay, int*, int*>("vaInitialize");
        vaapi_gotcha_t::configure<14, VAStatus, VADisplay, VAProfile, VAEntrypoint*,
                                  int*>("vaQueryConfigEntrypoints");
        vaapi_gotcha_t::configure<15, VAStatus, VADisplay, VAConfigID, VASurfaceAttrib*,
                                  unsigned int*>("vaQuerySurfaceAttributes");
        vaapi_gotcha_t::configure<16, VAStatus, VADisplay, VASurfaceID, VASurfaceStatus*>(
            "vaQuerySurfaceStatus");
        vaapi_gotcha_t::configure<17, VAStatus, VADisplay, VAContextID, VABufferID*, int>(
            "vaRenderPicture");
        vaapi_gotcha_t::configure<18, VAStatus, VADisplay>("vaTerminate");
        vaapi_gotcha_t::configure<19, int, VADisplay>("vaDisplayIsValid");
    };
}

void
vaapi_gotcha::shutdown()
{
    vaapi_gotcha_t::disable();
}

void
vaapi_gotcha::start()
{
    if(!get_vaapi_gotcha().get<vaapi_gotcha_t>()->get_is_running())
    {
        configure();
        get_vaapi_gotcha().start();
    }
}

void
vaapi_gotcha::stop()
{}

// vaBeginPicture
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAContextID context, VASurfaceID render_target)
{
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "context", context, "render_target",
                                            render_target);
}

// vaCreateBuffer
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAContextID context, VABufferType type, unsigned int size,
                    unsigned int num_elements, void* data, VABufferID* buf_id)
{
    category_region<category::vaapi>::start(
        std::string_view{ _data.tool_id }, "dpy", dpy, "context", context, "buffer_type",
        type, "size", size, "num_elements", num_elements, "data", data, "buf_id", buf_id);
}

// vaCreateConfig
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAProfile profile, VAEntrypoint entrypoint,
                    VAConfigAttrib* attrib_list, int num_attribs, VAConfigID* config_id)
{
    (void) attrib_list;  // unused
    category_region<category::vaapi>::start(
        std::string_view{ _data.tool_id }, "dpy", dpy, "profile", profile, "entrypoint",
        entrypoint, "num_attribs", num_attribs, "config_id", config_id);
}

// vaCreateContext
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAConfigID config_id, int picture_width, int picture_height, int flag,
                    VASurfaceID* render_targets, int num_render_targets,
                    VAContextID* context)
{
    category_region<category::vaapi>::start(
        std::string_view{ _data.tool_id }, "dpy", dpy, "config_id", config_id,
        "picture_width", picture_width, "picture_height", picture_height, "flag", flag,
        "render_targets", render_targets, "num_render_targets", num_render_targets,
        "context", context);
}

// vaCreateSurfaces
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    unsigned int format, unsigned int width, unsigned int height,
                    VASurfaceID* surfaces, unsigned int num_surfaces,
                    VASurfaceAttrib* attrib_list, unsigned int num_attribs)
{
    (void) attrib_list;  // unused
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "format", format, "width", width, "height",
                                            height, "surfaces", surfaces, "num_surfaces",
                                            num_surfaces, "num_attribs", num_attribs);
}

// vaDestroyBuffer
// vaDestroyConfig
// vaDestroyContext
// vaEndPicture
// vaSyncSurface
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAContextID context)
{
    if(_data.tool_id == "vaDestroyBuffer")
        category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                                dpy, "buffer_id", context);
    else if(_data.tool_id == "vaDestroyConfig")
        category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                                dpy, "config_id", context);
    else if(_data.tool_id == "vaDestroyContext")
        category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                                dpy, "context", context);
    else if(_data.tool_id == "vaEndPicture")
        category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                                dpy, "context", context);
    else if(_data.tool_id == "vaSyncSurface")
        category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                                dpy, "render_target", context);
}

// vaDestroySurfaces
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VASurfaceID* surfaces, int num_surfaces)
{
    (void) surfaces;  // unused
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "num_surfaces", num_surfaces);
}

// vaExportSurfaceHandle
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VASurfaceID surface_id, uint32_t mem_type, uint32_t flags,
                    void* descriptor)
{
    category_region<category::vaapi>::start(
        std::string_view{ _data.tool_id }, "dpy", dpy, "surface_id", surface_id,
        "mem_type", mem_type, "flags", flags, "descriptor", descriptor);
}

// vaGetConfigAttributes
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAProfile profile, VAEntrypoint entrypoint,
                    VAConfigAttrib* attrib_list, int num_attribs)
{
    (void) attrib_list;  // unused
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "profile", profile, "entrypoint", entrypoint,
                                            "num_attribs", num_attribs);
}

// vaInitialize
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    int* major_version, int* minor_version)
{
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "major_version", major_version,
                                            "minor_version", minor_version);
}

// vaQueryConfigEntrypoints
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAProfile profile, VAEntrypoint* entrypoint_list,
                    int* num_entrypoints)
{
    category_region<category::vaapi>::start(
        std::string_view{ _data.tool_id }, "dpy", dpy, "profile", profile,
        "entrypoint_list", entrypoint_list, "num_entrypoints", num_entrypoints);
}

// vaQuerySurfaceAttributes
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAConfigID config, VASurfaceAttrib* attrib_list,
                    unsigned int* num_attribs)
{
    (void) attrib_list;  // unused
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "config", config, "num_attribs", num_attribs);
}

// vaQuerySurfaceStatus
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VASurfaceID render_target, VASurfaceStatus* status)
{
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "render_target", render_target, "status",
                                            status);
}

// vaRenderPicture
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy,
                    VAContextID context, VABufferID* buffers, int num_buffers)
{
    (void) buffers;  // unused
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy", dpy,
                                            "context", context, "num_buffers",
                                            num_buffers);
}

// vaTerminate
// vaDisplayIsValid
void
vaapi_gotcha::audit(const gotcha_data& _data, audit::incoming, VADisplay dpy)
{
    category_region<category::vaapi>::start(std::string_view{ _data.tool_id }, "dpy",
                                            dpy);
}

void
vaapi_gotcha::audit(const gotcha_data& _data, audit::outgoing, VAStatus ret)
{
    category_region<category::vaapi>::stop(std::string_view{ _data.tool_id }, "return",
                                           ret);
}

}  // namespace component
}  // namespace rocprofsys

TIMEMORY_STORAGE_INITIALIZER(rocprofsys::component::vaapi_gotcha)
