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

/**
 * @file mocked_types.hpp
 * @brief Mock types and components for gotcha component unit tests
 *
 * This file contains mock API types and mock gotcha components for testing.

 */

#pragma once

#include "common/tests/mock_trace_sink.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

// Mock VAAPI types for testing
using VADisplay       = void*;
using VAContextID     = uint32_t;
using VASurfaceID     = uint32_t;
using VABufferID      = uint32_t;
using VAConfigID      = uint32_t;
using VAImageID       = uint32_t;
using VAProfile       = int;
using VAEntrypoint    = int;
using VABufferType    = int;
using VAStatus        = int;
using VASurfaceAttrib = int;
using VAConfigAttrib  = int;

namespace rocprofsys
{
namespace component
{
namespace audit
{
struct incoming
{};
struct outgoing
{};
}  // namespace audit

/**
 * @brief Helper to convert pointer to string for trace arguments
 */
inline std::string
ptr_to_str(void* ptr)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%p", ptr);
    return std::string(buf);
}

/**
 * @brief Mock vaapi_gotcha component for unit testing
 *
 * This mock implementation allows testing the trace event emission logic
 * without requiring actual VAAPI runtime or gotcha library initialization.
 */
class vaapi_gotcha_mock
{
public:
    using trace_sink_ptr = std::shared_ptr<testing::mock_trace_sink>;

    vaapi_gotcha_mock()
    : m_sink(std::make_shared<testing::mock_trace_sink>())
    , m_is_started(false)
    {}

    explicit vaapi_gotcha_mock(trace_sink_ptr sink)
    : m_sink(std::move(sink))
    , m_is_started(false)
    {}

    // Lifecycle methods
    void start()
    {
        if(!m_is_started.load())
        {
            m_is_started.store(true);
        }
    }

    void stop() { m_is_started.store(false); }

    bool is_running() const { return m_is_started.load(); }

    // Get the sink for verification
    trace_sink_ptr get_sink() const { return m_sink; }

    // Mock gotcha_data structure
    struct mock_gotcha_data
    {
        const char* tool_id;
        void*       wrapper_pointer;
        void*       wrappee_pointer;
    };

    // Audit functions that emit trace events for VAAPI calls
    void audit_vaBeginPicture(const mock_gotcha_data& data, audit::incoming,
                              VADisplay dpy, VAContextID context,
                              VASurfaceID render_target)
    {
        if(!m_is_started.load() || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "context", context, "render_target", render_target);
    }

    void audit_vaBeginPicture(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started.load() || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaCreateBuffer(const mock_gotcha_data& data, audit::incoming,
                              VADisplay dpy, VAContextID context, VABufferType type,
                              unsigned int size, unsigned int num_elements,
                              void* buffer_data, VABufferID* buf_id)
    {
        if(!m_is_started.load() || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "context", context, "buffer_type", type, "size",
                                     size, "num_elements", num_elements);
    }

    void audit_vaCreateBuffer(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started.load() || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaCreateSurfaces(const mock_gotcha_data& data, audit::incoming,
                                VADisplay dpy, unsigned int format, unsigned int width,
                                unsigned int height, VASurfaceID* surfaces,
                                unsigned int num_surfaces, VASurfaceAttrib* attrib_list,
                                unsigned int num_attribs)
    {
        if(!m_is_started.load() || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "format", format, "width", width, "height", height,
                                     "num_surfaces", num_surfaces);
    }

    void audit_vaCreateSurfaces(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started.load() || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaSyncSurface(const mock_gotcha_data& data, audit::incoming, VADisplay dpy,
                             VASurfaceID surface)
    {
        if(!m_is_started.load() || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "surface", surface);
    }

    void audit_vaSyncSurface(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started.load() || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

private:
    trace_sink_ptr    m_sink;
    std::atomic<bool> m_is_started;
};

}  // namespace component
}  // namespace rocprofsys