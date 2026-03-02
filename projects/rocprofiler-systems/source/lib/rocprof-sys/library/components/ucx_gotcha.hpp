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
#include "library/components/comm_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

namespace rocprofsys
{
namespace component
{
struct ucx_gotcha : tim::component::base<ucx_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 100;

    using gotcha_data = tim::component::gotcha_data;

    ROCPROFSYS_DEFAULT_OBJECT(ucx_gotcha)

    // string id for component
    static std::string label() { return "ucx_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    // Generic template audit function for UCX operations with void* parameters
    template <typename... Args>
    static void audit(const gotcha_data& _data, audit::incoming, Args...)
    {
        category_region<category::ucx>::start(std::string_view{ _data.tool_id });
    }

public:
    // Specific audit functions for tag operations (with uint64_t tags)
    // ucp_tag_send_nbx: (void* ep, const void* buffer, size_t count, uint64_t tag, const
    // void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, const void*, size_t,
                      uint64_t, const void*);
    // ucp_tag_recv_nbx: (void* worker, void* buffer, size_t count, uint64_t tag, uint64_t
    // tag_mask, const void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, void*, size_t, uint64_t,
                      uint64_t, const void*);

    // RMA operations
    // ucp_put_nbx: (void* ep, const void* buffer, size_t count, uint64_t remote_addr,
    // void* rkey, const void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, const void*, size_t,
                      uint64_t, void*, const void*);
    // ucp_get_nbx: (void* ep, void* buffer, size_t count, uint64_t remote_addr, void*
    // rkey, const void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, void*, size_t, uint64_t,
                      void*, const void*);

    // Active message send
    // ucp_am_send_nbx: (void* ep, unsigned id, const void* header, size_t header_length,
    // const void* buffer, size_t count, const void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, unsigned, const void*,
                      size_t, const void*, size_t, const void*);

    // Stream operations
    // ucp_stream_send_nbx: (void* ep, const void* buffer, size_t count, const void*
    // param)
    static void audit(const gotcha_data&, audit::incoming, void*, const void*, size_t,
                      const void*);
    // ucp_stream_recv_nbx: (void* ep, void* buffer, size_t count, size_t* length, const
    // void* param)
    static void audit(const gotcha_data&, audit::incoming, void*, void*, size_t, size_t*,
                      const void*);

    // Outgoing audit for return values
    static void audit(const gotcha_data&, audit::outgoing, void*);
    static void audit(const gotcha_data&, audit::outgoing, int);
};
}  // namespace component

using ucx_bundle_t =
    tim::component_bundle<category::ucx, component::ucx_gotcha, component::comm_data>;
using ucx_gotcha_t = tim::component::gotcha<component::ucx_gotcha::gotcha_capacity,
                                            ucx_bundle_t, category::ucx>;
}  // namespace rocprofsys
