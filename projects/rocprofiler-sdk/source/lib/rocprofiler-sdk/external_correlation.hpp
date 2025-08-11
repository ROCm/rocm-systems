// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/defines.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"

#include <bitset>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct context;
}

namespace external_correlation
{
static constexpr bool enable_const_wlock_v = true;

using external_correlation_stack_t = std::vector<rocprofiler_user_data_t>;

// we enable the wlock(...) const for the mapped type so that we can use wlock on the mapped type
// within a rlock of the external correlation map
using external_correlation_map_t =
    std::unordered_map<rocprofiler_thread_id_t,
                       common::Synchronized<external_correlation_stack_t, enable_const_wlock_v>>;

struct external_correlation
{
    using request_cb_t   = rocprofiler_external_correlation_id_request_cb_t;
    using request_kind_t = rocprofiler_external_correlation_id_request_kind_t;

    static constexpr size_t request_kind_size = ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST - 1;

    rocprofiler_user_data_t  get(rocprofiler_thread_id_t thr_id,
                                 const context::context* ctx,
                                 request_kind_t          kind,
                                 uint32_t                op,
                                 uint64_t                internal_corr_id) const;
    rocprofiler_user_data_t& update(rocprofiler_user_data_t& data,
                                    rocprofiler_thread_id_t  thr_id,
                                    request_kind_t           kind) const;

    void                    push(rocprofiler_thread_id_t, rocprofiler_user_data_t);
    rocprofiler_user_data_t pop(rocprofiler_thread_id_t);

    rocprofiler_status_t configure_request(request_cb_t                       callback_v,
                                           void*                              callback_data_v,
                                           const std::vector<request_kind_t>& kinds_v);

    bool requires_request(request_kind_t kind) const;

private:
    rocprofiler_user_data_t get(rocprofiler_thread_id_t thr_id) const;

    std::optional<rocprofiler_user_data_t> invoke_callback(
        rocprofiler_thread_id_t                            thr_id,
        const context::context*                            ctx,
        rocprofiler_external_correlation_id_request_kind_t kind,
        uint32_t                                           op,
        uint64_t                                           internal_corr_id) const;

    request_cb_t                                     callback      = nullptr;
    void*                                            callback_data = nullptr;
    std::bitset<request_kind_size>                   request       = 0;
    common::Synchronized<external_correlation_map_t> data          = {};
};
}  // namespace external_correlation
}  // namespace rocprofiler
