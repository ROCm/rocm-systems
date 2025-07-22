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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/container/small_vector.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct context;
struct correlation_tracing_service;
}  // namespace context
namespace tracing
{
template <typename Tp, size_t N>
using small_vector_t                = common::container::small_vector<Tp, N>;
using correlation_service           = context::correlation_tracing_service;
using context_t                     = context::context;
using context_array_t               = common::container::small_vector<const context_t*>;
using external_correlation_id_map_t = std::unordered_map<const context_t*, rocprofiler_user_data_t>;

constexpr auto context_data_vec_size = 2;
constexpr auto empty_user_data       = rocprofiler_user_data_t{.value = 0};

struct callback_context_data
{
    const context_t*                      ctx       = nullptr;
    rocprofiler_callback_tracing_record_t record    = {};
    rocprofiler_user_data_t               user_data = {.value = 0};
};

struct buffered_context_data
{
    const context_t* ctx = nullptr;
};

using callback_context_data_vec_t = small_vector_t<callback_context_data, context_data_vec_size>;
using buffered_context_data_vec_t = small_vector_t<buffered_context_data, context_data_vec_size>;

struct tracing_data
{
    callback_context_data_vec_t   callback_contexts        = {};
    buffered_context_data_vec_t   buffered_contexts        = {};
    external_correlation_id_map_t external_correlation_ids = {};

    bool empty() const { return (callback_contexts.empty() && buffered_contexts.empty()); }
};
}  // namespace tracing
}  // namespace rocprofiler
