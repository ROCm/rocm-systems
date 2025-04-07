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

#include "lib/rocprofiler-sdk/code_object/hsa/kernel_symbol.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
namespace hsa
{
using context_t               = context::context;
using user_data_t             = rocprofiler_user_data_t;
using context_user_data_map_t = std::unordered_map<const context_t*, user_data_t>;
using context_array_t         = context::context_array_t;
using context_user_data_map_t = std::unordered_map<const context_t*, user_data_t>;

struct code_object
{
    using code_object_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
    using symbol_array_t     = std::vector<std::unique_ptr<kernel_symbol>>;

    code_object()  = default;
    ~code_object() = default;

    code_object(const code_object&) = delete;
    code_object(code_object&&) noexcept;

    code_object& operator=(const code_object&) = delete;
    code_object& operator                      =(code_object&&) noexcept;

    bool                     beg_notified    = false;
    bool                     end_notified    = false;
    const std::string*       uri             = nullptr;
    hsa_executable_t         hsa_executable  = {};
    hsa_loaded_code_object_t hsa_code_object = {};
    code_object_data_t       rocp_data       = common::init_public_api_struct(code_object_data_t{});
    symbol_array_t           symbols         = {};
    context_array_t          contexts        = {};
    context_user_data_map_t  user_data       = {};
};

struct code_object_unload
{
    code_object*                object  = nullptr;
    std::vector<kernel_symbol*> symbols = {};
};

bool
operator==(const code_object& lhs, const code_object& rhs);
}  // namespace hsa
}  // namespace code_object
}  // namespace rocprofiler
