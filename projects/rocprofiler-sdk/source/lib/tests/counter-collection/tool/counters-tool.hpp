// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/fwd.h>

#define ROCPROFILER_HANDLE_HASH_AND_EQ(TYPE)                                                       \
    template <>                                                                                    \
    struct std::hash<TYPE>                                                                         \
    {                                                                                              \
        size_t operator()(TYPE id) const noexcept { return std::hash<uint64_t>{}(id.handle); }     \
    };                                                                                             \
    inline bool operator==(TYPE lhs, TYPE rhs) noexcept { return lhs.handle == rhs.handle; }

ROCPROFILER_HANDLE_HASH_AND_EQ(rocprofiler_agent_id_t)
ROCPROFILER_HANDLE_HASH_AND_EQ(rocprofiler_counter_id_t)
ROCPROFILER_HANDLE_HASH_AND_EQ(rocprofiler_counter_config_id_t)
#undef ROCPROFILER_HANDLE_HASH_AND_EQ

namespace tool
{
using counters_t = std::vector<std::string>;

struct counter_results_t
{
    std::vector<double> values{};

    double sum() const;
    double min() const;
    double max() const;
    size_t size() const;
};

std::string
kernel_name(uint64_t kernel_id);

using dispatch_results_t   = std::unordered_map<std::string, counter_results_t>;
using kernel_dispatches_t  = std::unordered_map<rocprofiler_dispatch_id_t, dispatch_results_t>;
using agent_results_t      = std::unordered_map<rocprofiler_kernel_id_t, kernel_dispatches_t>;
using collection_results_t = std::unordered_map<rocprofiler_agent_id_t, agent_results_t>;

std::future<collection_results_t>
set_counters_config(const counters_t&);

void
start_counter_collection();

void
stop_counter_collection();

template <typename Fn>
std::future<collection_results_t>
collect(const counters_t& counters, Fn&& func)
{
    auto future = tool::set_counters_config(counters);
    tool::start_counter_collection();
    func();
    tool::stop_counter_collection();
    return future;
}

}  // namespace tool
