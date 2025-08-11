// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/output/metadata.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <cereal/cereal.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace rocprofiler
{
namespace tool
{
struct execution_profile_data
{
    using extern_corr_id_request_t = rocprofiler_external_correlation_id_request_kind_t;
    using operation_set_t          = std::unordered_set<rocprofiler_tracing_operation_t>;

    std::unordered_map<extern_corr_id_request_t, uint64_t>        category_count    = {};
    std::unordered_map<extern_corr_id_request_t, operation_set_t> category_op_count = {};
    std::unordered_set<rocprofiler_thread_id_t>                   threads           = {};
    std::unordered_set<rocprofiler_context_id_t>                  contexts          = {};
};

struct execution_profile_category_data
{
    uint64_t count  = 0;  // total invocations of a given category
    uint64_t unique = 0;  // number of unique operations
};
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
template <typename ArchiveT>
void
save(ArchiveT& ar, ::rocprofiler::tool::execution_profile_category_data data)
{
    ar(cereal::make_nvp("count", data.count));
    ar(cereal::make_nvp("unique", data.unique));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::execution_profile_data& data)
{
    namespace tool = ::rocprofiler::tool;

    using category_count_map_t = std::map<std::string, tool::execution_profile_category_data>;

    auto _category_count = category_count_map_t{};
    for(auto itr : data.category_count)
    {
        const char* _name = nullptr;
        ROCPROFILER_CHECK(rocprofiler_query_external_correlation_id_request_kind_name(
            itr.first, &_name, nullptr));
        if(_name)
        {
            auto _unique_ops = data.category_op_count.at(itr.first).size();
            auto _kind_name  = std::string{_name};
            std::for_each(
                _kind_name.begin(), _kind_name.end(), [](auto& v) { v = ::std::tolower(v); });

            _category_count.emplace(_kind_name,
                                    tool::execution_profile_category_data{itr.second, _unique_ops});
        }
    }

    ar(cereal::make_nvp("threads", data.threads.size()));
    ar(cereal::make_nvp("contexts", data.contexts.size()));
    for(auto itr : _category_count)
        ar(cereal::make_nvp(itr.first.c_str(), itr.second));
}
}  // namespace cereal
