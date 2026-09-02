// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "kernel_filter_range.hpp"

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <string>

namespace rocprofiler
{
namespace tool
{
std::unordered_set<size_t>
get_kernel_filter_range(const std::string& kernel_filter)
{
    if(kernel_filter.empty()) return {};

    auto delim     = rocprofiler::sdk::parse::tokenize(kernel_filter, "[], ");
    auto range_set = std::unordered_set<size_t>{};
    for(const auto& itr : delim)
    {
        if(itr.find('-') != std::string::npos)
        {
            auto drange = rocprofiler::sdk::parse::tokenize(itr, "- ");

            ROCP_FATAL_IF(drange.size() != 2)
                << "bad range format for '" << itr << "'. Expected [A-B] where A and B are numbers";

            size_t start_range = std::stoul(drange.front());
            size_t end_range   = std::stoul(drange.back());
            for(auto i = start_range; i <= end_range; i++)
                range_set.emplace(i);
        }
        else
        {
            ROCP_FATAL_IF(itr.find_first_not_of("0123456789") != std::string::npos)
                << "expected integer for " << itr << ". Non-integer value detected";
            range_set.emplace(std::stoul(itr));
        }
    }
    return range_set;
}
}  // namespace tool
}  // namespace rocprofiler
