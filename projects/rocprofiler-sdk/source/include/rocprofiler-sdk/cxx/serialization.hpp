// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
//

#pragma once

#include <rocprofiler-sdk/cxx/name_info.hpp>

#include <cereal/cereal.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace cereal
{
template <typename ArchiveT, typename EnumT, typename ValueT>
void
save(ArchiveT& ar, const rocprofiler::sdk::utility::name_info<EnumT, ValueT>& data)
{
    ar.makeArray();
    for(const auto& itr : data)
        ar(cereal::make_nvp("entry", itr));
}

template <typename ArchiveT, typename EnumT, typename ValueT>
void
save(ArchiveT& ar, const rocprofiler::sdk::utility::name_info_impl<EnumT, ValueT>& data)
{
    auto _name = std::string{data.name};
    auto _ops  = std::vector<std::string>{};
    _ops.reserve(data.operations.size());

    ar(cereal::make_nvp("kind", _name));
    for(auto itr : data.operations)
        _ops.emplace_back(itr);
    ar(cereal::make_nvp("operations", _ops));
}
}  // namespace cereal
