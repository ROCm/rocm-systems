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

#pragma once

#include "lib/common/mpl.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <string_view>

namespace rocprofiler
{
namespace tool
{
namespace csv
{
template <typename TupleT, size_t... Idx>
std::ostream&
write_csv_entry(std::ostream& ofs, TupleT&& _data, std::index_sequence<Idx...>)
{
    auto _write = [&ofs](size_t idx, auto&& _val) {
        if(idx > 0) ofs << ",";
        if constexpr(rocprofiler::common::mpl::is_string_type<decltype(_val)>::value) ofs << "\"";
        ofs << _val;
        if constexpr(rocprofiler::common::mpl::is_string_type<decltype(_val)>::value) ofs << "\"";
    };

    (_write(Idx, std::get<Idx>(_data)), ...);
    return (ofs << '\n');
}

template <size_t NumCols>
struct csv_encoder
{
    static constexpr auto columns = NumCols;

    template <typename... Args,
              typename Tp                                       = void,
              std::enable_if_t<sizeof...(Args) == columns, int> = 0>
    static auto write_row(std::ostream& ofs, Args&&... args)
    {
        write_csv_entry(
            ofs, std::make_tuple(std::forward<Args>(args)...), std::make_index_sequence<columns>{});
        return csv_encoder<columns>{};
    }

    template <typename Tp, size_t N>
    static auto write_row(std::ostream& ofs, const std::array<Tp, N>& arr)
    {
        static_assert(N == columns, "Error! too many/few args passed");
        write_csv_entry(ofs, arr, std::make_index_sequence<columns>{});
        return csv_encoder<columns>{};
    }
};

using hsa_csv_encoder                = csv_encoder<7>;
using kernel_trace_csv_encoder       = csv_encoder<16>;
using counter_collection_csv_encoder = csv_encoder<14>;
using memory_copy_csv_encoder        = csv_encoder<7>;
}  // namespace csv
}  // namespace tool
}  // namespace rocprofiler
