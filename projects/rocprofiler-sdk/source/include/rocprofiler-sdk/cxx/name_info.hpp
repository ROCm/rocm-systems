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
//

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/mpl.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace sdk
{
namespace utility
{
template <typename EnumT, typename ValueT = std::string_view>
struct name_info_impl
{
    using support_type = mpl::string_support<ValueT>;
    using enum_type    = EnumT;
    using value_type   = ValueT;
    using return_type  = typename support_type::return_type;
    using item_type    = std::pair<rocprofiler_tracing_operation_t, const value_type*>;
    using item_array_t = std::vector<item_type>;

    static_assert(support_type::value,
                  "value_type must be supported by rocprofiler::sdk::mpl::string_support");

    return_type operator()() const { return name; }
    return_type operator()(size_t idx) const { return operations.at(idx); }
    return_type operator[](size_t idx) const { return operations.at(idx); }

    item_array_t items() const;

    EnumT                   value      = static_cast<EnumT>(0);
    value_type              name       = {};
    std::vector<value_type> operations = {};
};

template <typename EnumT, typename ValueT = std::string_view>
struct name_info
{
    using value_type   = name_info_impl<EnumT, ValueT>;
    using enum_type    = EnumT;
    using support_type = typename value_type::support_type;
    using return_type  = typename value_type::return_type;
    using item_type    = const value_type*;
    using item_array_t = std::vector<item_type>;

    void emplace(EnumT idx, const char* name);
    void emplace(EnumT idx, rocprofiler_tracing_operation_t opidx, const char* name);

    return_type at(EnumT idx) const;
    return_type at(EnumT idx, rocprofiler_tracing_operation_t opidx) const;

    item_array_t items() const;

    decltype(auto) size() const { return impl.size(); }
    decltype(auto) begin() { return impl.begin(); }
    decltype(auto) begin() const { return impl.begin(); }
    decltype(auto) end() { return impl.end(); }
    decltype(auto) end() const { return impl.end(); }

    value_type&       operator[](size_t idx) { return impl.at(idx); }
    const value_type& operator[](size_t idx) const { return impl.at(idx); }

private:
    std::vector<value_type> impl = {};
};
}  // namespace utility

template <typename Tp = std::string_view>
using callback_name_info_t = utility::name_info<rocprofiler_callback_tracing_kind_t, Tp>;

template <typename Tp = std::string_view>
using buffer_name_info_t = utility::name_info<rocprofiler_buffer_tracing_kind_t, Tp>;

using callback_name_info = callback_name_info_t<std::string_view>;
using buffer_name_info   = buffer_name_info_t<std::string_view>;

template <typename Tp = std::string_view>
callback_name_info_t<Tp>
get_callback_tracing_names();

template <typename Tp = std::string_view>
buffer_name_info_t<Tp>
get_buffer_tracing_names();
}  // namespace sdk
}  // namespace rocprofiler

#define ROCPROFILER_SDK_CXX_NAME_INFO_HPP_ 1
#include <rocprofiler-sdk/cxx/details/name_info.hpp>
