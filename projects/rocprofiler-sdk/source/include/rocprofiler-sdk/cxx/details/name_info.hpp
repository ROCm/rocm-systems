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

#if !defined(ROCPROFILER_SDK_CXX_NAME_INFO_HPP_)
#    include <rocprofiler-sdk/cxx/name_info.hpp>
#endif

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <string_view>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace sdk
{
namespace utility
{
template <typename EnumT, typename ValueT>
typename name_info_impl<EnumT, ValueT>::item_array_t
name_info_impl<EnumT, ValueT>::items() const
{
    auto ret = item_array_t{};
    ret.reserve(operations.size());
    rocprofiler_tracing_operation_t _idx = 0;
    for(const auto& itr : operations)
        ret.emplace_back(_idx++, &itr);
    return ret;
}

template <typename EnumT, typename ValueT>
inline void
name_info<EnumT, ValueT>::emplace(EnumT idx, const char* name)
{
    impl.resize(idx + 1, value_type{});
    impl.at(idx).value = idx;
    impl.at(idx).name  = support_type{}(name);
}

template <typename EnumT, typename ValueT>
inline void
name_info<EnumT, ValueT>::emplace(EnumT                           idx,
                                  rocprofiler_tracing_operation_t opidx,
                                  const char*                     name)
{
    impl.resize(idx + 1, value_type{});
    impl.at(idx).operations.resize(opidx + 1, support_type::default_value());
    impl.at(idx).operations.at(opidx) = support_type{}(name);
}

template <typename EnumT, typename ValueT>
typename name_info<EnumT, ValueT>::return_type
name_info<EnumT, ValueT>::at(EnumT idx) const
{
    return impl.at(idx).name;
}

template <typename EnumT, typename ValueT>
typename name_info<EnumT, ValueT>::return_type
name_info<EnumT, ValueT>::at(EnumT idx, rocprofiler_tracing_operation_t opidx) const
{
    return impl.at(idx).operations.at(opidx);
}

template <typename EnumT, typename ValueT>
typename name_info<EnumT, ValueT>::item_array_t
name_info<EnumT, ValueT>::items() const
{
    auto ret = item_array_t{};
    ret.reserve(impl.size());
    for(const auto& itr : impl)
        ret.emplace_back(&itr);
    return ret;
}
}  // namespace utility

constexpr auto success_v = ROCPROFILER_STATUS_SUCCESS;

template <typename Tp>
inline callback_name_info_t<Tp>
get_callback_tracing_names()
{
    auto cb_name_info = callback_name_info_t<Tp>{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_callback_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t     operation,
                                               void*                               data_v) {
        auto* name_info_v = static_cast<callback_name_info_t<Tp>*>(data_v);

        const char* name   = nullptr;
        auto        status = rocprofiler_query_callback_tracing_kind_operation_name(
            kindv, operation, &name, nullptr);
        if(status == success_v && name) name_info_v->emplace(kindv, operation, name);
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<callback_name_info_t<Tp>*>(data);
        const char* name        = nullptr;
        auto        status = rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr);
        if(status == success_v && name) name_info_v->emplace(kind, name);

        rocprofiler_iterate_callback_tracing_kind_operations(kind, tracing_kind_operation_cb, data);
        return 0;
    };

    rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb, &cb_name_info);

    return cb_name_info;
}

template <typename Tp>
inline buffer_name_info_t<Tp>
get_buffer_tracing_names()
{
    auto cb_name_info = buffer_name_info_t<Tp>{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_buffer_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t   operation,
                                               void*                             data_v) {
        auto* name_info_v = static_cast<buffer_name_info_t<Tp>*>(data_v);

        const char* name = nullptr;
        auto        status =
            rocprofiler_query_buffer_tracing_kind_operation_name(kindv, operation, &name, nullptr);
        if(status == success_v && name) name_info_v->emplace(kindv, operation, name);
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info_t<Tp>*>(data);
        const char* name        = nullptr;
        auto        status      = rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr);
        if(status == success_v && name) name_info_v->emplace(kind, name);

        rocprofiler_iterate_buffer_tracing_kind_operations(kind, tracing_kind_operation_cb, data);
        return 0;
    };

    rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb, &cb_name_info);

    return cb_name_info;
}
}  // namespace sdk
}  // namespace rocprofiler
