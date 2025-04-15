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

#include "lib/common/mpl.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <bitset>
#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace context
{
// number of bits to reserve all op codes
constexpr size_t domain_ops_padding = 512;

template <typename Tp>
struct domain_info;

template <>
struct domain_info<rocprofiler_callback_tracing_kind_t>
{
    static constexpr size_t none    = ROCPROFILER_CALLBACK_TRACING_NONE;
    static constexpr size_t last    = ROCPROFILER_CALLBACK_TRACING_LAST;
    static constexpr auto   padding = domain_ops_padding;
};

template <>
struct domain_info<rocprofiler_buffer_tracing_kind_t>
{
    static constexpr size_t none    = ROCPROFILER_BUFFER_TRACING_NONE;
    static constexpr size_t last    = ROCPROFILER_BUFFER_TRACING_LAST;
    static constexpr auto   padding = domain_ops_padding;
};

/// how the tools specify the tracing domain and (optionally) which operations in the
/// domain they want to trace
template <typename DomainT>
struct domain_context
{
    using supported_domains_v = common::mpl::type_list<rocprofiler_callback_tracing_kind_t,
                                                       rocprofiler_buffer_tracing_kind_t>;
    static_assert(common::mpl::is_one_of<DomainT, supported_domains_v>::value,
                  "Unsupported domain type");
    static constexpr auto none = domain_info<DomainT>::none;
    static constexpr auto last = domain_info<DomainT>::last;

    static_assert(last > none, "last must be > none");

    static constexpr int64_t array_size = (last - none - 1);
    static constexpr auto    span_size  = domain_info<DomainT>::padding;

    using bitset_type = std::bitset<span_size>;
    using array_type  = std::array<bitset_type, array_size>;

    /// check if domain is enabled
    bool operator()(DomainT) const;

    /// check if op in a domain is enabled
    bool operator()(DomainT, uint32_t) const;

    uint64_t   domains = 0;
    array_type opcodes = {};
};

template <typename DomainT>
rocprofiler_status_t
add_domain(domain_context<DomainT>&, DomainT);

template <typename DomainT>
rocprofiler_status_t
add_domain_op(domain_context<DomainT>&, DomainT, uint32_t);
}  // namespace context
}  // namespace rocprofiler
