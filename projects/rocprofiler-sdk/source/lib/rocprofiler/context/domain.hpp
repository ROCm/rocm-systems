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

#include <rocprofiler/rocprofiler.h>

#include "lib/common/mpl.hpp"

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
struct domain_info<rocprofiler_service_callback_tracing_kind_t>
{
    static constexpr size_t none    = ROCPROFILER_CALLBACK_TRACING_NONE;
    static constexpr size_t last    = ROCPROFILER_CALLBACK_TRACING_LAST;
    static constexpr auto   padding = domain_ops_padding;
};

template <>
struct domain_info<rocprofiler_service_buffer_tracing_kind_t>
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
    using supported_domains_v = common::mpl::type_list<rocprofiler_service_callback_tracing_kind_t,
                                                       rocprofiler_service_buffer_tracing_kind_t>;
    static_assert(common::mpl::is_one_of<DomainT, supported_domains_v>::value,
                  "Unsupported domain type");
    static constexpr auto opcode_padding_v = domain_info<DomainT>::padding;
    static constexpr auto max_opcodes_v    = opcode_padding_v * domain_info<DomainT>::last;

    /// check if domain is enabled
    bool operator()(DomainT) const;

    /// check if op in a domain is enabled
    bool operator()(DomainT, uint32_t) const;

    int64_t                    domains = 0;
    std::bitset<max_opcodes_v> opcodes = {};
};

template <typename DomainT>
rocprofiler_status_t
add_domain(domain_context<DomainT>&, DomainT);

template <typename DomainT>
rocprofiler_status_t
add_domain_op(domain_context<DomainT>&, DomainT, uint32_t);
}  // namespace context
}  // namespace rocprofiler
