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

#include "lib/rocprofiler-sdk/context/domain.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <rocprofiler-sdk/hip/compiler_api_id.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/hsa/amd_ext_api_id.h>
#include <rocprofiler-sdk/hsa/core_api_id.h>
#include <rocprofiler-sdk/hsa/finalize_ext_api_id.h>
#include <rocprofiler-sdk/hsa/image_ext_api_id.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rccl/api_id.h>
#include <rocprofiler-sdk/rocdecode/api_id.h>
#include <rocprofiler-sdk/rocjpeg/api_id.h>

#include <algorithm>
#include <limits>

namespace rocprofiler
{
namespace context
{
// Validate that domain_ops_padding is large enough for all API operation IDs.
// If this static_assert fails, increase domain_ops_padding in domain.hpp.
static_assert(std::max({static_cast<size_t>(ROCPROFILER_HIP_RUNTIME_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_HIP_COMPILER_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_HSA_CORE_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_MARKER_CORE_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_MARKER_CONTROL_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_MARKER_NAME_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_RCCL_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_ROCDECODE_API_ID_LAST),
                        static_cast<size_t>(ROCPROFILER_ROCJPEG_API_ID_LAST)}) < domain_ops_padding,
              "domain_ops_padding is too small for API operation IDs - increase it in domain.hpp");

template <typename DomainT>
bool
domain_context<DomainT>::operator()(DomainT _domain) const
{
    constexpr uint64_t one = 1;

    if(_domain <= domain_info<DomainT>::none) return false;

    auto _didx = (_domain - 1);
    return ((one << _didx) & domains) == (one << _didx);
}

template <typename DomainT>
bool
domain_context<DomainT>::operator()(DomainT _domain, uint32_t _op) const
{
    if(_domain <= domain_info<DomainT>::none) return false;

    auto _didx = (_domain - 1);

    if(_didx >= array_size) return false;

    return (*this)(_domain) && (opcodes.at(_didx).none() || opcodes.at(_didx).test(_op));
}

template <typename DomainT>
rocprofiler_status_t
add_domain(domain_context<DomainT>& _cfg, DomainT _domain)
{
    static_assert((1UL << domain_info<DomainT>::last) < std::numeric_limits<uint64_t>::max(),
                  "uint64_t cannot handle all the domains");

    if(_domain <= domain_info<DomainT>::none) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

    auto _didx = (_domain - 1);

    if(_didx >= _cfg.array_size) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

    _cfg.domains |= (1UL << _didx);
    return ROCPROFILER_STATUS_SUCCESS;
}

template <typename DomainT>
rocprofiler_status_t
add_domain_op(domain_context<DomainT>& _cfg, DomainT _domain, uint32_t _op)
{
    if(_domain <= domain_info<DomainT>::none || (_domain - 1) >= _cfg.array_size)
        return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

    if(_op >= domain_info<DomainT>::padding) return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;

    auto _didx = (_domain - 1);
    _cfg.opcodes.at(_didx).set(_op, true);
    return ROCPROFILER_STATUS_SUCCESS;
}

// instantiate the templates
template struct domain_context<rocprofiler_callback_tracing_kind_t>;

template rocprofiler_status_t
add_domain<rocprofiler_callback_tracing_kind_t>(
    domain_context<rocprofiler_callback_tracing_kind_t>&,
    rocprofiler_callback_tracing_kind_t);

template rocprofiler_status_t
add_domain<rocprofiler_buffer_tracing_kind_t>(domain_context<rocprofiler_buffer_tracing_kind_t>&,
                                              rocprofiler_buffer_tracing_kind_t);

template rocprofiler_status_t
add_domain_op<rocprofiler_callback_tracing_kind_t>(
    domain_context<rocprofiler_callback_tracing_kind_t>&,
    rocprofiler_callback_tracing_kind_t,
    uint32_t);

template struct domain_context<rocprofiler_buffer_tracing_kind_t>;

template rocprofiler_status_t
add_domain_op<rocprofiler_buffer_tracing_kind_t>(domain_context<rocprofiler_buffer_tracing_kind_t>&,
                                                 rocprofiler_buffer_tracing_kind_t,
                                                 uint32_t);
}  // namespace context
}  // namespace rocprofiler
