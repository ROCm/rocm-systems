// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include "lib/rocprofiler/context/domain.hpp"
#include <rocprofiler/rocprofiler.h>

namespace rocprofiler
{
namespace context
{
template <typename DomainT>
bool
domain_context<DomainT>::operator()(DomainT _domain) const
{
    return ((1 << _domain) & domains) == (1 << _domain);
}

template <typename DomainT>
bool
domain_context<DomainT>::operator()(DomainT _domain, uint32_t _op) const
{
    auto _offset = (_domain * opcode_padding_v);
    return (*this)(_domain) && (opcodes.none() || opcodes.test(_offset + _op));
}

template <typename DomainT>
rocprofiler_status_t
add_domain(domain_context<DomainT>& _cfg, DomainT _domain)
{
    if(_domain <= domain_info<DomainT>::none || _domain >= domain_info<DomainT>::last)
        return ROCPROFILER_STATUS_ERROR_DOMAIN_NOT_FOUND;

    _cfg.domains |= (1 << _domain);
    return ROCPROFILER_STATUS_SUCCESS;
}

template <typename DomainT>
rocprofiler_status_t
add_domain_op(domain_context<DomainT>& _cfg, DomainT _domain, uint32_t _op)
{
    if(_domain <= domain_info<DomainT>::none || _domain >= domain_info<DomainT>::last)
        return ROCPROFILER_STATUS_ERROR_DOMAIN_NOT_FOUND;

    if(_op >= domain_info<DomainT>::padding) return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;

    auto _offset = (_domain * domain_info<DomainT>::padding);
    if(_offset >= _cfg.opcodes.size()) return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;

    _cfg.opcodes.set(_offset + _op, true);
    return ROCPROFILER_STATUS_SUCCESS;
}

// instantiate the templates
template struct domain_context<rocprofiler_service_callback_tracing_kind_t>;

template rocprofiler_status_t
add_domain<rocprofiler_service_callback_tracing_kind_t>(
    domain_context<rocprofiler_service_callback_tracing_kind_t>&,
    rocprofiler_service_callback_tracing_kind_t);

template rocprofiler_status_t
add_domain<rocprofiler_service_buffer_tracing_kind_t>(
    domain_context<rocprofiler_service_buffer_tracing_kind_t>&,
    rocprofiler_service_buffer_tracing_kind_t);

template rocprofiler_status_t
add_domain_op<rocprofiler_service_callback_tracing_kind_t>(
    domain_context<rocprofiler_service_callback_tracing_kind_t>&,
    rocprofiler_service_callback_tracing_kind_t,
    uint32_t);

template struct domain_context<rocprofiler_service_buffer_tracing_kind_t>;

template rocprofiler_status_t
add_domain_op<rocprofiler_service_buffer_tracing_kind_t>(
    domain_context<rocprofiler_service_buffer_tracing_kind_t>&,
    rocprofiler_service_buffer_tracing_kind_t,
    uint32_t);
}  // namespace context
}  // namespace rocprofiler
