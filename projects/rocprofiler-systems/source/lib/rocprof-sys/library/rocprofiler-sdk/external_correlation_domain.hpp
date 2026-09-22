// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"

namespace rocprofsys::domains
{

/// Configures the rocprofiler-sdk external-correlation-id-request service for
/// a single request kind, against a single context. Owned and configured by
/// domain_service on behalf of a buffered_domain/callback_domain that declares
/// this as its `correlation_dependency` -- see
/// external_correlation_domain_definition.
template <policies::domain_service::backend SdkBackend>
class external_correlation_domain
{
public:
    external_correlation_domain(
        external_correlation_domain_definition<SdkBackend> definition,
        SdkBackend::context_id_t                           context)
    : m_definition{ definition }
    , m_context{ context }
    {}

    void configure()
    {
        SdkBackend::configure_external_correlation_id_request_service(
            m_context, &m_definition.kind, 1, m_definition.on_request, nullptr);
    }

private:
    external_correlation_domain_definition<SdkBackend> m_definition;
    SdkBackend::context_id_t                           m_context;
};

}  // namespace rocprofsys::domains
