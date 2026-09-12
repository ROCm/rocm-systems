// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service.hpp"

#include <string_view>
#include <vector>

namespace rocprofsys::domains
{

template <policies::rocprofiler_sdk::domain_service_backend SdkBackend>
class callback_domain
{
public:
    callback_domain(callback_domain_definition<SdkBackend>                definition,
                    SdkBackend::context_id_t                              context,
                    std::vector<typename SdkBackend::tracing_operation_t> operations)
    : m_definition{ definition }
    , m_context{ context }
    , m_operations{ std::move(operations) }
    {}

    void configure()
    {
        const auto kind =
            static_cast<SdkBackend::callback_tracing_kind_t>(m_definition.meta.id);

        SdkBackend::configure_callback_tracing_service(
            m_context, kind, m_operations.data(), m_operations.size(),
            m_definition.on_record, nullptr);
    }

    [[nodiscard]] std::string_view name() const noexcept
    {
        return m_definition.meta.name;
    }

private:
    callback_domain_definition<SdkBackend>                m_definition;
    SdkBackend::context_id_t                              m_context;
    std::vector<typename SdkBackend::tracing_operation_t> m_operations;
};

}  // namespace rocprofsys::domains
