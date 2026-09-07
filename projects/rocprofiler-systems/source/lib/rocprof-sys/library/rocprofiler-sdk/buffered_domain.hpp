// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"

#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::domains
{

template <typename SdkBackend>
class buffered_domain
{
public:
    buffered_domain(buffered_domain_definition<SdkBackend>                definition,
                    SdkBackend::context_id_t                              context,
                    std::vector<typename SdkBackend::tracing_operation_t> operations)
    : m_definition{ definition }
    , m_context{ context }
    , m_operations{ std::move(operations) }
    {}

    buffered_domain(const buffered_domain&)            = delete;
    buffered_domain& operator=(const buffered_domain&) = delete;
    buffered_domain(buffered_domain&& other) noexcept

    : m_definition{ other.m_definition }
    , m_context{ other.m_context }
    , m_operations{ std::move(other.m_operations) }
    , m_buffer{ std::exchange(other.m_buffer, {}) }
    {}

    buffered_domain& operator=(buffered_domain&& other) noexcept
    {
        if(this != &other)
        {
            destroy();
            m_definition = other.m_definition;
            m_context    = other.m_context;
            m_operations = std::move(other.m_operations);
            m_buffer     = std::exchange(other.m_buffer, {});
        }
        return *this;
    }

    ~buffered_domain() { destroy(); }

    void configure()
    {
        const auto& properties = m_definition.buffer;

        SdkBackend::create_buffer(
            m_context, properties.buffer_size, properties.buffer_watermark,
            k_buffer_policy, m_definition.on_records, nullptr,
            &m_buffer);  // create_buffer -> Add throw upon failure in SdkBackend

        const auto kind =
            static_cast<SdkBackend::buffer_tracing_kind_t>(m_definition.meta.id);

        SdkBackend::configure_buffer_tracing_service(
            m_context, kind, m_operations.data(), m_operations.size(),
            m_buffer);  // configure_buffer_tracing_service -> Add throw upon failure in
                        // SdkBackend

        typename SdkBackend::callback_thread_t thread{};
        SdkBackend::create_callback_thread(
            &thread);  // create_callback_thread -> Add throw upon failure in SdkBackend

        SdkBackend::assign_callback_thread(
            m_buffer,
            thread);  // assign_callback_thread -> Add throw upon failure in SdkBackend
    }

    void flush() const;

    [[nodiscard]] std::string_view name() const noexcept
    {
        return m_definition.meta.name;
    }
    [[nodiscard]] SdkBackend::buffer_id_t buffer_id() const noexcept { return m_buffer; }

private:
    void destroy() noexcept
    {
        if(!is_valid(m_buffer))
        {
            return;
        }

        static_cast<void>(SdkBackend::destroy_buffer(m_buffer));
        m_buffer = {};
    }
    buffered_domain_definition<SdkBackend>                m_definition;
    SdkBackend::context_id_t                              m_context;
    std::vector<typename SdkBackend::tracing_operation_t> m_operations;
    SdkBackend::buffer_id_t                               m_buffer{};

    constexpr static auto k_buffer_policy = SdkBackend::BUFFER_POLICY_LOSSLESS;
};

}  // namespace rocprofsys::domains
