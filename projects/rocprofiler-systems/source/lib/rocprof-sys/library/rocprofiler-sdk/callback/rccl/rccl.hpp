// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "library/rocprofiler-sdk/callback/rccl/device_resolver.hpp"
#include "library/rocprofiler-sdk/callback/rccl/event_info.hpp"
#include "library/rocprofiler-sdk/callback/rccl/transferred_bytes_tracker.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace rocprofsys::domains::callback::rccl
{

template <policies::domain_service::externals Externals>
void
register_gpu_pmc(std::uint32_t device_id)
{
    constexpr size_t k_event_code  = 0;
    constexpr size_t k_instance_id = 0;
    constexpr auto*  k_long_description =
        "Per-GPU RCCL communication data with transfer_bytes in extdata JSON";
    constexpr auto* k_component   = "";
    constexpr auto* k_block       = "";
    constexpr auto* k_expression  = "";
    constexpr auto* k_msg         = "bytes";
    constexpr auto* k_target_arch = "GPU";

    auto& metadata_registry = Externals::get_metadata_registry();

    auto register_rccl_info = [&](std::string_view direction_label,
                                  const char*      description) {
        const std::string label = fmt::format("{} GPU {}", direction_label, device_id);
        metadata_registry.add_pmc_info(typename Externals::pmc_info_t{
            .type             = Externals::k_agent_type_gpu,
            .agent_type_index = device_id,
            .target_arch      = k_target_arch,
            .event_code       = k_event_code,
            .instance_id      = k_instance_id,
            .name             = label,
            .symbol           = description,
            .description      = std::string{ Externals::comm_data_description },
            .long_description = k_long_description,
            .component        = k_component,
            .units            = k_msg,
            .value_type       = std::string{ Externals::k_pmc_value_type_absolute },
            .block            = k_block,
            .expression       = k_expression,
            .is_constant      = 0,
            .is_derived       = 0,
            .extdata          = "{}" });
    };

    register_rccl_info(Externals::rccl_send_label,
                       "Tracks RCCL communication data sizes (send)");
    register_rccl_info(Externals::rccl_recv_label,
                       "Tracks RCCL communication data sizes (recv)");
}

template <policies::domain_service::externals Externals>
struct rccl_api_category
{
    using type = Externals::rocm_rccl_api_category;

    static constexpr std::string_view k_name = Externals::rocm_rccl_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_rccl_configure()
{
    constexpr auto k_empty_json   = "{}";
    constexpr auto k_no_thread_id = std::nullopt;

    auto& metadata_registry = Externals::get_metadata_registry();
    metadata_registry.add_string(Externals::comm_data_name);
    metadata_registry.add_track(typename Externals::track_t{
        .track_name = std::string{ Externals::rccl_send_track_name },
        .thread_id  = k_no_thread_id,
        .extdata    = k_empty_json });
    metadata_registry.add_track(typename Externals::track_t{
        .track_name = std::string{ Externals::rccl_recv_track_name },
        .thread_id  = k_no_thread_id,
        .extdata    = k_empty_json });

    rccl::device_resolver<SdkBackend>::template configure_comm_cu_device_function<
        Externals>();
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
void
on_rccl_exit(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* user_data, void* callback_data,
             typename SdkBackend::timestamp_t timestamp)
{
    on_tracing_api_exit<SdkBackend, Externals, rccl_api_category>(
        record, user_data, callback_data, timestamp);

    const auto info = rccl::extract_event_info<SdkBackend>(record);

    if(info.size == 0)
    {
        // We don't want to process events with 0 size
        // Cumulative value will not change
        return;
    }

    const auto device_id =
        rccl::device_resolver<SdkBackend>::resolve_device_id(info.comm);

    const auto [cumulative, is_first_registration] =
        rccl::transferred_bytes_tracker<Externals>::add_bytes(device_id, info.size);

    if(is_first_registration)
    {
        rccl::register_gpu_pmc<Externals>(device_id);
    }

    const auto event_metadata = fmt::format(R"({{"transfer_bytes":{}}})", info.size);

    const auto label =
        info.type == rccl::event_type::send ? "RCCL Comm Send" : "RCCL Comm Recv";
    const auto pmc_label = fmt::format("{} GPU {}", label, device_id);

    constexpr size_t           k_stack_id        = 0;
    constexpr size_t           k_parent_stack_id = 0;
    constexpr size_t           k_correlation_id  = 0;
    constexpr std::string_view k_call_stack      = "{}";
    constexpr std::string_view k_line_info       = "{}";

    Externals::get_buffer_storage().store(typename Externals::pmc_event_with_sample{
        Externals::comm_data_enum_value, label, timestamp, event_metadata, k_stack_id,
        k_parent_stack_id, k_correlation_id, k_call_stack, k_line_info, device_id,
        static_cast<std::uint8_t>(Externals::k_agent_type_gpu), pmc_label,
        static_cast<double>(cumulative), std::nullopt });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_rccl_api = callback_domain_definition<SdkBackend>{
    .meta      = domain_descriptor{ .name  = "rccl_api",
                                    .id    = SdkBackend::CALLBACK_TRACING_RCCL_API,
                                    .mode  = collection_mode::callback,
                                    .group = std::nullopt },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_tracing_api_enter<SdkBackend, Externals, rccl_api_category>,
        on_rccl_exit<SdkBackend, Externals>>::callback,
    .on_configure = on_rccl_configure<SdkBackend, Externals>
};

}  // namespace rocprofsys::domains::callback::rccl
