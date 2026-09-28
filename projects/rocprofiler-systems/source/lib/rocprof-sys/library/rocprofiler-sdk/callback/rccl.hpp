// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace rocprofsys::domains::callback
{

// maybe move this to backend
namespace detail
{
enum class event_type
{
    recv,
    send
};

template <policies::domain_service::backend SdkBackend>
struct rccl_event_info
{
    template <typename EventT>
    rccl_event_info(const EventT& event, event_type ev_type)
    : comm(event.comm)
    , type(ev_type)
    {
        auto data_type_size = SdkBackend::rccl_type_size(event.datatype);
        if constexpr(requires { event.count; })
        {
            size = data_type_size * event.count;
        }
        else if constexpr(requires { event.sendcount; })
        {
            size = data_type_size * event.sendcount;
        }
        else if constexpr(requires { event.recvcount; })
        {
            size = data_type_size * event.recvcount;
        }
    }

    rccl_event_info() = default;

    size_t                  size = 0;  ///< Transfer size in bytes
    event_type              type{ event_type::recv };
    SdkBackend::nccl_comm_t comm = nullptr;  ///< RCCL communicator handle
};

template <policies::domain_service::backend SdkBackend>
[[nodiscard]] inline rccl_event_info<SdkBackend>
extract_event_info(const typename SdkBackend::callback_tracing_record_t& record)
{
    if(record.payload == nullptr)
    {
        return {};
    }

    typename SdkBackend::rccl_api_id_t operation = record.operation;
    auto payload = *static_cast<SdkBackend::rccl_api_data*>(record.payload);

    // <rocprofiler-sdk/rccl/api_args.h> <- source of truth for nccl types
    switch(operation)
    {
        case SdkBackend::RCCL_API_ID_ncclAllGather:
            return rccl_event_info<SdkBackend>{ payload.args.ncclAllGather,
                                                event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclAllToAll:
            return rccl_event_info<SdkBackend>{ payload.args.ncclAllToAll,
                                                event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclAllReduce:
            return rccl_event_info<SdkBackend>{ payload.args.ncclAllReduce,
                                                event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclGather:
            return rccl_event_info<SdkBackend>{ payload.args.ncclGather,
                                                event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclRecv:
            return rccl_event_info<SdkBackend>{ payload.args.ncclRecv, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclReduce:
            return rccl_event_info<SdkBackend>{ payload.args.ncclReduce,
                                                event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclBroadcast:
            return rccl_event_info<SdkBackend>{ payload.args.ncclBroadcast,
                                                event_type::send };
        case SdkBackend::RCCL_API_ID_ncclReduceScatter:
            return rccl_event_info<SdkBackend>{ payload.args.ncclReduceScatter,
                                                event_type::send };
        case SdkBackend::RCCL_API_ID_ncclSend:
            return rccl_event_info<SdkBackend>{ payload.args.ncclSend, event_type::send };
        default: break;
    }

    // RCCL renamed ncclAllToAll to ncclAlltoAll (note the lowercase 't'). The deprecated
    // ncclAllToAll now forwards to ncclAlltoAll, so on toolchains new enough to define
    // this id the SDK reports the collective under it too, and it must be handled here
    // as well. Expressed via SdkBackend so this header stays SDK-agnostic: SdkBackend
    // (backend<Wrapper>) only defines RCCL_API_ID_ncclAlltoAll when the underlying
    // rocprofiler-sdk headers define ROCPROFILER_RCCL_API_ID_ncclAlltoAll.
    if constexpr(requires { SdkBackend::RCCL_API_ID_ncclAlltoAll; })
    {
        if(operation == SdkBackend::RCCL_API_ID_ncclAlltoAll)
        {
            return rccl_event_info<SdkBackend>{ payload.args.ncclAlltoAll,
                                                event_type::recv };
        }
    }

    return {};
}

}  // namespace detail

namespace detail2
{

template <policies::domain_service::backend SdkBackend>
struct device_id_resolver
{
    template <policies::domain_service::externals Externals>
    static void configure_comm_cu_device_function()
    {
        auto func                 = Externals::dlsym("ncclCommCuDevice");
        s_nccl_comm_cu_device_ptr = reinterpret_cast<nccl_comm_cu_device_fn>(func);
        if(s_nccl_comm_cu_device_ptr == nullptr)
        {
            const char* error = Externals::dlerror();
            LOG_DEBUG(
                "ncclCommCuDevice not found via dlsym ({}), using default device_id",
                error ? error : "unknown error");
        }
    }

    [[nodiscard]] static std::uint32_t resolve_device_id(
        SdkBackend::nccl_comm_t comm) noexcept
    {
        constexpr std::uint32_t k_default_device_id = 0;

        if(comm == nullptr)
        {
            return k_default_device_id;
        }

        if(s_nccl_comm_cu_device_ptr == nullptr)
        {
            return k_default_device_id;
        }

        int                                device_id = k_default_device_id;
        typename SdkBackend::nccl_result_t result =
            s_nccl_comm_cu_device_ptr(comm, &device_id);
        if(result != SdkBackend::NCCL_SUCCESS)
        {
            LOG_DEBUG("ncclCommCuDevice failed with error {}, using default device_id",
                      static_cast<int>(result));
            return k_default_device_id;
        }
        return static_cast<std::uint32_t>(device_id);
    }

private:
    using nccl_comm_cu_device_fn =
        SdkBackend::nccl_result_t (*)(typename SdkBackend::nccl_comm_t, int*);

    static inline nccl_comm_cu_device_fn s_nccl_comm_cu_device_ptr{ nullptr };
};

}  // namespace detail2

namespace detail3
{

template <policies::domain_service::externals Externals>
struct rccl_device_transferred_bytes_tracking
{
    static bool register_gpu(std::uint32_t rccl_device_idx)
    {
        auto thread_state_guard =
            Externals::state_thread::scoped(Externals::state_thread::Internal);
        const std::unique_lock<std::mutex> lock{ m_registered_gpus_mutex };
        if(m_registered_gpus.contains(rccl_device_idx))
        {
            return false;
        }
        m_registered_gpus.insert(rccl_device_idx);

        return true;
    }

    [[nodiscard]] static std::uint64_t add_bytes(std::uint32_t rccl_device_idx,
                                                 size_t        bytes)
    {
        auto thread_state_guard =
            Externals::state_thread::scoped(Externals::state_thread::Internal);
        const std::unique_lock<std::mutex> lock{ m_cumulative_mutex };
        auto& device_bytes = m_cumulative_bytes_per_device[rccl_device_idx];
        device_bytes += bytes;
        return device_bytes;
    }

private:
    static inline std::mutex                        m_registered_gpus_mutex;
    static inline std::unordered_set<std::uint32_t> m_registered_gpus;
    static inline std::mutex                        m_cumulative_mutex;
    static inline std::unordered_map<std::uint32_t, std::uint64_t>
        m_cumulative_bytes_per_device;
};

template <policies::domain_service::externals Externals>
void
register_gpu_pmc(std::uint32_t rccl_device_idx)
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

    auto register_rccl_info = [&](std::string_view direction_label,
                                  const char*      description) {
        const std::string label =
            fmt::format("{} GPU {}", direction_label, rccl_device_idx);
        Externals::get_metadata_registry().add_pmc_info(typename Externals::pmc_info_t{
            .type             = Externals::k_agent_type_gpu,
            .agent_type_index = rccl_device_idx,
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
}  // namespace detail3

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
    constexpr auto          k_empty_json   = "{}";
    constexpr std::uint64_t k_no_thread_id = 0;
    Externals::get_metadata_registry().add_string(Externals::comm_data_name);
    Externals::get_metadata_registry().add_track(typename Externals::track_t{
        std::string{ Externals::rccl_send_track_name }, k_no_thread_id, k_empty_json });
    Externals::get_metadata_registry().add_track(typename Externals::track_t{
        std::string{ Externals::rccl_recv_track_name }, k_no_thread_id, k_empty_json });

    detail2::device_id_resolver<SdkBackend>::template configure_comm_cu_device_function<
        Externals>();

    LOG_CRITICAL("RCCL CONFIGURE");
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

    const auto info = detail::extract_event_info<SdkBackend>(record);
    const auto device_id =
        detail2::device_id_resolver<SdkBackend>::resolve_device_id(info.comm);

    const auto is_present =
        detail3::rccl_device_transferred_bytes_tracking<Externals>::register_gpu(
            device_id);
    if(!is_present)
    {
        detail3::register_gpu_pmc<Externals>(device_id);
    }

    const auto cumulative =
        detail3::rccl_device_transferred_bytes_tracking<Externals>::add_bytes(device_id,
                                                                              info.size);

    const auto event_metadata = fmt::format(R"({{"transfer_bytes":{}}})", info.size);

    const auto label =
        info.type == detail::event_type::send ? "RCCL Comm Send" : "RCCL Comm Recv";
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
inline constexpr auto k_rccl = callback_domain_definition<SdkBackend>{
    .meta      = domain_descriptor{ .name  = "rccl",
                                    .id    = SdkBackend::CALLBACK_TRACING_RCCL_API,
                                    .mode  = collection_mode::callback,
                                    .group = std::nullopt },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_tracing_api_enter<SdkBackend, Externals, rccl_api_category>,
        on_rccl_exit<SdkBackend, Externals>>::callback,
    .on_configure = on_rccl_configure<SdkBackend, Externals>
};

}  // namespace rocprofsys::domains::callback
