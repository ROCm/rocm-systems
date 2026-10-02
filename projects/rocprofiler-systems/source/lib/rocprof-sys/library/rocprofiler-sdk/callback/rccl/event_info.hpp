// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"

namespace rocprofsys::domains::callback::rccl
{
enum class event_type
{
    recv,
    send
};

template <policies::domain_service::backend SdkBackend>
struct event_info
{
    template <typename EventT>
    event_info(const EventT& event, event_type ev_type)
    : type(ev_type)
    , comm(event.comm)
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

    event_info() = default;

    size_t                  size = 0;  ///< Transfer size in bytes
    event_type              type{ event_type::recv };
    SdkBackend::nccl_comm_t comm = nullptr;  ///< RCCL communicator handle
};

template <policies::domain_service::backend SdkBackend>
[[nodiscard]] inline event_info<SdkBackend>
extract_event_info(const typename SdkBackend::callback_tracing_record_t& record)
{
    if(record.payload == nullptr)
    {
        return {};
    }

    auto operation = static_cast<SdkBackend::rccl_api_id_t>(record.operation);
    auto payload   = *static_cast<SdkBackend::rccl_api_data*>(record.payload);

    // <rocprofiler-sdk/rccl/api_args.h> <- source of truth for nccl types
    switch(operation)
    {
        case SdkBackend::RCCL_API_ID_ncclAllGather:
            return event_info<SdkBackend>{ payload.args.ncclAllGather, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclAllToAll:
            return event_info<SdkBackend>{ payload.args.ncclAllToAll, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclAllReduce:
            return event_info<SdkBackend>{ payload.args.ncclAllReduce, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclGather:
            return event_info<SdkBackend>{ payload.args.ncclGather, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclRecv:
            return event_info<SdkBackend>{ payload.args.ncclRecv, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclReduce:
            return event_info<SdkBackend>{ payload.args.ncclReduce, event_type::recv };
        case SdkBackend::RCCL_API_ID_ncclBroadcast:
            return event_info<SdkBackend>{ payload.args.ncclBroadcast, event_type::send };
        case SdkBackend::RCCL_API_ID_ncclReduceScatter:
            return event_info<SdkBackend>{ payload.args.ncclReduceScatter,
                                           event_type::send };
        case SdkBackend::RCCL_API_ID_ncclSend:
            return event_info<SdkBackend>{ payload.args.ncclSend, event_type::send };
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
            return event_info<SdkBackend>{ payload.args.ncclAlltoAll, event_type::recv };
        }
    }

    return {};
}

}  // namespace rocprofsys::domains::callback::rccl
