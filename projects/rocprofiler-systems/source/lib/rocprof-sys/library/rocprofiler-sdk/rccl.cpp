// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/rocprofiler-sdk/rccl.hpp"
#include "library/rocprofiler-sdk/rccl_internal.hpp"

#include "core/categories.hpp"
#include "core/config.hpp"
#include "core/perfetto.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"

#include "logger/debug.hpp"

#include <rocprofiler-sdk/rccl/api_args.h>

#include <dlfcn.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

struct rccl_recv
{
    static constexpr auto value = "comm_data";
    static constexpr auto label = "RCCL Comm Recv";
};

struct rccl_send
{
    static constexpr auto value = "comm_data";
    static constexpr auto label = "RCCL Comm Send";
};

namespace
{

[[nodiscard]] size_t
rccl_type_size_or_abort(ncclDataType_t datatype) noexcept;

void
rccl_metadata_initialize_categories()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    trace_cache::get_metadata_registry().add_string(
        trait::name<category::comm_data>::value);

    _is_initialized = true;
}

}  // anonymous namespace

/**
 * @brief Production PMC registrar implementation
 *
 * This struct registers RCCL communication counters with the trace cache
 * metadata registry. Uses duck typing for template-based dependency injection.
 */
struct production_pmc_registrar
{
    void register_gpu_pmc(uint32_t rccl_device_idx)
    {
        constexpr size_t EVENT_CODE  = 0;
        constexpr size_t INSTANCE_ID = 0;
        constexpr auto*  LONG_DESCRIPTION =
            "Per-GPU RCCL communication data with transfer_bytes in extdata JSON";
        constexpr auto* COMPONENT   = "";
        constexpr auto* BLOCK       = "";
        constexpr auto* EXPRESSION  = "";
        constexpr auto* MSG         = "bytes";
        constexpr auto* TARGET_ARCH = "GPU";

        auto register_rccl_info = [&](const char* direction_label,
                                      const char* description) {
            std::string label =
                fmt::format("{} GPU {}", direction_label, rccl_device_idx);
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, rccl_device_idx, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  label.c_str(), description,
                  trait::name<category::comm_data>::description, LONG_DESCRIPTION,
                  COMPONENT, MSG, trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
        };

        register_rccl_info(rccl_send::label,
                           "Tracks RCCL communication data sizes (send)");
        register_rccl_info(rccl_recv::label,
                           "Tracks RCCL communication data sizes (recv)");
    }
};

using rccl_gpu_tracking_state = rccl_gpu_tracking_state_t<production_pmc_registrar>;

namespace
{

rccl_gpu_tracking_state&
rccl_get_gpu_tracking_state();

[[nodiscard]] rccl_event_info
rccl_get_event_info_impl(
    uint32_t                                            operation,
    const rocprofiler_callback_tracing_rccl_api_data_t& payload) noexcept
{
    rccl_event_info info{};

    const auto set_event = [](rccl_event_info& out, bool send, size_t count,
                              ncclDataType_t _dt, ncclComm_t _comm) {
        out.is_send = send;
        out.size    = count * rccl_type_size_or_abort(_dt);
        out.comm    = _comm;
    };

    switch(operation)
    {
        case ROCPROFILER_RCCL_API_ID_ncclAllGather:
            set_event(info, false, payload.args.ncclAllGather.sendcount,
                      payload.args.ncclAllGather.datatype,
                      payload.args.ncclAllGather.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclAllToAll:
            set_event(info, false, payload.args.ncclAllToAll.count,
                      payload.args.ncclAllToAll.datatype, payload.args.ncclAllToAll.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclAllReduce:
            set_event(info, false, payload.args.ncclAllReduce.count,
                      payload.args.ncclAllReduce.datatype,
                      payload.args.ncclAllReduce.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclGather:
            set_event(info, false, payload.args.ncclGather.sendcount,
                      payload.args.ncclGather.datatype, payload.args.ncclGather.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclRecv:
            set_event(info, false, payload.args.ncclRecv.count,
                      payload.args.ncclRecv.datatype, payload.args.ncclRecv.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclReduce:
            set_event(info, false, payload.args.ncclReduce.count,
                      payload.args.ncclReduce.datatype, payload.args.ncclReduce.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclBroadcast:
            set_event(info, true, payload.args.ncclBroadcast.count,
                      payload.args.ncclBroadcast.datatype,
                      payload.args.ncclBroadcast.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclReduceScatter:
            set_event(info, true, payload.args.ncclReduceScatter.recvcount,
                      payload.args.ncclReduceScatter.datatype,
                      payload.args.ncclReduceScatter.comm);
            break;
        case ROCPROFILER_RCCL_API_ID_ncclSend:
            set_event(info, true, payload.args.ncclSend.count,
                      payload.args.ncclSend.datatype, payload.args.ncclSend.comm);
            break;
        default: break;
    }

    return info;
}

template <typename Track>
void
rccl_metadata_initialize_track()
{
    trace_cache::get_metadata_registry().add_track({ Track::label, std::nullopt, "{}" });
}

template <typename Tp, typename... Args>
void
write_perfetto_counter_track(uint64_t _val, uint64_t _begin_ts, uint64_t _end_ts)
{
    using counter_track = rocprofsys::perfetto_counter_track<Tp>;

    if(rocprofsys::get_use_perfetto() &&
       rocprofsys::get_state() == rocprofsys::State::Active)
    {
        const size_t _idx = 0;

        if(!counter_track::exists(_idx))
        {
            std::string _label =
                (_idx > 0) ? fmt::format("{} [{}]", Tp::label, _idx) : Tp::label;
            counter_track::emplace(_idx, _label, "bytes");
        }

        TRACE_COUNTER(Tp::value, counter_track::at(_idx, 0), _begin_ts, _val);
        TRACE_COUNTER(Tp::value, counter_track::at(_idx, 0), _end_ts, 0);
    }
}

template <typename Track>
void
cache_rccl_comm_data_events(uint32_t rccl_device_idx, size_t bytes, uint64_t timestamp_ns)
{
    auto&  tracking_state = rccl_get_gpu_tracking_state();
    size_t transfer_bytes = bytes;

    tracking_state.register_gpu(rccl_device_idx);
    uint64_t cumulative = tracking_state.add_bytes(rccl_device_idx, bytes);

    const auto event_metadata = fmt::format(R"({{"transfer_bytes":{}}})", transfer_bytes);

    const auto pmc_label = fmt::format("{} GPU {}", Track::label, rccl_device_idx);

    const size_t stack_id        = 0;
    const size_t parent_stack_id = 0;
    const size_t correlation_id  = 0;
    const auto*  call_stack      = "{}";
    const auto*  line_info       = "{}";

    trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
        static_cast<size_t>(category_enum_id<category::comm_data>::value), Track::label,
        timestamp_ns, event_metadata.c_str(), stack_id, parent_stack_id, correlation_id,
        call_stack, line_info, rccl_device_idx, static_cast<uint8_t>(agent_type::GPU),
        pmc_label.c_str(), static_cast<double>(cumulative) });
}

rccl_gpu_tracking_state&
rccl_get_gpu_tracking_state()
{
    static auto registrar = std::make_shared<production_pmc_registrar>();
    static rccl_gpu_tracking_state state{ registrar };
    return state;
}

[[nodiscard]] size_t
rccl_type_size_or_abort(ncclDataType_t datatype) noexcept
{
    auto size = rccl_type_size(datatype);
    if(size == 0)
    {
        LOG_WARNING("Unsupported RCCL datatype: {}", static_cast<int>(datatype));
        return 0;
    }
    return size;
}

}  // anonymous namespace

[[nodiscard]] uint32_t
rccl_get_device_id(ncclComm_t comm) noexcept
{
    constexpr uint32_t DEFAULT_DEVICE_ID = 0;

    if(comm == nullptr) return DEFAULT_DEVICE_ID;

    using ncclCommCuDevice_fn = ncclResult_t (*)(ncclComm_t, int*);

    static ncclCommCuDevice_fn ncclCommCuDevice_ptr = nullptr;
    static std::once_flag      lookup_flag;

    std::call_once(lookup_flag, []() {
        ncclCommCuDevice_ptr = reinterpret_cast<ncclCommCuDevice_fn>(
            dlsym(RTLD_DEFAULT, "ncclCommCuDevice"));
        if(ncclCommCuDevice_ptr == nullptr)
        {
            const char* error = dlerror();
            LOG_DEBUG(
                "ncclCommCuDevice not found via dlsym ({}), using default device_id",
                error ? error : "unknown error");
        }
    });

    if(ncclCommCuDevice_ptr == nullptr) return DEFAULT_DEVICE_ID;

    int          device_id = DEFAULT_DEVICE_ID;
    ncclResult_t result    = ncclCommCuDevice_ptr(comm, &device_id);
    if(result != ncclSuccess)
    {
        LOG_DEBUG("ncclCommCuDevice failed with error {}, using default device_id",
                  static_cast<int>(result));
        return DEFAULT_DEVICE_ID;
    }
    return static_cast<uint32_t>(device_id);
}

/**
 * @brief Initialize RCCL communication data tracking metadata
 *
 * This function performs one-time initialization of metadata categories
 * and tracking infrastructure for RCCL send/recv operations.
 * Called once during SDK initialization when RCCL callbacks are configured.
 */
void
rccl_comm_data_initialize()
{
    rccl_metadata_initialize_categories();
    rccl_metadata_initialize_track<rccl_send>();
    rccl_metadata_initialize_track<rccl_recv>();
}

/**
 * @brief Main callback handler for RCCL API tracing events
 *
 * This function is invoked by the profiling framework for each RCCL API call.
 * It determines the device ID from the communicator and records both cache
 * events and Perfetto counter tracks for send/recv operations.
 *
 * @param operation The RCCL operation ID
 * @param payload The RCCL API-specific payload data
 * @param begin_ts Timestamp when the API call began (nanoseconds)
 * @param end_ts Timestamp when the API call ended (nanoseconds)
 */
void
tool_tracing_callback_rccl(uint32_t                                      operation,
                           rocprofiler_callback_tracing_rccl_api_data_t* payload,
                           uint64_t begin_ts, uint64_t end_ts)
{
    rccl_event_info info = rccl_get_event_info_impl(operation, *payload);

    if(info.size > 0 && info.comm != nullptr)
    {
        uint32_t device_id = rccl_get_device_id(info.comm);

        if(info.is_send)
        {
            cache_rccl_comm_data_events<rccl_send>(device_id, info.size, end_ts);
            write_perfetto_counter_track<rccl_send>(info.size, begin_ts, end_ts);
        }
        else
        {
            cache_rccl_comm_data_events<rccl_recv>(device_id, info.size, end_ts);
            write_perfetto_counter_track<rccl_recv>(info.size, begin_ts, end_ts);
        }
    }
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
