// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/rocprofiler-sdk/rccl.hpp"

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
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
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

void
rccl_metadata_initialize_categories()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    trace_cache::get_metadata_registry().add_string(
        trait::name<category::comm_data>::value);

    _is_initialized = true;
}

std::mutex                   g_registered_gpus_mutex{};
std::unordered_set<uint32_t> g_registered_gpus{};

void
rccl_metadata_initialize_pmc_for_gpu(uint32_t rccl_device_idx)
{
    {
        std::unique_lock<std::mutex> _lk{ g_registered_gpus_mutex };
        if(g_registered_gpus.count(rccl_device_idx) > 0) return;
        g_registered_gpus.insert(rccl_device_idx);
    }

    [[maybe_unused]] const size_t EVENT_CODE  = 0;
    [[maybe_unused]] const size_t INSTANCE_ID = 0;
    [[maybe_unused]] const auto*  LONG_DESCRIPTION =
        "Per-GPU RCCL communication data with transfer_bytes in extdata JSON";
    [[maybe_unused]] const auto* COMPONENT   = "";
    [[maybe_unused]] const auto* BLOCK       = "";
    [[maybe_unused]] const auto* EXPRESSION  = "";
    [[maybe_unused]] const auto* MSG         = "bytes";
    [[maybe_unused]] const auto* TARGET_ARCH = "GPU";

    auto register_rccl_info = [&](const char* direction_label, const char* description) {
        std::string label = fmt::format("{} GPU {}", direction_label, rccl_device_idx);
        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, rccl_device_idx, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              label.c_str(), description, trait::name<category::comm_data>::description,
              LONG_DESCRIPTION, COMPONENT, MSG, trace_cache::ABSOLUTE, BLOCK, EXPRESSION,
              0, 0 });
    };

    register_rccl_info(rccl_send::label, "Tracks RCCL communication data sizes (send)");
    register_rccl_info(rccl_recv::label, "Tracks RCCL communication data sizes (recv)");
}

template <typename Track>
void
rccl_metadata_initialize_track()
{
    auto _init_track = [](const char* label) {
        trace_cache::get_metadata_registry().add_track({ label, std::nullopt, "{}" });
    };

    static std::once_flag _once{};
    std::call_once(_once, _init_track, Track::label);
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
    rccl_metadata_initialize_pmc_for_gpu(rccl_device_idx);

    static std::mutex                             _mutex{};
    static std::unordered_map<uint32_t, uint64_t> cumulative_bytes_per_device{};
    uint64_t                                      cumulative     = 0;
    size_t                                        transfer_bytes = bytes;
    {
        std::unique_lock<std::mutex> _lk{ _mutex };
        auto& device_bytes = cumulative_bytes_per_device[rccl_device_idx];
        device_bytes += bytes;
        cumulative = device_bytes;
    }

    std::string event_metadata =
        fmt::format(R"({{"transfer_bytes":{}}})", transfer_bytes);

    std::string pmc_label = fmt::format("{} GPU {}", Track::label, rccl_device_idx);

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

size_t
rccl_type_size(ncclDataType_t datatype)
{
    switch(datatype)
    {
        case ncclInt8:
        case ncclUint8: return 1;
        case ncclFloat16: return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32: return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64: return 8;
        default:
        {
            LOG_CRITICAL("Unsupported RCCL datatype: {}", static_cast<int>(datatype));
            ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
            std::abort();
        }
    };
}

/**
 * @brief Get device ID from RCCL communicator
 *
 * Dynamically loads ncclCommCuDevice() to query the device associated with
 * the communicator. Falls back to device 0 if the function is unavailable
 * or the query fails.
 *
 * @param comm The RCCL communicator
 * @return The device ID associated with the communicator
 */
uint32_t
rccl_get_device_id(ncclComm_t comm)
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

}  // namespace

void
rccl_comm_data_initialize()
{
    static std::once_flag _once{};
    std::call_once(_once, []() {
        rccl_metadata_initialize_categories();
        rccl_metadata_initialize_track<rccl_send>();
        rccl_metadata_initialize_track<rccl_recv>();
    });
}

void
tool_tracing_callback_rccl(rocprofiler_callback_tracing_record_t record,
                           uint64_t begin_ts, uint64_t end_ts)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_RCCL_API)
    {
        rccl_comm_data_initialize();

        auto* payload =
            static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);

        size_t     size    = 0;
        bool       is_send = false;
        ncclComm_t comm    = nullptr;

        auto set_recv = [&](size_t count, ncclDataType_t _dt, ncclComm_t _comm) {
            is_send = false;
            size    = count * rccl_type_size(_dt);
            comm    = _comm;
        };

        auto set_send = [&](size_t count, ncclDataType_t _dt, ncclComm_t _comm) {
            is_send = true;
            size    = count * rccl_type_size(_dt);
            comm    = _comm;
        };

        switch(record.operation)
        {
            case ROCPROFILER_RCCL_API_ID_ncclAllGather:
                set_recv(payload->args.ncclAllGather.sendcount,
                         payload->args.ncclAllGather.datatype,
                         payload->args.ncclAllGather.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclAllToAll:
                set_recv(payload->args.ncclAllToAll.count,
                         payload->args.ncclAllToAll.datatype,
                         payload->args.ncclAllToAll.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclAllReduce:
                set_recv(payload->args.ncclAllReduce.count,
                         payload->args.ncclAllReduce.datatype,
                         payload->args.ncclAllReduce.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclGather:
                set_recv(payload->args.ncclGather.sendcount,
                         payload->args.ncclGather.datatype,
                         payload->args.ncclGather.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclRecv:
                set_recv(payload->args.ncclRecv.count, payload->args.ncclRecv.datatype,
                         payload->args.ncclRecv.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclReduce:
                set_recv(payload->args.ncclReduce.count,
                         payload->args.ncclReduce.datatype,
                         payload->args.ncclReduce.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclBroadcast:
                set_send(payload->args.ncclBroadcast.count,
                         payload->args.ncclBroadcast.datatype,
                         payload->args.ncclBroadcast.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclReduceScatter:
                set_send(payload->args.ncclReduceScatter.recvcount,
                         payload->args.ncclReduceScatter.datatype,
                         payload->args.ncclReduceScatter.comm);
                break;
            case ROCPROFILER_RCCL_API_ID_ncclSend:
                set_send(payload->args.ncclSend.count, payload->args.ncclSend.datatype,
                         payload->args.ncclSend.comm);
                break;

            default: break;
        }

        if(size > 0 && comm != nullptr)
        {
            uint32_t device_id = rccl_get_device_id(comm);

            if(is_send)
            {
                cache_rccl_comm_data_events<rccl_send>(device_id, size, end_ts);
                write_perfetto_counter_track<rccl_send>(size, begin_ts, end_ts);
            }
            else
            {
                cache_rccl_comm_data_events<rccl_recv>(device_id, size, end_ts);
                write_perfetto_counter_track<rccl_recv>(size, begin_ts, end_ts);
            }
        }
    }
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
