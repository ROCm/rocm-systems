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

#include "generatePerfetto.hpp"
#include "output_stream.hpp"
#include "timestamps.hpp"

#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/perfetto.hpp>

#include <atomic>
#include <future>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace
{
auto main_tid = common::get_tid();

template <typename Tp>
size_t
get_hash_id(Tp&& _val)
{
    if constexpr(!std::is_pointer<Tp>::value)
        return std::hash<Tp>{}(std::forward<Tp>(_val));
    else if constexpr(std::is_same<Tp, const char*>::value)
        return get_hash_id(std::string_view{_val});
    else
        return get_hash_id(*_val);
}
}  // namespace

void
write_perfetto(
    const output_config&                                                  ocfg,
    const metadata&                                                       tool_metadata,
    std::vector<agent_info>                                               agent_data,
    const generator<rocprofiler_buffer_tracing_hip_api_record_t>&         hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&         hsa_api_gen,
    const generator<rocprofiler_buffer_tracing_kernel_dispatch_record_t>& kernel_dispatch_gen,
    const generator<rocprofiler_buffer_tracing_memory_copy_record_t>&     memory_copy_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&      marker_api_gen,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>& /*scratch_memory_gen*/,
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<rocprofiler_buffer_tracing_memory_allocation_record_t>& memory_allocation_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_record_t>&     rocdecode_api_gen)
{
    namespace sdk = ::rocprofiler::sdk;

    // auto     root_process_track = ::perfetto::Track{};
    // uint64_t process_uuid       = tool_metadata.process_start_ns ^ tool_metadata.process_id;
    // auto     process_track      = ::perfetto::Track{process_uuid, root_process_track};
    // auto     process_track      = ::perfetto::ProcessTrack::Current();

    auto agents_map = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>{};
    for(auto itr : agent_data)
        agents_map.emplace(itr.id, itr);

    auto args            = ::perfetto::TracingInitArgs{};
    auto track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto cfg             = ::perfetto::TraceConfig{};

    // environment settings
    auto shmem_size_hint = ocfg.perfetto_shmem_size_hint;
    auto buffer_size_kb  = ocfg.perfetto_buffer_size;

    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size_kb);

    if(ocfg.perfetto_buffer_fill_policy == "discard" || ocfg.perfetto_buffer_fill_policy.empty())
        buffer_config->set_fill_policy(
            ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);
    else if(ocfg.perfetto_buffer_fill_policy == "ring_buffer")
        buffer_config->set_fill_policy(
            ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER);
    else
        ROCP_FATAL << "Unsupport perfetto buffer fill policy: '" << ocfg.perfetto_buffer_fill_policy
                   << "'. Supported: discard, ring_buffer";

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.shmem_size_hint_kb = shmem_size_hint;

    if(ocfg.perfetto_backend == "inprocess" || ocfg.perfetto_backend.empty())
        args.backends |= ::perfetto::kInProcessBackend;
    else if(ocfg.perfetto_backend == "system")
        args.backends |= ::perfetto::kSystemBackend;
    else
        ROCP_FATAL << "Unsupport perfetto backend: '" << ocfg.perfetto_backend
                   << "'. Supported: inprocess, system";

    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();

    auto tracing_session = ::perfetto::Tracing::NewTrace();

    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();

    auto tids                   = std::set<rocprofiler_thread_id_t>{};
    auto demangled              = std::unordered_map<std::string_view, std::string>{};
    auto agent_thread_ids       = std::unordered_map<rocprofiler_agent_id_t, std::set<uint64_t>>{};
    auto agent_thread_ids_alloc = std::unordered_map<rocprofiler_agent_id_t, std::set<uint64_t>>{};
    auto agent_queue_ids =
        std::unordered_map<rocprofiler_agent_id_t, std::unordered_set<rocprofiler_queue_id_t>>{};
    auto thread_indexes = std::unordered_map<rocprofiler_thread_id_t, uint64_t>{};

    auto thread_tracks = std::unordered_map<rocprofiler_thread_id_t, ::perfetto::Track>{};
    auto agent_thread_tracks =
        std::unordered_map<rocprofiler_agent_id_t,
                           std::unordered_map<uint64_t, ::perfetto::Track>>{};
    auto agent_thread_tracks_alloc =
        std::unordered_map<rocprofiler_agent_id_t,
                           std::unordered_map<uint64_t, ::perfetto::Track>>{};
    auto agent_queue_tracks =
        std::unordered_map<rocprofiler_agent_id_t,
                           std::unordered_map<rocprofiler_queue_id_t, ::perfetto::Track>>{};

    auto _get_agent = [&agent_data](rocprofiler_agent_id_t _id) -> const rocprofiler_agent_t* {
        for(const auto& itr : agent_data)
        {
            if(_id == itr.id) return &itr;
        }
        return CHECK_NOTNULL(nullptr);
    };

    {
        for(auto ditr : hsa_api_gen)
            for(auto itr : hsa_api_gen.get(ditr))
                tids.emplace(itr.thread_id);
        for(auto ditr : hip_api_gen)
            for(auto itr : hip_api_gen.get(ditr))
                tids.emplace(itr.thread_id);
        for(auto ditr : marker_api_gen)
            for(auto itr : marker_api_gen.get(ditr))
                tids.emplace(itr.thread_id);
        for(auto ditr : rccl_api_gen)
            for(auto itr : rccl_api_gen.get(ditr))
                tids.emplace(itr.thread_id);
        for(auto ditr : rocdecode_api_gen)
            for(auto itr : rocdecode_api_gen.get(ditr))
                tids.emplace(itr.thread_id);

        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                tids.emplace(itr.thread_id);
                agent_thread_ids[itr.dst_agent_id].emplace(itr.thread_id);
            }

        for(auto ditr : memory_allocation_gen)
            for(auto itr : memory_allocation_gen.get(ditr))
            {
                tids.emplace(itr.thread_id);
                agent_thread_ids_alloc[itr.agent_id].emplace(itr.thread_id);
            }

        for(auto ditr : kernel_dispatch_gen)
            for(auto itr : kernel_dispatch_gen.get(ditr))
            {
                tids.emplace(itr.thread_id);
                agent_queue_ids[itr.dispatch_info.agent_id].emplace(itr.dispatch_info.queue_id);
            }
    }

    uint64_t nthrn = 0;
    for(auto itr : tids)
    {
        if(itr == main_tid)
        {
            thread_indexes.emplace(main_tid, 0);
            thread_tracks.emplace(main_tid, ::perfetto::ThreadTrack::Current());
        }
        else
        {
            auto _idx = ++nthrn;
            thread_indexes.emplace(itr, _idx);
            auto _track  = ::perfetto::Track{itr};
            auto _desc   = _track.Serialize();
            auto _namess = std::stringstream{};
            _namess << "THREAD " << _idx << " (" << itr << ")";
            _desc.set_name(_namess.str());
            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            thread_tracks.emplace(itr, _track);
        }
    }

    for(const auto& itr : agent_thread_ids)
    {
        const auto* _agent = _get_agent(itr.first);

        for(auto titr : itr.second)
        {
            auto _namess = std::stringstream{};
            _namess << "COPY to AGENT [" << _agent->logical_node_id << "] THREAD ["
                    << thread_indexes.at(titr) << "] ";

            if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _namess << "(CPU)";
            else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _namess << "(GPU)";
            else
                _namess << "(UNK)";

            auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_thread_tracks[itr.first].emplace(titr, _track);
        }
    }

    for(const auto& aitr : agent_queue_ids)
    {
        uint32_t nqueue = 0;
        for(auto qitr : aitr.second)
        {
            const auto* _agent = _get_agent(aitr.first);

            auto _namess = std::stringstream{};
            _namess << "COMPUTE AGENT [" << _agent->logical_node_id << "] QUEUE [" << nqueue++
                    << "] ";

            if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _namess << "(CPU)";
            else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _namess << "(GPU)";
            else
                _namess << "(UNK)";

            auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_queue_tracks[aitr.first].emplace(qitr, _track);
        }
    }

    // trace events
    {
        auto buffer_names     = sdk::get_buffer_tracing_names();
        auto callbk_name_info = sdk::get_callback_tracing_names();

        for(auto ditr : hsa_api_gen)
            for(auto itr : hsa_api_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = thread_tracks.at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::hsa_api>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "tid",
                                  itr.thread_id,
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "corr_id",
                                  itr.correlation_id.internal);
                TRACE_EVENT_END(
                    sdk::perfetto_category<sdk::category::hsa_api>::name, track, itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : hip_api_gen)
            for(auto itr : hip_api_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = thread_tracks.at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::hip_api>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "tid",
                                  itr.thread_id,
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "corr_id",
                                  itr.correlation_id.internal);
                TRACE_EVENT_END(
                    sdk::perfetto_category<sdk::category::hip_api>::name, track, itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : marker_api_gen)
            for(auto itr : marker_api_gen.get(ditr))
            {
                auto& track = thread_tracks.at(itr.thread_id);
                auto  name  = (itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API &&
                             itr.operation != ROCPROFILER_MARKER_CORE_API_ID_roctxGetThreadId)
                                  ? tool_metadata.get_marker_message(itr.correlation_id.internal)
                                  : buffer_names.at(itr.kind, itr.operation);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::marker_api>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "tid",
                                  itr.thread_id,
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "corr_id",
                                  itr.correlation_id.internal);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::marker_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : rccl_api_gen)
            for(auto itr : rccl_api_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = thread_tracks.at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rccl_api>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "tid",
                                  itr.thread_id,
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "corr_id",
                                  itr.correlation_id.internal);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rccl_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : rocdecode_api_gen)
            for(auto itr : rocdecode_api_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = thread_tracks.at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rocdecode_api>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "tid",
                                  itr.thread_id,
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "corr_id",
                                  itr.correlation_id.internal);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rocdecode_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = agent_thread_tracks.at(itr.dst_agent_id).at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::memory_copy>::name,
                                  ::perfetto::StaticString(name.data()),
                                  track,
                                  itr.start_timestamp,
                                  ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                                  "begin_ns",
                                  itr.start_timestamp,
                                  "end_ns",
                                  itr.end_timestamp,
                                  "delta_ns",
                                  (itr.end_timestamp - itr.start_timestamp),
                                  "kind",
                                  itr.kind,
                                  "operation",
                                  itr.operation,
                                  "src_agent",
                                  agents_map.at(itr.src_agent_id).logical_node_id,
                                  "dst_agent",
                                  agents_map.at(itr.dst_agent_id).logical_node_id,
                                  "copy_bytes",
                                  itr.bytes,
                                  "corr_id",
                                  itr.correlation_id.internal,
                                  "tid",
                                  itr.thread_id);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::memory_copy>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : kernel_dispatch_gen)
            for(auto itr : kernel_dispatch_gen.get(ditr))
            {
                const auto&               info = itr.dispatch_info;
                const kernel_symbol_info* sym  = tool_metadata.get_kernel_symbol(info.kernel_id);

                CHECK(sym != nullptr);

                auto  name  = std::string_view{sym->kernel_name};
                auto& track = agent_queue_tracks.at(info.agent_id).at(info.queue_id);

                if(demangled.find(name) == demangled.end())
                {
                    demangled.emplace(name, common::cxx_demangle(name));
                }

                TRACE_EVENT_BEGIN(
                    sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                    ::perfetto::StaticString(demangled.at(name).c_str()),
                    track,
                    itr.start_timestamp,
                    ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                    "begin_ns",
                    itr.start_timestamp,
                    "end_ns",
                    itr.end_timestamp,
                    "delta_ns",
                    (itr.end_timestamp - itr.start_timestamp),
                    "kind",
                    itr.kind,
                    "agent",
                    agents_map.at(info.agent_id).logical_node_id,
                    "corr_id",
                    itr.correlation_id.internal,
                    "queue",
                    info.queue_id.handle,
                    "tid",
                    itr.thread_id,
                    "kernel_id",
                    info.kernel_id,
                    "private_segment_size",
                    info.private_segment_size,
                    "group_segment_size",
                    info.group_segment_size,
                    "workgroup_size",
                    info.workgroup_size.x * info.workgroup_size.y * info.workgroup_size.z,
                    "grid_size",
                    info.grid_size.x * info.grid_size.y * info.grid_size.z);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }
    }

    // counter tracks
    {
        // memory copy counter track
        auto mem_cpy_endpoints = std::map<rocprofiler_agent_id_t, std::map<uint64_t, uint64_t>>{};
        auto mem_cpy_extremes  = std::pair<uint64_t, uint64_t>{};
        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                uint64_t _mean_timestamp =
                    itr.start_timestamp + (0.5 * (itr.end_timestamp - itr.start_timestamp));

                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.start_timestamp - 1000, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.start_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(_mean_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.end_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.end_timestamp + 1000, 0);

                mem_cpy_extremes =
                    std::make_pair(std::min(mem_cpy_extremes.first, itr.start_timestamp),
                                   std::max(mem_cpy_extremes.second, itr.end_timestamp));
            }

        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                auto mbeg = mem_cpy_endpoints.at(itr.dst_agent_id).lower_bound(itr.start_timestamp);
                auto mend = mem_cpy_endpoints.at(itr.dst_agent_id).upper_bound(itr.end_timestamp);

                LOG_IF(FATAL, mbeg == mend)
                    << "Missing range for timestamp [" << itr.start_timestamp << ", "
                    << itr.end_timestamp << "]";

                for(auto mitr = mbeg; mitr != mend; ++mitr)
                    mitr->second += itr.bytes;
            }

        constexpr auto bytes_multiplier = 1024;

        auto mem_cpy_tracks =
            std::unordered_map<rocprofiler_agent_id_t, ::perfetto::CounterTrack>{};
        auto mem_cpy_cnt_names = std::vector<std::string>{};
        mem_cpy_cnt_names.reserve(mem_cpy_endpoints.size());
        for(auto& mitr : mem_cpy_endpoints)
        {
            mem_cpy_endpoints[mitr.first].emplace(mem_cpy_extremes.first - 5000, 0);
            mem_cpy_endpoints[mitr.first].emplace(mem_cpy_extremes.second + 5000, 0);

            auto        _track_name = std::stringstream{};
            const auto* _agent      = _get_agent(mitr.first);

            if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _track_name << "COPY BYTES to AGENT [" << _agent->logical_node_id << "] (CPU)";
            else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _track_name << "COPY BYTES to AGENT [" << _agent->logical_node_id << "] (GPU)";

            constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
            auto&          _name = mem_cpy_cnt_names.emplace_back(_track_name.str());
            mem_cpy_tracks.emplace(mitr.first,
                                   ::perfetto::CounterTrack{_name.c_str()}
                                       .set_unit(_unit)
                                       .set_unit_multiplier(bytes_multiplier)
                                       .set_is_incremental(false));
        }

        for(auto& mitr : mem_cpy_endpoints)
        {
            for(auto itr : mitr.second)
            {
                TRACE_COUNTER(sdk::perfetto_category<sdk::category::memory_copy>::name,
                              mem_cpy_tracks.at(mitr.first),
                              itr.first,
                              itr.second / bytes_multiplier);
                tracing_session->FlushBlocking();
            }
        }

        // memory allocation counter track
        auto mem_alloc_endpoints = std::map<rocprofiler_agent_id_t, std::map<uint64_t, uint64_t>>{};
        auto mem_alloc_extremes  = std::pair<uint64_t, uint64_t>{};
        auto address_to_size     = std::unordered_map<uint64_t, uint64_t>{};
        for(auto ditr : memory_allocation_gen)
            for(auto itr : memory_allocation_gen.get(ditr))
            {
                uint64_t _mean_timestamp =
                    itr.start_timestamp + (0.5 * (itr.end_timestamp - itr.start_timestamp));

                mem_alloc_endpoints[itr.agent_id].emplace(itr.start_timestamp - 1000, 0);
                mem_alloc_endpoints[itr.agent_id].emplace(itr.start_timestamp, 0);
                mem_alloc_endpoints[itr.agent_id].emplace(_mean_timestamp, 0);
                mem_alloc_endpoints[itr.agent_id].emplace(itr.end_timestamp, 0);
                mem_alloc_endpoints[itr.agent_id].emplace(itr.end_timestamp + 1000, 0);

                mem_alloc_extremes =
                    std::make_pair(std::min(mem_alloc_extremes.first, itr.start_timestamp),
                                   std::max(mem_alloc_extremes.second, itr.end_timestamp));
                if(itr.operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE ||
                   itr.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE)
                {
                    address_to_size.emplace(itr.address.value, itr.allocation_size);
                }
            }

        for(auto ditr : memory_allocation_gen)
            for(auto itr : memory_allocation_gen.get(ditr))
            {
                auto alloc_beg =
                    mem_alloc_endpoints.at(itr.agent_id).lower_bound(itr.start_timestamp);
                auto alloc_end =
                    mem_alloc_endpoints.at(itr.agent_id).upper_bound(itr.end_timestamp);

                LOG_IF(FATAL, alloc_beg == alloc_end)
                    << "Missing range for timestamp [" << itr.start_timestamp << ", "
                    << itr.end_timestamp << "]";

                for(auto alloc_itr = alloc_beg; alloc_itr != alloc_end; ++alloc_itr)
                {
                    if(address_to_size.count(itr.address.value) > 0)
                    {
                        alloc_itr->second += address_to_size.at(itr.address.value);
                    }
                }
            }

        auto mem_alloc_tracks =
            std::unordered_map<rocprofiler_agent_id_t, ::perfetto::CounterTrack>{};
        auto           mem_alloc_cnt_names = std::vector<std::string>{};
        constexpr auto null_rocp_agent_id =
            rocprofiler_agent_id_t{.handle = std::numeric_limits<uint64_t>::max()};
        mem_alloc_cnt_names.reserve(mem_alloc_endpoints.size());
        for(auto& alloc_itr : mem_alloc_endpoints)
        {
            mem_alloc_endpoints[alloc_itr.first].emplace(mem_alloc_extremes.first - 5000, 0);
            mem_alloc_endpoints[alloc_itr.first].emplace(mem_alloc_extremes.second + 5000, 0);

            auto                       _track_name = std::stringstream{};
            const rocprofiler_agent_t* _agent      = nullptr;
            if(alloc_itr.first != null_rocp_agent_id)
            {
                _agent = _get_agent(alloc_itr.first);
            }

            if(_agent != nullptr && _agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _track_name << "ALLOCATE BYTES on AGENT [" << _agent->logical_node_id << "] (CPU)";
            else if(_agent != nullptr && _agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _track_name << "ALLOCATE BYTES on AGENT [" << _agent->logical_node_id << "] (GPU)";
            else
                _track_name << "FREE BYTES";

            constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
            auto&          _name = mem_alloc_cnt_names.emplace_back(_track_name.str());
            mem_alloc_tracks.emplace(alloc_itr.first,
                                     ::perfetto::CounterTrack{_name.c_str()}
                                         .set_unit(_unit)
                                         .set_unit_multiplier(bytes_multiplier)
                                         .set_is_incremental(false));
        }

        for(auto& alloc_itr : mem_alloc_endpoints)
        {
            for(auto itr : alloc_itr.second)
            {
                TRACE_COUNTER(sdk::perfetto_category<sdk::category::memory_allocation>::name,
                              mem_alloc_tracks.at(alloc_itr.first),
                              itr.first,
                              itr.second / bytes_multiplier);
                tracing_session->FlushBlocking();
            }
        }
    }

    ::perfetto::TrackEvent::Flush();
    tracing_session->FlushBlocking();
    tracing_session->StopBlocking();

    auto filename = std::string{"results"};
    auto ofs      = get_output_stream(ocfg, filename, ".pftrace");

    auto amount_read = std::atomic<size_t>{0};
    auto is_done     = std::promise<void>{};
    auto _mtx        = std::mutex{};
    auto _reader     = [&ofs, &_mtx, &is_done, &amount_read](
                       ::perfetto::TracingSession::ReadTraceCallbackArgs _args) {
        auto _lk = std::unique_lock<std::mutex>{_mtx};
        if(_args.data && _args.size > 0)
        {
            ROCP_TRACE << "Writing " << _args.size << " B to trace...";
            // Write the trace data into file
            ofs.stream->write(_args.data, _args.size);
            amount_read += _args.size;
        }
        ROCP_INFO_IF(!_args.has_more && amount_read > 0)
            << "Wrote " << amount_read << " B to perfetto trace file";
        if(!_args.has_more) is_done.set_value();
    };

    for(size_t i = 0; i < 2; ++i)
    {
        ROCP_TRACE << "Reading trace...";
        amount_read = 0;
        is_done     = std::promise<void>{};
        tracing_session->ReadTrace(_reader);
        is_done.get_future().wait();
    }

    ROCP_TRACE << "Destroying tracing session...";
    tracing_session.reset();

    ROCP_TRACE << "Flushing trace output stream...";
    (*ofs.stream) << std::flush;

    ROCP_TRACE << "Destroying trace output stream...";
    ofs.close();
}

}  // namespace tool
}  // namespace rocprofiler

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
