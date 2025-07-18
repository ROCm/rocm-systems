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

#include <fmt/core.h>

#include <atomic>
#include <future>
#include <iostream>
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
    const output_config&                                                    ocfg,
    const metadata&                                                         tool_metadata,
    std::vector<agent_info>                                                 agent_data,
    const generator<tool_buffer_tracing_hip_api_ext_record_t>&              hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&           hsa_api_gen,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>&      kernel_dispatch_gen,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>&          memory_copy_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&        marker_api_gen,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>&    scratch_memory_gen,
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<tool_buffer_tracing_memory_allocation_ext_record_t>&    memory_allocation_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>&       rocjpeg_api_gen)
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
    const auto is_hip_initialized =
        tool_metadata.is_runtime_initialized(ROCPROFILER_RUNTIME_INITIALIZATION_HIP);
    const auto group_by_queue   = ocfg.group_by_queue || !is_hip_initialized;
    auto       tids             = std::set<rocprofiler_thread_id_t>{};
    auto       demangled        = std::unordered_map<std::string_view, std::string>{};
    auto       agent_thread_ids = std::unordered_map<rocprofiler_agent_id_t, std::set<uint64_t>>{};
    auto agent_thread_ids_alloc = std::unordered_map<rocprofiler_agent_id_t, std::set<uint64_t>>{};
    auto agent_queue_ids =
        std::unordered_map<rocprofiler_agent_id_t, std::unordered_set<rocprofiler_queue_id_t>>{};
    auto agent_stream_ids = std::unordered_set<rocprofiler_stream_id_t>{};
    auto thread_indexes   = std::unordered_map<rocprofiler_thread_id_t, uint64_t>{};

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
    auto stream_tracks = std::unordered_map<rocprofiler_stream_id_t, ::perfetto::Track>{};

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
        for(auto ditr : rocjpeg_api_gen)
            for(auto itr : rocjpeg_api_gen.get(ditr))
                tids.emplace(itr.thread_id);

        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                tids.emplace(itr.thread_id);
                agent_stream_ids.emplace(itr.stream_id);
                if(group_by_queue)
                {
                    agent_thread_ids[itr.dst_agent_id].emplace(itr.thread_id);
                }
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
                agent_stream_ids.emplace(itr.stream_id);
                if(group_by_queue)
                {
                    agent_queue_ids[itr.dispatch_info.agent_id].emplace(itr.dispatch_info.queue_id);
                }
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
            auto agent_index_info =
                tool_metadata.get_agent_index(_agent->id, ocfg.agent_index_value);
            _namess << "COMPUTE " << agent_index_info.label << " [" << agent_index_info.index
                    << "] QUEUE [" << nqueue++ << "] ";
            _namess << agent_index_info.type;

            auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_queue_tracks[aitr.first].emplace(qitr, _track);
        }
    }

    for(const auto& sitr : agent_stream_ids)
    {
        const auto stream_id = sitr.handle;

        {
            auto _namess = std::stringstream{};
            _namess << fmt::format("STREAM [\" {} \"] ", stream_id);

            auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            stream_tracks.emplace(sitr, _track);
        }
    }

    auto counter_id_to_name = std::unordered_map<rocprofiler_counter_id_t, std::string_view>{};
    for(const auto& itr : tool_metadata.get_counter_info())
        counter_id_to_name.emplace(itr.id, itr.name);

    // Map: correlation_id -> map<counter_id, value>
    auto dispatch_counter_id_value =
        std::unordered_map<uint64_t, std::unordered_map<rocprofiler_counter_id_t, double>>{};

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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor);

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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor,
                                  "stream_ID",
                                  itr.stream_id.handle);

                TRACE_EVENT_END(
                    sdk::perfetto_category<sdk::category::hip_api>::name, track, itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : marker_api_gen)
            for(auto itr : marker_api_gen.get(ditr))
            {
                auto& track = thread_tracks.at(itr.thread_id);
                auto  name  = (itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API &&
                             itr.operation != ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId)
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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor);
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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rccl_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : rocdecode_api_gen)
            for(auto itr : rocdecode_api_gen.get(ditr))
            {
                auto  name           = buffer_names.at(itr.kind, itr.operation);
                auto& track          = thread_tracks.at(itr.thread_id);
                auto  rocdecode_args = sdk::serialization::get_buffer_tracing_args(itr);

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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor,
                                  [&](::perfetto::EventContext ctx) {
                                      for(const auto& rocdecode_arg : rocdecode_args)
                                      {
                                          sdk::add_perfetto_annotation(
                                              ctx, rocdecode_arg.name, rocdecode_arg.value);
                                      }
                                  });
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rocdecode_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : rocjpeg_api_gen)
            for(auto itr : rocjpeg_api_gen.get(ditr))
            {
                auto  name  = buffer_names.at(itr.kind, itr.operation);
                auto& track = thread_tracks.at(itr.thread_id);

                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rocjpeg_api>::name,
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
                                  itr.correlation_id.internal,
                                  "ancestor_id",
                                  itr.correlation_id.ancestor);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rocjpeg_api>::name,
                                track,
                                itr.end_timestamp);
                tracing_session->FlushBlocking();
            }

        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                auto name = buffer_names.at(itr.kind, itr.operation);

                ::perfetto::Track* _track = nullptr;
                if(group_by_queue)
                {
                    _track = &agent_thread_tracks.at(itr.dst_agent_id).at(itr.thread_id);
                }
                else
                {
                    _track = &stream_tracks.at(itr.stream_id);
                }

                TRACE_EVENT_BEGIN(
                    sdk::perfetto_category<sdk::category::memory_copy>::name,
                    ::perfetto::StaticString(name.data()),
                    *_track,
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
                    tool_metadata.get_agent_index(itr.src_agent_id, ocfg.agent_index_value)
                        .as_string("-"),
                    "dst_agent",
                    tool_metadata.get_agent_index(itr.dst_agent_id, ocfg.agent_index_value)
                        .as_string("-"),
                    "copy_bytes",
                    itr.bytes,
                    "corr_id",
                    itr.correlation_id.internal,
                    "tid",
                    itr.thread_id,
                    "stream_ID",
                    itr.stream_id.handle);
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::memory_copy>::name,
                                *_track,
                                itr.end_timestamp);

                tracing_session->FlushBlocking();
            }

        for(auto ditr : counter_collection_gen)
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                auto& counter_id_value =
                    dispatch_counter_id_value[record.dispatch_data.correlation_id.internal];
                auto record_vector = record.read();

                // Accumulate counters based on ID for this dispatch
                for(auto& count : record_vector)
                {
                    counter_id_value[count.id] += count.value;
                }
            }

        for(auto ditr : kernel_dispatch_gen)
        {
            auto generator = kernel_dispatch_gen.get(ditr);
            // Group kernels on the same queue and agent. Temporary fix for firmware timestamp bug
            // Can be removed once bug is resolved.
            auto dispatch_bins = std::unordered_map<
                rocprofiler_agent_id_t,
                std::unordered_map<
                    rocprofiler_queue_id_t,
                    std::vector<tool_buffer_tracing_kernel_dispatch_ext_record_t*>>>{};
            for(auto& itr : generator)
            {
                const auto& info = itr.dispatch_info;
                dispatch_bins[info.agent_id][info.queue_id].emplace_back(&itr);
            }

            for(const auto& aitr : dispatch_bins)
            {
                for(auto qitr : aitr.second)
                {
                    // Sort kernels on the same queue and agent by timestamp
                    std::sort(qitr.second.begin(),
                              qitr.second.end(),
                              [](const auto* lhs, const auto* rhs) {
                                  return lhs->start_timestamp < rhs->start_timestamp;
                              });

                    // Loop over the kernels (qitr.second) and put them into perfetto.
                    for(auto it = qitr.second.begin(); it != qitr.second.end(); ++it)
                    {
                        auto&                     current = **it;
                        const auto&               info    = current.dispatch_info;
                        const kernel_symbol_info* sym =
                            tool_metadata.get_kernel_symbol(info.kernel_id);

                        CHECK(sym != nullptr);

                        auto name = std::string_view{sym->kernel_name};

                        ::perfetto::Track* _track    = nullptr;
                        auto               stream_id = (*it)->stream_id;
                        if(group_by_queue)
                        {
                            _track = &agent_queue_tracks.at(info.agent_id).at(info.queue_id);
                        }
                        else
                        {
                            _track = &stream_tracks.at(stream_id);
                        }

                        // Temporary fix until timestamp issues are resolved: Set timestamps to be
                        // halfway between ending timestamp and starting timestamp of overlapping
                        // kernel dispatches. Perfetto displays slices incorrectly if overlapping
                        // slices on the same track are not completely enveloped.
                        auto next = std::next(it);
                        if(next != qitr.second.end() &&
                           (*next)->start_timestamp < (*it)->end_timestamp)
                        {
                            auto start = (*next)->start_timestamp;
                            auto end   = std::min((*it)->end_timestamp, (*next)->end_timestamp);
                            auto mid   = start + (end - start) / 2;
                            // Report changed timestamps to ROCP INFO
                            ROCP_INFO << fmt::format(
                                "Kernel ending timestamp increased by {} ns to {} ns with "
                                "following kernel starting timestamp decreased by {} ns to {} ns "
                                "due to firmware timestamp error.",
                                ((*it)->end_timestamp - mid),
                                mid,
                                (mid - (*next)->start_timestamp),
                                mid);
                            (*it)->end_timestamp     = mid;
                            (*next)->start_timestamp = mid;
                        }

                        if(demangled.find(name) == demangled.end())
                        {
                            demangled.emplace(name, common::cxx_demangle(name));
                        }
                        // Queue IDs are 1 higher than the track name. Subtracting 1 for consistency
                        auto queue_id = info.queue_id.handle > 0 ? info.queue_id.handle - 1 : 0;

                        TRACE_EVENT_BEGIN(
                            sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                            ::perfetto::StaticString(demangled.at(name).c_str()),
                            *_track,
                            current.start_timestamp,
                            ::perfetto::Flow::ProcessScoped(current.correlation_id.internal),
                            "begin_ns",
                            current.start_timestamp,
                            "end_ns",
                            current.end_timestamp,
                            "delta_ns",
                            (current.end_timestamp - current.start_timestamp),
                            "kind",
                            current.kind,
                            "agent",
                            tool_metadata
                                .get_agent_index(agents_map.at(info.agent_id).id,
                                                 ocfg.agent_index_value)
                                .as_string("-"),
                            "agent_type",
                            tool_metadata
                                .get_agent_index(agents_map.at(info.agent_id).id,
                                                 ocfg.agent_index_value)
                                .type,
                            "corr_id",
                            current.correlation_id.internal,
                            "queue",
                            queue_id,
                            "tid",
                            current.thread_id,
                            "kernel_id",
                            info.kernel_id,
                            "Scratch_Size",
                            info.private_segment_size,
                            "LDS_Block_Size",
                            info.group_segment_size,
                            "VGPR_Count",
                            sym->arch_vgpr_count,
                            "Accum_VGPR_Count",
                            sym->accum_vgpr_count,
                            "SGPR_Count",
                            sym->sgpr_count,
                            "workgroup_size",
                            info.workgroup_size.x * info.workgroup_size.y * info.workgroup_size.z,
                            "grid_size",
                            info.grid_size.x * info.grid_size.y * info.grid_size.z,
                            "stream_ID",
                            stream_id.handle,
                            [&](::perfetto::EventContext ctx) {
                                auto corr_id    = current.correlation_id.internal;
                                auto counter_it = dispatch_counter_id_value.find(corr_id);
                                if(counter_it != dispatch_counter_id_value.end())
                                {
                                    for(auto& [counter_id, counter_value] : counter_it->second)
                                    {
                                        auto name_it = counter_id_to_name.find(counter_id);
                                        if(name_it != counter_id_to_name.end())
                                        {
                                            sdk::add_perfetto_annotation(
                                                ctx, name_it->second, counter_value);
                                        }
                                    }
                                }
                            });
                        TRACE_EVENT_END(
                            sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                            *_track,
                            current.end_timestamp);
                        tracing_session->FlushBlocking();
                    }
                }
            }
        }
    }

    // counter tracks
    {
        // memory copy counter track
        auto mem_cpy_endpoints =
            std::map<rocprofiler_agent_id_t, std::map<rocprofiler_timestamp_t, uint64_t>>{};
        auto mem_cpy_extremes = std::pair<uint64_t, uint64_t>{std::numeric_limits<uint64_t>::max(),
                                                              std::numeric_limits<uint64_t>::min()};
        auto constexpr timestamp_buffer = 1000;
        for(auto ditr : memory_copy_gen)
            for(auto itr : memory_copy_gen.get(ditr))
            {
                uint64_t _mean_timestamp =
                    itr.start_timestamp + (0.5 * (itr.end_timestamp - itr.start_timestamp));

                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.start_timestamp - timestamp_buffer,
                                                            0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.start_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(_mean_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.end_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_id].emplace(itr.end_timestamp + timestamp_buffer,
                                                            0);

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

        constexpr auto bytes_multiplier         = 1024;
        constexpr auto extremes_endpoint_buffer = 5000;

        auto mem_cpy_tracks =
            std::unordered_map<rocprofiler_agent_id_t, ::perfetto::CounterTrack>{};
        auto mem_cpy_cnt_names = std::vector<std::string>{};
        mem_cpy_cnt_names.reserve(mem_cpy_endpoints.size());
        for(auto& mitr : mem_cpy_endpoints)
        {
            mem_cpy_endpoints[mitr.first].emplace(mem_cpy_extremes.first - extremes_endpoint_buffer,
                                                  0);
            mem_cpy_endpoints[mitr.first].emplace(
                mem_cpy_extremes.second + extremes_endpoint_buffer, 0);

            auto        _track_name = std::stringstream{};
            const auto* _agent      = _get_agent(mitr.first);
            auto        agent_index_info =
                tool_metadata.get_agent_index(_agent->id, ocfg.agent_index_value);
            _track_name << "COPY BYTES to " << agent_index_info.label << " ["
                        << agent_index_info.index << "] (" << agent_index_info.type << ")";

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
        constexpr auto null_rocp_agent_id = rocprofiler_agent_id_t{.handle = 0};
        struct free_memory_information
        {
            rocprofiler_timestamp_t start_timestamp = 0;
            rocprofiler_timestamp_t end_timestamp   = 0;
            rocprofiler_address_t   address         = {.handle = 0};
        };

        struct memory_information
        {
            uint64_t              alloc_size  = {0};
            rocprofiler_address_t address     = {.handle = 0};
            bool                  is_alloc_op = {false};
        };

        struct agent_and_size
        {
            rocprofiler_agent_id_t agent_id = rocprofiler_agent_id_t{.handle = 0};
            uint64_t               size     = {0};
        };

        auto mem_alloc_endpoints =
            std::unordered_map<rocprofiler_agent_id_t,
                               std::map<rocprofiler_timestamp_t, memory_information>>{};
        auto mem_alloc_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};
        auto address_to_agent_and_size =
            std::unordered_map<rocprofiler_address_t, agent_and_size>{};
        auto free_mem_info = std::vector<free_memory_information>{};

        // Load memory allocation endpoints
        for(auto ditr : memory_allocation_gen)
            for(auto itr : memory_allocation_gen.get(ditr))
            {
                if(itr.operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE ||
                   itr.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE)
                {
                    LOG_IF(FATAL, itr.agent_id == null_rocp_agent_id)
                        << "Missing agent id for memory allocation trace";
                    mem_alloc_endpoints[itr.agent_id].emplace(
                        itr.start_timestamp,
                        memory_information{itr.allocation_size, itr.address, true});
                    mem_alloc_endpoints[itr.agent_id].emplace(
                        itr.end_timestamp,
                        memory_information{itr.allocation_size, itr.address, true});
                    address_to_agent_and_size.emplace(
                        itr.address, agent_and_size{itr.agent_id, itr.allocation_size});
                }
                else if(itr.operation == ROCPROFILER_MEMORY_ALLOCATION_FREE ||
                        itr.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE)
                {
                    // Store free memory operations in seperate vector to pair with agent
                    // and allocation size in following loop
                    free_mem_info.push_back(free_memory_information{
                        itr.start_timestamp, itr.end_timestamp, itr.address});
                }
                else
                {
                    ROCP_CI_LOG(WARNING) << "unhandled memory allocation type " << itr.operation;
                }
            }
        // Add free memory operations to the endpoint map
        for(const auto& itr : free_mem_info)
        {
            if(address_to_agent_and_size.count(itr.address) == 0)
            {
                if(itr.address.handle == 0)
                {
                    // Freeing null pointers is expected behavior and is occurs in HSA functions
                    // like hipStreamDestroy
                    ROCP_INFO << "null pointer freed due to HSA operation";
                }
                else
                {
                    // Following should not occur
                    ROCP_INFO << "Unpaired free operation occurred";
                }
                continue;
            }
            auto [agent_id, allocation_size] = address_to_agent_and_size[itr.address];
            mem_alloc_endpoints[agent_id].emplace(
                itr.start_timestamp, memory_information{allocation_size, itr.address, false});
            mem_alloc_endpoints[agent_id].emplace(
                itr.end_timestamp, memory_information{allocation_size, itr.address, false});
        }
        // Create running sum of allocated memory
        for(auto& [_, endpoint_map] : mem_alloc_endpoints)
        {
            if(!endpoint_map.empty())
            {
                auto earliest_agent_timestamp = endpoint_map.begin()->first;
                auto latest_agent_timestamp   = (--endpoint_map.end())->first;
                mem_alloc_extremes =
                    std::make_pair(std::min(mem_alloc_extremes.first, earliest_agent_timestamp),
                                   std::max(mem_alloc_extremes.second, latest_agent_timestamp));
            }
            if(endpoint_map.size() <= 1)
            {
                continue;
            }

            auto prev = endpoint_map.begin();
            auto itr  = std::next(prev);
            for(; itr != endpoint_map.end(); ++itr, ++prev)
            {
                // If address or allocation type are different, add or subtract from running sum
                if(prev->second.address != itr->second.address ||
                   prev->second.is_alloc_op != itr->second.is_alloc_op)
                {
                    if(itr->second.is_alloc_op)
                    {
                        itr->second.alloc_size += prev->second.alloc_size;
                    }
                    else if(prev->second.alloc_size >= itr->second.alloc_size)
                    {
                        itr->second.alloc_size = prev->second.alloc_size - itr->second.alloc_size;
                    }
                }
                else
                {
                    itr->second.alloc_size = prev->second.alloc_size;
                }
            }
        }

        auto mem_alloc_tracks =
            std::unordered_map<rocprofiler_agent_id_t, ::perfetto::CounterTrack>{};
        auto mem_alloc_cnt_names = std::vector<std::string>{};
        mem_alloc_cnt_names.reserve(mem_alloc_endpoints.size());
        for(auto& alloc_itr : mem_alloc_endpoints)
        {
            mem_alloc_endpoints[alloc_itr.first].emplace(
                mem_alloc_extremes.first - extremes_endpoint_buffer,
                memory_information{0, {0}, false});
            mem_alloc_endpoints[alloc_itr.first].emplace(
                mem_alloc_extremes.second + extremes_endpoint_buffer,
                memory_information{0, {0}, false});

            auto                       _track_name = std::stringstream{};
            const rocprofiler_agent_t* _agent      = _get_agent(alloc_itr.first);

            if(_agent != nullptr)
            {
                auto agent_index_info =
                    tool_metadata.get_agent_index(_agent->id, ocfg.agent_index_value);
                _track_name << "ALLOCATE BYTES on " << agent_index_info.label << " ["
                            << agent_index_info.index << "] (" << agent_index_info.type << ")";
            }
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
                              itr.second.alloc_size / bytes_multiplier);
                tracing_session->FlushBlocking();
            }
        }

        // scratch memory counter track
        auto scratch_mem_endpoints =
            std::unordered_map<rocprofiler_agent_id_t,
                               std::map<rocprofiler_timestamp_t, uint64_t>>{};
        auto scratch_mem_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};

        // Load scratch memory usage endpoints
        for(auto ditr : scratch_memory_gen)
            for(auto itr : scratch_memory_gen.get(ditr))
            {
                // Track start and end timestamps for this scratch memory record
                scratch_mem_endpoints[itr.agent_id].emplace(itr.start_timestamp, 0);
                scratch_mem_endpoints[itr.agent_id].emplace(itr.end_timestamp, 0);

                // Update overall time range
                scratch_mem_extremes =
                    std::make_pair(std::min(scratch_mem_extremes.first, itr.start_timestamp),
                                   std::max(scratch_mem_extremes.second, itr.end_timestamp));
            }

        // Load values at each endpoint
        for(auto ditr : scratch_memory_gen)
            for(auto itr : scratch_memory_gen.get(ditr))
            {
                // For each timestamp in the range of this record
                auto begin =
                    scratch_mem_endpoints.at(itr.agent_id).lower_bound(itr.start_timestamp);
                auto end = scratch_mem_endpoints.at(itr.agent_id).upper_bound(itr.end_timestamp);

                for(auto mitr = begin; mitr != end; ++mitr)
                {
                    // Add scratch memory size to the counter value at this timestamp
                    if(itr.operation == ROCPROFILER_SCRATCH_MEMORY_ALLOC)
                        mitr->second = itr.allocation_size;
                    else if(itr.operation == ROCPROFILER_SCRATCH_MEMORY_FREE)
                        mitr->second = 0;  // For all free events current allocation drops to 0.
                }
            }

        // Create counter tracks for visualization
        auto scratch_mem_tracks =
            std::unordered_map<rocprofiler_agent_id_t, ::perfetto::CounterTrack>{};
        auto scratch_mem_names = std::vector<std::string>{};
        scratch_mem_names.reserve(scratch_mem_endpoints.size());

        for(auto& mitr : scratch_mem_endpoints)
        {
            // Add buffer timestamps for better visualization
            if(!mitr.second.empty())
            {
                scratch_mem_endpoints[mitr.first].emplace(
                    scratch_mem_extremes.first - extremes_endpoint_buffer, 0);
                scratch_mem_endpoints[mitr.first].emplace(
                    scratch_mem_extremes.second + extremes_endpoint_buffer, 0);

                auto        _track_name = std::stringstream{};
                const auto* _agent      = _get_agent(mitr.first);
                auto        agent_index_info =
                    tool_metadata.get_agent_index(_agent->id, ocfg.agent_index_value);
                _track_name << "SCRATCH MEMORY on " << agent_index_info.label << " ["
                            << agent_index_info.index << "] (" << agent_index_info.type << ")";

                constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
                auto&          _name = scratch_mem_names.emplace_back(_track_name.str());
                scratch_mem_tracks.emplace(mitr.first,
                                           ::perfetto::CounterTrack{_name.c_str()}
                                               .set_unit(_unit)
                                               .set_unit_multiplier(bytes_multiplier)
                                               .set_is_incremental(false));
            }
        }

        // Write counter values to perfetto trace
        for(auto& mitr : scratch_mem_endpoints)
        {
            if(scratch_mem_tracks.count(mitr.first) > 0)
            {
                for(auto itr : mitr.second)
                {
                    TRACE_COUNTER(sdk::perfetto_category<sdk::category::scratch_memory>::name,
                                  scratch_mem_tracks.at(mitr.first),
                                  itr.first,
                                  itr.second / bytes_multiplier);
                    tracing_session->FlushBlocking();

                    PERFETTO_LOG("TRACE_COUNTER rocprofv3 %s %llu %llu %llu %llu",
                                 sdk::perfetto_category<sdk::category::scratch_memory>::name,
                                 (unsigned long long) (mitr.first.handle),
                                 (unsigned long long) (scratch_mem_tracks.at(mitr.first).uuid),
                                 (unsigned long long) itr.first,
                                 (unsigned long long) itr.second);
                }
            }
        }
    }

    // Create counter tracks per agent
    {
        auto counters_endpoints = std::unordered_map<
            rocprofiler_agent_id_t,
            std::unordered_map<rocprofiler_counter_id_t, std::map<uint64_t, uint64_t>>>{};

        auto counters_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};

        auto constexpr timestamp_buffer = 1000;

        for(auto ditr : counter_collection_gen)
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                const auto& info = record.dispatch_data.dispatch_info;

                const auto& start_timestamp = record.dispatch_data.start_timestamp;
                const auto& end_timestamp   = record.dispatch_data.end_timestamp;

                uint64_t _mean_timestamp =
                    start_timestamp + (0.5 * (end_timestamp - start_timestamp));

                auto corr_id = record.dispatch_data.correlation_id.internal;
                auto it      = dispatch_counter_id_value.find(corr_id);
                if(it != dispatch_counter_id_value.end())
                {
                    for(auto& [counter_id, counter_value] : it->second)
                    {
                        counters_endpoints[info.agent_id][counter_id].emplace(
                            start_timestamp - timestamp_buffer, 0);
                        counters_endpoints[info.agent_id][counter_id].emplace(start_timestamp,
                                                                              counter_value);
                        counters_endpoints[info.agent_id][counter_id].emplace(_mean_timestamp,
                                                                              counter_value);
                        counters_endpoints[info.agent_id][counter_id].emplace(end_timestamp, 0);
                        counters_endpoints[info.agent_id][counter_id].emplace(
                            end_timestamp + timestamp_buffer, 0);
                    }
                }

                counters_extremes = std::make_pair(
                    std::min(counters_extremes.first, record.dispatch_data.start_timestamp),
                    std::max(counters_extremes.second, record.dispatch_data.end_timestamp));
            }

        auto counter_tracks = std::unordered_map<rocprofiler_agent_id_t,
                                                 std::map<std::string, ::perfetto::CounterTrack>>{};

        constexpr auto extremes_endpoint_buffer = 5000;

        for(auto ditr : counter_collection_gen)
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                const auto& info = record.dispatch_data.dispatch_info;
                const auto& sym  = tool_metadata.get_kernel_symbol(info.kernel_id);

                CHECK(sym != nullptr);

                auto name = sym->formatted_kernel_name;

                auto corr_id = record.dispatch_data.correlation_id.internal;
                auto it      = dispatch_counter_id_value.find(corr_id);
                if(it != dispatch_counter_id_value.end())
                {
                    for(auto& [counter_id, counter_value] : it->second)
                    {
                        counters_endpoints[info.agent_id][counter_id].emplace(
                            counters_extremes.first - extremes_endpoint_buffer, 0);
                        counters_endpoints[info.agent_id][counter_id].emplace(
                            counters_extremes.second + extremes_endpoint_buffer, 0);

                        auto agent_index_info =
                            tool_metadata.get_agent_index(info.agent_id, ocfg.agent_index_value);
                        auto track_name_ss = std::stringstream{};
                        track_name_ss << agent_index_info.label << " [" << agent_index_info.index
                                      << "] "
                                      << "PMC " << counter_id_to_name.at(counter_id);

                        auto track_name = track_name_ss.str();

                        counter_tracks[info.agent_id].emplace(
                            track_name, ::perfetto::CounterTrack(track_name.c_str()));
                        auto& endpoints = counters_endpoints[info.agent_id][counter_id];
                        for(auto& counter_itr : endpoints)
                        {
                            TRACE_COUNTER(
                                sdk::perfetto_category<sdk::category::counter_collection>::name,
                                counter_tracks[info.agent_id].at(track_name),
                                counter_itr.first,
                                counter_itr.second);
                            tracing_session->FlushBlocking();
                        }
                    }
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
