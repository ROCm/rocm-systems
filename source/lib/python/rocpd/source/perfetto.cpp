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

#include "lib/python/rocpd/source/perfetto.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/mpl.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <fmt/format.h>

#include <atomic>
#include <future>
#include <mutex>
#include <vector>

namespace rocpd
{
namespace output
{
namespace
{
auto
to_string(rocprofiler_dim3_t v)
{
    return fmt::format("{} (x={}, y={}, z={})", (v.x * v.y * v.z), v.x, v.y, v.z);
}

template <typename Tp>
size_t
get_hash_id(Tp&& _val)
{
    using value_type = rocprofiler::common::mpl::unqualified_type_t<Tp>;

    if constexpr(!std::is_pointer<Tp>::value)
        return std::hash<value_type>{}(std::forward<Tp>(_val));
    else if constexpr(std::is_same<Tp, const char*>::value)
        return get_hash_id(std::string_view{_val});
    else
        return get_hash_id(*_val);
}
}  // namespace

PerfettoSession::PerfettoSession(const tool::output_config& output_cfg)
: config{output_cfg}
{
    auto args            = ::perfetto::TracingInitArgs{};
    auto track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto cfg             = ::perfetto::TraceConfig{};

    // environment settings
    auto shmem_size_hint = config.perfetto_shmem_size_hint;
    auto buffer_size_kb  = config.perfetto_buffer_size;

    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size_kb);

    args.supports_multiple_data_source_instances = true;
    // track_event_cfg.clear_disabled_categories();
    // track_event_cfg.clear_disabled_tags();

    if(config.perfetto_buffer_fill_policy == "discard" ||
       config.perfetto_buffer_fill_policy.empty())
        buffer_config->set_fill_policy(
            ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);
    else if(config.perfetto_buffer_fill_policy == "ring_buffer")
        buffer_config->set_fill_policy(
            ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER);
    else
        ROCP_FATAL << "Unsupport perfetto buffer fill policy: '"
                   << config.perfetto_buffer_fill_policy << "'. Supported: discard, ring_buffer";

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.shmem_size_hint_kb = shmem_size_hint;

    if(config.perfetto_backend == "inprocess" || config.perfetto_backend.empty())
        args.backends |= ::perfetto::kInProcessBackend;
    else if(config.perfetto_backend == "system")
        args.backends |= ::perfetto::kSystemBackend;
    else
        ROCP_FATAL << "Unsupport perfetto backend: '" << config.perfetto_backend
                   << "'. Supported: inprocess, system";

    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();

    tracing_session = ::perfetto::Tracing::NewTrace();

    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
}

PerfettoSession::~PerfettoSession()
{
    tracing_session->StopBlocking();
    auto filename = std::string{"results"};
    auto ofs      = tool::get_output_stream(config, filename, ".pftrace", std::ios::binary);

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

void
write_perfetto(
    const PerfettoSession& perfetto_session,
    const types::process&  process,
    const std::unordered_map<uint64_t, std::pair<rocpd::types::agent, tool::agent_index>>&
                                                     agent_data,
    const tool::generator<types::thread>&            thread_gen,
    const tool::generator<types::region>&            region_gen,
    const tool::generator<types::sample>&            sample_gen,
    const tool::generator<types::kernel_dispatch>&   kernel_dispatch_gen,
    const tool::generator<types::memory_copies>&     memory_copy_gen,
    const tool::generator<types::scratch_memory>&    scratch_memory_gen,
    const tool::generator<types::memory_allocation>& memory_allocation_gen,
    const tool::generator<types::counter>&           counter_collection_gen)
{
    namespace sdk    = ::rocprofiler::sdk;
    namespace common = ::rocprofiler::common;

    static auto orig_process_track = ::perfetto::ProcessTrack::Current();
    static auto orig_process_desc  = orig_process_track.Serialize();

    const auto&    tracing_session  = perfetto_session.tracing_session;
    const auto&    ocfg             = perfetto_session.config;
    const uint64_t this_pid         = process.pid;
    const uint64_t this_pid_init_ns = process.init;
    const uint64_t this_nid         = process.nid;
    auto           command_line     = ::rocprofiler::sdk::parse::tokenize(process.command, " ");

    auto uuid_pid       = common::fnv1a_hasher::combine(this_nid, this_pid_init_ns, this_pid);
    auto this_pid_track = ::perfetto::Track{uuid_pid, ::perfetto::Track{}};

    {
        auto desc = orig_process_desc;
        desc.set_uuid(uuid_pid);
        desc.set_parent_uuid(0);
        desc.mutable_process()->set_pid(this_pid);
        desc.mutable_process()->set_start_timestamp_ns(this_pid_init_ns);

        desc.mutable_process()->set_process_name(command_line.front());
        desc.mutable_process()->clear_cmdline();
        for(const auto& itr : command_line)
            desc.mutable_process()->add_cmdline(itr);

        ::perfetto::TrackEvent::SetTrackDescriptor(this_pid_track, desc);
    }

    auto agent_thread_ids       = std::unordered_map<uint64_t, std::set<uint64_t>>{};
    auto agent_thread_ids_alloc = std::unordered_map<uint64_t, std::set<uint64_t>>{};
    auto agent_queue_ids =
        std::unordered_map<uint64_t, std::unordered_set<rocprofiler_queue_id_t>>{};
    auto agent_stream_ids = std::unordered_set<rocprofiler_stream_id_t>{};
    auto thread_indexes   = std::unordered_map<uint64_t, uint64_t>{};

    auto thread_tracks = std::unordered_map<uint64_t, ::perfetto::Track>{};
    auto agent_thread_tracks =
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, ::perfetto::Track>>{};
    auto agent_queue_tracks =
        std::unordered_map<uint64_t,
                           std::unordered_map<rocprofiler_queue_id_t, ::perfetto::Track>>{};
    auto stream_tracks = std::unordered_map<rocprofiler_stream_id_t, ::perfetto::Track>{};

    {
        for(auto ditr : memory_copy_gen)
            for(const auto& itr : memory_copy_gen.get(ditr))
            {
                auto stream_id = rocprofiler_stream_id_t{.handle = itr.stream_id};
                agent_stream_ids.emplace(stream_id);
                if(ocfg.group_by_queue)
                {
                    agent_thread_ids[itr.dst_agent_abs_index].emplace(itr.tid);
                }
            }
    }

    for(auto ditr : memory_allocation_gen)
        for(const auto& itr : memory_allocation_gen.get(ditr))
        {
            agent_thread_ids_alloc[itr.agent_abs_index].emplace(itr.tid);
        }

    {
        for(auto ditr : kernel_dispatch_gen)
            for(const auto& itr : kernel_dispatch_gen.get(ditr))
            {
                auto stream_id = rocprofiler_stream_id_t{.handle = itr.stream_id};
                auto queue_id  = rocprofiler_queue_id_t{.handle = itr.queue_id};
                agent_stream_ids.emplace(stream_id);
                if(ocfg.group_by_queue)
                {
                    agent_queue_ids[itr.agent_abs_index].emplace(queue_id);
                }
            }
    }

    uint64_t nthrn = 0;
    for(auto ditr : thread_gen)
        for(const auto& itr : thread_gen.get(ditr))
        {
            auto is_main_thread = (static_cast<uint64_t>(itr.tid) == this_pid);
            auto _idx           = (is_main_thread) ? 0 : ++nthrn;
            thread_indexes.emplace(itr.tid, _idx);
            auto _track = ::perfetto::Track{static_cast<uint64_t>(itr.tid), this_pid_track};
            auto _desc  = _track.Serialize();
            if(is_main_thread)
                _desc.set_name(fmt::format("{}", ::basename(command_line.front().c_str())));
            else
                _desc.set_name(fmt::format("THREAD {}", _idx));
            _desc.mutable_thread()->set_pid(this_pid);
            _desc.mutable_thread()->set_tid(itr.tid);
            if(is_main_thread)
                _desc.mutable_thread()->set_thread_name(
                    fmt::format("{}", ::basename(command_line.front().c_str())));
            else
                _desc.mutable_thread()->set_thread_name(fmt::format("THREAD {}", _idx));
            ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            thread_tracks.emplace(itr.tid, _track);
        }

    for(const auto& [abs_index, thread_ids] : agent_thread_ids)
    {
        const auto _agent = agent_data.at(abs_index).first;

        for(auto titr : thread_ids)
        {
            auto _namess = std::stringstream{};
            _namess << "COPY to AGENT [" << _agent.logical_node_id << "] THREAD ["
                    << thread_indexes.at(titr) << "] ";

            if(_agent.type == "CPU")
                _namess << "(CPU)";
            else if(_agent.type == "GPU")
                _namess << "(GPU)";
            else
                _namess << "(UNK)";

            auto _track = ::perfetto::Track{get_hash_id(_namess.str()), this_pid_track};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_thread_tracks[abs_index].emplace(titr, _track);
        }
    }

    for(const auto& [abs_index, queue_ids] : agent_queue_ids)
    {
        uint32_t   nqueue           = 0;
        const auto _agent           = agent_data.at(abs_index).first;
        auto       agent_index_info = agent_data.at(abs_index).second;

        for(auto qitr : queue_ids)
        {
            auto _namess = std::stringstream{};

            _namess << "COMPUTE " << agent_index_info.label << " [" << agent_index_info.index
                    << "] QUEUE [" << nqueue++ << "] ";
            _namess << agent_index_info.type;

            auto _track = ::perfetto::Track{get_hash_id(_namess.str()), this_pid_track};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_queue_tracks[abs_index].emplace(qitr, _track);
        }
    }

    for(const auto& sitr : agent_stream_ids)
    {
        const auto stream_id = sitr.handle;

        auto _name = fmt::format("STREAM [{}]", stream_id);

        auto _track = ::perfetto::Track{get_hash_id(_name), this_pid_track};
        auto _desc  = _track.Serialize();
        _desc.set_name(_name);

        ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        stream_tracks.emplace(sitr, _track);
    }

    // Fetch counter values
    auto counter_id_value = std::map<uint32_t, double>{};
    auto counter_id_name  = std::map<uint32_t, std::string>{};
    for(auto ditr : counter_collection_gen)
        for(const auto& record : counter_collection_gen.get(ditr))
        {
            // Accumulate counters based on ID
            counter_id_value[record.counter_id] += record.value;
            counter_id_name[record.counter_id] = std::string{record.counter_name};
        }

    // trace events
    {
        auto get_category_string = [](std::string_view _category) {
            static auto buffer_names  = sdk::get_buffer_tracing_names();
            auto        _category_idx = ROCPROFILER_BUFFER_TRACING_NONE;
            for(const auto& citr : buffer_names)
            {
                if(_category == citr.name) _category_idx = citr.value;
            }
            return sdk::get_perfetto_category(_category_idx);
        };

        for(auto ditr : region_gen)
        {
            for(auto itr : region_gen.get(ditr))
            {
                auto& track = thread_tracks.at(itr.tid);
                auto  _name = itr.name;

                if(itr.has_extdata())
                {
                    if(auto _extdata = itr.get_extdata(); !_extdata.message.empty())
                        _name = _extdata.message;
                }

                auto _category = ::perfetto::DynamicCategory{get_category_string(itr.category)};
                TRACE_EVENT_BEGIN(_category,
                                  ::perfetto::DynamicString{_name},
                                  track,
                                  itr.start,
                                  ::perfetto::Flow::Global(itr.stack_id ^ uuid_pid),
                                  "begin_ns",
                                  itr.start,
                                  "end_ns",
                                  itr.end,
                                  "delta_ns",
                                  (itr.end - itr.start),
                                  "tid",
                                  itr.tid,
                                  "kind",
                                  itr.category,
                                  "operation",
                                  _name,
                                  "corr_id",
                                  itr.stack_id,
                                  "ancestor_id",
                                  itr.parent_stack_id,
                                  [&](::perfetto::EventContext ctx) { (void) ctx; });

                TRACE_EVENT_END(_category, track, itr.end);

                tracing_session->FlushBlocking();
            }
        }

        for(auto ditr : sample_gen)
        {
            for(auto itr : sample_gen.get(ditr))
            {
                auto& track = thread_tracks.at(itr.tid);
                auto  _name = itr.name;

                if(itr.has_extdata())
                {
                    if(auto _extdata = itr.get_extdata(); !_extdata.message.empty())
                        _name = _extdata.message;
                }

                auto _category = ::perfetto::DynamicCategory{get_category_string(itr.category)};
                TRACE_EVENT_INSTANT(_category,
                                    ::perfetto::DynamicString{_name},
                                    track,
                                    itr.timestamp,
                                    ::perfetto::Flow::Global(itr.stack_id ^ uuid_pid),
                                    "begin_ns",
                                    itr.timestamp,
                                    "end_ns",
                                    itr.timestamp,
                                    "delta_ns",
                                    0,
                                    "tid",
                                    itr.tid,
                                    "kind",
                                    itr.category,
                                    "operation",
                                    _name,
                                    "corr_id",
                                    itr.stack_id,
                                    "ancestor_id",
                                    itr.parent_stack_id,
                                    [&](::perfetto::EventContext ctx) { (void) ctx; });

                tracing_session->FlushBlocking();
            }
        }

        for(auto ditr : memory_copy_gen)
        {
            for(auto itr : memory_copy_gen.get(ditr))
            {
                ::perfetto::Track* _track = nullptr;
                if(ocfg.group_by_queue)
                {
                    _track = &agent_thread_tracks.at(itr.dst_agent_abs_index).at(itr.tid);
                }
                else
                {
                    auto stream_id = rocprofiler_stream_id_t{.handle = itr.stream_id};
                    _track         = &stream_tracks.at(stream_id);
                }

                auto src_agent_index = agent_data.at(itr.src_agent_abs_index).second;
                auto dst_agent_index = agent_data.at(itr.dst_agent_abs_index).second;
                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::memory_copy>::name,
                                  ::perfetto::DynamicString{itr.name},
                                  *_track,
                                  itr.start,
                                  ::perfetto::Flow::Global(itr.stack_id ^ uuid_pid),
                                  "begin_ns",
                                  itr.start,
                                  "end_ns",
                                  itr.end,
                                  "delta_ns",
                                  (itr.end - itr.start),
                                  "kind",
                                  itr.category,
                                  "operation",
                                  itr.name,
                                  "src_agent",
                                  src_agent_index.as_string("-"),
                                  "dst_agent",
                                  dst_agent_index.as_string("-"),
                                  "copy_bytes",
                                  itr.size,
                                  "corr_id",
                                  itr.stack_id,
                                  "tid",
                                  itr.tid,
                                  "stream_id",
                                  itr.stream_id);
                TRACE_EVENT_END(
                    sdk::perfetto_category<sdk::category::memory_copy>::name, *_track, itr.end);
            }
            tracing_session->FlushBlocking();
        }

        for(auto ditr : kernel_dispatch_gen)
        {
            auto gen = kernel_dispatch_gen.get(ditr);
            for(auto it = begin(gen); it != end(gen); ++it)
            {
                auto& current = *it;

                ::perfetto::Track* _track    = nullptr;
                auto               agent_id  = current.agent_abs_index;
                auto               queue_id  = rocprofiler_queue_id_t{.handle = current.queue_id};
                auto               stream_id = rocprofiler_stream_id_t{.handle = current.stream_id};
                if(ocfg.group_by_queue)
                {
                    _track = &agent_queue_tracks.at(agent_id).at(queue_id);
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
                if(next != end(gen) && next->agent_abs_index == it->agent_abs_index &&
                   ((ocfg.group_by_queue && next->queue_id == it->queue_id) ||
                    (!ocfg.group_by_queue && next->stream_id == it->stream_id)) &&
                   next->start < it->end)
                {
                    auto start = next->start;
                    auto end   = it->end;
                    auto mid   = start + (end - start) / 2;
                    // Report changed timestamps to ROCP INFO
                    ROCP_INFO << fmt::format(
                        "Kernel ending timestamp increased by {} ns to {} ns with "
                        "following kernel starting timestamp decreased by {} ns to {} ns "
                        "due to firmware timestamp error.",
                        (it->end - mid),
                        mid,
                        (mid - next->start),
                        mid);
                    it->end     = mid;
                    next->start = mid;
                }

                auto agent_index = agent_data.at(current.agent_abs_index).second;
                auto _name =
                    (ocfg.kernel_rename && !current.region.empty()) ? current.region : current.name;
                TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                                  ::perfetto::DynamicString{_name},
                                  *_track,
                                  current.start,
                                  ::perfetto::Flow::Global(current.stack_id ^ uuid_pid),
                                  "begin_ns",
                                  current.start,
                                  "end_ns",
                                  current.end,
                                  "delta_ns",
                                  (current.end - current.start),
                                  "kind",
                                  current.category,
                                  "agent",
                                  agent_index.as_string("-"),
                                  "corr_id",
                                  current.stack_id,
                                  "queue",
                                  current.queue_id,
                                  "tid",
                                  current.tid,
                                  "kernel_id",
                                  current.kernel_id,
                                  "Scratch_Size",
                                  current.scratch_size,
                                  "LDS_Block_Size",
                                  current.lds_size,
                                  "workgroup_size",
                                  to_string(current.workgroup_size),
                                  "grid_size",
                                  to_string(current.grid_size),
                                  "stream_id",
                                  current.stream_id,
                                  [&](::perfetto::EventContext ctx) {
                                      for(auto& [counter_id, counter_value] : counter_id_value)
                                      {
                                          rocprofiler::sdk::add_perfetto_annotation(
                                              ctx, counter_id_name.at(counter_id), counter_value);
                                      }
                                  });
                TRACE_EVENT_END(sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                                *_track,
                                current.end);
            }
            tracing_session->FlushBlocking();
        }
    }

    // counter tracks
    {
        // memory copy counter track
        auto mem_cpy_endpoints = std::map<uint64_t, std::map<rocprofiler_timestamp_t, uint64_t>>{};
        auto mem_cpy_extremes  = std::pair<uint64_t, uint64_t>{std::numeric_limits<uint64_t>::max(),
                                                              std::numeric_limits<uint64_t>::min()};
        auto constexpr timestamp_buffer = 1000;
        for(auto ditr : memory_copy_gen)
        {
            for(const auto& itr : memory_copy_gen.get(ditr))
            {
                uint64_t _mean_timestamp = itr.start + (0.5 * (itr.end - itr.start));

                mem_cpy_endpoints[itr.dst_agent_abs_index].emplace(itr.start - timestamp_buffer, 0);
                mem_cpy_endpoints[itr.dst_agent_abs_index].emplace(itr.start, 0);
                mem_cpy_endpoints[itr.dst_agent_abs_index].emplace(_mean_timestamp, 0);
                mem_cpy_endpoints[itr.dst_agent_abs_index].emplace(itr.end, 0);
                mem_cpy_endpoints[itr.dst_agent_abs_index].emplace(itr.end + timestamp_buffer, 0);

                mem_cpy_extremes = std::make_pair(std::min(mem_cpy_extremes.first, itr.start),
                                                  std::max(mem_cpy_extremes.second, itr.end));
            }
        }

        for(auto ditr : memory_copy_gen)
        {
            for(const auto& itr : memory_copy_gen.get(ditr))
            {
                auto mbeg = mem_cpy_endpoints.at(itr.dst_agent_abs_index).lower_bound(itr.start);
                auto mend = mem_cpy_endpoints.at(itr.dst_agent_abs_index).upper_bound(itr.end);

                LOG_IF(FATAL, mbeg == mend)
                    << "Missing range for timestamp [" << itr.start << ", " << itr.end << "]";

                for(auto mitr = mbeg; mitr != mend; ++mitr)
                    mitr->second += itr.size;
            }
        }

        constexpr auto bytes_multiplier         = 1024;
        constexpr auto extremes_endpoint_buffer = 5000;

        auto mem_cpy_tracks    = std::unordered_map<uint64_t, ::perfetto::CounterTrack>{};
        auto mem_cpy_cnt_names = std::vector<std::string>{};
        mem_cpy_cnt_names.reserve(mem_cpy_endpoints.size());

        for(auto& [abs_index, ts_map] : mem_cpy_endpoints)
        {
            mem_cpy_endpoints[abs_index].emplace(mem_cpy_extremes.first - extremes_endpoint_buffer,
                                                 0);
            mem_cpy_endpoints[abs_index].emplace(mem_cpy_extremes.second + extremes_endpoint_buffer,
                                                 0);

            auto       _track_name      = std::stringstream{};
            const auto _agent           = agent_data.at(abs_index).first;
            auto       agent_index_info = agent_data.at(abs_index).second;
            _track_name << "COPY BYTES to " << agent_index_info.label << " ["
                        << agent_index_info.index << "] (" << agent_index_info.type << ")";

            constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
            auto&          _name = mem_cpy_cnt_names.emplace_back(_track_name.str());
            mem_cpy_tracks.emplace(abs_index,
                                   ::perfetto::CounterTrack{_name.c_str(), this_pid_track}
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
            }
            tracing_session->FlushBlocking();
        }

        // memory allocation counter track
        struct free_memory_information
        {
            rocprofiler_timestamp_t start_timestamp = 0;
            rocprofiler_timestamp_t end_timestamp   = 0;
            rocprofiler_address_t   address         = {.handle = 0};
            rocprofiler_queue_id_t  queue           = {.handle = 0};
        };

        struct memory_information
        {
            uint64_t               alloc_size  = {0};
            rocprofiler_address_t  address     = {.handle = 0};
            rocprofiler_queue_id_t queue       = {.handle = 0};
            bool                   is_alloc_op = {false};
        };

        struct agent_and_size
        {
            uint64_t agent_abs_index = {};
            uint64_t size            = {0};
        };

        auto mem_alloc_endpoints =
            std::unordered_map<uint64_t, std::map<rocprofiler_timestamp_t, memory_information>>{};
        auto mem_alloc_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};
        auto address_to_agent_and_size =
            std::unordered_map<rocprofiler_address_t, agent_and_size>{};
        auto queue_to_agent_and_size = std::unordered_map<rocprofiler_queue_id_t, agent_and_size>{};
        auto free_mem_info           = std::vector<free_memory_information>{};

        // Load memory allocation endpoints
        for(auto ditr : memory_allocation_gen)
        {
            for(const auto& itr : memory_allocation_gen.get(ditr))
            {
                if(itr.type == "ALLOC")
                {
                    LOG_IF(FATAL, itr.agent_name.empty())
                        << "Missing agent id for memory allocation trace";

                    if(itr.level == "REAL")
                    {
                        mem_alloc_endpoints[itr.agent_abs_index].emplace(
                            itr.start,
                            memory_information{itr.size,
                                               rocprofiler_address_t{.handle = itr.address},
                                               rocprofiler_queue_id_t{.handle = itr.queue_id},
                                               true});
                        mem_alloc_endpoints[itr.agent_abs_index].emplace(
                            itr.end,
                            memory_information{itr.size,
                                               rocprofiler_address_t{.handle = itr.address},
                                               rocprofiler_queue_id_t{.handle = itr.queue_id},
                                               true});
                    }

                    // Agent and allocation size are stored to pair with free memory operations
                    if(itr.level == "REAL")
                    {
                        address_to_agent_and_size.emplace(
                            rocprofiler_address_t{.handle = itr.address},
                            agent_and_size{itr.agent_abs_index, itr.size});
                    }
                    // Scratch memory operations operations are indexed by queue id as agent
                    // id is not available
                    else if(itr.level == "SCRATCH")
                    {
                        queue_to_agent_and_size.emplace(
                            rocprofiler_queue_id_t{.handle = itr.queue_id},
                            agent_and_size{itr.agent_abs_index, itr.size});
                    }
                }
                else if(itr.type == "FREE")
                {
                    // Store free memory operations in seperate vector to pair with agent
                    // and allocation size in following loop
                    if(itr.level == "REAL")
                    {
                        free_mem_info.push_back(free_memory_information{
                            itr.start,
                            itr.end,
                            rocprofiler_address_t{.handle = itr.address},
                            rocprofiler_queue_id_t{.handle = itr.queue_id}});
                    }
                }
                else
                {
                    ROCP_CI_LOG(WARNING) << "unhandled memory allocation type " << itr.type;
                }
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
            auto [agent_abs_index, size] = address_to_agent_and_size[itr.address];
            mem_alloc_endpoints[agent_abs_index].emplace(
                itr.start_timestamp, memory_information{size, itr.address, itr.queue, false});
            mem_alloc_endpoints[agent_abs_index].emplace(
                itr.end_timestamp, memory_information{size, itr.address, itr.queue, false});
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

        auto mem_alloc_tracks    = std::unordered_map<uint64_t, ::perfetto::CounterTrack>{};
        auto mem_alloc_cnt_names = std::vector<std::string>{};
        mem_alloc_cnt_names.reserve(mem_alloc_endpoints.size());

        for(auto& [abs_index, ts_map] : mem_alloc_endpoints)
        {
            mem_alloc_endpoints[abs_index].emplace(
                mem_alloc_extremes.first - extremes_endpoint_buffer,
                memory_information{0, {0}, {0}, false});
            mem_alloc_endpoints[abs_index].emplace(
                mem_alloc_extremes.second + extremes_endpoint_buffer,
                memory_information{0, {0}, {0}, false});

            auto _track_name = std::stringstream{};

            if(agent_data.find(abs_index) != agent_data.end())
            {
                const auto _agent           = agent_data.at(abs_index).first;
                auto       agent_index_info = agent_data.at(abs_index).second;
                _track_name << "ALLOCATE BYTES on " << agent_index_info.label << " ["
                            << agent_index_info.index << "] (" << agent_index_info.type << ")";
            }
            else
            {
                _track_name << "FREE BYTES";
            }

            constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
            auto&          _name = mem_alloc_cnt_names.emplace_back(_track_name.str());
            mem_alloc_tracks.emplace(abs_index,
                                     ::perfetto::CounterTrack{_name.c_str(), this_pid_track}
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
            std::unordered_map<uint64_t, std::map<rocprofiler_timestamp_t, uint64_t>>{};
        auto scratch_mem_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};

        // Load scratch memory usage endpoints
        for(auto ditr : scratch_memory_gen)
            for(auto itr : scratch_memory_gen.get(ditr))
            {
                auto agent_abs_index = itr.agent_abs_index;
                if(itr.operation == "FREE")
                {
                    auto [agent_index, size] =
                        queue_to_agent_and_size[rocprofiler_queue_id_t{.handle = itr.queue_id}];
                    agent_abs_index = agent_index;
                }

                // Track start and end timestamps for this scratch memory record
                scratch_mem_endpoints[agent_abs_index].emplace(itr.start, 0);
                scratch_mem_endpoints[agent_abs_index].emplace(itr.end, 0);

                // Update overall time range
                scratch_mem_extremes =
                    std::make_pair(std::min(scratch_mem_extremes.first, itr.start),
                                   std::max(scratch_mem_extremes.second, itr.end));
            }

        // Load values at each endpoint
        for(auto ditr : scratch_memory_gen)
            for(auto itr : scratch_memory_gen.get(ditr))
            {
                auto agent_abs_index = itr.agent_abs_index;
                if(itr.operation == "FREE")
                {
                    auto [agent_index, size] =
                        queue_to_agent_and_size[rocprofiler_queue_id_t{.handle = itr.queue_id}];
                    agent_abs_index = agent_index;
                }
                // For each timestamp in the range of this record
                auto begin = scratch_mem_endpoints.at(agent_abs_index).lower_bound(itr.start);
                auto end   = scratch_mem_endpoints.at(agent_abs_index).upper_bound(itr.end);

                for(auto mitr = begin; mitr != end; ++mitr)
                {
                    // Add scratch memory size to the counter value at this timestamp
                    if(itr.operation == "ALLOC")
                        mitr->second = itr.size;
                    else if(itr.operation == "FREE")
                        mitr->second = 0;  // For all free events current allocation drops to 0.
                }
            }

        // Create counter tracks for visualization
        auto scratch_mem_tracks = std::unordered_map<uint64_t, ::perfetto::CounterTrack>{};
        auto scratch_mem_names  = std::vector<std::string>{};
        scratch_mem_names.reserve(scratch_mem_endpoints.size());

        for(auto& [abs_index, ts_map] : scratch_mem_endpoints)
        {
            // Add buffer timestamps for better visualization
            if(!ts_map.empty())
            {
                scratch_mem_endpoints[abs_index].emplace(
                    scratch_mem_extremes.first - extremes_endpoint_buffer, 0);
                scratch_mem_endpoints[abs_index].emplace(
                    scratch_mem_extremes.second + extremes_endpoint_buffer, 0);

                auto       _track_name      = std::stringstream{};
                const auto _agent           = agent_data.at(abs_index).first;
                auto       agent_index_info = agent_data.at(abs_index).second;

                _track_name << "SCRATCH MEMORY on " << agent_index_info.label << " ["
                            << agent_index_info.index << "] (" << agent_index_info.type << ")";

                constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_SIZE_BYTES;
                auto&          _name = scratch_mem_names.emplace_back(_track_name.str());
                scratch_mem_tracks.emplace(abs_index,
                                           ::perfetto::CounterTrack{_name.c_str(), this_pid_track}
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
                }
            }
        }
    }

    // Create counter tracks per agent
    {
        auto counters_endpoints =
            std::unordered_map<uint64_t,
                               std::unordered_map<uint32_t, std::map<uint64_t, uint64_t>>>{};

        auto counters_extremes = std::pair<uint64_t, uint64_t>{
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min()};

        auto constexpr timestamp_buffer = 1000;

        for(auto ditr : counter_collection_gen)
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                // const auto& info = record.;

                const auto& start_timestamp = record.start;
                const auto& end_timestamp   = record.end;

                uint64_t _mean_timestamp =
                    start_timestamp + (0.5 * (end_timestamp - start_timestamp));

                for(auto& [counter_id, counter_value] : counter_id_value)
                {
                    counters_endpoints[record.agent_abs_index][counter_id].emplace(
                        start_timestamp - timestamp_buffer, 0);
                    counters_endpoints[record.agent_abs_index][counter_id].emplace(start_timestamp,
                                                                                   counter_value);
                    counters_endpoints[record.agent_abs_index][counter_id].emplace(_mean_timestamp,
                                                                                   counter_value);
                    counters_endpoints[record.agent_abs_index][counter_id].emplace(end_timestamp,
                                                                                   0);
                    counters_endpoints[record.agent_abs_index][counter_id].emplace(
                        end_timestamp + timestamp_buffer, 0);
                }

                counters_extremes = std::make_pair(std::min(counters_extremes.first, record.start),
                                                   std::max(counters_extremes.second, record.end));
            }

        auto counter_tracks =
            std::unordered_map<uint64_t, std::map<std::string, ::perfetto::CounterTrack>>{};

        constexpr auto extremes_endpoint_buffer = 5000;

        for(auto ditr : counter_collection_gen)
        {
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                // const auto& info = record.dispatch_data.dispatch_info;
                // const auto& sym  = tool_metadata.get_kernel_symbol(info.kernel_id);

                // CHECK(sym != nullptr);

                auto name = record.kernel_name;

                for(auto& [counter_id, counter_value] : counter_id_value)
                {
                    counters_endpoints[record.agent_id][counter_id].emplace(
                        counters_extremes.first - extremes_endpoint_buffer, 0);
                    counters_endpoints[record.agent_id][counter_id].emplace(
                        counters_extremes.second + extremes_endpoint_buffer, 0);

                    const auto _agent           = agent_data.at(record.agent_abs_index).first;
                    auto       agent_index_info = agent_data.at(record.agent_abs_index).second;
                    auto       track_name_ss    = std::stringstream{};
                    track_name_ss << agent_index_info.label << " [" << agent_index_info.index
                                  << "] "
                                  << "PMC " << record.counter_name;

                    auto track_name = track_name_ss.str();

                    counter_tracks[record.agent_abs_index].emplace(
                        track_name, ::perfetto::CounterTrack{track_name.c_str(), this_pid_track});
                    auto& endpoints = counters_endpoints[record.agent_id][counter_id];
                    for(auto& counter_itr : endpoints)
                    {
                        TRACE_COUNTER(
                            sdk::perfetto_category<sdk::category::counter_collection>::name,
                            counter_tracks[record.agent_abs_index].at(track_name),
                            counter_itr.first,
                            counter_itr.second);
                    }
                }
            }
            tracing_session->FlushBlocking();
        }
    }

    ::perfetto::TrackEvent::Flush();
    tracing_session->FlushBlocking();
}
}  // namespace output
}  // namespace rocpd
