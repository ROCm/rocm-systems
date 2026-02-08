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

#include "core/trace_cache/perfetto_processor.hpp"
#include "core/agent_manager.hpp"
#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/gpu_metrics.hpp"
#include "core/perfetto.hpp"
#include "core/utility.hpp"
#include "library/tracing.hpp"
#include "trace_cache/metadata_registry.hpp"
#include "trace_cache/sample_type.hpp"

#include "logger/debug.hpp"
#include <cstdint>
#include <nlohmann/json.hpp>

#include <memory>
#include <mutex>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include "library/rocprofiler-sdk/fwd.hpp"
#    include <rocprofiler-sdk/context.h>
#endif

namespace rocprofsys
{
namespace trace_cache
{
namespace
{
struct annotation_entry
{
    const char*                                                             key;
    std::variant<std::string, uint64_t, int64_t, double, int32_t, uint32_t> value;
};

void
annotate_perfetto(::perfetto::EventContext&            ctx,
                  const std::vector<annotation_entry>& annotations)
{
    for(const auto& ann : annotations)
    {
        std::visit(
            [&](auto&& val) { tracing::add_perfetto_annotation(ctx, ann.key, val); },
            ann.value);
    }
}  // close annotate_perfetto

template <typename CategoryT>
::perfetto::Track
get_track(CategoryT, std::string name, uint64_t hash_arg)
{
    auto  _uuid        = tracing::get_perfetto_category_uuid<CategoryT>(hash_arg);
    auto& _track_uuids = tracing::get_perfetto_track_uuids();

    if(_track_uuids.find(_uuid) == _track_uuids.end())
    {
        const auto _track = ::perfetto::Track(_uuid, ::perfetto::ProcessTrack::Current());
        auto       _desc  = _track.Serialize();

        _desc.set_name(name);
        ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        _track_uuids.emplace(_uuid, name);
    }
    return ::perfetto::Track(_uuid, ::perfetto::ProcessTrack::Current());
}

using amd_smi_gfx_track   = perfetto_counter_track<category::amd_smi_gfx_busy>;
using amd_smi_umc_track   = perfetto_counter_track<category::amd_smi_umc_busy>;
using amd_smi_mm_track    = perfetto_counter_track<category::amd_smi_mm_busy>;
using amd_smi_temp_track  = perfetto_counter_track<category::amd_smi_temp>;
using amd_smi_power_track = perfetto_counter_track<category::amd_smi_power>;
using amd_smi_mem_track   = perfetto_counter_track<category::amd_smi_memory_usage>;
using amd_smi_vcn_track   = perfetto_counter_track<category::amd_smi_vcn_activity>;
using amd_smi_jpeg_track  = perfetto_counter_track<category::amd_smi_jpeg_activity>;
using amd_smi_xgmi_link_width_track =
    perfetto_counter_track<category::amd_smi_xgmi_link_width>;
using amd_smi_xgmi_link_speed_track =
    perfetto_counter_track<category::amd_smi_xgmi_link_speed>;
using amd_smi_xgmi_read_track = perfetto_counter_track<category::amd_smi_xgmi_read_data>;
using amd_smi_xgmi_write_track =
    perfetto_counter_track<category::amd_smi_xgmi_write_data>;
using amd_smi_pcie_link_width_track =
    perfetto_counter_track<category::amd_smi_pcie_link_width>;
using amd_smi_pcie_link_speed_track =
    perfetto_counter_track<category::amd_smi_pcie_link_speed>;
using amd_smi_pcie_bandwidth_acc_track =
    perfetto_counter_track<category::amd_smi_pcie_bandwidth_acc>;
using amd_smi_pcie_bandwidth_inst_track =
    perfetto_counter_track<category::amd_smi_pcie_bandwidth_inst>;

void
setup_amd_smi_tracks(const uint32_t _device_id, bool is_busy_enabled,
                     bool is_temp_enabled, bool is_power_enabled,
                     bool is_mem_usage_enabled)
{
    if(amd_smi_gfx_track::exists(_device_id)) return;

    auto make_track_name = [&](const char* metric) {
        return fmt::format("GPU [{}] {} (S)", _device_id, metric);
    };

    if(is_busy_enabled)
    {
        amd_smi_gfx_track::emplace(_device_id, make_track_name("GFX Busy"), "%");
        amd_smi_umc_track::emplace(_device_id, make_track_name("UMC Busy"), "%");
        amd_smi_mm_track::emplace(_device_id, make_track_name("MM Busy"), "%");
    }
    if(is_temp_enabled)
    {
        amd_smi_temp_track::emplace(_device_id, make_track_name("Temperature"), "deg C");
    }
    if(is_power_enabled)
    {
        amd_smi_power_track::emplace(_device_id, make_track_name("Power"), "W");
    }
    if(is_mem_usage_enabled)
    {
        amd_smi_mem_track::emplace(_device_id, make_track_name("Memory Usage"), "MB");
    }
}

template <typename Category>
void
write_sampling_track_data(const struct backtrace_region_sample& _sample,
                          bool                                  use_annotations)
{
    auto _track_name = _sample.track_name;
    auto _thread_id  = _sample.thread_id;
    auto _main_name  = _sample.name;

    auto _track = get_track(Category{}, _track_name, _thread_id);

    auto add_annotations = [&](::perfetto::EventContext& ctx) {
        if(!use_annotations) return;

        std::vector<annotation_entry> annotations = {
            { "begin_ns", _sample.start_timestamp }, { "end_ns", _sample.end_timestamp }
        };

        auto _call_stack = _sample.call_stack;
        if(!_call_stack.empty())
        {
            try
            {
                auto backtrace = nlohmann::json::parse(_call_stack);
                for(const auto& [key, val] : backtrace.items())
                {
                    annotations.push_back(
                        { key.c_str(), val.template get<std::string>() });
                }
            } catch(const std::exception& e)
            {
                LOG_WARNING("Failed to parse call_stack JSON: {}", e.what());
            }
        }
        annotate_perfetto(ctx, annotations);
    };

    tracing::push_perfetto_track(Category{}, _main_name.c_str(), _track,
                                 _sample.start_timestamp, add_annotations);
    tracing::pop_perfetto_track(Category{}, _main_name.c_str(), _track,
                                _sample.end_timestamp);
}

template <typename CategoryT>
void
write_in_time_sample_data(CategoryT, const in_time_sample& _sample, bool use_annotations)
{
    const auto event_metadata = nlohmann::json::parse(_sample.event_metadata);

    const auto _track_name = _sample.track_name;
    const auto _timestamp  = _sample.timestamp_ns;

    const std::string _name       = event_metadata.value("name", "");
    const std::string _event_type = event_metadata.value("event_type", "");
    const std::string _target     = event_metadata.value("target", "");

    const auto _track_uuid = std::hash<std::string>{}(_track_name);

    auto _track                   = get_track(CategoryT{}, _track_name, _track_uuid);
    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
        if(!use_annotations) return;

        annotate_perfetto(ctx, { { "timestamp_ns", _timestamp },
                                 { "event_type", _event_type },
                                 { "target", _target } });
    };

    TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, ::perfetto::DynamicString{ _name },
                        _track, _timestamp, add_perfetto_annotations);
}

// Dispatch to write_in_time_sample_data with the correct category type
// based on runtime category_enum_id, using category_type_id mapping from categories.hpp
template <size_t... Idx>
bool
dispatch_in_time_sample(size_t category_enum_id, const in_time_sample& _sample,
                        bool use_annotations, std::index_sequence<Idx...>)
{
    return ((category_enum_id == Idx
                 ? (write_in_time_sample_data(category_type_id_t<Idx>{}, _sample,
                                              use_annotations),
                    true)
                 : false) ||
            ...);
}

inline bool
dispatch_in_time_sample(size_t category_enum_id, const in_time_sample& _sample,
                        bool use_annotations)
{
    return dispatch_in_time_sample(
        category_enum_id, _sample, use_annotations,
        rocprofsys::utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}
}  // namespace

perfetto_processor_t::perfetto_processor_t(
    const std::shared_ptr<metadata_registry>& metadata,
    const std::shared_ptr<agent_manager>& agent_mngr, int pid, int ppid)
: processor_t<perfetto_processor_t>()
, m_metadata(*metadata)
, m_process_id(pid)
, m_parrent_pid(ppid)
, m_agent_manager(*agent_mngr)
, m_tmp_file(nullptr)
, m_tracing_session(nullptr)
, m_use_annotations(config::get_perfetto_annotations())
{}

void
perfetto_processor_t::initialize_perfetto()
{
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        LOG_DEBUG("Initializing perfetto tracing backend");
        auto args               = ::perfetto::TracingInitArgs{};
        args.backends           = ::perfetto::kInProcessBackend;
        args.shmem_size_hint_kb = config::get_perfetto_shmem_size_hint();

        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();
        LOG_TRACE("Perfetto tracing backend initialized");
    });
}

void
perfetto_processor_t::setup_perfetto()
{
    LOG_DEBUG("Setting up perfetto configuration for pid={}", m_process_id);

    auto  track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto& cfg             = m_session_config;

    auto perfetto_buffer_size = config::get_perfetto_buffer_size();
    auto flush_period         = config::get_perfetto_flush_period();

    LOG_TRACE("Perfetto buffer size: {} KB, flush period: {} ms", perfetto_buffer_size,
              flush_period);

    auto _policy =
        config::get_perfetto_fill_policy() == "discard"
            ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
            : ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(perfetto_buffer_size);
    buffer_config->set_fill_policy(_policy);

    for(const auto& itr : config::get_disabled_categories())
    {
        LOG_TRACE("Disabling perfetto track event category: {}", itr);
        track_event_cfg.add_disabled_categories(itr);
    }

    cfg.set_flush_period_ms(flush_period);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    LOG_TRACE("Perfetto configuration setup complete");
}

void
perfetto_processor_t::start_session()
{
    if(config::get_perfetto_backend() != "inprocess")
    {
        LOG_TRACE("Perfetto backend is not 'inprocess', skipping session start");
        return;
    }

    LOG_DEBUG("Starting perfetto tracing session for pid={}", m_process_id);

    if(!m_tracing_session)
    {
        m_tracing_session = ::perfetto::Tracing::NewTrace();
        LOG_TRACE("Created new perfetto trace");
    }

    int temp_fd = -1;
    if(config::get_use_tmp_files())
    {
        auto _base = fmt::format("cached-perfetto-trace-{}", m_process_id);
        m_tmp_file = config::get_tmp_file(_base, "proto");
        m_tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, 0600);
        temp_fd = m_tmp_file->fd;
        LOG_TRACE("Using temp file for perfetto trace: {}", m_tmp_file->filename);
    }
    m_tracing_session->Setup(m_session_config, temp_fd);
    m_tracing_session->StartBlocking();

    LOG_TRACE("Perfetto tracing session started for pid={}", m_process_id);
}

void
perfetto_processor_t::stop_session()
{
    if(!m_tracing_session)
    {
        LOG_TRACE("No active perfetto session to stop");
        return;
    }

    LOG_DEBUG("Stopping perfetto tracing session for pid={}", m_process_id);
    ::perfetto::TrackEvent::Flush();
    m_tracing_session->FlushBlocking();
    m_tracing_session->StopBlocking();
    LOG_TRACE("Perfetto tracing session stopped");
}

char_vec_t
perfetto_processor_t::get_session_data()
{
    auto _data = char_vec_t{};
    if(m_tmp_file && *m_tmp_file)
    {
        m_tmp_file->close();
        FILE* _fdata = ::fopen(m_tmp_file->filename.c_str(), "rb");

        if(!_fdata)
        {
            LOG_ERROR("Perfetto temp trace file '{}' could not be read",
                      m_tmp_file->filename);
            return char_vec_t{ m_tracing_session->ReadTraceBlocking() };
        }

        ::fseek(_fdata, 0, SEEK_END);
        size_t _fnum_elem = ::ftell(_fdata);
        ::fseek(_fdata, 0, SEEK_SET);

        _data.resize(_fnum_elem, '\0');
        auto _fnum_read = ::fread(_data.data(), sizeof(char), _fnum_elem, _fdata);
        ::fclose(_fdata);

        if(get_is_continuous_integration() && _fnum_read != _fnum_elem)
        {
            throw std::runtime_error(fmt::format(
                "Error! read {} elements from perfetto trace file '{}'. Expected {}",
                _fnum_read, m_tmp_file->filename, _fnum_elem));
        }
    }
    else
    {
        _data = char_vec_t{ m_tracing_session->ReadTraceBlocking() };
    }

    return _data;
}

void
perfetto_processor_t::flush(bool& _perfetto_output_error)
{
    if(!m_tracing_session) return;

    stop_session();

    auto trace_data = char_vec_t{};
    trace_data      = get_session_data();

    // If processing parrent process, use default filename (respects MPI rank/USE_PID
    // settings) Otherwise, use PID-based suffix for child process traces
    auto _filename = (m_process_id == m_parrent_pid)
                         ? config::get_perfetto_output_filename()
                         : config::get_perfetto_output_filename_with_suffix(
                               std::to_string(m_process_id));

    if(!trace_data.empty())
    {
        operation::file_output_message<tim::project::rocprofsys> _fom{};
        // Write the trace into a file.
        if(config::get_verbose() >= 0)
            _fom(_filename, std::string{ "perfetto" },
                 " (%.2f KB / %.2f MB / %.2f GB)... ",
                 static_cast<double>(trace_data.size()) / units::KB,
                 static_cast<double>(trace_data.size()) / units::MB,
                 static_cast<double>(trace_data.size()) / units::GB);
        std::ofstream ofs{};
        if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
        {
            _fom.append("Error opening '%s'...", _filename.c_str());
            _perfetto_output_error = true;
        }
        else
        {
            // Write the trace into a file.
            ofs.write(trace_data.data(), trace_data.size());
            if(config::get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
        }
        ofs.close();
    }
    else
    {
        LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                  _filename.c_str());
    }

    if(m_tmp_file)
    {
        m_tmp_file->close();
        m_tmp_file->remove();
        m_tmp_file.reset();
    }

    m_tracing_session.reset();
}

void
perfetto_processor_t::prepare_for_processing()
{
    LOG_DEBUG("Preparing perfetto processor for pid={}", m_process_id);
    initialize_perfetto();
    setup_perfetto();
    start_session();
    LOG_TRACE("Perfetto processor prepared for processing");
}

void
perfetto_processor_t::finalize_processing()
{
    LOG_DEBUG("Finalizing perfetto processor for pid={}", m_process_id);
    bool _perfetto_output_error = false;
    flush(_perfetto_output_error);

    if(_perfetto_output_error)
    {
        LOG_ERROR("Perfetto trace generation failed for pid={}", m_process_id);
    }
    else
    {
        LOG_DEBUG("Perfetto processing finalized successfully for pid={}", m_process_id);
    }
}

void
perfetto_processor_t::handle([[maybe_unused]] const kernel_dispatch_sample& _kds)
{
#if ROCPROFSYS_USE_ROCM > 0
    static auto _track_desc = [](uint64_t _device_id_v, uint64_t _queue_id_v) {
        return fmt::format("GPU Kernel Dispatch [{}] Queue {}", _device_id_v,
                           _queue_id_v);
    };

    auto kernel_symbol = m_metadata.get_kernel_symbol(_kds.kernel_id);
    auto _agent_device_id =
        m_agent_manager.get_agent_by_handle(_kds.agent_id_handle).device_type_index;
    auto _queue_id_handle = _kds.queue_id_handle;
    auto _stream_handle   = _kds.stream_handle;
    auto _corr_id         = _kds.correlation_id_internal;
    auto _beg_ts          = _kds.start_timestamp;
    auto _end_ts          = _kds.end_timestamp;

    if(!kernel_symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }

    auto kernel_name = rocprofsys::utility::demangle(kernel_symbol->kernel_name);

    const auto _track =
        tracing::get_perfetto_track(category::rocm_kernel_dispatch{}, _track_desc,
                                    _agent_device_id, _queue_id_handle);

    auto add_annotations = [&](::perfetto::EventContext ctx) {
        if(!m_use_annotations) return;

        annotate_perfetto(
            ctx, { { "begin_ns", _beg_ts },
                   { "end_ns", _end_ts },
                   { "corr_id", _corr_id },
                   { "stream_id", _stream_handle },
                   { "queue", _queue_id_handle },
                   { "dispatch_id", _kds.dispatch_id },
                   { "kernel_id", _kds.kernel_id },
                   { "private_segment_size", _kds.private_segment_size },
                   { "group_segment_size", _kds.group_segment_size },
                   { "workgroup_size",
                     fmt::format("({},{},{})", _kds.workgroup_size_x,
                                 _kds.workgroup_size_y, _kds.workgroup_size_z) },
                   { "grid_size", fmt::format("({},{},{})", _kds.grid_size_x,
                                              _kds.grid_size_y, _kds.grid_size_z) } });
    };

    tracing::push_perfetto(category::rocm_kernel_dispatch{}, kernel_name.c_str(), _track,
                           _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                           add_annotations);

    tracing::pop_perfetto(category::rocm_kernel_dispatch{}, kernel_name.c_str(), _track,
                          _end_ts);
#endif
}

void
perfetto_processor_t::handle([[maybe_unused]] const scratch_memory_sample& _sms)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto        _corr_id           = _sms.correlation_id_internal;
    auto        _stream_id         = _sms.stream_handle;
    auto        _queue_id_handle   = _sms.queue_id_handle;
    const auto& _t_info            = thread_info::get(_sms.thread_id, SystemTID);
    const auto  _thread_id_sequent = _t_info->index_data->sequent_value;
    auto        _beg_ts            = _sms.start_timestamp;
    auto        _end_ts            = _sms.end_timestamp;

    auto _agent_device_id =
        m_agent_manager.get_agent_by_handle(_sms.agent_id_handle).device_type_index;
    auto _name = std::string{ m_metadata.get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_sms.kind),
        static_cast<rocprofiler_tracing_operation_t>(_sms.operation)) };

// Scratch memory samples from SDK versions prior to 7.0.2 do not include
// allocation_size field, so counter tracks are not needed
#    if ROCPROFSYS_ROCM_VERSION >= 70002
    using counter_track =
        perfetto_counter_track<rocprofiler_buffer_tracing_scratch_memory_record_t>;

    if(!counter_track::exists(_agent_device_id))
    {
        auto _track_desc_alloc_size = fmt::format("GPU Scratch Memory [{}] Thread {}",
                                                  _agent_device_id, _thread_id_sequent);
        counter_track::emplace(_agent_device_id, _track_desc_alloc_size, "bytes");
    }

    if(_sms.operation == ROCPROFILER_SCRATCH_MEMORY_ALLOC)
    {
        TRACE_COUNTER("rocm_scratch_memory", counter_track::at(_agent_device_id, 0),
                      _beg_ts, _sms.allocation_size);
    }
#    endif

    auto _track_desc_events = [&]() {
        return fmt::format("GPU Scratch Memory Events Thread {}", _thread_id_sequent);
    };

    const auto _track =
        tracing::get_perfetto_track(category::rocm_scratch_memory{}, _track_desc_events);

    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
        if(!m_use_annotations) return;

        annotate_perfetto(ctx, { { "begin_ns", _beg_ts },
                                 { "end_ns", _end_ts },
                                 { "corr_id", _corr_id },
                                 { "stream_id", _stream_id },
                                 { "queue", _queue_id_handle },
                                 { "allocation_size", _sms.allocation_size },
                                 { "agent_id", _agent_device_id },
                                 { "operation", _name },
                                 { "flags", _sms.flags } });
    };

    tracing::push_perfetto(category::rocm_scratch_memory{}, _name.c_str(), _track,
                           _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                           add_perfetto_annotations);
    tracing::pop_perfetto(category::rocm_scratch_memory{}, "", _track, _end_ts);
#endif
}

void
perfetto_processor_t::handle([[maybe_unused]] const memory_copy_sample& _mcs)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto _corr_id   = _mcs.correlation_id_internal;
    auto _thrd_id   = _mcs.thread_id;
    auto _stream_id = _mcs.stream_handle;
    auto _beg_ts    = _mcs.start_timestamp;
    auto _end_ts    = _mcs.end_timestamp;

    auto _src_agent_log_node_id =
        m_agent_manager.get_agent_by_handle(_mcs.src_agent_id_handle).logical_node_id;
    auto _dst_agent_log_node_id =
        m_agent_manager.get_agent_by_handle(_mcs.dst_agent_id_handle).logical_node_id;
    auto _name = std::string{ m_metadata.get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_mcs.kind),
        static_cast<rocprofiler_tracing_operation_t>(_mcs.operation)) };

    auto _track_desc = [](int32_t _device_id_v, rocprofiler_thread_id_t _tid) {
        const auto& _tid_v = thread_info::get(_tid, SystemTID);
        return fmt::format("GPU Memory Copy to Agent [{}] Thread {}", _device_id_v,
                           _tid_v->index_data->sequent_value);
    };

    const auto _track = tracing::get_perfetto_track(
        category::rocm_memory_copy{}, _track_desc, _dst_agent_log_node_id, _thrd_id);

    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
        if(!m_use_annotations) return;

        annotate_perfetto(ctx, { { "begin_ns", _beg_ts },
                                 { "end_ns", _end_ts },
                                 { "corr_id", _corr_id },
                                 { "stream_id", _stream_id },
                                 { "bytes", _mcs.bytes },
                                 { "src_agent_id", _src_agent_log_node_id },
                                 { "dst_agent_id", _dst_agent_log_node_id },
                                 { "operation", _name },
                                 { "src_address", _mcs.src_address_value },
                                 { "dst_address", _mcs.dst_address_value } });
    };

    tracing::push_perfetto(category::rocm_memory_copy{}, _name.c_str(), _track, _beg_ts,
                           ::perfetto::Flow::ProcessScoped(_corr_id),
                           add_perfetto_annotations);
    tracing::pop_perfetto(category::rocm_memory_copy{}, "", _track, _end_ts);
#endif
}

void
perfetto_processor_t::handle([[maybe_unused]] const memory_allocate_sample& _mas)
{
#if ROCPROFSYS_USE_ROCM > 0 && ROCPROFILER_VERSION >= 600
    auto memop_to_string =
        [](rocprofiler_memory_allocation_operation_t op) -> const char* {
        switch(op)
        {
            case ROCPROFILER_MEMORY_ALLOCATION_NONE: return "NONE";
            case ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE: return "ALLOCATE";
            case ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE: return "VMEM_ALLOCATE";
            case ROCPROFILER_MEMORY_ALLOCATION_FREE: return "FREE";
            case ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE: return "VMEM_FREE";
            default: return "UNKNOWN";
        }
    };

    const auto _thrd_id    = _mas.thread_id;
    const auto _corr_id    = _mas.correlation_id_internal;
    const auto _stream_id  = _mas.stream_handle;
    const auto _beg_ts     = _mas.start_timestamp;
    const auto _end_ts     = _mas.end_timestamp;
    const auto _addr_val   = _mas.address_value;
    const auto _alloc_size = _mas.allocation_size;

    const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
    if(_mas.agent_id_handle != invalid_context.handle)
    {
        const auto* operation = memop_to_string(
            static_cast<rocprofiler_memory_allocation_operation_t>(_mas.operation));

        auto _track_desc = [](int32_t _device_id_v, rocprofiler_thread_id_t _tid) {
            const auto& _tid_v = thread_info::get(_tid, SystemTID);
            return fmt::format("GPU Memory Allocation to Agent [{}] Thread {}",
                               _device_id_v, _tid_v->index_data->sequent_value);
        };

        auto _agent_logical_node_id =
            m_agent_manager.get_agent_by_handle(_mas.agent_id_handle).logical_node_id;

        const auto _track =
            tracing::get_perfetto_track(category::rocm_memory_allocate{}, _track_desc,
                                        _agent_logical_node_id, _thrd_id);

        auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
            if(!m_use_annotations) return;

            annotate_perfetto(ctx, { { "begin_ns", _beg_ts },
                                     { "end_ns", _end_ts },
                                     { "corr_id", _corr_id },
                                     { "stream_id", _stream_id },
                                     { "bytes", _alloc_size },
                                     { "agent_id", _agent_logical_node_id },
                                     { "address", _addr_val } });
        };

        tracing::push_perfetto(category::rocm_memory_allocate{}, operation, _track,
                               _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                               add_perfetto_annotations);
        tracing::pop_perfetto(category::rocm_memory_allocate{}, "", _track, _end_ts);
    }
#endif
}

void
perfetto_processor_t::handle(const region_sample& _rs)
{
    const auto _corr_id  = _rs.correlation_id_internal;
    const auto _beg_ts   = _rs.start_timestamp;
    const auto _end_ts   = _rs.end_timestamp;
    const auto _category = _rs.category;
    const auto _name     = _rs.name;

    auto args = process_arguments_string(_rs.args_str);

    auto add_annotations = [&](::perfetto::EventContext ctx) {
        if(!m_use_annotations) return;

        std::vector<annotation_entry> annotations = { { "begin_ns", _beg_ts },
                                                      { "corr_id", _corr_id } };
        for(const auto& arg : args)
        {
            annotations.push_back({ arg.arg_name.c_str(), arg.arg_value });
        }

        if(!_rs.call_stack.empty())
        {
            try
            {
                auto backtrace = nlohmann::json::parse(_rs.call_stack);
                for(const auto& [key, val] : backtrace.items())
                {
                    annotations.push_back(
                        { key.c_str(), val.template get<std::string>() });
                }
            } catch(const std::exception& e)
            {
                LOG_ERROR("Failed to parse call_stack JSON: {}", e.what());
            }
        }

        annotate_perfetto(ctx, annotations);
    };

    auto emit_trace = [&](auto category_tag) {
        using CategoryT = decltype(category_tag);
        if(_corr_id != 0)
        {
            tracing::push_perfetto_ts(CategoryT{}, _name.c_str(), _beg_ts,
                                      ::perfetto::Flow::ProcessScoped(_corr_id),
                                      add_annotations);
        }
        else
        {
            tracing::push_perfetto_ts(CategoryT{}, _name.c_str(), _beg_ts,
                                      add_annotations);
        }

        tracing::pop_perfetto_ts(CategoryT{}, _name.c_str(), _end_ts);
    };

    auto try_category = [&](auto category_tag) {
        using CategoryT = decltype(category_tag);
        if(_category == trait::name<CategoryT>::value)
        {
            emit_trace(category_tag);
            return true;
        }
        return false;
    };

    bool dispatched =
        (try_category(category::host{}) || try_category(category::user{}) ||
         try_category(category::python{}) || try_category(category::mpi{}) ||
         try_category(category::pthread{}) || try_category(category::kokkos{}) ||
         try_category(category::rocm_hip_api{}) ||
         try_category(category::rocm_hsa_api{}) ||
         try_category(category::rocm_marker_api{}) ||
         try_category(category::rocm_rccl{}) ||
         try_category(category::rocm_rocdecode_api{}) ||
         try_category(category::rocm_rocjpeg_api{}) || try_category(category::vaapi{}));

    if(!dispatched)
    {
        // Default to rocm category for backward compatibility
        emit_trace(category::rocm{});
    }
}

void
perfetto_processor_t::handle(const cpu_freq_sample& _cpu_sample)
{
    using process_page_track = perfetto_counter_track<category::process_page>;
    using process_virt_track = perfetto_counter_track<category::process_virt>;
    using process_peak_track = perfetto_counter_track<category::process_peak>;
    using process_cntx_track = perfetto_counter_track<category::process_context_switch>;
    using process_flts_track = perfetto_counter_track<category::process_page_fault>;
    using process_user_track = perfetto_counter_track<category::process_user_mode_time>;
    using process_kern_track = perfetto_counter_track<category::process_kernel_mode_time>;
    using cpu_freq_track     = perfetto_counter_track<category::cpu_freq>;

    struct core_freq_sample
    {
        size_t id;
        float  value;
    };

    auto deserialize_freqs = [](const std::vector<uint8_t>& buffer) {
        std::vector<core_freq_sample> result;
        size_t                        offset = 0;

        while(offset + sizeof(float) + sizeof(size_t) <= buffer.size())
        {
            core_freq_sample core_sample;
            std::memcpy(&core_sample.id, buffer.data() + offset, sizeof(size_t));
            offset += sizeof(size_t);
            std::memcpy(&core_sample.value, buffer.data() + offset, sizeof(float));
            offset += sizeof(float);
            result.push_back(core_sample);
        }
        return result;
    };

    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        process_page_track::emplace(0, "CPU Memory Usage (S)", "MB");
        process_virt_track::emplace(0, "CPU Virtual Memory (S)", "MB");
        process_peak_track::emplace(0, "CPU Peak Memory (S)", "MB");
        process_cntx_track::emplace(0, "CPU Context Switches (S)", "");
        process_flts_track::emplace(0, "CPU Page Faults (S)", "");
        process_user_track::emplace(0, "CPU User Time (S)", "sec");
        process_kern_track::emplace(0, "CPU Kernel Time (S)", "sec");
    });

    auto _ts = _cpu_sample.timestamp;

    TRACE_COUNTER(trait::name<category::process_page>::value,
                  process_page_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.page_rss) / units::megabyte);

    TRACE_COUNTER(trait::name<category::process_virt>::value,
                  process_virt_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.virt_mem_usage) / units::megabyte);

    TRACE_COUNTER(trait::name<category::process_peak>::value,
                  process_peak_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.peak_rss) / units::megabyte);

    TRACE_COUNTER(trait::name<category::process_context_switch>::value,
                  process_cntx_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.context_switch_count));

    TRACE_COUNTER(trait::name<category::process_page_fault>::value,
                  process_flts_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.page_faults));

    TRACE_COUNTER(trait::name<category::process_user_mode_time>::value,
                  process_user_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.user_mode_time) / units::sec);

    TRACE_COUNTER(trait::name<category::process_kernel_mode_time>::value,
                  process_kern_track::at(0, 0), _ts,
                  static_cast<double>(_cpu_sample.kernel_mode_time) / units::sec);

    auto cpu_freqs = deserialize_freqs(_cpu_sample.freqs);
    for(const auto& cpu_data : cpu_freqs)
    {
        size_t cpu_id = cpu_data.id;
        if(!cpu_freq_track::exists(cpu_id))
        {
            std::string track_name = "CPU Frequency [" + std::to_string(cpu_id) + "] (S)";
            cpu_freq_track::emplace(cpu_id, track_name, "MHz");
        }

        TRACE_COUNTER(trait::name<category::cpu_freq>::value,
                      cpu_freq_track::at(cpu_id, 0), _ts,
                      static_cast<double>(cpu_data.value));
    }
}

void
perfetto_processor_t::handle([[maybe_unused]] const backtrace_region_sample& _bts)
{
    (_bts.category == trait::name<category::timer_sampling>::value)
        ? write_sampling_track_data<category::timer_sampling>(_bts, m_use_annotations)
        : write_sampling_track_data<category::overflow_sampling>(_bts, m_use_annotations);
}

void
perfetto_processor_t::handle([[maybe_unused]] const pmc_event_with_sample& _pmc)
{
    using counter_collection_track =
        perfetto_counter_track<category::rocm_counter_collection>;
    using thread_cpu_time_track    = perfetto_counter_track<category::thread_cpu_time>;
    using thread_peak_memory_track = perfetto_counter_track<category::thread_peak_memory>;
    using thread_context_switch_track =
        perfetto_counter_track<category::thread_context_switch>;
    using thread_page_fault_track = perfetto_counter_track<category::thread_page_fault>;
    using thread_hardware_counter_track =
        perfetto_counter_track<category::thread_hardware_counter>;
    using comm_data_track = perfetto_counter_track<category::comm_data>;

    m_pmc_track_map = {
        { ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION,
          { "Unit Count", [](auto id) { return counter_collection_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                counter_collection_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::rocm_counter_collection>::value,
                              counter_collection_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_THREAD_CPU_TIME,
          { "sec", [](auto id) { return thread_cpu_time_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                thread_cpu_time_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::thread_cpu_time>::value,
                              thread_cpu_time_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_THREAD_PEAK_MEMORY,
          { "MB", [](auto id) { return thread_peak_memory_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                thread_peak_memory_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::thread_peak_memory>::value,
                              thread_peak_memory_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_THREAD_CONTEXT_SWITCH,
          { "", [](auto id) { return thread_context_switch_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                thread_context_switch_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::thread_context_switch>::value,
                              thread_context_switch_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_THREAD_PAGE_FAULT,
          { "", [](auto id) { return thread_page_fault_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                thread_page_fault_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::thread_page_fault>::value,
                              thread_page_fault_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_THREAD_HARDWARE_COUNTER,
          { "", [](auto id) { return thread_hardware_counter_track::exists(id); },
            [](auto id, auto& n, auto& u) {
                thread_hardware_counter_track::emplace(id, n, u.c_str());
            },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::thread_hardware_counter>::value,
                              thread_hardware_counter_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_COMM_DATA,
          { "bytes", [](auto id) { return comm_data_track::exists(id); },
            [](auto id, auto& n, auto& u) { comm_data_track::emplace(id, n, u.c_str()); },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::comm_data>::value,
                              comm_data_track::at(id, idx), ts, val);
            } } },

        { ROCPROFSYS_CATEGORY_MPI,
          { "bytes", [](auto id) { return comm_data_track::exists(id); },
            [](auto id, auto& n, auto& u) { comm_data_track::emplace(id, n, u.c_str()); },
            [](auto id, auto idx, auto ts, auto val) {
                TRACE_COUNTER(trait::name<category::comm_data>::value,
                              comm_data_track::at(id, idx), ts, val);
            } } }
    };

    const auto _track_name = _pmc.track_name;
    const auto _value      = _pmc.value;
    const auto _beg_ts     = _pmc.timestamp_ns;
    const auto _device_id  = _pmc.device_id;

    auto track_key = std::hash<std::string>{}(_track_name + std::to_string(_device_id));

    auto track_it = m_pmc_track_map.find(_pmc.category_enum_id);
    if(track_it != m_pmc_track_map.end())
    {
        const auto& track_info = track_it->second;

        if(!track_info.exists_fn(track_key))
        {
            track_info.emplace_fn(track_key, _track_name, track_info.default_units);
        }

        track_info.trace_fn(track_key, 0, _beg_ts, _value);
    }
    else
    {
        LOG_WARNING("Unknown PMC event category_enum_id: {} for track '{}'",
                    _pmc.category_enum_id, _track_name);
    }
}

void
perfetto_processor_t::handle([[maybe_unused]] const amd_smi_sample& _amd_smi)
{
    // Use the shared gpu_metrics_t from core/gpu_metrics.hpp
    using gpu_metrics_t = gpu::gpu_metrics_t;

    using pos = trace_cache::amd_smi_sample::settings_positions;
    std::bitset<8> settings_bits(_amd_smi.settings);
    bool           is_busy_enabled  = settings_bits.test(static_cast<int>(pos::busy));
    bool           is_temp_enabled  = settings_bits.test(static_cast<int>(pos::temp));
    bool           is_power_enabled = settings_bits.test(static_cast<int>(pos::power));
    bool is_mem_usage_enabled = settings_bits.test(static_cast<int>(pos::mem_usage));
    bool is_vcn_enabled       = settings_bits.test(static_cast<int>(pos::vcn_activity));
    bool is_jpeg_enabled      = settings_bits.test(static_cast<int>(pos::jpeg_activity));
    bool is_xgmi_enabled      = settings_bits.test(static_cast<int>(pos::xgmi));
    bool is_pcie_enabled      = settings_bits.test(static_cast<int>(pos::pcie));

    auto _ts        = _amd_smi.timestamp;
    auto _device_id = _amd_smi.device_id;

    setup_amd_smi_tracks(_device_id, is_busy_enabled, is_temp_enabled, is_power_enabled,
                         is_mem_usage_enabled);

    if(is_busy_enabled)
    {
        TRACE_COUNTER("device_busy_gfx", amd_smi_gfx_track::at(_device_id, 0), _ts,
                      _amd_smi.gfx_activity);
        TRACE_COUNTER("device_busy_umc", amd_smi_umc_track::at(_device_id, 0), _ts,
                      _amd_smi.umc_activity);
        TRACE_COUNTER("device_busy_mm", amd_smi_mm_track::at(_device_id, 0), _ts,
                      _amd_smi.mm_activity);
    }
    if(is_temp_enabled)
    {
        TRACE_COUNTER("device_temp", amd_smi_temp_track::at(_device_id, 0), _ts,
                      _amd_smi.temperature);
    }
    if(is_power_enabled)
    {
        TRACE_COUNTER("device_power", amd_smi_power_track::at(_device_id, 0), _ts,
                      _amd_smi.power);
    }
    if(is_mem_usage_enabled)
    {
        double mem_mb = _amd_smi.mem_usage / static_cast<double>(units::megabyte);
        TRACE_COUNTER("device_memory_usage", amd_smi_mem_track::at(_device_id, 0), _ts,
                      mem_mb);
    }

    if(!is_vcn_enabled && !is_jpeg_enabled && !is_xgmi_enabled && !is_pcie_enabled)
        return;

    gpu_metrics_t                   gpu_metrics;
    gpu::gpu_metrics_capabilities_t capabilities;
    gpu::deserialize_gpu_metrics(_amd_smi.gpu_activity, gpu_metrics, is_vcn_enabled,
                                 is_jpeg_enabled, is_xgmi_enabled, is_pcie_enabled,
                                 capabilities);

    // Helper lambda to insert VCN/JPEG activity metrics
    auto insert_decode_vector_metrics = [&](auto category, bool _is_enabled,
                                            const std::vector<uint16_t>& data,
                                            std::optional<size_t> _idx = std::nullopt) {
        if(!_is_enabled) return;

        using Category = std::decay_t<decltype(category)>;

        const char* metric_name = nullptr;
        if constexpr(std::is_same_v<Category, category::amd_smi_vcn_activity>)
            metric_name = "VCN Activity";
        else if constexpr(std::is_same_v<Category, category::amd_smi_jpeg_activity>)
            metric_name = "JPEG Activity";
        else
            metric_name = trait::name<Category>::value;

        for(size_t i = 0; i < data.size(); ++i)
        {
            const auto value = data[i];
            if(value == std::numeric_limits<uint16_t>::max()) continue;

            std::string track_name;
            if(_idx.has_value())
            {
                // Per-XCP format
                track_name = fmt::format("GPU [{}] {} XCP_{}: [{:02}] (S)", _device_id,
                                         metric_name, _idx.value(), i);
            }
            else
            {
                // Device-level format
                track_name =
                    fmt::format("GPU [{}] {} [{:02}] (S)", _device_id, metric_name, i);
            }

            auto generate_track_key = [](uint32_t _dev_idx, size_t _xcp_idx,
                                         size_t _clk_idx) {
                return (static_cast<uint64_t>(_dev_idx) << 16) |
                       (static_cast<uint64_t>(_xcp_idx) << 8) |
                       static_cast<uint64_t>(_clk_idx);
            };

            auto unique_key = generate_track_key(_device_id, _idx.value_or(0), i);

            if constexpr(std::is_same_v<Category, category::amd_smi_vcn_activity>)
            {
                if(!amd_smi_vcn_track::exists(unique_key))
                {
                    amd_smi_vcn_track::emplace(unique_key, track_name, "%");
                }
                TRACE_COUNTER("device_vcn_activity", amd_smi_vcn_track::at(unique_key, 0),
                              _ts, static_cast<double>(value));
            }
            else if constexpr(std::is_same_v<Category, category::amd_smi_jpeg_activity>)
            {
                if(!amd_smi_jpeg_track::exists(unique_key))
                {
                    amd_smi_jpeg_track::emplace(unique_key, track_name, "%");
                }
                TRACE_COUNTER("device_jpeg_activity",
                              amd_smi_jpeg_track::at(unique_key, 0), _ts,
                              static_cast<double>(value));
            }
        }
    };

    auto insert_xgmi_vector_metrics = [&](auto category, bool _is_enabled,
                                          const std::vector<uint64_t>& data) {
        if(!_is_enabled) return;

        using Category = std::decay_t<decltype(category)>;

        for(size_t i = 0; i < data.size(); ++i)
        {
            const auto value = data[i];
            if(value == std::numeric_limits<uint64_t>::max()) continue;

            std::string track_name = fmt::format("GPU [{}] {} [{:02}] (S)", _device_id,
                                                 trait::name<Category>::value, i);

            auto unique_key = (_device_id << 8) | i;

            if constexpr(std::is_same_v<Category, category::amd_smi_xgmi_read_data>)
            {
                if(!amd_smi_xgmi_read_track::exists(unique_key))
                {
                    amd_smi_xgmi_read_track::emplace(unique_key, track_name, "bytes");
                }
                TRACE_COUNTER("device_xgmi_read_data",
                              amd_smi_xgmi_read_track::at(unique_key, 0), _ts,
                              static_cast<double>(value));
            }
            else if constexpr(std::is_same_v<Category, category::amd_smi_xgmi_write_data>)
            {
                if(!amd_smi_xgmi_write_track::exists(unique_key))
                {
                    amd_smi_xgmi_write_track::emplace(unique_key, track_name, "bytes");
                }
                TRACE_COUNTER("device_xgmi_write_data",
                              amd_smi_xgmi_write_track::at(unique_key, 0), _ts,
                              static_cast<double>(value));
            }
        }
    };

    // Insert VCN activity metrics
    if(capabilities.flags.vcn_is_device_level_only)
    {
        insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                     gpu_metrics.vcn_activity, std::nullopt);
    }
    else
    {
        for(size_t xcp = 0; xcp < gpu_metrics.vcn_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                         gpu_metrics.vcn_busy[xcp], xcp);
        }
    }

    // Insert JPEG activity metrics
    if(capabilities.flags.jpeg_is_device_level_only)
    {
        insert_decode_vector_metrics(category::amd_smi_jpeg_activity{}, is_jpeg_enabled,
                                     gpu_metrics.jpeg_activity, std::nullopt);
    }
    else
    {
        for(size_t xcp = 0; xcp < gpu_metrics.jpeg_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_jpeg_activity{},
                                         is_jpeg_enabled, gpu_metrics.jpeg_busy[xcp],
                                         xcp);
        }
    }

    // Insert XGMI metrics
    if(is_xgmi_enabled)
    {
        auto make_track_name = [&](const char* metric) {
            return fmt::format("GPU [{}] {} (S)", _device_id, metric);
        };

        if(!amd_smi_xgmi_link_width_track::exists(_device_id))
        {
            amd_smi_xgmi_link_width_track::emplace(
                _device_id, make_track_name("XGMI Link Width"), "");
        }
        TRACE_COUNTER("device_xgmi_link_width",
                      amd_smi_xgmi_link_width_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.xgmi_link_width));

        if(!amd_smi_xgmi_link_speed_track::exists(_device_id))
        {
            amd_smi_xgmi_link_speed_track::emplace(
                _device_id, make_track_name("XGMI Link Speed"), "MT/s");
        }
        TRACE_COUNTER("device_xgmi_link_speed",
                      amd_smi_xgmi_link_speed_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.xgmi_link_speed));

        insert_xgmi_vector_metrics(category::amd_smi_xgmi_read_data{}, is_xgmi_enabled,
                                   gpu_metrics.xgmi_read_data_acc);

        insert_xgmi_vector_metrics(category::amd_smi_xgmi_write_data{}, is_xgmi_enabled,
                                   gpu_metrics.xgmi_write_data_acc);
    }

    // Insert PCIe metrics
    if(is_pcie_enabled)
    {
        auto make_track_name = [&](const char* metric) {
            return fmt::format("GPU [{}] {} (S)", _device_id, metric);
        };

        if(!amd_smi_pcie_link_width_track::exists(_device_id))
        {
            amd_smi_pcie_link_width_track::emplace(
                _device_id, make_track_name("PCIe Link Width"), "");
        }
        TRACE_COUNTER("device_pcie_link_width",
                      amd_smi_pcie_link_width_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.pcie_link_width));

        if(!amd_smi_pcie_link_speed_track::exists(_device_id))
        {
            amd_smi_pcie_link_speed_track::emplace(
                _device_id, make_track_name("PCIe Link Speed"), "MT/s");
        }
        TRACE_COUNTER("device_pcie_link_speed",
                      amd_smi_pcie_link_speed_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.pcie_link_speed));

        if(!amd_smi_pcie_bandwidth_acc_track::exists(_device_id))
        {
            amd_smi_pcie_bandwidth_acc_track::emplace(
                _device_id, make_track_name("PCIe Bandwidth Acc"), "bytes");
        }
        TRACE_COUNTER("device_pcie_bandwidth_acc",
                      amd_smi_pcie_bandwidth_acc_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.pcie_bandwidth_acc));

        if(!amd_smi_pcie_bandwidth_inst_track::exists(_device_id))
        {
            amd_smi_pcie_bandwidth_inst_track::emplace(
                _device_id, make_track_name("PCIe Bandwidth Inst"), "bytes");
        }
        TRACE_COUNTER("device_pcie_bandwidth_inst",
                      amd_smi_pcie_bandwidth_inst_track::at(_device_id, 0), _ts,
                      static_cast<double>(gpu_metrics.pcie_bandwidth_inst));
    }
}

void
perfetto_processor_t::handle([[maybe_unused]] const in_time_sample& _sample)
{
    // Dispatch based on category_enum_id using the category type mapping
    if(!dispatch_in_time_sample(_sample.category_enum_id, _sample, m_use_annotations))
    {
        LOG_DEBUG("Unknown in_time_sample category_enum_id: {}, using user category",
                  _sample.category_enum_id);
        write_in_time_sample_data(category::user{}, _sample, m_use_annotations);
    }
}

}  // namespace trace_cache
}  // namespace rocprofsys
