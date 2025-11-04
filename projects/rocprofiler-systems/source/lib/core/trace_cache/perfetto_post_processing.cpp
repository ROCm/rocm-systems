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

#include "core/trace_cache/perfetto_post_processing.hpp"
#include "common.hpp"
#include "config.hpp"
#include "core/agent_manager.hpp"
#include "library/tracing.hpp"
#include "perfetto.hpp"
#include "rocprofiler-sdk/fwd.h"
#include "trace_cache/metadata_registry.hpp"
#include "trace_cache/sample_type.hpp"
#include "trace_cache/storage_parser.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>

#include <memory>
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

bool&
get_perfetto_initialized()
{
    static bool _initialized = false;
    return _initialized;
}

auto&
get_perfetto_tmp_file()
{
    static std::shared_ptr<tmp_file> _tmp_file = nullptr;
    return _tmp_file;
}

int
get_perffeto_temp_fd(const std::string& _pid)
{
    auto& _tmp_file = get_perfetto_tmp_file();
    if(config::get_use_tmp_files())
    {
        auto _base = JOIN("-", "cached-perfetto-trace", _pid);
        _tmp_file  = config::get_tmp_file(_base, "proto");
        _tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, 0600);
    }
    return ((_tmp_file) ? _tmp_file->fd : -1);
}

void
initialize_perfetto()
{
    static std::mutex           init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if(!get_perfetto_initialized())
    {
        auto args               = ::perfetto::TracingInitArgs{};
        args.backends           = ::perfetto::kInProcessBackend;
        args.shmem_size_hint_kb = config::get_perfetto_shmem_size_hint();

        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();  // Only register once globally!

        get_perfetto_initialized() = true;
    }
}

#if ROCPROFSYS_USE_ROCM > 0
rocprofiler_sdk::function_args_t
process_arguments_string(const std::string& arg_str)
{
    rocprofiler_sdk::function_args_t args;
    const std::string                delimiter = ";;";

    auto split = [](const std::string& str, const std::string& _delimiter) {
        std::vector<std::string> tokens;
        size_t                   start = 0;
        size_t                   end   = str.find(_delimiter);

        while(end != std::string::npos)
        {
            tokens.push_back(str.substr(start, end - start));
            start = end + _delimiter.length();
            end   = str.find(_delimiter, start);
        }

        return tokens;
    };

    auto tokens = split(arg_str, delimiter);

    // Ensure the number of tokens is a multiple of 4
    if(tokens.size() % 4 != 0)
    {
        throw std::invalid_argument("Malformed argument string.");
    }

    for(auto it = tokens.begin(); it != tokens.end(); it += 4)
    {
        rocprofiler_sdk::argument_info arg = { static_cast<uint32_t>(std::stoi(*it)),
                                               *(it + 1), *(it + 2), *(it + 3) };
        args.push_back(arg);
    }

    return args;
}
#endif

struct annotation_entry
{
    const char*                                                             key;
    std::variant<std::string, uint64_t, int64_t, double, int32_t, uint32_t> value;
};

void
annotate_perfetto(::perfetto::EventContext&            ctx,
                  const std::vector<annotation_entry>& annotations)
{
    if(!config::get_perfetto_annotations()) return;

    for(const auto& ann : annotations)
    {
        std::visit(
            [&](auto&& val) { tracing::add_perfetto_annotation(ctx, ann.key, val); },
            ann.value);
    }
}

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

template <typename Category>
void
write_track_data(const struct backtrace_region_sample& _sample)
{
    auto _track_name = _sample.track_name;
    auto _thread_id  = _sample.thread_id;
    auto _main_name  = _sample.name;

    auto _track = get_track(Category{}, _track_name, _thread_id);

    auto add_annotations = [&](::perfetto::EventContext ctx) {
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
                ROCPROFSYS_VERBOSE_F(2, "Failed to parse call_stack JSON: %s\n",
                                     e.what());
            }
        }
        annotate_perfetto(ctx, annotations);
    };

    tracing::push_perfetto_track(Category{}, _main_name.c_str(), _track,
                                 _sample.start_timestamp, add_annotations);
    tracing::pop_perfetto_track(Category{}, _main_name.c_str(), _track,
                                _sample.end_timestamp);
}
}  // namespace

perfetto_post_processing::perfetto_post_processing(metadata_registry& metadata,
                                                   const uint64_t&    pid,
                                                   agent_manager&     agent_mngr)
: m_metadata(metadata)
, m_process_id(pid)
, m_agent_manager(agent_mngr)
, m_tracing_session(nullptr)
{
    if(get_caching_perfetto())
    {
        initialize_perfetto();
        setup_perfetto();
    }
}

perfetto_post_processing::~perfetto_post_processing()
{
    if(m_tracing_session)
    {
        stop_session();
        m_tracing_session.reset();
    }
}

void
perfetto_post_processing::setup_perfetto()
{
    auto  track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto& cfg             = m_session_config;

    auto buffer_size  = config::get_perfetto_buffer_size();
    auto flush_period = config::get_perfetto_flush_period();

    auto _policy =
        config::get_perfetto_fill_policy() == "discard"
            ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
            : ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size);
    buffer_config->set_fill_policy(_policy);

    for(const auto& itr : config::get_disabled_categories())
    {
        ROCPROFSYS_VERBOSE_F(1, "Disabling perfetto track event category: %s\n",
                             itr.c_str());
        track_event_cfg.add_disabled_categories(itr);
    }

    cfg.set_flush_period_ms(flush_period);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());
}

void
perfetto_post_processing::start_session()
{
    if(config::get_perfetto_backend() != "inprocess") return;

    if(!m_tracing_session)
    {
        m_tracing_session = ::perfetto::Tracing::NewTrace();
    }

    ROCPROFSYS_VERBOSE(2,
                       "Starting perfetto post-processing session with cached data...\n");

    auto temp_fd = get_perffeto_temp_fd(std::to_string(m_process_id));

    m_tracing_session->Setup(m_session_config, temp_fd);
    m_tracing_session->StartBlocking();
}

void
perfetto_post_processing::stop_session()
{
    if(!m_tracing_session) return;

    ROCPROFSYS_VERBOSE(2, "Stopping perfetto post-processing session...\n");
    ::perfetto::TrackEvent::Flush();
    m_tracing_session->FlushBlocking();
    m_tracing_session->StopBlocking();
}

void
perfetto_post_processing::post_process(bool& _perfetto_output_error)
{
    using char_vec_t = std::vector<char>;

    if(!m_tracing_session) return;

    stop_session();

    auto _get_session_data = [this]() {
        auto _data     = char_vec_t{};
        auto _tmp_file = get_perfetto_tmp_file();
        if(_tmp_file && *_tmp_file)
        {
            _tmp_file->close();
            FILE* _fdata = ::fopen(_tmp_file->filename.c_str(), "rb");

            if(!_fdata)
            {
                ROCPROFSYS_VERBOSE(
                    -1, "Error! perfetto temp trace file '%s' could not be read",
                    _tmp_file->filename.c_str());
                return char_vec_t{ m_tracing_session->ReadTraceBlocking() };
            }

            ::fseek(_fdata, 0, SEEK_END);
            size_t _fnum_elem = ::ftell(_fdata);
            ::fseek(_fdata, 0, SEEK_SET);

            _data.resize(_fnum_elem, '\0');
            auto _fnum_read = ::fread(_data.data(), sizeof(char), _fnum_elem, _fdata);
            ::fclose(_fdata);

            ROCPROFSYS_CI_THROW(
                _fnum_read != _fnum_elem,
                "Error! read %zu elements from perfetto trace file '%s'. Expected %zu\n",
                _fnum_read, _tmp_file->filename.c_str(), _fnum_elem);
        }
        else
        {
            _data = char_vec_t{ m_tracing_session->ReadTraceBlocking() };
        }

        return _data;
    };

    auto trace_data = char_vec_t{};
    trace_data      = _get_session_data();

    auto output_dir = filepath::dirname(config::get_perfetto_output_filename());
    auto _filename =
        JOIN("/", output_dir,
             JOIN("-", "perfetto-cached-trace", std::to_string(m_process_id) + ".proto"));

    if(!trace_data.empty())
    {
        ROCPROFSYS_VERBOSE(1, "Writing perfetto trace data (%zu bytes)...\n",
                           trace_data.size());

        operation::file_output_message<tim::project::rocprofsys> _fom{};
        if(config::get_verbose() >= 0)
            _fom(_filename, std::string{ "perfetto" },
                 " (%.2f KB / %.2f MB / %.2f GB)... ",
                 static_cast<double>(trace_data.size()) / units::KB,
                 static_cast<double>(trace_data.size()) / units::MB,
                 static_cast<double>(trace_data.size()) / units::GB);

        std::ofstream ofs{};
        if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
        {
            ROCPROFSYS_VERBOSE(-1, "Error opening '%s'...", _filename.c_str());
            _perfetto_output_error = true;
        }
        else
        {
            ofs.write(trace_data.data(), trace_data.size());
            if(config::get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
            ROCPROFSYS_VERBOSE(0, "Perfetto trace written to: %s (%.2f MB)\n",
                               _filename.c_str(),
                               static_cast<double>(trace_data.size()) / units::MB);
        }
        ofs.close();
    }
    else
    {
        ROCPROFSYS_VERBOSE(
            0, "perfetto trace data is empty. File '%s' will not be written...\n",
            _filename.c_str());
    }

    auto& _tmp_file = get_perfetto_tmp_file();
    if(_tmp_file)
    {
        _tmp_file->close();
        _tmp_file->remove();
        _tmp_file.reset();
    }

    m_tracing_session.reset();
}

postprocessing_callback
perfetto_post_processing::get_kernel_dispatch_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _kds = static_cast<const struct kernel_dispatch_sample&>(parsed);

        auto kernel_symbol = m_metadata.get_kernel_symbol(_kds.kernel_id);
        auto _agent_device_id =
            m_agent_manager.get_agent_by_handle(_kds.agent_id_handle).device_id;
        auto _queue_id_handle = _kds.queue_id_handle;
        auto _stream_handle   = _kds.stream_handle;
        auto _corr_id         = _kds.correlation_id_internal;
        auto _beg_ts          = _kds.start_timestamp;
        auto _end_ts          = _kds.end_timestamp;

        if(!kernel_symbol.has_value())
        {
            throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
        }

        auto kernel_name = tim::demangle(kernel_symbol->kernel_name);

        auto _track_desc = [](uint64_t _device_id_v, uint64_t _queue_id_v) {
            return JOIN("", "GPU Kernel Dispatch [", _device_id_v, "] Queue ",
                        _queue_id_v);
        };

        const auto _track =
            tracing::get_perfetto_track(category::rocm_kernel_dispatch{}, _track_desc,
                                        _agent_device_id, _queue_id_handle);

        auto add_annotations = [&](::perfetto::EventContext ctx) {
            annotate_perfetto(
                ctx, { { "begin_ns", _beg_ts },
                       { "end_ns", _end_ts },
                       { "corr_id", _corr_id },
                       { "stream_id", static_cast<uint64_t>(_stream_handle) },
                       { "queue", static_cast<uint64_t>(_queue_id_handle) },
                       { "dispatch_id", static_cast<uint64_t>(_kds.dispatch_id) },
                       { "kernel_id", static_cast<uint64_t>(_kds.kernel_id) },
                       { "private_segment_size",
                         static_cast<uint64_t>(_kds.private_segment_size) },
                       { "group_segment_size",
                         static_cast<uint64_t>(_kds.group_segment_size) },
                       { "workgroup_size",
                         JOIN("", "(",
                              JOIN(',', _kds.workgroup_size_x, _kds.workgroup_size_y,
                                   _kds.workgroup_size_z),
                              ")") },
                       { "grid_size", JOIN("", "(",
                                           JOIN(',', _kds.grid_size_x, _kds.grid_size_y,
                                                _kds.grid_size_z),
                                           ")") } });
        };

        tracing::push_perfetto(category::rocm_kernel_dispatch{}, kernel_name.c_str(),
                               _track, _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                               add_annotations);

        tracing::pop_perfetto(category::rocm_kernel_dispatch{}, kernel_name.c_str(),
                              _track, _end_ts);
#endif
    };
}

postprocessing_callback
perfetto_post_processing::get_memory_copy_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _mcs = static_cast<const struct memory_copy_sample&>(parsed);

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
            return JOIN("", "GPU Memory Copy to Agent [", _device_id_v, "] Thread ",
                        _tid_v->index_data->sequent_value);
        };

        const auto _track = tracing::get_perfetto_track(
            category::rocm_memory_copy{}, _track_desc, _dst_agent_log_node_id, _thrd_id);

        auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
            annotate_perfetto(
                ctx,
                { { "begin_ns", _beg_ts },
                  { "end_ns", _end_ts },
                  { "corr_id", _corr_id },
                  { "stream_id", static_cast<uint64_t>(_stream_id) },
                  { "bytes", static_cast<uint64_t>(_mcs.bytes) },
                  { "src_agent_id", static_cast<uint64_t>(_mcs.src_agent_id_handle) },
                  { "dst_agent_id", static_cast<uint64_t>(_mcs.dst_agent_id_handle) },
                  { "operation", _name },
                  { "src_address", static_cast<uint64_t>(_mcs.src_address_value) },
                  { "dst_address", static_cast<uint64_t>(_mcs.dst_address_value) } });
        };

        tracing::push_perfetto(category::rocm_memory_copy{}, _name.c_str(), _track,
                               _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                               add_perfetto_annotations);
        tracing::pop_perfetto(category::rocm_memory_copy{}, "", _track, _end_ts);
#endif
    };
}

#if(ROCPROFSYS_USE_ROCM > 0 && ROCPROFILER_VERSION >= 600)
postprocessing_callback
perfetto_post_processing::get_memory_allocate_callback() const
{
#    if ROCPROFSYS_USE_ROCM > 0
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
#    endif
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#    if ROCPROFSYS_USE_ROCM > 0
        auto _mas = static_cast<const struct memory_allocate_sample&>(parsed);

        auto _thrd_id    = _mas.thread_id;
        auto _corr_id    = _mas.correlation_id_internal;
        auto _stream_id  = _mas.stream_handle;
        auto _beg_ts     = _mas.start_timestamp;
        auto _end_ts     = _mas.end_timestamp;
        auto _addr_val   = _mas.address_value;
        auto _alloc_size = _mas.allocation_size;

        const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
        if(_mas.agent_id_handle != invalid_context.handle)
        {
            const auto* operation = memop_to_string(
                static_cast<rocprofiler_memory_allocation_operation_t>(_mas.operation));

            auto _track_desc = [](int32_t _device_id_v, rocprofiler_thread_id_t _tid) {
                const auto& _tid_v = thread_info::get(_tid, SystemTID);
                return JOIN("", "GPU Memory Allocation to Agent [", _device_id_v,
                            "] Thread ", _tid_v->index_data->sequent_value);
            };

            auto _agent_logical_node_id =
                m_agent_manager.get_agent_by_handle(_mas.agent_id_handle).logical_node_id;

            const auto _track =
                tracing::get_perfetto_track(category::rocm_memory_allocate{}, _track_desc,
                                            _agent_logical_node_id, _thrd_id);

            auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
                annotate_perfetto(
                    ctx, { { "begin_ns", _beg_ts },
                           { "end_ns", _end_ts },
                           { "corr_id", _corr_id },
                           { "stream_id", static_cast<uint64_t>(_stream_id) },
                           { "bytes", static_cast<uint64_t>(_mas.allocation_size) },
                           { "agent_id", static_cast<uint64_t>(_mas.agent_id_handle) },
                           { "address", static_cast<uint64_t>(_mas.address_value) } });
            };

            tracing::push_perfetto(category::rocm_memory_allocate{}, operation, _track,
                                   _beg_ts, ::perfetto::Flow::ProcessScoped(_corr_id),
                                   add_perfetto_annotations);
            tracing::pop_perfetto(category::rocm_memory_allocate{}, "", _track, _end_ts);
#    endif
        };
    };
}
#endif

postprocessing_callback
perfetto_post_processing::get_region_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _rs = static_cast<const struct region_sample&>(parsed);

        auto _corr_id  = _rs.correlation_id_internal;
        auto _beg_ts   = _rs.start_timestamp;
        auto _end_ts   = _rs.end_timestamp;
        auto _category = _rs.category;
        auto _name     = _rs.name;

        auto args = process_arguments_string(_rs.args_str);

        auto add_annotations = [&](::perfetto::EventContext ctx) {
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
                    ROCPROFSYS_VERBOSE_F(2, "Failed to parse call_stack JSON: %s\n",
                                         e.what());
                }
            }

            annotate_perfetto(ctx, annotations);
        };

        tracing::push_perfetto_ts(category::rocm{}, _name.c_str(), _beg_ts,
                                  ::perfetto::Flow::ProcessScoped(_corr_id),
                                  add_annotations);
        tracing::pop_perfetto_ts(category::rocm{}, _name.c_str(), _end_ts);
#endif
    };
}

postprocessing_callback
perfetto_post_processing::get_cpu_freq_sample_callback() const
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

    auto deserialize_freqs = [](std::vector<uint8_t>& buffer) {
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

    return [&](const storage_parsed_type_base& parsed) {
        auto _cpu_sample = static_cast<const struct cpu_freq_sample&>(parsed);

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
                std::string track_name =
                    "CPU Frequency [" + std::to_string(cpu_id) + "] (S)";
                cpu_freq_track::emplace(cpu_id, track_name, "MHz");
            }

            TRACE_COUNTER(trait::name<category::cpu_freq>::value,
                          cpu_freq_track::at(cpu_id, 0), _ts,
                          static_cast<double>(cpu_data.value));
        }
    };
}

postprocessing_callback
perfetto_post_processing::get_backtrace_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto _bts = static_cast<const struct backtrace_region_sample&>(parsed);

        (_bts.category == trait::name<category::timer_sampling>::value)
            ? write_track_data<category::timer_sampling>(_bts)
            : write_track_data<category::overflow_sampling>(_bts);
    };
}

postprocessing_callback
perfetto_post_processing::get_pmc_event_with_sample_callback() const
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

    struct pmc_track_info
    {
        const char*                   default_units;
        std::function<bool(uint64_t)> exists_fn;
        std::function<void(uint64_t, const std::string&, const std::string&)> emplace_fn;
        std::function<void(uint64_t, uint64_t, uint64_t, double)>             trace_fn;
    };

    static const std::map<size_t, pmc_track_info> pmc_track_map = {
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

    return [&](const storage_parsed_type_base& parsed) {
        auto _pmc = static_cast<const struct pmc_event_with_sample&>(parsed);

        auto _track_name = _pmc.track_name;
        auto _value      = _pmc.value;
        auto _beg_ts     = _pmc.timestamp_ns;
        auto _device_id  = _pmc.device_id;

        auto track_key =
            std::hash<std::string>{}(_track_name + std::to_string(_device_id));

        auto track_it = pmc_track_map.find(_pmc.category_enum_id);
        if(track_it != pmc_track_map.end())
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
            ROCPROFSYS_VERBOSE_F(
                2, "Unknown PMC event category_enum_id: %zu for track '%s'\n",
                _pmc.category_enum_id, _track_name.c_str());
        }
    };
}

postprocessing_callback
perfetto_post_processing::get_amd_smi_sample_callback() const
{
    using amd_smi_gfx_track   = perfetto_counter_track<category::amd_smi_gfx_busy>;
    using amd_smi_umc_track   = perfetto_counter_track<category::amd_smi_umc_busy>;
    using amd_smi_mm_track    = perfetto_counter_track<category::amd_smi_mm_busy>;
    using amd_smi_temp_track  = perfetto_counter_track<category::amd_smi_temp>;
    using amd_smi_power_track = perfetto_counter_track<category::amd_smi_power>;
    using amd_smi_mem_track   = perfetto_counter_track<category::amd_smi_memory_usage>;
    using amd_smi_vcn_track   = perfetto_counter_track<category::amd_smi_vcn_activity>;
    using amd_smi_jpeg_track  = perfetto_counter_track<category::amd_smi_jpeg_activity>;

    struct xcp_metrics_t
    {
        std::vector<uint16_t> vcn_busy;
        std::vector<uint16_t> jpeg_busy;
    };

    auto deserialize_xcp_metrics = [](const std::vector<uint8_t>& serialized_data,
                                      bool& _is_vcn_supported, bool& _is_jpeg_supported,
                                      std::vector<xcp_metrics_t>& result) {
        if(serialized_data.size() < 5)
        {
            throw std::runtime_error("Invalid serialized data: insufficient header size");
        }

        size_t offset = 0;

        _is_vcn_supported   = static_cast<bool>(serialized_data[offset++]);
        _is_jpeg_supported  = static_cast<bool>(serialized_data[offset++]);
        uint8_t chunk_count = serialized_data[offset++];
        uint8_t vcn_count   = serialized_data[offset++];
        uint8_t jpeg_count  = serialized_data[offset++];

        constexpr size_t elem_size  = sizeof(uint16_t) / sizeof(uint8_t);
        const size_t     chunk_size = (vcn_count + jpeg_count) * elem_size;

        const size_t expected_size = 5 + (chunk_count * chunk_size);
        if(serialized_data.size() != expected_size)
        {
            throw std::runtime_error("Invalid serialized data: size mismatch");
        }

        auto deserialize_uint16_array = [](const std::vector<uint8_t>& data,
                                           size_t& _offset, int array_size) {
            std::vector<uint16_t> _result;
            _result.reserve(array_size);

            for(int i = 0; i < array_size; ++i)
            {
                if(_offset + 1 >= data.size())
                {
                    throw std::runtime_error(
                        "Invalid serialized data: unexpected end of data");
                }

                uint16_t value = static_cast<uint16_t>(data[_offset]) |
                                 (static_cast<uint16_t>(data[_offset + 1]) << 8);
                _result.push_back(value);
                _offset += 2;
            }

            return _result;
        };

        result.reserve(chunk_count);

        for(size_t count = 0; count < chunk_count; ++count)
        {
            xcp_metrics_t entry;
            entry.vcn_busy = deserialize_uint16_array(serialized_data, offset, vcn_count);
            entry.jpeg_busy =
                deserialize_uint16_array(serialized_data, offset, jpeg_count);

            result.emplace_back(std::move(entry));
        }
    };

    return [&](const storage_parsed_type_base& parsed) {
        auto _amd_smi = static_cast<const struct amd_smi_sample&>(parsed);

        using pos = trace_cache::amd_smi_sample::settings_positions;
        std::bitset<8> settings_bits(_amd_smi.settings);
        bool           is_busy_enabled = settings_bits.test(static_cast<int>(pos::busy));
        bool           is_temp_enabled = settings_bits.test(static_cast<int>(pos::temp));
        bool is_power_enabled          = settings_bits.test(static_cast<int>(pos::power));
        bool is_mem_usage_enabled = settings_bits.test(static_cast<int>(pos::mem_usage));
        bool is_vcn_enabled  = settings_bits.test(static_cast<int>(pos::vcn_activity));
        bool is_jpeg_enabled = settings_bits.test(static_cast<int>(pos::jpeg_activity));

        auto _ts        = _amd_smi.timestamp;
        auto _device_id = _amd_smi.device_id;

        auto setup_tracks = [&]() {
            if(amd_smi_gfx_track::exists(_device_id)) return;

            auto make_track_name = [&](const char* metric) {
                return JOIN(" ", "GPU", JOIN("", '[', _device_id, ']'), metric, "(S)");
            };

            if(is_busy_enabled)
            {
                amd_smi_gfx_track::emplace(_device_id, make_track_name("GFX Busy"), "%");
                amd_smi_umc_track::emplace(_device_id, make_track_name("UMC Busy"), "%");
                amd_smi_mm_track::emplace(_device_id, make_track_name("MM Busy"), "%");
            }
            if(is_temp_enabled)
            {
                amd_smi_temp_track::emplace(_device_id, make_track_name("Temperature"),
                                            "deg C");
            }
            if(is_power_enabled)
            {
                amd_smi_power_track::emplace(_device_id, make_track_name("Power"), "W");
            }
            if(is_mem_usage_enabled)
            {
                amd_smi_mem_track::emplace(_device_id, make_track_name("Memory Usage"),
                                           "MB");
            }
        };

        setup_tracks();

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
            TRACE_COUNTER("device_memory_usage", amd_smi_mem_track::at(_device_id, 0),
                          _ts, mem_mb);
        }

        if(is_vcn_enabled || is_jpeg_enabled)
        {
            std::vector<xcp_metrics_t> xcp_metrics;
            bool is_vcn_activity_supported, is_jpeg_activity_supported;
            deserialize_xcp_metrics(_amd_smi.xcp_activity, is_vcn_activity_supported,
                                    is_jpeg_activity_supported, xcp_metrics);

            auto generate_track_key = [](uint32_t _dev_idx, size_t _xcp_idx,
                                         size_t _clk_idx) {
                return (static_cast<uint64_t>(_dev_idx) << 16) |
                       (static_cast<uint64_t>(_xcp_idx) << 8) |
                       static_cast<uint64_t>(_clk_idx);
            };

            if(is_vcn_enabled && !xcp_metrics.empty())
            {
                for(size_t xcp_idx = 0; xcp_idx < xcp_metrics.size(); ++xcp_idx)
                {
                    for(size_t clk = 0; clk < xcp_metrics[xcp_idx].vcn_busy.size(); ++clk)
                    {
                        auto val = xcp_metrics[xcp_idx].vcn_busy[clk];
                        if(val == std::numeric_limits<uint16_t>::max()) continue;

                        // WiP: Create track name pattern matching the one seen in
                        // amd_smi.cpp
                        auto track_name =
                            (xcp_metrics.size() == 1)
                                ? JOIN(" ", "GPU", JOIN("", '[', _device_id, ']'),
                                       "VCN Activity",
                                       JOIN("", "[", (clk < 10 ? "0" : ""), clk, ']'),
                                       "(S)")
                                : JOIN(" ", "GPU", JOIN("", '[', _device_id, ']'),
                                       "VCN Activity",
                                       JOIN("", "XCP_", xcp_idx, ": [",
                                            (clk < 10 ? "0" : ""), clk, ']'),
                                       "(S)");

                        auto unique_key = generate_track_key(_device_id, xcp_idx, clk);
                        if(!amd_smi_vcn_track::exists(unique_key))
                        {
                            amd_smi_vcn_track::emplace(unique_key, track_name, "%");
                        }
                        TRACE_COUNTER("device_vcn_activity",
                                      amd_smi_vcn_track::at(unique_key, 0), _ts, val);
                    }
                }
            }

            if(is_jpeg_enabled && !xcp_metrics.empty())
            {
                for(size_t xcp_idx = 0; xcp_idx < xcp_metrics.size(); ++xcp_idx)
                {
                    for(size_t clk = 0; clk < xcp_metrics[xcp_idx].jpeg_busy.size();
                        ++clk)
                    {
                        auto val = xcp_metrics[xcp_idx].jpeg_busy[clk];
                        if(val == std::numeric_limits<uint16_t>::max()) continue;

                        auto track_name =
                            (xcp_metrics.size() == 1)
                                ? JOIN(" ", "GPU", JOIN("", '[', _device_id, ']'),
                                       "JPEG Activity",
                                       JOIN("", "[", (clk < 10 ? "0" : ""), clk, ']'),
                                       "(S)")
                                : JOIN(" ", "GPU", JOIN("", '[', _device_id, ']'),
                                       "JPEG Activity",
                                       JOIN("", "XCP_", xcp_idx, ": [",
                                            (clk < 10 ? "0" : ""), clk, ']'),
                                       "(S)");
                        auto unique_key = generate_track_key(_device_id, xcp_idx, clk);
                        if(!amd_smi_jpeg_track::exists(unique_key))
                        {
                            amd_smi_jpeg_track::emplace(unique_key, track_name, "%");
                        }
                        TRACE_COUNTER("device_jpeg_activity",
                                      amd_smi_jpeg_track::at(unique_key, 0), _ts, val);
                    }
                }
            }
        }
    };
}

postprocessing_callback
perfetto_post_processing::get_in_time_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto _sample = static_cast<const struct in_time_sample&>(parsed);

        auto event_metadata = nlohmann::json::parse(_sample.event_metadata);

        auto _track_name = _sample.track_name;
        auto _timestamp  = _sample.timestamp_ns;

        std::string _name       = event_metadata.value("name", "");
        std::string _event_type = event_metadata.value("event_type", "");
        std::string _target     = event_metadata.value("target", "");

        auto _track_uuid = std::hash<std::string>{}(_track_name);

        auto _track = get_track(category::kokkos{}, _track_name, _track_uuid);
        auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
            annotate_perfetto(ctx, { { "timestamp_ns", _timestamp },
                                     { "event_type", _event_type },
                                     { "target", _target } });
        };

        TRACE_EVENT_INSTANT(trait::name<category::kokkos>::value,
                            ::perfetto::DynamicString{ _name }, _track, _timestamp,
                            add_perfetto_annotations);
    };
}

void
perfetto_post_processing::register_parser_callback(
    [[maybe_unused]] storage_parser& parser)
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_caching_perfetto())
    {
        return;
    }
    parser.register_type_callback(entry_type::region, get_region_callback());
    parser.register_type_callback(entry_type::kernel_dispatch,
                                  get_kernel_dispatch_callback());
    parser.register_type_callback(entry_type::memory_copy, get_memory_copy_callback());
    parser.register_type_callback(entry_type::memory_alloc,
                                  get_memory_allocate_callback());
    parser.register_type_callback(entry_type::cpu_freq_sample,
                                  get_cpu_freq_sample_callback());
    parser.register_type_callback(entry_type::amd_smi_sample,
                                  get_amd_smi_sample_callback());
    parser.register_type_callback(entry_type::backtrace_region_sample,
                                  get_backtrace_sample_callback());
    parser.register_type_callback(entry_type::pmc_event_with_sample,
                                  get_pmc_event_with_sample_callback());
    parser.register_type_callback(entry_type::in_time_sample,
                                  get_in_time_sample_callback());

    ROCPROFSYS_DEBUG("Buffer parser callbacks are registered..");
#endif
}

}  // namespace trace_cache
}  // namespace rocprofsys
