// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "perfetto.hpp"
#include "config.hpp"
#include "debug.hpp"
#include "library/runtime.hpp"
#include "perfetto_fwd.hpp"
#include "utility.hpp"

#include <chrono>

namespace rocprofsys
{
namespace perfetto
{
namespace
{
auto
is_system_backend()
{
    // if get_perfetto_backend() returns 'system' or 'all', this is true
    return (config::get_perfetto_backend() != "inprocess");
}

auto&
get_perfetto_tmp_file(pid_t _pid = process::get_id())
{
    static auto _v = std::unordered_map<pid_t, std::shared_ptr<tmp_file>>{};
    if(_v.find(_pid) == _v.end()) _v.emplace(_pid, std::shared_ptr<tmp_file>{});
    return _v.at(_pid);
}

auto&
get_config()
{
    static auto _v = ::perfetto::TraceConfig{};
    return _v;
}

auto&
get_session(pid_t _pid = process::get_id())
{
    static auto _v =
        std::unordered_map<pid_t, std::unique_ptr<::perfetto::TracingSession>>{};
    if(_v.find(_pid) == _v.end())
        _v.emplace(_pid, std::unique_ptr<::perfetto::TracingSession>{});
    return _v.at(_pid);
}
}  // namespace

void
setup()
{
    auto  args            = ::perfetto::TracingInitArgs{};
    auto  track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto& cfg             = get_config();

    // environment settings
    auto shmem_size_hint = config::get_perfetto_shmem_size_hint();
    auto buffer_size     = config::get_perfetto_buffer_size();

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

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.shmem_size_hint_kb = shmem_size_hint;

    if(get_perfetto_backend() != "inprocess") args.backends |= ::perfetto::kSystemBackend;
    if(get_perfetto_backend() != "system") args.backends |= ::perfetto::kInProcessBackend;

    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();
}

void
start()
{
    if(is_system_backend()) return;

    auto& tracing_session = get_session();

    if(!tracing_session) tracing_session = ::perfetto::Tracing::NewTrace();

    tracing_session = ::perfetto::Tracing::NewTrace();
    auto& _tmp_file = get_perfetto_tmp_file();
    if(config::get_use_tmp_files())
    {
        if(!_tmp_file)
        {
            _tmp_file = config::get_tmp_file("perfetto-trace", "proto");
            _tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, 0600);
        }
    }

    ROCPROFSYS_VERBOSE(2, "Setup perfetto...\n");
    int   _fd = (_tmp_file) ? _tmp_file->fd : -1;
    auto& cfg = get_config();
    tracing_session->SetOnErrorCallback([](::perfetto::TracingError _err) {
        if(_err.code == ::perfetto::TracingError::kTracingFailed)
            ROCPROFSYS_WARNING(0, "perfetto encountered a tracing error: %s\n",
                               _err.message.c_str());
    });
    tracing_session->Setup(cfg, _fd);
    tracing_session->StartBlocking();
}

void
stop()
{
    if(is_system_backend()) return;

    auto& tracing_session = get_perfetto_session();

    ROCPROFSYS_CI_THROW(tracing_session == nullptr,
                        "Null pointer to the tracing session");

    if(tracing_session)
    {
        // Make sure the last event is closed
        ROCPROFSYS_VERBOSE(2, "Flushing the perfetto trace data...\n");
        ::perfetto::TrackEvent::Flush();
        tracing_session->FlushBlocking();

        ROCPROFSYS_VERBOSE(2, "Stopping the perfetto trace session (blocking)...\n");
        tracing_session->StopBlocking();
    }
}

void
post_process(tim::manager* _timemory_manager, bool& _perfetto_output_error)
{
    using char_vec_t = std::vector<char>;

    stop();

    auto& tracing_session = get_perfetto_session();
    if(!tracing_session) return;

    auto _get_session_data = [&tracing_session]() {
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
                return char_vec_t{ tracing_session->ReadTraceBlocking() };
            }

            ::fseek(_fdata, 0, SEEK_END);
            size_t _fnum_elem = ::ftell(_fdata);
            ::fseek(_fdata, 0, SEEK_SET);  // same as rewind(f);

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
            _data = char_vec_t{ tracing_session->ReadTraceBlocking() };
        }

        return _data;
    };

    auto trace_data = char_vec_t{};
#if defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0
    if(get_perfetto_combined_traces())
    {
        using perfetto_mpi_get_t = tim::operation::finalize::mpi_get<char_vec_t, true>;

        auto _trace_data = _get_session_data();
        auto _rank_data  = std::vector<char_vec_t>{};
        auto _combine    = [](char_vec_t& _dst, const char_vec_t& _src) -> char_vec_t& {
            _dst.reserve(_dst.size() + _src.size());
            for(auto&& itr : _src)
                _dst.emplace_back(itr);
            return _dst;
        };

        perfetto_mpi_get_t{ get_perfetto_combined_traces(),
                            settings::node_count() }(_rank_data, _trace_data, _combine);
        for(auto& itr : _rank_data)
            trace_data =
                (trace_data.empty()) ? std::move(itr) : _combine(trace_data, itr);
    }
    else
    {
        trace_data = _get_session_data();
    }
#else
    trace_data = _get_session_data();
#endif

    auto _filename = config::get_perfetto_output_filename();
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
            if(_timemory_manager)
                _timemory_manager->add_file_output("protobuf", "perfetto", _filename);
        }
        ofs.close();
    }
    else if(dmp::rank() == 0)
    {
        ROCPROFSYS_VERBOSE(
            0, "perfetto trace data is empty. File '%s' will not be written...\n",
            _filename.c_str());
    }

    // Merge the output files, if rank 0
    if(dmp::rank() == 0)
    {
        auto _output_folder = filepath::dirname(_filename);
        auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
        auto _script_dir    = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{}, false);

        if(!_script_dir.empty())
        {
            _script_path = rocprofsys::common::join("/", _script_dir, _script_path);
        }

        // Test that the script exists
        if(!filepath::exists(_script_path))
        {
            ROCPROFSYS_VERBOSE(0, "Script not found: %s\n", _script_path.c_str());
        }
        else
        {
            auto _command = _script_path + " '" + _output_folder + "'";

            // Execute the merge script
            int result = system(_command.c_str());

            if(result != 0)
            {
                ROCPROFSYS_VERBOSE(0, "Failed to execute: %s\n", _command.c_str());
            }
            else
            {
                ROCPROFSYS_VERBOSE(0, "Successfully executed: %s\n", _command.c_str());
            }
        }
    }

    auto& _tmp_file = get_perfetto_tmp_file();
    if(_tmp_file)
    {
        _tmp_file->close();
        _tmp_file->remove();
        _tmp_file.reset();
    }
}

}  // namespace perfetto

std::unique_ptr<::perfetto::TracingSession>&
get_perfetto_session(pid_t _pid)
{
    return ::rocprofsys::perfetto::get_session(_pid);
}
}  // namespace rocprofsys

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
