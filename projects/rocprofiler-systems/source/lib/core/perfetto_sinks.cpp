// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto_sinks.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <timemory/manager/manager.hpp>
#include <timemory/operations/types/file_output_message.hpp>

#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

namespace rocprofsys
{
namespace core
{
// ----------------------------------------------------------------------------
// live_fd_sink
// ----------------------------------------------------------------------------

live_fd_sink::live_fd_sink(tim::manager*         timemory_manager,
                           bool*                 perfetto_output_error,
                           output_file_registry& registry)
: m_manager{ timemory_manager }
, m_output_error{ perfetto_output_error }
, m_registry{ &registry }
{}

void
live_fd_sink::on_source_drained(int /*source_id*/, std::vector<char> bytes)
{
    // Live mode produces exactly one source per engine.stop(): the
    // process-wide TracingSession bytes for the current pid.
    m_bytes   = std::move(bytes);
    m_drained = true;
}

void
live_fd_sink::finalize()
{
    if(!m_drained) return;

    using char_vec_t = std::vector<char>;

    char_vec_t trace_data{};

#if defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0
    if(config::get_perfetto_combined_traces())
    {
        using perfetto_mpi_get_t = tim::operation::finalize::mpi_get<char_vec_t, true>;

        auto _trace_data = std::move(m_bytes);
        auto _rank_data  = std::vector<char_vec_t>{};
        auto _combine    = [](char_vec_t& _dst, const char_vec_t& _src) -> char_vec_t& {
            _dst.reserve(_dst.size() + _src.size());
            for(auto&& itr : _src)
                _dst.emplace_back(itr);
            return _dst;
        };

        perfetto_mpi_get_t{ config::get_perfetto_combined_traces(),
                            settings::node_count() }(_rank_data, _trace_data, _combine);
        for(auto& itr : _rank_data)
            trace_data =
                (trace_data.empty()) ? std::move(itr) : _combine(trace_data, itr);
    }
    else
    {
        trace_data = std::move(m_bytes);
    }
#else
    trace_data = std::move(m_bytes);
#endif

    auto _filename = config::get_perfetto_output_filename();

    if(config::output_filtering::is_output_enabled_for_current_mpi_rank())
    {
        if(!trace_data.empty())
        {
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
                _fom.append("Error opening '%s'...", _filename.c_str());
                if(m_output_error) *m_output_error = true;
            }
            else
            {
                ofs.write(trace_data.data(), trace_data.size());
                if(config::get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
                if(m_manager)
                    m_manager->add_file_output("protobuf", "perfetto", _filename);
                if(m_registry)
                    m_registry->register_file(_filename, output_format::perfetto);
            }
            ofs.close();
        }
        else if(dmp::rank() == 0)
        {
            LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                      _filename);
        }
    }

    if(dmp::rank() == 0)
    {
        auto _output_folder = filepath::dirname(_filename);
        auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
        auto _script_dir    = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{}, false);

        if(!_script_dir.empty())
            _script_path = fmt::format("{}/{}", _script_dir, _script_path);

        if(!filepath::exists(_script_path))
        {
            LOG_WARNING("Script not found: {}", _script_path);
        }
        else
        {
            auto _command = _script_path + " '" + _output_folder + "'";
            int  result   = std::system(_command.c_str());
            if(result != 0)
                LOG_ERROR("Failed to execute: {}", _command);
            else
                LOG_INFO("Successfully executed: {}", _command);
        }
    }

    m_drained = false;
    m_bytes.clear();
}

// ----------------------------------------------------------------------------
// per_pid_file_sink
// ----------------------------------------------------------------------------

per_pid_file_sink::per_pid_file_sink(pid_t parent_pid, output_file_registry& registry)
: m_parent_pid{ parent_pid }
, m_registry{ &registry }
{}

void
per_pid_file_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    if(bytes.empty()) return;

    const auto pid = static_cast<pid_t>(source_id);
    auto       _filename =
        (pid == m_parent_pid)
            ? config::get_perfetto_output_filename()
            : config::get_perfetto_output_filename_with_suffix(std::to_string(pid));

    operation::file_output_message<tim::project::rocprofsys> _fom{};
    if(config::get_verbose() >= 0)
        _fom(_filename, std::string{ "perfetto" },
             " (%.2f KB / %.2f MB / %.2f GB)... ",
             static_cast<double>(bytes.size()) / units::KB,
             static_cast<double>(bytes.size()) / units::MB,
             static_cast<double>(bytes.size()) / units::GB);

    std::ofstream ofs{};
    if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
    {
        _fom.append("Error opening '%s'...", _filename.c_str());
        LOG_ERROR("per_pid_file_sink: failed to open '{}' for pid {}", _filename,
                  pid);
        return;
    }

    ofs.write(bytes.data(), bytes.size());
    if(config::get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
    if(m_registry) m_registry->register_file(_filename, output_format::perfetto);
    ofs.close();
}

void
per_pid_file_sink::finalize()
{}

// ----------------------------------------------------------------------------
// recording_sink
// ----------------------------------------------------------------------------

void
recording_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    m_records.emplace_back(source_id, std::move(bytes));
}

void
recording_sink::finalize()
{
    m_finalized = true;
}
}  // namespace core
}  // namespace rocprofsys
