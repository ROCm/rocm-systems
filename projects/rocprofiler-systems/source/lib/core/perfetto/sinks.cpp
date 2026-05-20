// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

namespace rocprofsys
{
namespace core
{
namespace
{
// Emits the user-facing "[rocprofsys][<rank>]> <file> perfetto (...)... Done"
// stderr line that announces a written output file. Local to the perfetto
// sinks until a project-wide file-write notification helper exists.
void
emit_size_line(const std::string& filename, std::size_t bytes)
{
    if(config::get_verbose() < 0) return;
    std::fprintf(
        stderr, "[rocprofsys][%i]> %s perfetto (%.2f KB / %.2f MB / %.2f GB)... Done\n",
        dmp::rank(), filename.c_str(), static_cast<double>(bytes) / units::KB,
        static_cast<double>(bytes) / units::MB, static_cast<double>(bytes) / units::GB);
    std::fflush(stderr);
}

void
emit_open_error_line(const std::string& filename)
{
    if(config::get_verbose() < 0) return;
    std::fprintf(stderr, "[rocprofsys][%i]> Error opening '%s'...\n", dmp::rank(),
                 filename.c_str());
    std::fflush(stderr);
}
}  // namespace

// ----------------------------------------------------------------------------
// live_fd_sink
// ----------------------------------------------------------------------------

live_fd_sink::live_fd_sink(bool* perfetto_output_error, output_file_registry& registry)
: m_output_error{ perfetto_output_error }
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

    auto trace_data = std::move(m_bytes);
    auto _filename  = config::get_perfetto_output_filename();

    if(config::output_filtering::is_output_enabled_for_current_mpi_rank())
    {
        if(!trace_data.empty())
        {
            std::ofstream ofs{};
            if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
            {
                emit_open_error_line(_filename);
                if(m_output_error) *m_output_error = true;
            }
            else
            {
                ofs.write(trace_data.data(), trace_data.size());
                emit_size_line(_filename, trace_data.size());
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

    std::ofstream ofs{};
    if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
    {
        emit_open_error_line(_filename);
        LOG_ERROR("per_pid_file_sink: failed to open '{}' for pid {}", _filename, pid);
        return;
    }

    ofs.write(bytes.data(), bytes.size());
    emit_size_line(_filename, bytes.size());
    if(m_registry) m_registry->register_file(_filename, output_format::perfetto);
    ofs.close();
}

void
per_pid_file_sink::finalize()
{}

// ----------------------------------------------------------------------------
// single_file_sink
// ----------------------------------------------------------------------------

single_file_sink::single_file_sink(output_file_registry& registry)
: m_registry{ &registry }
{}

void
single_file_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    if(bytes.empty()) return;

    auto [it, inserted] = m_source_seq_ids.try_emplace(source_id, m_next_seq_id);
    if(inserted) ++m_next_seq_id;
    const auto seq_id = it->second;

    std::size_t pos = 0;
    while(pos < bytes.size())
    {
        auto tag = static_cast<std::uint8_t>(bytes[pos]);
        if(tag != TRACE_PACKETS_TAG)
        {
            LOG_ERROR("single_file_sink: source {} has malformed Trace.packets "
                      "framing at offset {} (tag=0x{:02x}); dropping remainder",
                      source_id, pos, static_cast<unsigned>(tag));
            return;
        }
        ++pos;

        std::uint64_t len = 0;
        if(!read_varint(bytes.data(), bytes.size(), pos, len) || len > bytes.size() - pos)
        {
            LOG_ERROR("single_file_sink: source {} has truncated Trace.packets "
                      "frame at offset {}; dropping remainder",
                      source_id, pos);
            return;
        }

        if(!rewrite_trace_packet(m_buffer, bytes.data() + pos,
                                 static_cast<std::size_t>(len), seq_id))
        {
            LOG_ERROR("single_file_sink: source {} TracePacket malformed at "
                      "offset {}; dropping remainder",
                      source_id, pos);
            return;
        }
        pos += static_cast<std::size_t>(len);
    }
}

void
single_file_sink::finalize()
{
    auto _filename = config::get_perfetto_output_filename();

    if(!config::output_filtering::is_output_enabled_for_current_mpi_rank())
    {
        m_buffer.clear();
        m_source_seq_ids.clear();
        return;
    }

    if(m_buffer.empty())
    {
        if(dmp::rank() == 0)
            LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                      _filename);
        return;
    }

    std::ofstream ofs{};
    if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
    {
        emit_open_error_line(_filename);
        LOG_ERROR("single_file_sink: failed to open '{}'", _filename);
        return;
    }

    ofs.write(m_buffer.data(), m_buffer.size());
    emit_size_line(_filename, m_buffer.size());
    if(m_registry) m_registry->register_file(_filename, output_format::perfetto);
    ofs.close();

    m_buffer.clear();
    m_source_seq_ids.clear();
}

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
