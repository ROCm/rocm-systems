// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
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

    using char_vec_t = std::vector<char>;

    char_vec_t trace_data{};

#if defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0
    if(config::get_perfetto_combined_traces())
    {
        // Held pending a project-local mpi gather helper that handles
        // variable-size byte buffers and the collapse_processes setting.
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
            // fork+execlp avoids the shell-injection footprint of std::system
            // when _output_folder contains characters that would terminate the
            // single-quoted argument (a single quote, in particular).
            pid_t pid = ::fork();
            if(pid < 0)
            {
                LOG_ERROR("fork failed for merge script {}: errno={}", _script_path,
                          errno);
            }
            else if(pid == 0)
            {
                ::execlp(_script_path.c_str(), _script_path.c_str(),
                         _output_folder.c_str(), nullptr);
                // execlp only returns on failure
                ::_exit(127);
            }
            else
            {
                int status = 0;
                while(::waitpid(pid, &status, 0) < 0 && errno == EINTR)
                {
                }
                if(WIFEXITED(status) && WEXITSTATUS(status) == 0)
                    LOG_INFO("Successfully executed: {} {}", _script_path,
                             _output_folder);
                else
                    LOG_ERROR("Failed to execute: {} {} (status={})", _script_path,
                              _output_folder, status);
            }
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

namespace
{
// Trace.packets framing wire tag: field 1, wire type 2 (length-delimited).
constexpr std::uint8_t k_trace_packets_tag = 0x0A;
// TracePacket.trusted_packet_sequence_id wire tag: field 10, wire type 0 (varint).
constexpr std::uint8_t k_trusted_seq_id_tag = 0x50;

bool
read_varint(const char* data, std::size_t size, std::size_t& pos, std::uint64_t& out)
{
    out                 = 0;
    std::uint32_t shift = 0;
    while(pos < size)
    {
        auto b = static_cast<std::uint8_t>(data[pos++]);
        out |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if((b & 0x80) == 0) return true;
        shift += 7;
        if(shift >= 64) return false;
    }
    return false;
}

void
append_varint(std::vector<char>& dst, std::uint64_t v)
{
    while(v >= 0x80)
    {
        dst.push_back(static_cast<char>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    dst.push_back(static_cast<char>(v));
}

// Walks one TracePacket payload, copies every field verbatim EXCEPT
// trusted_packet_sequence_id (field 10), then appends a fresh field 10
// with `new_seq_id`. Returns false on malformed input (the caller drops
// the remainder of the source's bytes rather than risk emitting garbage).
bool
rewrite_trace_packet(std::vector<char>& dst, const char* packet, std::size_t size,
                     std::uint32_t new_seq_id)
{
    std::vector<char> rewritten;
    rewritten.reserve(size + 5);

    std::size_t pos = 0;
    while(pos < size)
    {
        std::size_t   tag_start = pos;
        std::uint64_t tag       = 0;
        if(!read_varint(packet, size, pos, tag)) return false;
        const std::uint32_t wire = tag & 0x7;

        std::size_t value_end = pos;
        switch(wire)
        {
            case 0:  // varint
            {
                std::uint64_t v = 0;
                if(!read_varint(packet, size, pos, v)) return false;
                value_end = pos;
                break;
            }
            case 2:  // length-delimited
            {
                std::uint64_t len = 0;
                if(!read_varint(packet, size, pos, len)) return false;
                if(len > size - pos) return false;
                pos += static_cast<std::size_t>(len);
                value_end = pos;
                break;
            }
            case 1:
                pos += 8;
                value_end = pos;
                break;
            case 5:
                pos += 4;
                value_end = pos;
                break;
            default: return false;  // group/unknown wire types
        }
        if(value_end > size) return false;

        if(tag == k_trusted_seq_id_tag) continue;  // re-emitted below
        rewritten.insert(rewritten.end(), packet + tag_start, packet + value_end);
    }

    rewritten.push_back(static_cast<char>(k_trusted_seq_id_tag));
    append_varint(rewritten, new_seq_id);

    dst.push_back(static_cast<char>(k_trace_packets_tag));
    append_varint(dst, rewritten.size());
    dst.insert(dst.end(), rewritten.begin(), rewritten.end());
    return true;
}
}  // namespace

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
        if(tag != k_trace_packets_tag)
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
