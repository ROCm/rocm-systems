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

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <ios>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <utility>

namespace rocprofsys
{
namespace core
{
namespace
{
// Gates both global verbosity and the per-rank log-output filter; either
// silencer suppresses the stderr lines emitted below.
[[nodiscard]] bool
stderr_log_allowed() noexcept
{
    return config::get_verbose() >= 0 &&
           config::output_filtering::is_log_output_enabled_for_current_mpi_rank();
}

void
emit_size_line(const std::string& filename, std::size_t bytes)
{
    if(!stderr_log_allowed()) return;
    std::fprintf(
        stderr, "[rocprofsys][%i]> %s perfetto (%.2f KB / %.2f MB / %.2f GB)... Done\n",
        dmp::rank(), filename.c_str(), static_cast<double>(bytes) / units::KB,
        static_cast<double>(bytes) / units::MB, static_cast<double>(bytes) / units::GB);
    std::fflush(stderr);
}

void
emit_open_error_line(const std::string& filename)
{
    if(!stderr_log_allowed()) return;
    std::fprintf(stderr, "[rocprofsys][%i]> Error opening '%s'...\n", dmp::rank(),
                 filename.c_str());
    std::fflush(stderr);
}

bool
write_proto_to(const std::string& filename, const char* data, std::size_t size,
               output_file_registry* registry)
{
    std::ofstream ofs{};
    if(!filepath::open(ofs, filename, std::ios::out | std::ios::binary))
    {
        emit_open_error_line(filename);
        return false;
    }
    ofs.write(data, static_cast<std::streamsize>(size));
    emit_size_line(filename, size);
    if(registry) registry->register_file(filename, output_format::perfetto);
    return true;
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
    // process-wide TracingSession bytes for the current pid. A second
    // call would silently lose either the prior bytes or the new ones,
    // and the only way to reach here is an engine misconfiguration;
    // surface it loudly and refuse to overwrite.
    if(m_drained)
    {
        LOG_ERROR("live_fd_sink received a second source drain; ignoring (the "
                  "single-source live-mode contract was violated)");
        return;
    }
    m_bytes   = std::move(bytes);
    m_drained = true;
}

void
live_fd_sink::finalize()
{
    if(!m_drained) return;

    auto trace_data = std::move(m_bytes);
    auto filename   = config::get_perfetto_output_filename();

    if(config::output_filtering::is_file_output_enabled_for_current_mpi_rank())
    {
        if(!trace_data.empty())
        {
            if(!write_proto_to(filename, trace_data.data(), trace_data.size(),
                               m_registry) &&
               m_output_error)
            {
                *m_output_error = true;
            }
        }
        else if(dmp::rank() == 0)
        {
            LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                      filename);
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
    auto       filename =
        (pid == m_parent_pid)
                  ? config::get_perfetto_output_filename()
                  : config::get_perfetto_output_filename_with_suffix(std::to_string(pid));

    if(!write_proto_to(filename, bytes.data(), bytes.size(), m_registry))
    {
        LOG_ERROR("per_pid_file_sink: failed to open '{}' for pid {}", filename, pid);
    }
}

void
per_pid_file_sink::finalize()
{}

namespace
{
// Append the entire payload to the file at `filename` under an exclusive
// flock that lives only for the duration of the write. flock auto-releases
// on process exit, so a crash mid-write leaves the partial bytes in place
// without holding the lock; the next process gets the lock immediately.
// The lock serializes writers; concurrent readers (e.g. a future merge
// inspection tool) see whatever flushed bytes existed at their read point.
bool
append_with_flock(const std::string& filename, const char* data, std::size_t size,
                  output_file_registry* registry)
{
    // TIME_OUTPUT=ON puts the merged file under a timestamped subdirectory
    // that no other code may have created yet on this process; ensure the
    // parent exists before O_CREAT, matching filepath::open()'s behavior for
    // the ofstream-based writers.
    auto parent = filepath::dirname(filename);
    if(!parent.empty()) filepath::makedir(parent);

    int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if(fd < 0)
    {
        emit_open_error_line(filename);
        return false;
    }

    if(::flock(fd, LOCK_EX) != 0)
    {
        const int err = errno;
        ::close(fd);
        LOG_ERROR("single_file_sink: flock(LOCK_EX) failed on '{}': {}", filename,
                  std::strerror(err));
        return false;
    }

    bool        ok     = true;
    const char* ptr    = data;
    std::size_t remain = size;
    while(remain > 0)
    {
        const ssize_t n = ::write(fd, ptr, remain);
        if(n < 0)
        {
            const int err = errno;
            if(err == EINTR) continue;
            LOG_ERROR("single_file_sink: write to '{}' failed: {}", filename,
                      std::strerror(err));
            ok = false;
            break;
        }
        ptr += n;
        remain -= static_cast<std::size_t>(n);
    }

    ::flock(fd, LOCK_UN);
    ::close(fd);

    if(ok)
    {
        emit_size_line(filename, size);
        if(registry) registry->register_file(filename, output_format::perfetto);
    }
    return ok;
}
}  // namespace

// ----------------------------------------------------------------------------
// single_file_sink
// ----------------------------------------------------------------------------

single_file_sink::single_file_sink(output_file_registry& registry,
                                   std::string           output_filename_override)
: m_registry{ &registry }
, m_output_filename_override{ std::move(output_filename_override) }
{}

void
single_file_sink::set_append_mode(std::uint32_t seq_id_base) noexcept
{
    m_append_mode = true;
    // Shift this process's seq_id namespace so concurrent appenders do not
    // collide on trusted_packet_sequence_id. The +1 preserves the existing
    // "seq_id starts at 1" invariant downstream consumers may rely on.
    m_next_seq_id = seq_id_base + 1;
}

void
single_file_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    if(bytes.empty()) return;

    // Rewriting preserves payload size and adds a small per-packet header,
    // so each source contributes at most slightly more than bytes.size() to
    // the buffer. Reserve up front to skip the geometric-growth realloc
    // cascade across many small per-packet appends.
    if(m_buffer.capacity() < m_buffer.size() + bytes.size())
        m_buffer.reserve(m_buffer.size() + bytes.size() + bytes.size() / 8);

    static constexpr std::size_t SINGLE_FILE_BUFFER_WARN_THRESHOLD =
        std::size_t{ 1 } * 1024 * 1024 * 1024;
    if(m_buffer.size() < SINGLE_FILE_BUFFER_WARN_THRESHOLD &&
       m_buffer.size() + bytes.size() >= SINGLE_FILE_BUFFER_WARN_THRESHOLD)
    {
        LOG_WARNING("single_file_sink in-memory buffer crossed 1 GiB; large MPI "
                    "traces may exhaust host memory. Consider switching to "
                    "ROCPROFSYS_PERFETTO_OUTPUT_LAYOUT=per_process.");
    }

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
    auto filename = m_output_filename_override.empty()
                        ? config::get_perfetto_output_filename()
                        : m_output_filename_override;

    if(!config::output_filtering::is_file_output_enabled_for_current_mpi_rank())
    {
        m_buffer.clear();
        m_source_seq_ids.clear();
        return;
    }

    if(m_buffer.empty())
    {
        if(dmp::rank() == 0)
            LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                      filename);
        return;
    }

    if(m_append_mode)
    {
        if(!append_with_flock(filename, m_buffer.data(), m_buffer.size(), m_registry))
        {
            LOG_ERROR("single_file_sink: append-with-flock failed for '{}'", filename);
        }
    }
    else if(!write_proto_to(filename, m_buffer.data(), m_buffer.size(), m_registry))
    {
        LOG_ERROR("single_file_sink: failed to open '{}'", filename);
    }

    m_buffer.clear();
    m_source_seq_ids.clear();
}

// ----------------------------------------------------------------------------
// tee_sink
// ----------------------------------------------------------------------------

tee_sink::tee_sink(per_pid_file_sink per_pid, single_file_sink single_file)
: m_per_pid{ std::move(per_pid) }
, m_single_file{ std::move(single_file) }
{}

void
tee_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    auto copy = bytes;
    m_per_pid.on_source_drained(source_id, std::move(copy));
    m_single_file.on_source_drained(source_id, std::move(bytes));
}

void
tee_sink::finalize()
{
    m_per_pid.finalize();
    m_single_file.finalize();
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
