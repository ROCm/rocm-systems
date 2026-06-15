// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks.hpp"
#include <cstdint>

#include "common/units.hpp"
#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/locked_file_append.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

#include <system_error>
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
    try
    {
        return config::get_verbose() >= 0 &&
               config::output_filtering::is_log_output_enabled_for_current_mpi_rank();
    } catch(const std::exception&)
    {
        return false;
    }
}

void
emit_size_line(const std::string& filename, std::size_t bytes)
{
    if(!stderr_log_allowed()) return;
    std::fprintf(
        stderr, "[rocprofsys][%i]> %s perfetto (%.2f KB / %.2f MB / %.2f GB)... Done\n",
        dmp::rank(), filename.c_str(), static_cast<double>(bytes) / units::kilobyte,
        static_cast<double>(bytes) / units::megabyte,
        static_cast<double>(bytes) / units::gigabyte);
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
               output_file_registry* registry, bool emit_status = true)
{
    const auto parent = std::filesystem::path{ filename }.parent_path();
    if(!parent.empty())
    {
        std::error_code ec{};
        std::filesystem::create_directories(parent, ec);
        if(ec)
        {
            LOG_ERROR("write_proto_to: could not create directory '{}': {}",
                      parent.string(), ec.message());
            if(emit_status) emit_open_error_line(filename);
            return false;
        }
    }

    std::ofstream ofs{ filename, std::ios::out | std::ios::binary };
    if(!ofs.is_open() || !ofs.good())
    {
        if(emit_status) emit_open_error_line(filename);
        return false;
    }
    ofs.write(data, static_cast<std::streamsize>(size));
    if(emit_status) emit_size_line(filename, size);
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

// ----------------------------------------------------------------------------
// single_file_sink
// ----------------------------------------------------------------------------

single_file_sink::single_file_sink(output_file_registry& registry,
                                   std::string           output_filename_override)
: m_registry{ &registry }
, m_output_filename_override{ std::move(output_filename_override) }
{}

void
single_file_sink::set_append_mode(append_mode_config config) noexcept
{
    m_append_mode     = true;
    m_output_disabled = false;

    const auto seq_id_base64 = static_cast<std::uint64_t>(config.seq_id_base);
    const auto window_size64 = static_cast<std::uint64_t>(config.seq_id_window_size);
    if(config.source_count == 0 || config.seq_id_window_size == 0 ||
       window_size64 < config.source_count ||
       seq_id_base64 + window_size64 >= TRUSTED_SEQ_ID_MAX_EXCLUSIVE)
    {
        LOG_ERROR("single_file_sink append mode disabled: invalid seq_id window "
                  "base={} window={} sources={}",
                  config.seq_id_base, config.seq_id_window_size, config.source_count);
        m_output_disabled = true;
        return;
    }

    // Shift this process's seq_id namespace so concurrent appenders do not
    // collide on trusted_packet_sequence_id. The +1 preserves the existing
    // "seq_id starts at 1" invariant downstream consumers may rely on.
    m_next_source_base = config.seq_id_base + 1;
    m_source_stride =
        config.seq_id_window_size / static_cast<std::uint32_t>(config.source_count);
    m_seq_id_window_limit_exclusive = seq_id_base64 + window_size64 + 1;

    // Rewriting adds each packet's original seq_id to the source base. A
    // one-wide source slice can only hold original seq_id 0, while cached
    // packets normally carry placeholder seq_id 1. Reject that unusable setup
    // before accepting any source.
    constexpr auto MIN_SOURCE_SEQ_ID_SLICE = std::uint32_t{ 2 };
    if(m_source_stride < MIN_SOURCE_SEQ_ID_SLICE)
    {
        LOG_ERROR("single_file_sink append mode disabled: source_count {} leaves "
                  "seq_id slice {} below minimum {} in window {}",
                  config.source_count, m_source_stride, MIN_SOURCE_SEQ_ID_SLICE,
                  config.seq_id_window_size);
        m_output_disabled = true;
    }
}

void
single_file_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    if(bytes.empty()) return;

    if(m_output_disabled)
    {
        LOG_ERROR("single_file_sink: output disabled after invalid append-mode setup; "
                  "dropping source {}",
                  source_id);
        return;
    }
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

    // Each source gets a disjoint seq_id sub-range; per-packet rewrites
    // add this base to the packet's original seq_id, preserving the
    // per-source iid namespace structure that Perfetto's interned-data
    // resolution depends on.
    auto [it, inserted] = m_source_seq_id_bases.try_emplace(source_id, 0);
    if(inserted)
    {
        const auto source_base64 = static_cast<std::uint64_t>(m_next_source_base);
        if(source_base64 + m_source_stride > m_seq_id_window_limit_exclusive)
        {
            m_source_seq_id_bases.erase(it);
            LOG_ERROR("single_file_sink: source {} would exceed append-mode seq_id "
                      "window (base={} stride={} limit={}); dropping source",
                      source_id, source_base64, m_source_stride,
                      m_seq_id_window_limit_exclusive);
            return;
        }

        it->second = static_cast<std::uint32_t>(m_next_source_base);
        m_next_source_base += m_source_stride;
    }
    const auto seq_id_offset = it->second;

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

        const auto rewrite_status = rewrite_trace_packet_checked(
            m_buffer, bytes.data() + pos, static_cast<std::size_t>(len), seq_id_offset,
            static_cast<std::uint64_t>(seq_id_offset) + m_source_stride);
        if(rewrite_status != rewrite_trace_packet_status::success)
        {
            if(rewrite_status == rewrite_trace_packet_status::seq_id_out_of_range)
            {
                LOG_ERROR("single_file_sink: source {} TracePacket at offset {} "
                          "exceeds seq_id slice [base={}, limit={}); dropping remainder",
                          source_id, pos, seq_id_offset,
                          static_cast<std::uint64_t>(seq_id_offset) + m_source_stride);
            }
            else
            {
                LOG_ERROR("single_file_sink: source {} TracePacket malformed at "
                          "offset {}; dropping remainder",
                          source_id, pos);
            }
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

    if(m_output_disabled)
    {
        m_buffer.clear();
        m_source_seq_id_bases.clear();
        m_output_disabled = false;
        return;
    }

    const auto explicit_non_append_output =
        !m_output_filename_override.empty() && !m_append_mode;
    if(!explicit_non_append_output &&
       !config::output_filtering::is_file_output_enabled_for_current_mpi_rank())
    {
        m_buffer.clear();
        m_source_seq_id_bases.clear();
        m_output_disabled = false;
        return;
    }

    if(m_buffer.empty())
    {
        if(dmp::rank() == 0)
            LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                      filename);
        m_output_disabled = false;
        return;
    }

    if(m_append_mode)
    {
        const auto status =
            append_with_file_lock(filename, m_buffer.data(), m_buffer.size());
        if(status == locked_append_status::success)
        {
            emit_size_line(filename, m_buffer.size());
            if(m_registry) m_registry->register_file(filename, output_format::perfetto);
        }
        else
        {
            if(status == locked_append_status::open_failed)
                emit_open_error_line(filename);
            LOG_ERROR("single_file_sink: append-with-flock failed for {} ({})", filename,
                      status_name(status));
        }
    }
    else if(!write_proto_to(filename, m_buffer.data(), m_buffer.size(), m_registry,
                            !explicit_non_append_output))
    {
        LOG_ERROR("single_file_sink: failed to open '{}'", filename);
    }

    m_buffer.clear();
    m_source_seq_id_bases.clear();
    m_output_disabled = false;
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
