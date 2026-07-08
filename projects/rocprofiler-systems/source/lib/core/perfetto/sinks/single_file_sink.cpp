// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks/single_file_sink.hpp"
#include "core/perfetto/sinks/io_helpers.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/locked_file_append.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <utility>

namespace rocprofsys::core
{
single_file_sink::single_file_sink(output_file_registry& registry,
                                   std::string           output_filename_override)
: m_registry{ registry }
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

    m_next_source_base = config.seq_id_base + 1;
    m_source_stride =
        config.seq_id_window_size / static_cast<std::uint32_t>(config.source_count);
    m_seq_id_window_limit_exclusive = seq_id_base64 + window_size64 + 1;

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

    if(m_buffer.capacity() < m_buffer.size() + bytes.size())
        m_buffer.reserve(m_buffer.size() + bytes.size() + bytes.size() / 8);

    static constexpr std::size_t SINGLE_FILE_BUFFER_WARN_THRESHOLD =
        std::size_t{ 1 } * 1024 * 1024 * 1024;  // 1 GiB
    if(m_buffer.size() < SINGLE_FILE_BUFFER_WARN_THRESHOLD &&
       m_buffer.size() + bytes.size() >= SINGLE_FILE_BUFFER_WARN_THRESHOLD)
    {
        LOG_WARNING("single_file_sink in-memory buffer crossed 1 GiB; large MPI "
                    "traces may exhaust host memory. Consider switching to "
                    "ROCPROFSYS_PERFETTO_COMBINE_TRACES=OFF.");
    }

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
            perfetto_sink_detail::emit_size_line(filename, m_buffer.size());
            m_registry.get().register_file(filename, output_format::perfetto);
        }
        else
        {
            if(status == locked_append_status::open_failed)
                perfetto_sink_detail::emit_open_error_line(filename);
            LOG_ERROR("single_file_sink: append-with-flock failed for {} ({})", filename,
                      status_name(status));
        }
    }
    else if(!perfetto_sink_detail::write_proto_to(filename, m_buffer.data(),
                                                  m_buffer.size(), m_registry.get(),
                                                  !explicit_non_append_output))
    {
        LOG_ERROR("single_file_sink: failed to open '{}'", filename);
    }

    m_buffer.clear();
    m_source_seq_id_bases.clear();
    m_output_disabled = false;
}
}  // namespace rocprofsys::core
