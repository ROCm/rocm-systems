// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/sinks/append_mode.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
class output_file_registry;

namespace core
{
// Cached-mode sink: concatenates per-pid bytes into one .proto file.
// Each source_id receives a disjoint trusted_packet_sequence_id range before
// packets are appended, preserving Perfetto interned-data namespaces.
class single_file_sink
{
public:
    // output_filename_override empty -> resolve via
    // config::get_perfetto_output_filename() at finalize time. Set to a concrete
    // path to write to a different location than the configured base.
    explicit single_file_sink(output_file_registry& registry,
                              std::string           output_filename_override = {});

    single_file_sink(single_file_sink&&) noexcept            = default;
    single_file_sink& operator=(single_file_sink&&) noexcept = default;
    single_file_sink(const single_file_sink&)                = delete;
    single_file_sink& operator=(const single_file_sink&)     = delete;
    ~single_file_sink()                                      = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

    // Switch the sink into append-with-file-lock mode for cross-process
    // aggregation. `seq_id_base` shifts this process's seq_id namespace so
    // concurrent appenders do not collide on trusted_packet_sequence_id.
    void set_append_mode(append_mode_config config) noexcept;

    [[nodiscard]] const std::vector<char>& buffer_for_testing() const noexcept
    {
        return m_buffer;
    }

private:
    static constexpr std::uint32_t PER_SOURCE_SEQ_ID_BASE_STRIDE = 1u << 16;

    static constexpr std::uint64_t TRUSTED_SEQ_ID_MAX_EXCLUSIVE =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    std::reference_wrapper<output_file_registry> m_registry;
    std::string                                  m_output_filename_override{};
    std::vector<char>                            m_buffer{};
    std::unordered_map<int, std::uint32_t>       m_source_seq_id_bases{};
    std::uint64_t                                m_next_source_base{ 1 };
    bool                                         m_append_mode{ false };
    std::uint32_t m_source_stride{ PER_SOURCE_SEQ_ID_BASE_STRIDE };
    std::uint64_t m_seq_id_window_limit_exclusive{ TRUSTED_SEQ_ID_MAX_EXCLUSIVE };
    bool          m_output_disabled{ false };
};
}  // namespace core
}  // namespace rocprofsys
