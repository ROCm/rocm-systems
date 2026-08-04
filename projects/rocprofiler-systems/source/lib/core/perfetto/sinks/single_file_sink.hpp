// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/packet_framing.hpp"
#include "core/perfetto/sinks/append_mode.hpp"
#include "core/perfetto/sinks/trace_sink.hpp"

#include <cstdint>
#include <functional>
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
class single_file_sink : public trace_sink
{
public:
    explicit single_file_sink(output_file_registry& registry,
                              std::string           output_filename_override = {});

    void on_source_drained(int source_id, std::vector<char> bytes) override;
    void finalize() override;

    void set_append_mode(append_mode_config config) noexcept;

    [[nodiscard]] const std::vector<char>& buffer_for_testing() const noexcept
    {
        return m_buffer;
    }

private:
    static constexpr std::uint32_t PER_SOURCE_SEQ_ID_BASE_STRIDE = 1u << 16;

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
