// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocprofsys::core
{
// Stride between per-process seq_id ranges in the cross-process merged output.
// Each rocprof-sys instance gets a disjoint slice of the
// trusted_packet_sequence_id space when writing to the shared merged file.
inline constexpr std::uint32_t MERGED_SEQ_ID_RANK_STRIDE = 1u << 20;

struct append_mode_config
{
    std::uint32_t seq_id_base        = 0;
    std::uint32_t seq_id_window_size = MERGED_SEQ_ID_RANK_STRIDE;
    std::size_t   source_count       = 1;
};

[[nodiscard]] constexpr std::optional<std::uint32_t>
append_seq_id_base_for_rank(
    std::uint32_t rank, std::uint32_t rank_stride = MERGED_SEQ_ID_RANK_STRIDE) noexcept
{
    if(rank_stride == 0) return std::nullopt;

    constexpr auto max_exclusive =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;

    const auto base = static_cast<std::uint64_t>(rank) * rank_stride;
    if(base > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;

    // set_append_mode starts the process slice at base+1, so a full rank window
    // is valid only when the last possible id (base + rank_stride) is still a
    // std::uint32_t value.
    if(base + rank_stride >= max_exclusive) return std::nullopt;

    return static_cast<std::uint32_t>(base);
}
}  // namespace rocprofsys::core
