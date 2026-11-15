// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace rocprofsys::trace_cache
{
/** Counter row written by collection_sink; views the caller's row. */
struct counter_batch_view
{
    // NOLINTNEXTLINE(readability-identifier-naming): name required by has_type_identifier
    static constexpr type_identifier_t type_identifier = type_identifier_t::counter_batch;

    /** metadata_registry counter schema id. */
    std::uint32_t schema_id = 0;
    /** Timeline clock, nanoseconds. */
    std::uint64_t timestamp = 0;
    /** One per column, NaN = no value. */
    std::span<const double> values;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(schema_id, timestamp, values);
    }
};

/** Counter row read back by the storage parser; same wire format as counter_batch_view.
 */
struct counter_batch
{
    // NOLINTNEXTLINE(readability-identifier-naming): name required by has_type_identifier
    static constexpr type_identifier_t type_identifier = type_identifier_t::counter_batch;

    std::uint32_t       schema_id = 0;
    std::uint64_t       timestamp = 0;
    std::vector<double> values;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(schema_id, timestamp, values);
    }
};

/** A pause of all sampled counters; one record per pause, written by the collector. */
struct gap_marker
{
    // NOLINTNEXTLINE(readability-identifier-naming): name required by has_type_identifier
    static constexpr type_identifier_t type_identifier = type_identifier_t::gap_marker;

    std::uint64_t begin_ns = 0;
    /** Not read yet: kept so a later change can draw the pause as a region. */
    std::uint64_t end_ns = 0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(begin_ns, end_ns);
    }
};
}  // namespace rocprofsys::trace_cache
