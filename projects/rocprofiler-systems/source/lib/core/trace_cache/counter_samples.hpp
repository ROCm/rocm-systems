// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace rocprofsys::trace_cache
{
template <typename T>
using counter_view_of = std::span<T>;

template <typename T>
using counter_owned_of = std::vector<std::remove_const_t<T>>;

template <template <typename> class Seq>
struct basic_counter_batch
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::counter_batch;

    std::uint32_t     schema_id = 0;
    std::uint64_t     timestamp = 0;  ///< timeline clock
    Seq<const double> values;         ///< one per column, NaN = unsupported

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(schema_id, timestamp, values);
    }
};

using counter_batch_view = basic_counter_batch<counter_view_of>;   // sink -> store()
using counter_batch      = basic_counter_batch<counter_owned_of>;  // storage parser

/** One column of a schema, stored once before the first batch of that schema. */
struct counter_column
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::counter_column;

    std::uint32_t source      = 0;
    std::uint32_t schema_id   = 0;
    std::uint32_t column      = 0;
    std::uint32_t category_id = 0;
    agent_type    agent       = agent_type::cpu;
    std::uint32_t agent_index = 0;
    std::string   pmc_name;
    std::string   units;
    std::string   perfetto_track_name;
    std::string   rocpd_track_name;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(source, schema_id, column, category_id, agent, agent_index, pmc_name,
                units, perfetto_track_name, rocpd_track_name);
    }
};

struct gap_marker
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::gap_marker;

    std::uint32_t source   = 0;
    std::uint64_t begin_ns = 0;
    std::uint64_t end_ns   = 0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(source, begin_ns, end_ns);
    }
};
}  // namespace rocprofsys::trace_cache
