// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/guarded_read.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace rocprofsys::pmc::pollers
{
/** The row one device of a poller writes. */
struct row_stream
{
    /** Registered schema of the device. */
    collection::schema_id schema = 0;
    /** One value per column. */
    std::vector<double> row;
    /** Poller name and device index, for the log line of a failed read. */
    std::string_view source;
    std::size_t      device_index = 0;
    /** Set when a read threw; the device is not polled again. */
    bool failed = false;

    /** Registers `description` and gives the row one k_no_value per column. */
    void open(const collection::schema& description, collection::sink_interface& out,
              std::string_view source_name, std::size_t index)
    {
        row.assign(description.columns.size(), collection::k_no_value);
        schema       = out.register_schema(description);
        source       = source_name;
        device_index = index;
    }
};

/**
 * Polls each device: fill(stream) reads it into stream.rows.row and returns false when
 * there is no row this time. A throw disables the device (guarded_read). Writes one batch
 * per filled row.
 */
template <typename Stream, typename FillFn>
void
poll_streams(std::vector<Stream>& streams, std::uint64_t timeline_ns,
             collection::sink_interface& out, FillFn&& fill)
{
    for(auto& stream : streams)
    {
        auto&      rows    = stream.rows;
        bool       has_row = false;
        const bool read    = collection::guarded_read(
            rows.failed, rows.source, rows.device_index,
            [&fill, &stream, &has_row] { has_row = fill(stream); });
        if(read && has_row)
        {
            out.write_batch(
                { .schema = rows.schema, .timestamp = timeline_ns, .values = rows.row });
        }
    }
}
}  // namespace rocprofsys::pmc::pollers
