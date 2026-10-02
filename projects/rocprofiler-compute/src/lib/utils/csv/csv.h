// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofiler_compute_tool::csv
{

/// Takes formatted text and returns false when it cannot take any more.
using Sink = std::function<bool(std::string_view)>;

// Amortizes the sink indirection over many rows rather than paying it per row.
// Not tuned; any size well above a row works.
inline constexpr std::size_t kBatchBytes = 256 * 1024;

/// Quotes a field per RFC 4180, for anything that can hold a comma or a quote.
std::string quote(std::string_view field);

/// Writes the header and then every row into sink, in batches.
///
/// write_row prints one row into the stream it is given, without the newline.
/// Returns false as soon as the sink refuses, leaving the rest unwritten.
template<typename Range, typename WriteRow>
bool format(std::string_view header, const Range& rows, WriteRow write_row, const Sink& sink)
{
    std::ostringstream batch;
    batch << header;

    const auto flush_batch = [&sink, &batch]()
    {
        const auto text = batch.str();
        batch.str(std::string{});
        return text.empty() || sink(text);
    };

    for (const auto& row : rows)
    {
        write_row(batch, row);
        batch << '\n';

        if (static_cast<std::size_t>(batch.tellp()) >= kBatchBytes && !flush_batch())
            return false;
    }

    return flush_batch();
}

}  // namespace rocprofiler_compute_tool::csv
