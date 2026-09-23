// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "wire_format.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>

namespace
{

constexpr std::string_view kExpectedMarker =
    "aten::add%2F%25:n/a|seqNr=7|tid=11|ftid=13|ltid=17|scope=FUNCTION|"
    "args=(value=%25%7C%3B%0D%0A)|torch";
constexpr std::string_view kExpectedMarkerWithoutSequence =
    "aten::add%2F%25:n/a|seqNr=n/a|tid=11|ftid=13|ltid=17|scope=FUNCTION|"
    "args=(value=%25%7C%3B%0D%0A)|torch";
constexpr std::string_view kExpectedMarkerWithoutLauncher =
    "aten::add%2F%25:n/a|seqNr=7|tid=11|ftid=13|ltid=n/a|scope=FUNCTION|"
    "args=(value=%25%7C%3B%0D%0A)|torch";

std::size_t render_golden(std::span<char>              destination,
                          std::int64_t                 sequence_number    = 7,
                          std::optional<std::uint64_t> launcher_thread_id = std::uint64_t{17})
{
    const torch_trace_collector::detail::RangeNameFields fields{
        .name               = "aten::add/%",
        .context            = "n/a",
        .sequence_number    = sequence_number,
        .thread_id          = 11,
        .forward_thread_id  = 13,
        .launcher_thread_id = launcher_thread_id,
        .scope              = "FUNCTION",
        .arguments          = "(value=%25%7C%3B%0D%0A)",
        .backend            = "torch",
    };
    return torch_trace_collector::detail::format_range_name(destination, fields);
}

bool full_buffer_matches()
{
    std::array<char, 256> output{};
    const std::size_t     required = render_golden(output);
    return required == kExpectedMarker.size() + 1 && std::string_view{output.data()} == kExpectedMarker;
}

bool short_buffer_is_terminated()
{
    std::array<char, 12> output{};
    const std::size_t    required = render_golden(output);
    return required == kExpectedMarker.size() + 1 && output.back() == '\0' &&
           std::string_view{output.data()} == kExpectedMarker.substr(0, output.size() - 1);
}

bool empty_buffer_reports_required_size()
{
    return render_golden({}) == kExpectedMarker.size() + 1;
}

bool negative_sequence_is_unavailable()
{
    std::array<char, 256> output{};
    render_golden(output, -1);
    return std::string_view{output.data()} == kExpectedMarkerWithoutSequence;
}

bool missing_launcher_is_unavailable()
{
    std::array<char, 256> output{};
    render_golden(output, 7, std::nullopt);
    return std::string_view{output.data()} == kExpectedMarkerWithoutLauncher;
}

}  // namespace

int main()
{
    if (!full_buffer_matches())
    {
        std::cerr << "full marker does not match the shared wire-format golden\n";
        return 1;
    }
    if (!short_buffer_is_terminated())
    {
        std::cerr << "short marker buffer did not report size and terminate safely\n";
        return 1;
    }
    if (!empty_buffer_reports_required_size())
    {
        std::cerr << "empty marker buffer did not report the required size\n";
        return 1;
    }
    if (!negative_sequence_is_unavailable())
    {
        std::cerr << "negative sequence number was not rendered as unavailable\n";
        return 1;
    }
    if (!missing_launcher_is_unavailable())
    {
        std::cerr << "missing launcher thread ID was not rendered as unavailable\n";
        return 1;
    }
    return 0;
}
