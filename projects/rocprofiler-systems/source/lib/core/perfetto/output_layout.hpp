// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace rocprofsys::core
{
enum class perfetto_output_layout
{
    single_file_only,
    per_process_only,
    full,
};

[[nodiscard]] constexpr perfetto_output_layout
parse_perfetto_output_layout(std::string_view value) noexcept
{
    if(value == "single_file_only" || value == "single_file")
        return perfetto_output_layout::single_file_only;
    if(value == "per_process_only" || value == "per_process")
        return perfetto_output_layout::per_process_only;
    return perfetto_output_layout::full;
}

[[nodiscard]] constexpr bool
writes_per_process_files(perfetto_output_layout layout) noexcept
{
    return layout != perfetto_output_layout::single_file_only;
}

[[nodiscard]] constexpr bool
writes_merged_file(perfetto_output_layout layout) noexcept
{
    return layout != perfetto_output_layout::per_process_only;
}
}  // namespace rocprofsys::core
