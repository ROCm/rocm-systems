// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace rocprofsys::core
{
enum class locked_append_status
{
    success,
    open_failed,
    lock_failed,
    write_failed,
};

[[nodiscard]] locked_append_status
append_with_file_lock(const std::string& filename, const char* data, std::size_t size);

[[nodiscard]] constexpr std::string_view
status_name(locked_append_status status) noexcept
{
    switch(status)
    {
        case locked_append_status::success: return "success";
        case locked_append_status::open_failed: return "open_failed";
        case locked_append_status::lock_failed: return "lock_failed";
        case locked_append_status::write_failed: return "write_failed";
    }
    return "unknown";
}
}  // namespace rocprofsys::core
