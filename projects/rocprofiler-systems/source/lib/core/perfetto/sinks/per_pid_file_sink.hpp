// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <vector>

#include <sys/types.h>

namespace rocprofsys
{
class output_file_registry;

namespace core
{
// Cached-mode sink: writes per-pid bytes to one .proto file per pid.
// The parent_pid receives the default filename; every other pid receives the
// suffix-stamped variant, matching the historical cached-output convention.
class per_pid_file_sink
{
public:
    per_pid_file_sink(pid_t parent_pid, output_file_registry& registry);

    per_pid_file_sink(per_pid_file_sink&&) noexcept            = default;
    per_pid_file_sink& operator=(per_pid_file_sink&&) noexcept = default;
    per_pid_file_sink(const per_pid_file_sink&)                = delete;
    per_pid_file_sink& operator=(const per_pid_file_sink&)     = delete;
    ~per_pid_file_sink()                                       = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

private:
    pid_t                                        m_parent_pid{ 0 };
    std::reference_wrapper<output_file_registry> m_registry;
};
}  // namespace core
}  // namespace rocprofsys
