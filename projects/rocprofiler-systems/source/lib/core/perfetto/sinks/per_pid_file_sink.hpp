// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/sinks/trace_sink.hpp"

#include <span>
#include <vector>

#include <sys/types.h>

namespace rocprofsys
{
namespace core
{
// Cached-mode sink: writes per-pid bytes to one .proto file per pid.
// The parent_pid receives the default filename; every other pid receives the
// suffix-stamped variant, matching the historical cached-output convention.
class per_pid_file_sink : public trace_sink_interface
{
public:
    explicit per_pid_file_sink(pid_t parent_pid);

    void on_source_drained(int source_id, std::span<const char> bytes) override;
    void finalize() override;

private:
    pid_t m_parent_pid{ 0 };
};
}  // namespace core
}  // namespace rocprofsys
