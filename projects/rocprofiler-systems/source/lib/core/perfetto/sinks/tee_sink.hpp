// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/sinks/per_pid_file_sink.hpp"
#include "core/perfetto/sinks/single_file_sink.hpp"

#include <vector>

namespace rocprofsys::core
{
// Composite sink that fans every drained source out to both wrapped sinks.
class tee_sink
{
public:
    tee_sink(per_pid_file_sink per_pid, single_file_sink single_file);

    tee_sink(tee_sink&&) noexcept            = default;
    tee_sink& operator=(tee_sink&&) noexcept = default;
    tee_sink(const tee_sink&)                = delete;
    tee_sink& operator=(const tee_sink&)     = delete;
    ~tee_sink()                              = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

private:
    per_pid_file_sink m_per_pid;
    single_file_sink  m_single_file;
};
}  // namespace rocprofsys::core
