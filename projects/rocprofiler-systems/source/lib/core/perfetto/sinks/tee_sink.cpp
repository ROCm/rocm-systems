// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks/tee_sink.hpp"

#include <utility>

namespace rocprofsys::core
{
tee_sink::tee_sink(per_pid_file_sink per_pid, single_file_sink single_file)
: m_per_pid{ std::move(per_pid) }
, m_single_file{ std::move(single_file) }
{}

void
tee_sink::on_source_drained(int source_id, std::vector<char> bytes)
{
    auto copy = bytes;
    m_per_pid.on_source_drained(source_id, std::move(copy));
    m_single_file.on_source_drained(source_id, std::move(bytes));
}

void
tee_sink::finalize()
{
    m_per_pid.finalize();
    m_single_file.finalize();
}
}  // namespace rocprofsys::core
