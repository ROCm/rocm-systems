// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks/per_pid_file_sink.hpp"
#include "core/perfetto/sinks/file_output.hpp"

#include "core/config.hpp"
#include "core/output/output_summary.hpp"
#include "logger/debug.hpp"

#include <span>
#include <string>
#include <utility>

namespace rocprofsys::core
{
per_pid_file_sink::per_pid_file_sink(pid_t parent_pid)
: m_parent_pid{ parent_pid }
{}

void
per_pid_file_sink::on_source_drained(int source_id, std::span<const char> bytes)
{
    if(bytes.empty())
    {
        return;
    }

    const auto pid = static_cast<pid_t>(source_id);
    auto       filename =
        (pid == m_parent_pid)
                  ? config::get_perfetto_output_filename()
                  : config::get_perfetto_output_filename_with_suffix(std::to_string(pid));

    if(!write_proto_to(filename, bytes.data(), bytes.size()))
    {
        LOG_ERROR("per_pid_file_sink: failed to open '{}' for pid {}", filename, pid);
        return;
    }
    output::registry::instance().register_file(std::move(filename), pid);
}

void
per_pid_file_sink::finalize()
{}
}  // namespace rocprofsys::core
