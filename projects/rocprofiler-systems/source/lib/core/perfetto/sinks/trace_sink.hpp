// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/sinks/per_pid_file_sink.hpp"
#include "core/perfetto/sinks/polymorphic_sink_view.hpp"
#include "core/perfetto/sinks/recording_sink.hpp"
#include "core/perfetto/sinks/single_file_sink.hpp"
#include "core/perfetto/sinks/tee_sink.hpp"

#include <variant>

namespace rocprofsys::core
{
// Cached trace bytes are dispatched to one of these alternatives via std::visit.
// polymorphic_sink_view lets tests inject arbitrary fixtures.
using trace_sink = std::variant<per_pid_file_sink, single_file_sink, recording_sink,
                                polymorphic_sink_view, tee_sink>;
}  // namespace rocprofsys::core
