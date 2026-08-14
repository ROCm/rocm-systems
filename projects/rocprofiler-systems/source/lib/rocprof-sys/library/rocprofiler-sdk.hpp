// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/control/session.hpp"
#include "core/timemory.hpp"

#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
using hardware_counter_info = ::tim::hardware_counters::info;

void
setup();

void
shutdown();

void
config();

void
post_process();

void
sample();

void
start();

void
stop();

void
pause();

void
resume();

void
bind_session(std::shared_ptr<control::session> sess);

void
reset_sdk_session_guards();

std::vector<hardware_counter_info>
get_rocm_events_info();
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
