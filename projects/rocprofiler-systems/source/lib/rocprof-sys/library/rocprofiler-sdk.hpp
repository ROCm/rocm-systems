// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/session.hpp"
#include "core/timemory.hpp"

#include <memory>
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

std::shared_ptr<control::session>
get_session();

/// Force lazy construction of the roctx_client so its trigger is attached to
/// the control session before subscribers' force_initial_pause runs. Without
/// this, the roctx region filter would not be in the votes table at
/// initial-broadcast time and subscribers (process_sampler, sampling, ...)
/// would not observe the paused initial state. No-op if no marker domain or
/// trace_region is configured.
void
ensure_roctx_initialized();

void
reset_sdk_session_guards();

std::vector<hardware_counter_info>
get_rocm_events_info();
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
