// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/state.hpp"
#include <atomic>

namespace rocprofsys::pmc
{

std::atomic<State>&
get_state();

void
setup();

void
config();

void
sample();

void
shutdown();

void
post_process();

void set_state(State);

void
pause();

void
postfork_child_cleanup();

void
postfork_parent_reinit();

}  // namespace rocprofsys::pmc
