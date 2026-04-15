// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/components/fwd.hpp"
#include "core/state.hpp"

#include <cstdint>

namespace rocprofsys
{
namespace pmc
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

}  // namespace pmc
}  // namespace rocprofsys
