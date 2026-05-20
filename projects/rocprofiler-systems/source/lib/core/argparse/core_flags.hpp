// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "flag_descriptor.hpp"

namespace rocprofsys
{
namespace argparse
{

const flag_group&
debug_group();

const flag_group&
general_group();

const flag_group&
launcher_group();

const flag_group&
tracing_group();

const flag_group&
profile_group();

const flag_group&
process_sampling_group();

const flag_group&
general_sampling_group();

const flag_group&
sampling_timer_group();

const flag_group&
hw_counter_group();

const flag_group&
misc_group();

}  // namespace argparse
}  // namespace rocprofsys
