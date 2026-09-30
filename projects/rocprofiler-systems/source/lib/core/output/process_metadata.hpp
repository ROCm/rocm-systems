// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <string>

namespace rocprofsys::output
{

// Sentinel for "no such pid" — an absent parent (root process) or a not-yet
// assigned pid field.
inline constexpr pid_t k_no_pid = -1;

struct process_metadata
{
    pid_t       pid{ k_no_pid };
    pid_t       ppid{ k_no_pid };
    std::string command;
};

}  // namespace rocprofsys::output
