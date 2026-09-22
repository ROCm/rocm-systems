// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <unistd.h>

namespace rocprofsys
{
namespace mproc
{
// get the concurrent processes from /proc/<PPID>/task/<PPID>/children
std::set<int>
get_concurrent_processes(int _ppid = getppid());

int
get_process_index(int _pid = getpid(), int _ppid = getppid());

int
wait_pid(pid_t _pid, int _opts = 0);

int
diagnose_status(pid_t _pid, int _status, int _verbose = 0);
}  // namespace mproc
}  // namespace rocprofsys
