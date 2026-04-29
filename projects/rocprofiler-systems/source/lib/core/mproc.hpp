// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <unistd.h>

namespace rocprofsys
{
// Process-identity helpers. Defined in core/mproc.cpp so that core-only callers
// (notably trace_cache::cache_manager) can link them without pulling in the
// main rocprof-sys library. library/runtime.hpp re-declares the same trio so
// pre-existing call-sites continue to compile unchanged.
pid_t
get_root_process_id();
bool
is_root_process();
bool
is_child_process();

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
