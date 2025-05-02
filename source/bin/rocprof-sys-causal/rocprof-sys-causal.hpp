// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#define TIMEMORY_PROJECT_NAME "rocprof-sys-causal"

#include <csignal>
#include <map>
#include <sched.h>
#include <set>
#include <string>
#include <string_view>
#include <vector>

int
get_verbose();

std::string
get_realpath(const std::string&);

void
print_command(const std::vector<char*>& _argv, std::string_view);

void print_updated_environment(std::vector<char*>, std::string_view);

std::vector<char*>
get_initial_environment();

void
prepare_command_for_run(char*, std::vector<char*>&);

void
prepare_environment_for_run(std::vector<char*>&);

std::string
get_internal_libpath(const std::string& _lib);

std::string
get_internal_script_path(void);

template <typename Tp>
void
update_env(std::vector<char*>&, std::string_view, Tp&&, bool _append = false,
           std::string_view _join_delim = ":");

template <typename Tp>
void
add_default_env(std::vector<char*>&, std::string_view, Tp&&);

void
remove_env(std::vector<char*>&, std::string_view);

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>&,
           std::vector<std::map<std::string_view, std::string>>&);

using sigaction_t = struct sigaction;

struct signal_handler
{
    sigaction_t m_custom_sigaction   = {};
    sigaction_t m_original_sigaction = {};
};

void
forward_signals(const std::set<int>&);

void add_child_pid(pid_t);

void remove_child_pid(pid_t);

int
wait_pid(pid_t, int = 0);

int
diagnose_status(pid_t, int);
