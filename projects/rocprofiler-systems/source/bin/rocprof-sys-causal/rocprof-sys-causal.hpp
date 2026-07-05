// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#define TIMEMORY_PROJECT_NAME "rocprof-sys-causal"

#include <csignal>
#include <map>
#include <sched.h>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

template <typename Tp>
void
update_env(std::vector<std::string>& _environ, std::string_view _env_var, Tp&& _env_val,
           bool _append = false, std::string_view _join_delim = ":");

int
get_verbose();

const std::unordered_set<std::string_view>&
get_updated_envs();

std::vector<std::string>
get_initial_environment();

void
prepare_command_for_run(char*, std::vector<char*>&);

void
prepare_environment_for_run(std::vector<std::string>&);

std::vector<char*>
parse_args(int argc, char** argv, std::vector<std::string>&,
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
