// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "fwd.hpp"

#include <string>
#include <vector>

enum class analysis_type
{
    insertion_set,
};

namespace analysis
{
// Used to determine if the current process is child
constexpr const char* child_analysis_env = "ROCPROFSYS_CHILD_ANALYSIS";
// Indicate that the child process failed to handle the given subset
constexpr int child_analysis_exit_code = 42;
// Developer only var. Used to print the full logs of each child fork+exec subprocess
constexpr const char* analysis_output_env = "ROCPROFSYS_ANALYSIS_OUTPUT";

// A trial is a subprocess running a subset of functions
enum class trial_result
{
    pass,        // Subset passed
    fail,        // Subset failed
    unexpected,  // Error not related to subset bisecting occurred, terminate analysis
};

// Result from an operation run under the analysis signal guard
enum class guarded_result
{
    pass,
    fail,
    signaled,
};

// Stores minimal information from a module_function
struct procedure_id
{
    string_t library_name  = {};
    string_t function_name = {};
};

// Wrapper around finalizeInsertionSet(). In debug-analysis mode, SIGSEGV/SIGBUS
// are caught so fork+exec child trials can report failing subsets; otherwise the
// call only prepends a diagnostic before forwarding crashes to the normal handler
guarded_result
finalize_insertion_set(address_space_t* addr_space, bool* modified_out = nullptr,
                       bool debug_analysis = false);

// True if the current process is a subprocess trial spawned by the analysis
bool
is_analysis_child();

// Entry point from rocprof-sys-instrument.cpp
void
run_analysis(analysis_type type, fmodset_t& instrumented_module_functions);

// function-only insertion-set analysis
// The primary cost of this analysis is from dyninst creating the new process
// and the number of functions that it is analyzing
void
run_insertion_analysis(fmodset_t& instrumented_module_functions);

// Run one subprocess trial restricted by "function-restrict"
trial_result
run_trial(const std::vector<std::string>& parent_argv, const std::string& restrict_regex);

// Print a short summary of a single trial
void
print_trial_result(size_t trial_idx, trial_result result, double elapsed_seconds);

// Print the final analysis result. Each failing subset contains one or more functions
void
print_analysis_result(const std::vector<std::vector<procedure_id>>& failing_subsets);

}  // namespace analysis
