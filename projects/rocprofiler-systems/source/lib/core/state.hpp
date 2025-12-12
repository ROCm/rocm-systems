// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"

#include <cstdint>
#include <string>

namespace rocprofsys
{
// used for specifying the state of rocprof-sys
enum class State : unsigned short
{
    PreInit = 0,
    Init,
    Active,
    Finalized,
    Disabled,
};

// used for specifying the state of rocprof-sys
enum class ThreadState : unsigned short
{
    Enabled = 0,
    Internal,
    Completed,
    Disabled,
};

enum class Mode : unsigned short
{
    Trace = 0,
    Sampling,
    Causal,
    Coverage
};

enum class CausalBackend : unsigned short
{
    Perf = 0,
    Timer,
    Auto,
};

enum class CausalMode : unsigned short
{
    Line = 0,
    Function
};

//
//      Runtime configuration data
//
State
get_state() ROCPROFSYS_HOT;

ThreadState
get_thread_state() ROCPROFSYS_HOT;

/// returns old state
State set_state(State) ROCPROFSYS_COLD;  // does not change often

/// Reset state to PreInit (for re-attach scenarios). Bypasses state validation.
State
reset_state() ROCPROFSYS_COLD;

/// returns old state
ThreadState set_thread_state(ThreadState) ROCPROFSYS_HOT;  // changes often

/// return current state (state change may be ignored)
ThreadState push_thread_state(ThreadState) ROCPROFSYS_HOT;

/// return current state (state change may be ignored)
ThreadState
pop_thread_state() ROCPROFSYS_HOT;

struct scoped_thread_state
{
    ROCPROFSYS_INLINE scoped_thread_state(ThreadState _v) { push_thread_state(_v); }
    ROCPROFSYS_INLINE ~scoped_thread_state() { pop_thread_state(); }
};
}  // namespace rocprofsys

#define ROCPROFSYS_SCOPED_THREAD_STATE(STATE)                                            \
    ::rocprofsys::scoped_thread_state ROCPROFSYS_VARIABLE(_scoped_thread_state_,         \
                                                          __LINE__)                      \
    {                                                                                    \
        ::rocprofsys::STATE                                                              \
    }

namespace std
{
std::string
to_string(rocprofsys::State _v);

std::string
to_string(rocprofsys::ThreadState _v);

std::string
to_string(rocprofsys::Mode _v);

std::string
to_string(rocprofsys::CausalMode _v);
}  // namespace std
