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

#include "common/defines.h"
#include "defines.hpp"

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
