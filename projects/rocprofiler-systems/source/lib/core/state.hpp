// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

namespace omnitrace
{
// used for specifying the state of omnitrace
enum class State : unsigned short
{
    PreInit = 0,
    Init,
    Active,
    Finalized,
    Disabled,
};

// used for specifying the state of omnitrace
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

enum class CausalMode : unsigned short
{
    Line = 0,
    Function
};

//
//      Runtime configuration data
//
State
get_state() OMNITRACE_HOT;

ThreadState
get_thread_state() OMNITRACE_HOT;

/// returns old state
State set_state(State) OMNITRACE_COLD;  // does not change often

/// returns old state
ThreadState set_thread_state(ThreadState) OMNITRACE_HOT;  // changes often

/// return current state (state change may be ignored)
ThreadState push_thread_state(ThreadState) OMNITRACE_HOT;

/// return current state (state change may be ignored)
ThreadState
pop_thread_state() OMNITRACE_HOT;

struct scoped_thread_state
{
    OMNITRACE_INLINE scoped_thread_state(ThreadState _v) { push_thread_state(_v); }
    OMNITRACE_INLINE ~scoped_thread_state() { pop_thread_state(); }
};
}  // namespace omnitrace

#define OMNITRACE_SCOPED_THREAD_STATE(STATE)                                             \
    ::omnitrace::scoped_thread_state OMNITRACE_VARIABLE(_scoped_thread_state_, __LINE__) \
    {                                                                                    \
        ::omnitrace::STATE                                                               \
    }

namespace std
{
std::string
to_string(omnitrace::State _v);

std::string
to_string(omnitrace::ThreadState _v);

std::string
to_string(omnitrace::Mode _v);

std::string
to_string(omnitrace::CausalMode _v);
}  // namespace std
