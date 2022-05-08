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

#include "library/state.hpp"

#include <string>

namespace std
{
std::string
to_string(omnitrace::State _v)
{
    switch(_v)
    {
        case omnitrace::State::DelayedInit: return "DelayedInit";
        case omnitrace::State::PreInit: return "PreInit";
        case omnitrace::State::Init: return "Init";
        case omnitrace::State::Active: return "Active";
        case omnitrace::State::Finalized: return "Finalized";
    }
    return {};
}

std::string
to_string(omnitrace::ThreadState _v)
{
    switch(_v)
    {
        case omnitrace::ThreadState::Enabled: return "Enabled";
        case omnitrace::ThreadState::Internal: return "Internal";
        case omnitrace::ThreadState::Disabled: return "Disabled";
        case omnitrace::ThreadState::Completed: return "Completed";
    }
    return {};
}

std::string
to_string(omnitrace::Mode _v)
{
    switch(_v)
    {
        case omnitrace::Mode::Trace: return "Trace";
        case omnitrace::Mode::Sampling: return "Sampling";
        case omnitrace::Mode::Coverage: return "Coverage";
    }
    return {};
}
}  // namespace std
