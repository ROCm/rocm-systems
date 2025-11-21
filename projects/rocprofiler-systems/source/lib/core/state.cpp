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

#include "state.hpp"
#include "common/static_object.hpp"
#include "config.hpp"
#include "debug.hpp"
#include "utility.hpp"

#include <atomic>
#include <string>

namespace rocprofsys
{
namespace
{
auto&
get_state_value()
{
    static auto*& _v = common::static_object<std::atomic<State>>::construct(
        common::do_not_destroy{}, State::PreInit);
    return *_v;
}

ThreadState&
get_thread_state_value()
{
    static thread_local auto _v = ThreadState{ ThreadState::Enabled };
    return _v;
}

auto&
get_thread_state_history(int64_t _idx = utility::get_thread_index())
{
    static auto _v = utility::get_filled_array<ROCPROFSYS_MAX_THREADS>(
        []() { return utility::get_reserved_vector<ThreadState>(32); });

    if(_idx >= ROCPROFSYS_MAX_THREADS)
    {
        static thread_local auto _tl_v = utility::get_reserved_vector<ThreadState>(32);
        return _tl_v;
    }

    return _v.at(_idx);
}
}  // namespace

State
get_state()
{
    return get_state_value().load(std::memory_order_relaxed);
}

ThreadState
get_thread_state()
{
    return get_thread_state_value();
}

State
set_state(State _n)
{
    ROCPROFSYS_CONDITIONAL_PRINT_F(get_debug_init(), "Setting state :: %s -> %s\n",
                                   std::to_string(get_state()).c_str(),
                                   std::to_string(_n).c_str());
    // state should always be increased, not decreased
    ROCPROFSYS_CI_BASIC_THROW(
        _n < get_state(), "State is being assigned to a lesser value :: %s -> %s",
        std::to_string(get_state()).c_str(), std::to_string(_n).c_str());
    auto _v = get_state();
    get_state_value().store(_n, std::memory_order_relaxed);
    // std::swap(get_state_value(), _n);
    return _v;
}

ThreadState
set_thread_state(ThreadState _n)
{
    std::swap(get_thread_state_value(), _n);
    return _n;
}

ThreadState
push_thread_state(ThreadState _v)
{
    if(get_thread_state() >= ThreadState::Completed) return get_thread_state();

    return get_thread_state_history().emplace_back(set_thread_state(_v));
}

ThreadState
pop_thread_state()
{
    if(get_thread_state() >= ThreadState::Completed) return get_thread_state();

    auto& _hist = get_thread_state_history();
    if(!_hist.empty())
    {
        set_thread_state(_hist.back());
        _hist.pop_back();
    }
    return get_thread_state();
}
}  // namespace rocprofsys

namespace std
{
std::string
to_string(rocprofsys::State _v)
{
    switch(_v)
    {
        case rocprofsys::State::PreInit: return "PreInit";
        case rocprofsys::State::Init: return "Init";
        case rocprofsys::State::Active: return "Active";
        case rocprofsys::State::Disabled: return "Disabled";
        case rocprofsys::State::Finalized: return "Finalized";
    }
    return {};
}

std::string
to_string(rocprofsys::ThreadState _v)
{
    switch(_v)
    {
        case rocprofsys::ThreadState::Enabled: return "Enabled";
        case rocprofsys::ThreadState::Internal: return "Internal";
        case rocprofsys::ThreadState::Completed: return "Completed";
        case rocprofsys::ThreadState::Disabled: return "Disabled";
    }
    return {};
}

std::string
to_string(rocprofsys::Mode _v)
{
    switch(_v)
    {
        case rocprofsys::Mode::Trace: return "Trace";
        case rocprofsys::Mode::Sampling: return "Sampling";
        case rocprofsys::Mode::Causal: return "Causal";
        case rocprofsys::Mode::Coverage: return "Coverage";
    }
    return {};
}

std::string
to_string(rocprofsys::CausalMode _v)
{
    switch(_v)
    {
        case rocprofsys::CausalMode::Line: return "Line";
        case rocprofsys::CausalMode::Function: return "Function";
    }
    return {};
}
}  // namespace std
