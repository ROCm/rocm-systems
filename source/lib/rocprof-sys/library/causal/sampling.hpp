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

#include "core/concepts.hpp"
#include "core/defines.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>

namespace rocprofsys
{
namespace causal
{
namespace sampling
{
std::set<int>
get_signal_types(int64_t _tid);

void
block_samples();

void
unblock_samples();

void
block_backtrace_samples();

void
unblock_backtrace_samples();

template <typename Tp = tim::scope::thread_scope>
void pause(Tp = {});

template <typename Tp = tim::scope::thread_scope>
void resume(Tp = {});

void block_signals(std::set<int> = {});

void unblock_signals(std::set<int> = {});

std::set<int>
setup();

std::set<int>
shutdown();

void
post_process();
}  // namespace sampling
}  // namespace causal
}  // namespace rocprofsys
