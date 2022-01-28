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

#include "library/common.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <timemory/macros/language.hpp>
#include <timemory/sampling/sampler.hpp>
#include <timemory/variadic/types.hpp>

#include <cstdint>
#include <memory>
#include <set>

namespace omnitrace
{
namespace sampling
{
using component::backtrace;
using component::backtrace_cpu_clock;   // NOLINT
using component::backtrace_fraction;    // NOLINT
using component::backtrace_wall_clock;  // NOLINT
using component::sampling_cpu_clock;
using component::sampling_percent;
using component::sampling_wall_clock;

std::unique_ptr<std::set<int>>&
get_signal_types(int64_t _tid);

std::set<int>
setup();

std::set<int>
shutdown();

void block_signals(std::set<int> = {});

void unblock_signals(std::set<int> = {});

using bundle_t          = tim::lightweight_tuple<backtrace>;
using sampler_t         = tim::sampling::sampler<bundle_t, tim::sampling::dynamic>;
using sampler_instances = omnitrace_thread_data<sampler_t, api::sampling>;

std::unique_ptr<sampler_t>&
get_sampler(int64_t _tid = threading::get_id());

}  // namespace sampling
}  // namespace omnitrace

TIMEMORY_DEFINE_CONCRETE_TRAIT(prevent_reentry, omnitrace::sampling::sampler_t,
                               std::true_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(check_signals, omnitrace::sampling::sampler_t,
                               std::false_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(buffer_size, omnitrace::sampling::sampler_t,
                               TIMEMORY_ESC(std::integral_constant<size_t, 256>))
