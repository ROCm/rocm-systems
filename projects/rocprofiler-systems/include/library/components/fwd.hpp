// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/defines.hpp"
#include "timemory/components/user_bundle/types.hpp"

#include <timemory/components/data_tracker/types.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/enum.h>

TIMEMORY_DECLARE_COMPONENT(roctracer)

namespace omnitrace
{
namespace component
{
template <typename... Tp>
using data_tracker = tim::component::data_tracker<Tp...>;

struct omnitrace;
struct backtrace;
struct backtrace_wall_clock
{};
struct backtrace_cpu_clock
{};
struct backtrace_fraction
{};
using sampling_wall_clock = data_tracker<double, backtrace_wall_clock>;
using sampling_cpu_clock  = data_tracker<double, backtrace_cpu_clock>;
using sampling_percent    = data_tracker<double, backtrace_fraction>;
using roctracer           = tim::component::roctracer;
}  // namespace component
}  // namespace omnitrace

#if !defined(OMNITRACE_USE_ROCTRACER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::roctracer, false_type)
#endif

#if !defined(TIMEMORY_USE_LIBUNWIND)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, omnitrace::api::sampling, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, omnitrace::sampling::backtrace, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, omnitrace::component::sampling_wall_clock,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, omnitrace::component::sampling_cpu_clock,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, omnitrace::component::sampling_percent,
                               false_type)
#endif

TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::omnitrace, OMNITRACE_COMPONENT,
                                 "omnitrace", "omnitrace_component")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::roctracer, OMNITRACE_ROCTRACER,
                                 "roctracer", "omnitrace_roctracer")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_wall_clock,
                                 OMNITRACE_SAMPLING_WALL_CLOCK, "sampling_wall_clock", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_cpu_clock,
                                 OMNITRACE_SAMPLING_CPU_CLOCK, "sampling_cpu_clock", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_percent,
                                 OMNITRACE_SAMPLING_PERCENT, "sampling_percent", "")

TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_wall_clock,
                                 "sampling_wall_clock", "Wall-clock timing",
                                 "derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_cpu_clock,
                                 "sampling_cpu_clock", "CPU-clock timing",
                                 "derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_percent,
                                 "sampling_percent",
                                 "Fraction of wall-clock time spent in functions",
                                 "derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::roctracer, "roctracer",
                                 "High-precision ROCm API and kernel tracing", "")

TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_wall_clock, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_cpu_clock, double)

// enable timing units
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category,
                               omnitrace::component::sampling_wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units,
                               omnitrace::component::sampling_wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category,
                               omnitrace::component::sampling_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units,
                               omnitrace::component::sampling_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, omnitrace::component::sampling_percent,
                               true_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(report_mean, omnitrace::component::sampling_percent,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, omnitrace::component::sampling_percent,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_statistics, omnitrace::component::sampling_percent,
                               false_type)
