// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/components/fwd.hpp"
#include "core/state.hpp"

#include <cstdint>

namespace rocprofsys
{
namespace pmc
{

std::atomic<State>&
get_state();

void
setup();

void
config();

void
sample();

void
shutdown();

void
post_process();

void set_state(State);

void
postfork_child_cleanup();

void
postfork_parent_reinit();

#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0

inline void
setup()
{}

inline void
config()
{}

inline void
sample()
{}

inline void
shutdown()
{}

inline void
post_process()
{}

inline void
set_state(State)
{}

inline void
postfork_child_cleanup()
{}

inline void
postfork_parent_reinit()
{}

#endif

}  // namespace pmc
}  // namespace rocprofsys

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                        \
        (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#        include <timemory/components/base.hpp>
#        include <timemory/components/data_tracker/components.hpp>
#        include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)

#    endif
#endif
