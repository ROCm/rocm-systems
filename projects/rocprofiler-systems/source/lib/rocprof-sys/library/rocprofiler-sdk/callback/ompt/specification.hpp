// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys::domains::callback::ompt
{

enum class ompt_parallel_flag_t : std::size_t
{
    ompt_parallel_invoker_program = 0x00000001,
    ompt_parallel_invoker_runtime = 0x00000002,
    ompt_parallel_league          = 0x40000000,
    ompt_parallel_team            = 0x80000000
};

enum class ompt_task_flag_t : std::size_t
{
    ompt_task_initial    = 0x00000001,
    ompt_task_implicit   = 0x00000002,
    ompt_task_explicit   = 0x00000004,
    ompt_task_target     = 0x00000008,
    ompt_task_taskwait   = 0x00000010,
    ompt_task_undeferred = 0x08000000,
    ompt_task_untied     = 0x10000000,
    ompt_task_final      = 0x20000000,
    ompt_task_mergeable  = 0x40000000,
    ompt_task_merged     = 0x80000000
};

enum class ompt_cancel_flag_t : std::size_t
{
    ompt_cancel_parallel       = 0x01,
    ompt_cancel_sections       = 0x02,
    ompt_cancel_loop           = 0x04,
    ompt_cancel_taskgroup      = 0x08,
    ompt_cancel_activated      = 0x10,
    ompt_cancel_detected       = 0x20,
    ompt_cancel_discarded_task = 0x40
};

enum class ompt_thread_t
{
    ompt_thread_initial = 1,
    ompt_thread_worker  = 2,
    ompt_thread_other   = 3,
    ompt_thread_unknown = 4
};

}  // namespace rocprofsys::domains::callback::ompt
