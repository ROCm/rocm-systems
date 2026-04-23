// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "library/thread_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>

#include <atomic>
#include <cstdint>

namespace rocprofsys
{
namespace causal
{
struct delay : comp::empty_base
{
    using value_type = void;

    ROCPROFSYS_DEFAULT_OBJECT(delay)

    static void    setup();
    static void    process();
    static void    credit();
    static void    preblock();
    static void    postblock(int64_t);
    static int64_t sync();

    static std::atomic<int64_t>& get_global();
    static int64_t&              get_local(int64_t _tid = threading::get_id());
    static bool                  is_local_available();

    static int64_t  get(int64_t _tid = threading::get_id());
    static uint64_t compute_total_delay(uint64_t);
};
}  // namespace causal
}  // namespace rocprofsys
