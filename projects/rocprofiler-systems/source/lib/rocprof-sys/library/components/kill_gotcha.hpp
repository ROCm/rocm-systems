// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/defines.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <csignal>
#include <sys/types.h>

namespace rocprofsys
{
namespace component
{
struct kill_gotcha : tim::component::base<kill_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 1;

    using gotcha_data = tim::component::gotcha_data;
    using kill_func_t = int (*)(pid_t, int);

    ROCPROFSYS_DEFAULT_OBJECT(kill_gotcha)

    static std::string label() { return "kill_gotcha"; }

    static void configure();

    static inline void start() {}
    static inline void stop() {}

    int operator()(const gotcha_data&, kill_func_t, pid_t, int) const;
};
}  // namespace component

using kill_gotcha_t = tim::component::gotcha<component::kill_gotcha::gotcha_capacity,
                                             std::tuple<>, component::kill_gotcha>;
}  // namespace rocprofsys
