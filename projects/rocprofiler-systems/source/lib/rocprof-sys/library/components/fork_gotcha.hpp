// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/timemory.hpp"

namespace rocprofsys
{
namespace component
{
// this is used to wrap fork()
struct fork_gotcha : comp::base<fork_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 1;

    using gotcha_data_t = comp::gotcha_data;

    // string id for component
    static std::string label() { return "fork_gotcha"; }

    // generate the gotcha wrappers
    static void configure();

    // this will get called right before fork
    pid_t operator()(const gotcha_data_t&, pid_t (*)()) const;

    // silence SFINAE disabled for rocprofsys::fork_gotcha warnings
    static inline void start() {}
    static inline void stop() {}
};
}  // namespace component

using fork_gotcha_t = comp::gotcha<component::fork_gotcha::gotcha_capacity, std::tuple<>,
                                   component::fork_gotcha>;
}  // namespace rocprofsys
