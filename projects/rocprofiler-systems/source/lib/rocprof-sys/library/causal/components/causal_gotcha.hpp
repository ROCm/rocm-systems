// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <cstdint>
#include <future>

namespace rocprofsys
{
namespace causal
{
namespace component
{
struct causal_gotcha : tim::component::base<causal_gotcha, void>
{
    ROCPROFSYS_DEFAULT_OBJECT(causal_gotcha)

    // string id for component
    static std::string label() { return "causal_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    static void remove_signals(sigset_t*);
};
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys
