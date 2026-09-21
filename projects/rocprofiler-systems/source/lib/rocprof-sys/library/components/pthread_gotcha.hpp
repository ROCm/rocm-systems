// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/timemory.hpp"
#include "library/thread_info.hpp"

#include <cstdint>
#include <future>

namespace rocprofsys
{
struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    using native_handle_t = std::thread::native_handle_type;

    // string id for component
    static std::string label() { return "pthread_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    static void pause();
    static void resume();

    static std::set<native_handle_t> get_native_handles();
};
}  // namespace rocprofsys
