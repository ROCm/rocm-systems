// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <cstdint>
#include <cstdlib>

namespace rocprofsys
{
namespace component
{
struct numa_gotcha : tim::component::base<numa_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 11;

    using gotcha_data  = tim::component::gotcha_data;
    using exit_func_t  = void (*)(int);
    using abort_func_t = void (*)();

    ROCPROFSYS_DEFAULT_OBJECT(numa_gotcha)

    // string id for component
    static std::string label() { return "numa_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    static void pause();
    static void resume();

    static void audit(const gotcha_data&, audit::incoming, void* start, unsigned long len,
                      int mode, const unsigned long* nmask, unsigned long maxnode,
                      unsigned flags);
    static void audit(const gotcha_data&, audit::incoming, int pid, unsigned long maxnode,
                      const unsigned long* frommask, const unsigned long* tomask);
    static void audit(const gotcha_data&, audit::incoming, int pid, unsigned long count,
                      void** pages, const int* nodes, int* status, int flags);
    static void audit(const gotcha_data&, audit::incoming, int pid, struct bitmask* from,
                      struct bitmask* to);
    static void audit(const gotcha_data&, audit::incoming, size_t);
    static void audit(const gotcha_data&, audit::incoming, size_t, int);
    static void audit(const gotcha_data&, audit::incoming, void*, size_t);
    static void audit(const gotcha_data&, audit::incoming, void*, size_t, size_t);
    static void audit(const gotcha_data&, audit::outgoing, int);
    static void audit(const gotcha_data&, audit::outgoing, long);
    static void audit(const gotcha_data&, audit::outgoing, void*);

private:
    static std::mutex s_mutex;
};
}  // namespace component

using numa_bundle_t = tim::component_bundle<category::numa, component::numa_gotcha>;
using numa_gotcha_t = tim::component::gotcha<component::numa_gotcha::gotcha_capacity,
                                             numa_bundle_t, category::numa>;
}  // namespace rocprofsys
