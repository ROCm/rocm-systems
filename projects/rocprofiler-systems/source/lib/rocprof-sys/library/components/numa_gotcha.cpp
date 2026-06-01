// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/numa_gotcha.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdlib>

// int
// numa_migrate_pages(int pid, struct bitmask* from, struct bitmask* to);

// int
// numa_move_pages(int pid, unsigned long count, void** pages, const int* nodes, int*
// status,
//                int flags);

namespace rocprofsys
{
namespace component
{
namespace
{
auto&
get_numa_gotcha()
{
    static auto _v = tim::lightweight_tuple<numa_gotcha_t>{};
    return _v;
}
}  // namespace

void
numa_gotcha::configure()
{
    // don't emit warnings for missing MPI functions unless debug or verbosity >= 3
    if(get_verbose_env() < 3 && !get_debug_env())
    {
        for(size_t i = 0; i < numa_gotcha_t::capacity(); ++i)
        {
            auto* itr = numa_gotcha_t::at(i);
            if(itr) itr->verbose = -1;
        }
    }

    numa_gotcha_t::get_initializer() = []() {
        numa_gotcha_t::configure<0, long, void*, unsigned long, int, const unsigned long*,
                                 unsigned long, unsigned>("mbind");
        numa_gotcha_t::configure<1, long, int, unsigned long, const unsigned long*,
                                 const unsigned long*>("migrate_pages");
        numa_gotcha_t::configure<2, long, int, unsigned long, void**, const int*, int*,
                                 int>("move_pages");
        numa_gotcha_t::configure<3, int, int, struct bitmask*, struct bitmask*>(
            "numa_migrate_pages");
        numa_gotcha_t::configure<4, int, int, unsigned long, void**, const int*, int*,
                                 int>("numa_move_pages");
        numa_gotcha_t::configure<5, void*, size_t>("numa_alloc");
        numa_gotcha_t::configure<6, void*, size_t>("numa_alloc_local");
        numa_gotcha_t::configure<7, void*, size_t>("numa_alloc_interleaved");
        numa_gotcha_t::configure<8, void*, size_t, int>("numa_alloc_onnode");
        numa_gotcha_t::configure<9, void*, void*, size_t, size_t>("numa_realloc");
        numa_gotcha_t::configure<10, void, void*, size_t>("numa_free");
    };
}

void
numa_gotcha::shutdown()
{
    numa_gotcha_t::disable();
}

std::mutex numa_gotcha::s_mutex = {};

void
numa_gotcha::pause()
{
    std::scoped_lock<std::mutex> _lk{ s_mutex };
    numa_gotcha_t::set_ready(false);
}

void
numa_gotcha::resume()
{
    std::scoped_lock<std::mutex> _lk{ s_mutex };
    numa_gotcha_t::set_ready(true);
}

void
numa_gotcha::start()
{
    if(!get_numa_gotcha().get<numa_gotcha_t>()->get_is_running())
    {
        configure();
        get_numa_gotcha().start();
    }
}

void
numa_gotcha::stop()
{
    // get_numa_gotcha().stop();
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, void* start,
                   unsigned long len, int mode, const unsigned long* nmask,
                   unsigned long maxnode, unsigned flags)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "start",
                                           start, "len", len, "mode", mode, "nmask",
                                           nmask, "maxnode", maxnode, "flags", flags);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, int pid,
                   unsigned long maxnode, const unsigned long* frommask,
                   const unsigned long* tomask)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "pid", pid,
                                           "maxnode", maxnode, "frommask", frommask,
                                           "tomask", tomask);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, int pid,
                   unsigned long count, void** pages, const int* nodes, int* status,
                   int flags)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "pid", pid,
                                           "count", count, "pages", pages, "nodes", nodes,
                                           "status", status, "flags", flags);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, int pid,
                   struct bitmask* from, struct bitmask* to)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "pid", pid,
                                           "from", fmt::ptr(from), "to", fmt::ptr(to));
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, size_t _size)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "size",
                                           _size);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, size_t _size, int _node)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "size",
                                           _size, "node", _node);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, void* _addr, size_t _size)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id }, "address",
                                           _addr, "size", _size);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::incoming, void* _old_addr,
                   size_t _old_size, size_t _new_size)
{
    category_region<category::numa>::start(std::string_view{ _data.tool_id },
                                           "old_address", _old_addr, "old_size",
                                           _old_size, "new_size", _new_size);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::outgoing, int ret)
{
    category_region<category::numa>::stop(std::string_view{ _data.tool_id }, "return",
                                          ret);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::outgoing, long ret)
{
    category_region<category::numa>::stop(std::string_view{ _data.tool_id }, "return",
                                          ret);
}

void
numa_gotcha::audit(const gotcha_data& _data, audit::outgoing, void* ret)
{
    category_region<category::numa>::stop(std::string_view{ _data.tool_id }, "return",
                                          ret);
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_STORAGE_INITIALIZER(rocprofsys::component::numa_gotcha)
