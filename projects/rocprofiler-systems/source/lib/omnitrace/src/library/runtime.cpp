// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/runtime.hpp"
#include "library/api.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/declaration.hpp>
#include <timemory/utility/signals.hpp>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <string>
#include <unistd.h>

namespace omnitrace
{
std::atomic<uint64_t>&
get_cpu_cid()
{
    static std::atomic<uint64_t> _v{ 0 };
    return _v;
}

unique_ptr_t<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid, int64_t _parent)
{
    struct omnitrace_cpu_cid_stack
    {};
    using thread_data_t = thread_data<std::vector<uint64_t>, omnitrace_cpu_cid_stack>;
    static auto&             _v      = thread_data_t::instances();
    static thread_local auto _v_copy = [_tid, _parent]() {
        auto _parent_tid = _parent;
        // if tid != parent and there is not a valid pointer for the provided parent
        // thread id set it to zero since that will always be valid
        if(_tid != _parent_tid && !_v.at(_parent_tid)) _parent_tid = 0;
        // copy over the thread ids from the parent if tid != parent
        thread_data_t::construct((_tid != _parent_tid) ? *(_v.at(_parent_tid))
                                                       : std::vector<uint64_t>{});
        return true;
    }();
    return _v.at(_tid);
    (void) _v_copy;
}

unique_ptr_t<cpu_cid_parent_map_t>&
get_cpu_cid_parents(int64_t _tid)
{
    struct omnitrace_cpu_cid_stack
    {};
    using thread_data_t = thread_data<cpu_cid_parent_map_t, omnitrace_cpu_cid_stack>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{},
                                               cpu_cid_parent_map_t{});
    return _v.at(_tid);
}

std::tuple<uint64_t, uint64_t, uint16_t>
create_cpu_cid_entry(int64_t _tid)
{
    auto&& _cid        = get_cpu_cid()++;
    auto&& _parent_cid = (get_cpu_cid_stack(_tid)->empty()) ? get_cpu_cid_stack(0)->back()
                                                            : get_cpu_cid_stack()->back();
    uint16_t&& _depth  = (get_cpu_cid_stack(_tid)->empty())
                             ? get_cpu_cid_stack(0)->size()
                             : get_cpu_cid_stack()->size() - 1;
    get_cpu_cid_parents(_tid)->emplace(_cid, std::make_tuple(_parent_cid, _depth));
    return std::make_tuple(_cid, _parent_cid, _depth);
}

cpu_cid_pair_t
get_cpu_cid_entry(uint64_t _cid, int64_t _tid)
{
    return get_cpu_cid_parents(_tid)->at(_cid);
}

namespace
{
void
setup_gotchas()
{
    static bool _initialized = false;
    if(_initialized) return;
    _initialized = true;

    OMNITRACE_CONDITIONAL_PRINT_F(
        get_debug_env(),
        "Configuring gotcha wrapper around fork, MPI_Init, and MPI_Init_thread\n");

    mpi_gotcha::configure();
    fork_gotcha::configure();
    pthread_gotcha::configure();
}
}  // namespace

std::unique_ptr<main_bundle_t>&
get_main_bundle()
{
    static auto _v =
        std::make_unique<main_bundle_t>("omnitrace", quirk::config<quirk::auto_start>{});
    return _v;
}

std::unique_ptr<gotcha_bundle_t>&
get_gotcha_bundle()
{
    static auto _v =
        (setup_gotchas(), std::make_unique<gotcha_bundle_t>(
                              "omnitrace", quirk::config<quirk::auto_start>{}));
    return _v;
}
}  // namespace omnitrace
