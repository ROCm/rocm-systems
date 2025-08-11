// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "api.hpp"
#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/causal/components/causal_gotcha.hpp"
#include "library/components/exit_gotcha.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/numa_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/components/vaapi_gotcha.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/macros/language.hpp>

#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rocprofsys
{
// started during preinit phase
using preinit_bundle_t =
    tim::lightweight_tuple<exit_gotcha_t, fork_gotcha_t, mpi_gotcha_t>;

// started during init phase
using init_bundle_t = tim::lightweight_tuple<causal::component::causal_gotcha,
                                             pthread_gotcha, component::numa_gotcha>;

// bundle of components around rocprofsys_init and rocprofsys_finalize
using main_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::page_rss,
                           comp::cpu_clock, comp::cpu_util>;

// bundle of components around each thread
#if defined(TIMEMORY_RUSAGE_THREAD) && TIMEMORY_RUSAGE_THREAD > 0
using thread_bundle_t = tim::lightweight_tuple<comp::wall_clock, comp::thread_cpu_clock,
                                               comp::thread_cpu_util, comp::peak_rss>;
#else
using thread_bundle_t = tim::lightweight_tuple<comp::wall_clock, comp::thread_cpu_clock,
                                               comp::thread_cpu_util>;
#endif

std::unique_ptr<main_bundle_t>&
get_main_bundle();

std::unique_ptr<init_bundle_t>&
get_init_bundle();

std::unique_ptr<preinit_bundle_t>&
get_preinit_bundle();

std::atomic<uint64_t>&
get_cpu_cid() TIMEMORY_HOT;

unique_ptr_t<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid = threading::get_id(), int64_t _parent = 0) TIMEMORY_HOT;

using cpu_cid_data_t       = std::tuple<uint64_t, uint64_t, uint32_t>;
using cpu_cid_pair_t       = std::tuple<uint64_t, uint32_t>;
using cpu_cid_parent_map_t = std::unordered_map<uint64_t, cpu_cid_pair_t>;

unique_ptr_t<cpu_cid_parent_map_t>&
get_cpu_cid_parents(int64_t _tid = threading::get_id()) TIMEMORY_HOT;

cpu_cid_data_t
create_cpu_cid_entry(int64_t _tid = threading::get_id()) TIMEMORY_HOT;

cpu_cid_pair_t
get_cpu_cid_entry(uint64_t _cid, int64_t _tid = threading::get_id()) TIMEMORY_HOT;

tim::mutex_t&
get_cpu_cid_stack_lock(int64_t _tid = threading::get_id()) TIMEMORY_HOT;

// query current value
bool
sampling_enabled_on_child_threads();

// use this to disable sampling in a region (e.g. right before thread creation)
bool
push_enable_sampling_on_child_threads(bool _v);

// use this to restore previous setting
bool
pop_enable_sampling_on_child_threads();

// make sure every newly created thead starts with this value
void
set_sampling_on_all_future_threads(bool _v);

struct scoped_child_sampling
{
    scoped_child_sampling(bool _v) { push_enable_sampling_on_child_threads(_v); }
    ~scoped_child_sampling() { pop_enable_sampling_on_child_threads(); }
};

pid_t
get_root_process_id();

bool
is_root_process();

bool
is_child_process();
}  // namespace rocprofsys

#define ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(VALUE)                               \
    ::rocprofsys::scoped_child_sampling ROCPROFSYS_VARIABLE(_scoped_child_sampling_,     \
                                                            __LINE__)                    \
    {                                                                                    \
        VALUE                                                                            \
    }
