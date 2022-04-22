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

#pragma once

#include "library/api.hpp"
#include "library/common.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/components/roctracer.hpp"
#include "library/defines.hpp"
#include "library/state.hpp"
#include "library/timemory.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/macros/language.hpp>

#include <string>
#include <string_view>
#include <unordered_set>

namespace omnitrace
{
// bundle of components around omnitrace_init and omnitrace_finalize
using main_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::cpu_clock,
                           comp::cpu_util, pthread_gotcha_t>;

using gotcha_bundle_t = tim::lightweight_tuple<fork_gotcha_t, mpi_gotcha_t>;

// bundle of components used in instrumentation
using instrumentation_bundle_t =
    tim::component_bundle<api::omnitrace, comp::wall_clock*, comp::user_global_bundle*>;

// allocator for instrumentation_bundle_t
using bundle_allocator_t = tim::data::ring_buffer_allocator<instrumentation_bundle_t>;

// bundle of components around each thread
#if defined(TIMEMORY_RUSAGE_THREAD) && TIMEMORY_RUSAGE_THREAD > 0
using omnitrace_thread_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::thread_cpu_clock,
                           comp::thread_cpu_util, comp::peak_rss>;
#else
using omnitrace_thread_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::thread_cpu_clock,
                           comp::thread_cpu_util>;
#endif

std::unique_ptr<main_bundle_t>&
get_main_bundle();

std::unique_ptr<gotcha_bundle_t>&
get_gotcha_bundle();

std::atomic<uint64_t>&
get_cpu_cid();

std::unique_ptr<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid = threading::get_id(), int64_t _parent = 0);

using cpu_cid_data_t       = std::tuple<uint64_t, uint64_t, uint16_t>;
using cpu_cid_pair_t       = std::tuple<uint64_t, uint16_t>;
using cpu_cid_parent_map_t = std::unordered_map<uint64_t, cpu_cid_pair_t>;

std::unique_ptr<cpu_cid_parent_map_t>&
get_cpu_cid_parents(int64_t _tid = threading::get_id());

cpu_cid_data_t
create_cpu_cid_entry(int64_t _tid = threading::get_id());

cpu_cid_pair_t
get_cpu_cid_entry(uint64_t _cid, int64_t _tid = threading::get_id());

}  // namespace omnitrace
