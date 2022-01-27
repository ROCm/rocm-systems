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

#include <string_view>

namespace omnitrace
{
// bundle of components around omnitrace_init and omnitrace_finalize
using main_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::cpu_clock,
                           comp::cpu_util, comp::roctracer, comp::user_global_bundle,
                           fork_gotcha_t, mpi_gotcha_t, pthread_gotcha_t>;

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

//
//      Initialization routines
//
void
configure_settings() TIMEMORY_VISIBILITY("default");

void
print_config_settings(
    std::ostream&                                                                _os,
    std::function<bool(const std::string_view&, const std::set<std::string>&)>&& _filter);

std::string&
get_exe_name();

//
//      User-configurable settings
//
std::string
get_config_file();

bool
get_debug_env();

bool
get_debug();

int
get_verbose_env();

int
get_verbose();

bool&
get_use_perfetto();

bool&
get_use_timemory();

bool&
get_use_roctracer();

bool&
get_use_sampling();

bool&
get_use_pid();

bool&
get_use_mpip();

bool&
get_use_critical_trace();

bool
get_timeline_sampling();

bool
get_flat_sampling();

bool
get_roctracer_timeline_profile();

bool
get_roctracer_flat_profile();

bool
get_trace_hsa_api();

bool
get_trace_hsa_activity();

bool
get_critical_trace_debug();

bool
get_critical_trace_serialize_names();

size_t
get_perfetto_shmem_size_hint();

size_t
get_perfetto_buffer_size();

uint64_t
get_critical_trace_update_freq();

uint64_t
get_critical_trace_num_threads();

std::string
get_trace_hsa_api_types();

std::string&
get_backend();

std::string&
get_perfetto_output_filename();

int64_t
get_critical_trace_count();

size_t&
get_instrumentation_interval();

double&
get_sampling_freq();

double&
get_sampling_delay();

int64_t
get_critical_trace_per_row();

//
//      Runtime configuration data
//
State&
get_state();

std::unique_ptr<main_bundle_t>&
get_main_bundle();

std::atomic<uint64_t>&
get_cpu_cid();

std::unique_ptr<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid = threading::get_id());
}  // namespace omnitrace
