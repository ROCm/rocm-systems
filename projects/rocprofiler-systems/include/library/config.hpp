// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/api.hpp"
#include "library/common.hpp"
#include "library/fork_gotcha.hpp"
#include "library/mpi_gotcha.hpp"
#include "library/roctracer.hpp"
#include "library/state.hpp"
#include "library/timemory.hpp"

#include <timemory/backends/threading.hpp>

#include <string_view>

// bundle of components around omnitrace_init and omnitrace_finalize
using main_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::cpu_clock,
                           comp::cpu_util, comp::roctracer, papi_tot_ins,
                           comp::user_global_bundle, fork_gotcha_t, mpi_gotcha_t>;

// bundle of components used in instrumentation
using instrumentation_bundle_t =
    tim::component_bundle<omnitrace, comp::wall_clock*, comp::user_global_bundle*>;

// allocator for instrumentation_bundle_t
using bundle_allocator_t = tim::data::ring_buffer_allocator<instrumentation_bundle_t>;

// bundle of components around each thread
using omnitrace_thread_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::thread_cpu_clock,
                           comp::thread_cpu_util,
#if defined(TIMEMORY_RUSAGE_THREAD) && TIMEMORY_RUSAGE_THREAD > 0
                           comp::peak_rss,
#endif
                           papi_tot_ins>;

//
//      Initialization routines
//
void
configure_settings();

void
print_config_settings(std::ostream&                                  _os,
                      std::function<bool(const std::string_view&)>&& _filter);

std::string&
get_exe_name();

//
//      User-configurable settings
//
std::string
get_config_file();

bool
get_debug();

bool
get_use_perfetto();

bool
get_use_timemory();

bool&
get_use_pid();

bool
get_use_mpip();

bool
get_use_critical_trace();

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

std::string
get_perfetto_output_filename();

int64_t
get_critical_trace_count();

size_t&
get_sample_rate();

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
