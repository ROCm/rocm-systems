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
#include "timemory/macros/language.hpp"

#include <timemory/backends/threading.hpp>

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

//
//      Initialization routines
//
inline namespace config
{
void
configure_settings();

void
print_banner(std::ostream& _os = std::cout);

void
print_settings(
    std::ostream&                                                                _os,
    std::function<bool(const std::string_view&, const std::set<std::string>&)>&& _filter);

void
print_settings();

std::string&
get_exe_name();

template <typename Tp>
bool
set_setting_value(const std::string& _name, Tp&& _v)
{
    auto _instance = tim::settings::shared_instance();
    auto _setting  = _instance->find(_name);
    if(_setting == _instance->end()) return false;
    if(!_setting->second) return false;
    return _setting->second->set(std::forward<Tp>(_v));
}

//
//      User-configurable settings
//
std::string
get_config_file();

Mode
get_mode();

bool&
is_attached();

bool&
is_binary_rewrite();

bool
get_is_continuous_integration();

bool
get_debug_env();

bool
get_debug_init();

bool
get_debug_finalize();

bool
get_debug();

bool
get_debug_tid();

bool
get_debug_pid();

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
get_use_rocm_smi();

bool&
get_use_sampling();

bool&
get_use_pid();

bool&
get_use_mpip();

bool&
get_use_critical_trace();

bool
get_use_kokkosp();

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

// make this visible so omnitrace-avail can call it
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

double&
get_thread_sampling_freq();

std::string
get_rocm_smi_devices();

int64_t
get_critical_trace_per_row();
}  // namespace config

//
//      Runtime configuration data
//
State&
get_state();

std::unique_ptr<main_bundle_t>&
get_main_bundle();

std::unique_ptr<gotcha_bundle_t>&
get_gotcha_bundle();

std::atomic<uint64_t>&
get_cpu_cid();

std::unique_ptr<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid = threading::get_id());
}  // namespace omnitrace
