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
//
//      Initialization routines
//
inline namespace config
{
void
configure_settings(bool _init = true);

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
get_debug_sampling();

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
get_use_ompt();

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

std::string
get_sampling_cpus();

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

/// returns old state
State set_state(State);
}  // namespace omnitrace
