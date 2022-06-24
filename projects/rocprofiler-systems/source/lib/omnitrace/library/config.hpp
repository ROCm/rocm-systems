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
bool
settings_are_configured() OMNITRACE_HOT;

void
configure_settings(bool _init = true);

void
configure_mode_settings();

void
configure_signal_handler();

void
configure_disabled_settings();

void
handle_deprecated_setting(const std::string& _old, const std::string& _new,
                          int _verbose = 0);

void
print_banner(std::ostream& _os = std::cerr);

void
print_settings(
    std::ostream&                                                                _os,
    std::function<bool(const std::string_view&, const std::set<std::string>&)>&& _filter);

void
print_settings(bool include_env = true);

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

template <typename Tp>
bool
set_default_setting_value(const std::string& _name, Tp&& _v)
{
    auto _instance = tim::settings::shared_instance();
    auto _setting  = _instance->find(_name);
    if(_setting == _instance->end()) return false;
    if(!_setting->second) return false;
    if(_setting->second->get_config_updated() || _setting->second->get_environ_updated())
        return false;
    return _setting->second->set(std::forward<Tp>(_v));
}

template <typename Tp>
std::pair<bool, Tp>
get_setting_value(const std::string& _name)
{
    auto _instance = tim::settings::shared_instance();
    if(!_instance) return std::make_pair(false, Tp{});
    auto _setting = _instance->find(_name);
    if(_setting == _instance->end() || !_setting->second)
        return std::make_pair(false, Tp{});
    return _setting->second->get<Tp>();
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
get_is_continuous_integration() OMNITRACE_HOT;

bool
get_debug_env() OMNITRACE_HOT;

bool
get_debug_init();

bool
get_debug_finalize();

bool
get_debug() OMNITRACE_HOT;

bool
get_debug_sampling() OMNITRACE_HOT;

bool
get_debug_tid() OMNITRACE_HOT;

bool
get_debug_pid() OMNITRACE_HOT;

int
get_verbose_env() OMNITRACE_HOT;

int
get_verbose() OMNITRACE_HOT;

bool&
get_use_perfetto() OMNITRACE_HOT;

bool&
get_use_timemory() OMNITRACE_HOT;

bool&
get_use_roctracer() OMNITRACE_HOT;

bool&
get_use_rocm_smi() OMNITRACE_HOT;

bool&
get_use_sampling() OMNITRACE_HOT;

bool&
get_use_process_sampling() OMNITRACE_HOT;

bool&
get_use_pid();

bool&
get_use_mpip();

bool&
get_use_critical_trace() OMNITRACE_HOT;

bool
get_use_kokkosp();

bool
get_use_kokkosp_kernel_logger();

bool
get_use_ompt();

bool
get_use_code_coverage();

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

bool
get_perfetto_combined_traces();

std::string
get_perfetto_fill_policy();

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
get_process_sampling_freq();

std::string
get_sampling_gpus();

int64_t
get_critical_trace_per_row();

bool
get_trace_thread_locks();
}  // namespace config

//
//      Runtime configuration data
//
State&
get_state() TIMEMORY_HOT;

/// returns old state
State set_state(State);
}  // namespace omnitrace
