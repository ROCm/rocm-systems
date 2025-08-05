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

#include "common.hpp"
#include "defines.hpp"
#include "state.hpp"
#include "timemory.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/macros/language.hpp>

#include <fstream>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rocprofsys
{
//
//      Initialization routines
//
inline namespace config
{
using signal_handler_t = void (*)(void);

// if arg is nullptr, returns current signal handler
// if arg is non-null, returns replaced signal handler
signal_handler_t set_signal_handler(signal_handler_t);

bool
settings_are_configured() ROCPROFSYS_HOT;

void
configure_settings(bool _init = true);

void
configure_mode_settings(const std::shared_ptr<settings>&);

void
configure_signal_handler(const std::shared_ptr<settings>&);

void
configure_disabled_settings(const std::shared_ptr<settings>&);

int
get_sampling_overflow_signal();

int
get_sampling_realtime_signal();

int
get_sampling_cputime_signal();

std::set<int>
get_sampling_signals(int64_t _tid = 0);

void
finalize();

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

std::string&
get_exe_realpath();

template <typename Tp>
bool
set_setting_value(const std::string& _name, Tp&& _v,
                  settings::update_type _upd = settings::update_type::user)
{
    auto* _instance = tim::settings::instance();
    if(!_instance) return false;

    auto _setting = _instance->find(_name);
    if(_setting == _instance->end()) return false;
    if(!_setting->second) return false;

    auto& itr      = _setting->second;
    auto  _old_upd = itr->get_updated_type();

    auto _success = itr->set(std::forward<Tp>(_v), _upd);
    if(!_success) itr->set_updated(_old_upd);

    return _success;
}

template <typename Tp>
bool
set_default_setting_value(const std::string& _name, Tp&& _v)
{
    auto* _instance = tim::settings::instance();
    if(!_instance) return false;

    auto _setting = _instance->find(_name);
    if(_setting == _instance->end()) return false;
    if(!_setting->second) return false;

    if(_setting->second->get_config_updated() || _setting->second->get_environ_updated())
        return false;
    return _setting->second->set(std::forward<Tp>(_v));
}

template <typename Tp>
std::optional<Tp>
get_setting_value(const std::string& _name)
{
    auto* _instance = tim::settings::instance();
    if(!_instance) return std::nullopt;

    auto _setting = _instance->find(_name);
    if(_setting == _instance->end() || !_setting->second) return std::optional<Tp>{};

    auto&& _ret = _setting->second->get<Tp>();
    return (_ret.first) ? std::optional<Tp>{ _ret.second } : std::optional<Tp>{};
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
get_is_continuous_integration() ROCPROFSYS_HOT;

bool
get_debug_env() ROCPROFSYS_HOT;

bool
get_debug_init();

bool
get_debug_finalize();

bool
get_debug() ROCPROFSYS_HOT;

bool
get_debug_sampling() ROCPROFSYS_HOT;

bool
get_debug_tid() ROCPROFSYS_HOT;

bool
get_debug_pid() ROCPROFSYS_HOT;

int
get_verbose_env() ROCPROFSYS_HOT;

int
get_verbose() ROCPROFSYS_HOT;

bool&
get_use_perfetto() ROCPROFSYS_HOT;

bool&
get_use_timemory() ROCPROFSYS_HOT;

bool&
get_use_causal() ROCPROFSYS_HOT;

bool
get_use_rocm() ROCPROFSYS_HOT;

bool
get_use_amd_smi() ROCPROFSYS_HOT;

bool&
get_use_sampling() ROCPROFSYS_HOT;

bool&
get_use_process_sampling() ROCPROFSYS_HOT;

bool&
get_cpu_freq_enabled();

bool&
get_use_pid();

bool&
get_use_mpip();

bool
get_use_kokkosp();

bool
get_use_kokkosp_kernel_logger();

bool
get_use_vaapi_tracing();

bool
get_use_ompt();

bool
get_use_code_coverage();

bool
get_sampling_keep_internal();

bool
get_use_rcclp();

size_t
get_perfetto_shmem_size_hint();

size_t
get_perfetto_buffer_size();

bool
get_perfetto_combined_traces();

std::string
get_perfetto_fill_policy();

std::set<std::string>
get_enabled_categories();

std::set<std::string>
get_disabled_categories();

bool
get_perfetto_annotations() ROCPROFSYS_HOT;

uint64_t
get_thread_pool_size();

std::string&
get_perfetto_backend();

// make this visible so rocprof-sys-avail can call it
std::string
get_perfetto_output_filename();

double
get_trace_delay();

double
get_trace_duration();

double
get_sampling_freq();

double
get_sampling_cputime_freq();

double
get_sampling_realtime_freq();

double
get_sampling_overflow_freq();

double
get_sampling_delay();

double
get_sampling_cputime_delay();

double
get_sampling_realtime_delay();

double
get_sampling_duration();

std::string
get_sampling_cpus();

std::set<int64_t>
get_sampling_cputime_tids();

std::set<int64_t>
get_sampling_realtime_tids();

std::set<int64_t>
get_sampling_overflow_tids();

bool
get_sampling_include_inlines();

size_t
get_num_threads_hint();

size_t
get_sampling_allocator_size();

double
get_process_sampling_freq();

double
get_process_sampling_duration();

std::string
get_sampling_gpus();

bool
get_trace_thread_locks();

bool
get_trace_thread_rwlocks();

bool
get_trace_thread_spin_locks();

bool
get_trace_thread_barriers();

bool
get_trace_thread_join();

bool
get_use_tmp_files();

std::string
get_tmpdir();

struct tmp_file
{
    tmp_file(std::string);
    ~tmp_file();

    bool open(int, int);
    bool open(std::ios::openmode = std::ios::binary | std::ios::in | std::ios::out);
    bool fopen(const char* = "r+");
    bool flush();
    bool close();
    bool remove();

    explicit operator bool() const;

    std::string  filename = {};
    std::fstream stream   = {};
    FILE*        file     = nullptr;
    int          fd       = -1;

private:
    void touch() const;

private:
    pid_t m_pid = getpid();
};

std::shared_ptr<tmp_file>
get_tmp_file(std::string _basename, std::string _ext = "dat");

CausalBackend
get_causal_backend();

CausalMode
get_causal_mode();

bool
get_causal_end_to_end();

std::vector<int64_t>
get_causal_fixed_speedup();

std::string
get_causal_output_filename();

std::vector<std::string>
get_causal_binary_scope();

std::vector<std::string>
get_causal_source_scope();

std::vector<std::string>
get_causal_function_scope();

std::vector<std::string>
get_causal_binary_exclude();

std::vector<std::string>
get_causal_source_exclude();

std::vector<std::string>
get_causal_function_exclude();
}  // namespace config
}  // namespace rocprofsys
