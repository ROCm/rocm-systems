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

#include "library/sampling.hpp"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/node_info.hpp"
#include "core/perf.hpp"
#include "core/rocpd/data_processor.hpp"
#include "core/rocpd/json.hpp"
#include "core/sample_cache/cache_manager.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/backtrace_metrics.hpp"
#include "library/components/backtrace_timestamp.hpp"
#include "library/components/callchain.hpp"
#include "library/perf.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"
#include "timemory/process/threading.hpp"

#include <string_view>
#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/sampling/overflow.hpp>
#include <timemory/sampling/sampler.hpp>
#include <timemory/sampling/timer.hpp>
#include <timemory/storage.hpp>
#include <timemory/units.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>

#include <pthread.h>
#include <signal.h>

namespace tim
{
namespace math
{
template <typename Tp, typename Up>
TIMEMORY_INLINE Tp
plus(Tp&& _lhs, const Up& _rhs)
{
    Tp _v = _lhs;
    plus(_v, _rhs);
    return _v;
}
}  // namespace math
}  // namespace tim
namespace rocprofsys
{
namespace sampling
{
using ::tim::sampling::dynamic;
using ::tim::sampling::overflow;
using ::tim::sampling::timer;

using hw_counters               = typename component::backtrace_metrics::hw_counters;
using signal_type_instances     = thread_data<std::set<int>, category::sampling>;
using sampler_running_instances = thread_data<bool, category::sampling>;
using bundle_t =
    tim::lightweight_tuple<component::backtrace_timestamp, component::backtrace,
                           component::backtrace_metrics, component::callchain>;
using sampler_t              = tim::sampling::sampler<bundle_t, dynamic>;
using sampler_instances      = thread_data<sampler_t, category::sampling>;
using sampler_init_instances = thread_data<bundle_t, category::sampling>;

using component::backtrace;
using component::backtrace_cpu_clock;  // NOLINT
using component::backtrace_fraction;   // NOLINT
using component::backtrace_metrics;
using component::backtrace_timestamp;
using component::backtrace_wall_clock;  // NOLINT
using component::callchain;
using component::sampling_cpu_clock;
using component::sampling_gpu_busy_gfx;
using component::sampling_gpu_busy_mm;
using component::sampling_gpu_busy_umc;
using component::sampling_gpu_jpeg;
using component::sampling_gpu_memory;
using component::sampling_gpu_power;
using component::sampling_gpu_temp;
using component::sampling_gpu_vcn;
using component::sampling_percent;
using component::sampling_wall_clock;
}  // namespace sampling
}  // namespace rocprofsys

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(prevent_reentry, sampling::sampler_t, std::true_type)

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(provide_backtrace, sampling::sampler_t, std::false_type)

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(buffer_size, sampling::sampler_t,
                                 TIMEMORY_ESC(std::integral_constant<size_t, 2048>))

namespace rocprofsys
{
namespace sampling
{
namespace
{
using sampler_allocator_t = typename sampler_t::allocator_t;

template <typename Category>
inline std::string
get_category_track_name(uint64_t tid)
{
    return std::string(trait::name<Category>::value) + "_" + std::to_string(tid);
}

std::string
generate_call_stack_json(const tim::unwind::processed_entry& stack_entry)
{
    auto call_stack = ::rocpd::json::create();

    call_stack->set("name", std::string(demangle(stack_entry.name)));
    call_stack->set("pc", as_hex(stack_entry.address));
    call_stack->set("file", std::string(stack_entry.location));

    return call_stack->to_string();
}

std::string
generate_line_info_json(const tim::unwind::processed_entry& line_info_entry)
{
    auto line_info = ::rocpd::json::create();
    line_info->set("line_address", as_hex(line_info_entry.line_address));
    line_info->set("name", std::string(demangle(line_info_entry.name)));

    if(line_info_entry.lineinfo && !line_info_entry.lineinfo.lines.empty())
    {
        auto _lines = line_info_entry.lineinfo.lines;
        std::reverse(_lines.begin(), _lines.end());
        for(const auto& line : _lines)
        {
            auto inlined = ::rocpd::json::create();
            inlined->set("name", std::string(demangle(line.name)));
            inlined->set("location", std::string(line.location));
            inlined->set("line", std::to_string(line.line));
            line_info->set("inlined", inlined);
        }
    }

    return line_info->to_string();
}

std::string
generate_hw_counter_json(int64_t _tid, const backtrace_metrics& metrics)
{
    auto extdata = ::rocpd::json::create();

    if(!metrics.get_hw_counters().empty())
    {
        auto _labels      = backtrace_metrics::get_hw_counter_labels(_tid);
        auto _hw_cnt_vals = metrics.get_hw_counters();

        auto hw_counters = ::rocpd::json::create();
        for(size_t i = 0; i < _labels.size(); ++i)
        {
            hw_counters->set(_labels.at(i), _hw_cnt_vals.at(i));
        }

        extdata->set("hw_counters", hw_counters);
    }

    return extdata->to_string();
}

rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

std::string
get_track_name(const thread_info& _thread_info)
{
    size_t thread_id     = _thread_info.index_data->system_value;
    size_t sequent_value = _thread_info.index_data->sequent_value;
    return JOIN(" ", "Thread", sequent_value, "Overflow", "(S)", thread_id);
}

void
rocpd_initialize_sampling_category()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    sample_cache::get_cache_metadata().add_string(trait::name<category::sampling>::value);
    sample_cache::get_cache_metadata().add_string(
        trait::name<category::overflow_sampling>::value);
    sample_cache::get_cache_metadata().add_string(
        trait::name<category::timer_sampling>::value);

    _is_initialized = true;
}

void
rocpd_initialize_thread_info(size_t tid)
{
    const auto& _thread_info = thread_info::get(tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", tid);
    if(!_thread_info) return;

    sample_cache::get_cache_metadata().add_thread_info(
        { getppid(), getpid(),
          static_cast<size_t>(_thread_info->index_data->system_value),
          static_cast<uint32_t>(_thread_info->get_start()),
          static_cast<uint32_t>(_thread_info->get_stop()), "{}" });
}

void
rocpd_init_track(int64_t tid)
{
    const auto& _thread_info = thread_info::get(tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", tid);
    if(!_thread_info) return;

    size_t thread_id = _thread_info->index_data->system_value;

    const auto& _track_name = get_track_name(*_thread_info);

    sample_cache::get_cache_metadata().add_track({ _track_name, thread_id, "{}" });
}

template <typename Category>
void
rocpd_insert_region(size_t thread_id, size_t start_time, size_t end_time, size_t name_id,
                    const char* track, const char* call_stack = "{}",
                    const char* line_info = "{}", const char* extdata = "{}")
{
    auto& data_processor     = get_data_processor();
    auto& n_info             = node_info::get_instance();
    auto  string_primary_key = data_processor.insert_string(trait::name<Category>::value);

    auto event_id = data_processor.insert_event(string_primary_key, 0, 0, 0, call_stack,
                                                line_info, extdata);

    data_processor.insert_region(n_info.id, getpid(), thread_id, start_time, end_time,
                                 name_id, event_id);
    data_processor.insert_sample(track, start_time, event_id);
}

auto&
get_sampler_allocators()
{
    static auto _v = std::vector<std::shared_ptr<sampler_allocator_t>>{};
    return _v;
}

std::set<int>
configure(bool _setup, int64_t _tid = threading::get_id());

void
configure_sampler_allocator(std::shared_ptr<sampler_allocator_t>& _v)
{
    if(_v) return;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    _v = std::make_shared<sampler_allocator_t>();
    _v->reserve(config::get_sampling_allocator_size());
}

void
configure_sampler_allocators()
{
    auto& _allocators = get_sampler_allocators();
    if(_allocators.empty())
    {
        // avoid lock until necessary
        auto_lock_t _alloc_lk{ type_mutex<decltype(_allocators)>() };
        if(_allocators.empty())
        {
            _allocators.resize(std::ceil(config::get_num_threads_hint() /
                                         config::get_sampling_allocator_size()));
            for(auto& itr : _allocators)
                configure_sampler_allocator(itr);
        }
    }
}

std::shared_ptr<sampler_allocator_t>
get_sampler_allocator()
{
    configure_sampler_allocators();

    auto& _allocators = get_sampler_allocators();

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto_lock_t _lk{ type_mutex<sampler_allocator_t>() };

    for(auto& itr : _allocators)
    {
        if(!itr) configure_sampler_allocator(itr);
        if(itr->size() < config::get_sampling_allocator_size()) return itr;
    }

    auto& _v = _allocators.emplace_back();
    configure_sampler_allocator(_v);
    return _v;
}

template <typename... Args>
void
thread_sigmask(Args... _args)
{
    auto _err = pthread_sigmask(_args...);
    if(_err != 0)
    {
        errno = _err;
        perror("pthread_sigmask");
        exit(EXIT_FAILURE);
    }
}

template <typename Tp>
sigset_t
get_signal_set(Tp&& _v)
{
    sigset_t _sigset;
    sigemptyset(&_sigset);
    for(auto itr : _v)
        sigaddset(&_sigset, itr);
    return _sigset;
}

template <typename Tp>
std::string
get_signal_names(Tp&& _v)
{
    std::string _sig_names{};
    for(auto&& itr : _v)
        _sig_names += std::get<0>(tim::signals::signal_settings::get_info(
                          static_cast<tim::signals::sys_signal>(itr))) +
                      " ";
    return (_sig_names.empty()) ? _sig_names
                                : _sig_names.substr(0, _sig_names.length() - 1);
}

unique_ptr_t<sampler_t>&
get_sampler(int64_t _tid = threading::get_id())
{
    static auto* _v = sampler_instances::get();
    return _v->at(_tid);
}

unique_ptr_t<bundle_t>&
get_sampler_init(int64_t _tid = threading::get_id())
{
    return sampler_init_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<bool>&
get_sampler_running(int64_t _tid)
{
    return sampler_running_instances::instance(construct_on_thread{ _tid }, false);
}

auto&
get_duration_disabled()
{
    static auto _v = std::atomic<bool>{ false };
    return _v;
}

auto&
get_is_duration_thread()
{
    static thread_local auto _v = false;
    return _v;
}

auto&
get_duration_cv()
{
    static auto _v = std::condition_variable{};
    return _v;
}

auto&
get_duration_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

auto&
get_duration_thread()
{
    static auto _v = std::unique_ptr<std::thread>{};
    return _v;
}

auto
notify_duration_thread()
{
    if(get_duration_thread() && !get_is_duration_thread())
    {
        std::unique_lock<std::mutex> _lk{ get_duration_mutex(), std::defer_lock };
        if(!_lk.owns_lock()) _lk.lock();
        get_duration_cv().notify_all();
    }
}

void
stop_duration_thread()
{
    if(get_duration_thread() && !get_is_duration_thread())
    {
        notify_duration_thread();
        get_duration_thread()->join();
        get_duration_thread().reset();
    }
}

void
start_duration_thread()
{
    static std::mutex            _start_mutex{};
    std::unique_lock<std::mutex> _start_lk{ _start_mutex, std::defer_lock };
    if(!_start_lk.owns_lock()) _start_lk.lock();

    if(!get_duration_thread() && config::get_sampling_duration() > 0.0)
    {
        // we may need to protect against recursion bc of pthread wrapper
        static bool _protect = false;
        if(_protect) return;
        _protect   = true;
        auto _now  = std::chrono::steady_clock::now();
        auto _end  = _now + std::chrono::nanoseconds{ static_cast<uint64_t>(
                               config::get_sampling_duration() * units::sec) };
        auto _func = [_end]() {
            thread_info::init(true);
            threading::set_thread_name("omni.samp.dur");
            get_is_duration_thread() = true;
            bool _wait               = true;
            while(_wait)
            {
                _wait = false;
                std::unique_lock<std::mutex> _lk{ get_duration_mutex(), std::defer_lock };
                if(!_lk.owns_lock()) _lk.lock();
                get_duration_cv().wait_until(_lk, _end);
                auto _premature = (std::chrono::steady_clock::now() < _end);
                auto _finalized = (get_state() >= State::Finalized);
                if(_premature && !_finalized)
                {
                    // protect against spurious wakeups
                    ROCPROFSYS_VERBOSE(
                        2, "%sSpurious wakeup of sampling duration thread...\n",
                        tim::log::color::warning());
                    _wait = true;
                }
                else if(_finalized)
                {
                    break;
                }
                else
                {
                    get_duration_disabled().store(true);
                    ROCPROFSYS_VERBOSE(1,
                                       "Sampling duration of %f seconds has elapsed. "
                                       "Shutting down sampling...\n",
                                       config::get_sampling_duration());
                    configure(false, 0);
                }
            }
        };

        ROCPROFSYS_VERBOSE(1, "Sampling will be disabled after %f seconds...\n",
                           config::get_sampling_duration());

        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        get_duration_thread() = std::make_unique<std::thread>(_func);
        _protect              = false;
    }
}

auto&
get_offload_file()
{
    static auto _v = []() {
        auto _tmp_v = config::get_tmp_file("sampling");
        if(get_use_tmp_files())
        {
            auto _success = _tmp_v->open();
            ROCPROFSYS_CI_FAIL(!_success,
                               "Error opening sampling offload temporary file '%s'\n",
                               _tmp_v->filename.c_str());
        }
        return _tmp_v;
    }();
    return _v;
}

locking::atomic_mutex&
get_offload_mutex()
{
    static auto _v = locking::atomic_mutex{};
    return _v;
}

using sampler_bundle_t = typename sampler_t::bundle_type;
using sampler_buffer_t = tim::data_storage::ring_buffer<sampler_bundle_t>;
using pos_type         = typename std::fstream::pos_type;

auto offload_seq_data = std::unordered_map<int64_t, std::set<pos_type>>{};

void
offload_buffer(int64_t _seq, sampler_buffer_t&& _buf)
{
    ROCPROFSYS_REQUIRE(get_use_tmp_files())
        << "Error! sampling allocator tries to offload buffer of samples but "
           "rocprof-sys was configured to not use temporary files\n";

    // use homemade atomic_mutex/atomic_lock since contention will be low
    // and using pthread_lock might trigger our wrappers
    auto  _lk   = locking::atomic_lock{ get_offload_mutex() };
    auto& _file = get_offload_file();

    ROCPROFSYS_REQUIRE(_file)
        << "Error! sampling allocator tried to offload buffer of samples for thread "
        << _seq << " but the offload file does not exist\n";

    ROCPROFSYS_VERBOSE_F(2, "Offloading %zu samples for thread %li to %s...\n",
                         _buf.count(), _seq, _file->filename.c_str());
    auto& _fs = _file->stream;

    ROCPROFSYS_REQUIRE(_fs.good()) << "Error! temporary file for offloading buffer is in "
                                      "an invalid state during offload for thread "
                                   << _seq << "\n";

    offload_seq_data[_seq].emplace(_fs.tellg());
    _fs.write(reinterpret_cast<char*>(&_seq), sizeof(_seq));
    auto _data = std::move(_buf);
    _data.save(_fs);
    _data.destroy();
    _buf.destroy();
}

auto
load_offload_buffer(int64_t _thread_idx)
{
    auto _data = std::vector<sampler_buffer_t>{};
    if(!get_use_tmp_files())
    {
        ROCPROFSYS_WARNING_F(
            2, "[sampling] returning no data because using temporary files is disabled");
        return _data;
    }

    // use homemade atomic_mutex/atomic_lock since contention will be low
    // and using pthread_lock might trigger our wrappers
    auto  _lk   = locking::atomic_lock{ get_offload_mutex() };
    auto& _file = get_offload_file();
    if(!_file)
    {
        ROCPROFSYS_WARNING_F(
            0, "[sampling] returning no data because the offload file no longer exists");
        return _data;
    }

    auto& _fs = _file->stream;

    if(_fs.is_open()) _fs.close();

    if(!_file->open(std::ios::binary | std::ios::in))
    {
        ROCPROFSYS_WARNING_F(0, "[sampling] %s failed to open", _file->filename.c_str());
        return _data;
    }

    if(offload_seq_data.count(_thread_idx) == 0) return _data;

    size_t _count = 0;
    for(auto itr : offload_seq_data.at(_thread_idx))
    {
        _fs.seekg(itr);  // set to the absolute position

        int64_t _seq = 0;
        _fs.read(reinterpret_cast<char*>(&_seq), sizeof(_seq));
        if(_fs.eof()) break;

        sampler_buffer_t _buffer{};
        _buffer.load(_fs);

        if(_seq != _thread_idx)
        {
            ROCPROFSYS_WARNING_F(
                0,
                "[sampling] file position %zu returned %zi instead of (expected) %zi\n",
                static_cast<uintptr_t>(itr), _seq, _thread_idx);
            continue;
        }
        _count += _buffer.count();
        _data.emplace_back(std::move(_buffer));
    }

    ROCPROFSYS_VERBOSE_F(2, "[sampling] Loaded %zu samples for thread %li...\n", _count,
                         _thread_idx);

    _file->close();

    return _data;
}

std::set<int>
configure(bool _setup, int64_t _tid)
{
    const auto& _info         = thread_info::get(_tid, SequentTID);
    auto&       _sampler      = sampling::get_sampler(_tid);
    auto&       _perf_sampler = perf::get_instance(_tid);
    auto&       _running      = get_sampler_running(_tid);
    bool        _is_running   = (!_running) ? false : *_running;
    auto&       _signal_types = sampling::get_signal_types(_tid);

    ROCPROFSYS_CONDITIONAL_THROW(
        get_use_causal(), "Internal error! configuring sampling not permitted when "
                          "causal profiling is enabled");

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto&& _cputime_tids  = get_sampling_cputime_tids();
    auto&& _realtime_tids = get_sampling_realtime_tids();
    auto&& _overflow_tids = get_sampling_overflow_tids();

    auto _erase_tid_signal = [_tid, &_signal_types](auto& _tids, int _signum) {
        if(!_tids.empty())
        {
            if(_tids.count(_tid) == 0)
            {
                ROCPROFSYS_VERBOSE(3, "Disabling SIG%i from thread %li\n", _signum, _tid);
                _signal_types->erase(_signum);
            }
        }
    };

    _erase_tid_signal(_cputime_tids, get_sampling_cputime_signal());
    _erase_tid_signal(_realtime_tids, get_sampling_realtime_signal());
    _erase_tid_signal(_overflow_tids, get_sampling_overflow_signal());

    if(_setup && !_sampler && !_is_running && !_signal_types->empty())
    {
        if(get_duration_disabled()) return std::set<int>{};

        // if this thread has an offset ID, that means it was created internally
        // and is probably here bc it called a function which was instrumented.
        // thus we should not start a sampler for it
        if(_tid > 0 && _info && _info->is_offset) return std::set<int>{};
        // if the thread state is disabled or completed, return
        if(_info && _info->index_data->sequent_value == _tid &&
           get_thread_state() == ThreadState::Disabled)
            return std::set<int>{};

        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());

        if(trait::runtime_enabled<backtrace_metrics>::get())
            backtrace_metrics::configure(_setup, _tid);

        // NOTE: signals need to be unblocked by calling function
        sampling::block_signals(*_signal_types);

        auto _verbose = std::min<int>(get_verbose() - 2, 2);
        if(get_debug_sampling()) _verbose = 2;

        ROCPROFSYS_DEBUG("Requesting allocator for sampler on thread %lu...\n", _tid);
        auto _alloc = get_sampler_allocator();

        ROCPROFSYS_DEBUG("Configuring sampler for thread %lu...\n", _tid);
        sampling::sampler_instances::construct(construct_on_thread{ _tid }, _alloc,
                                               "rocprofsys", _tid, _verbose);

        _sampler->set_flags(SA_RESTART);
        _sampler->set_verbose(_verbose);

        if(_signal_types->count(get_sampling_realtime_signal()) > 0)
        {
            _sampler->configure(timer{ get_sampling_realtime_signal(), CLOCK_REALTIME,
                                       SIGEV_THREAD_ID, get_sampling_realtime_freq(),
                                       get_sampling_realtime_delay(), _tid,
                                       threading::get_sys_tid() });
        }

        if(_signal_types->count(get_sampling_cputime_signal()) > 0)
        {
            _sampler->configure(
                timer{ get_sampling_cputime_signal(), CLOCK_THREAD_CPUTIME_ID,
                       SIGEV_THREAD_ID, get_sampling_cputime_freq(),
                       get_sampling_cputime_delay(), _tid, threading::get_sys_tid() });
        }

        if(_signal_types->count(get_sampling_overflow_signal()) > 0)
        {
            if(_signal_types->size() == 1)
                trait::runtime_enabled<backtrace_metrics>::set(false);

            _perf_sampler = std::make_unique<perf::perf_event>();

            struct perf_event_attr _pe;
            memset(&_pe, 0, sizeof(_pe));

            auto _freq = get_sampling_overflow_freq();
            auto _overflow_event =
                get_setting_value<std::string>("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT")
                    .value_or("perf::PERF_COUNT_HW_CACHE_REFERENCES");

            perf::config_overflow_sampling(_pe, _overflow_event, _freq);

            _pe.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN;

            _pe.wakeup_events            = 10;
            _pe.exclude_idle             = 1;
            _pe.exclude_kernel           = 1;
            _pe.exclude_hv               = 1;
            _pe.exclude_callchain_kernel = 1;
            _pe.disabled                 = 1;
            _pe.inherit                  = 0;

            if(_pe.type == PERF_TYPE_SOFTWARE)
            {
                _pe.use_clockid = 1;
                _pe.clockid     = CLOCK_REALTIME;
            }

            auto _perf_open_error =
                _perf_sampler->open(_pe, _info->index_data->system_value);

            ROCPROFSYS_REQUIRE(!_perf_open_error)
                << "perf backend for overflow failed to activate: " << *_perf_open_error;

            _perf_sampler->set_ready_signal(get_sampling_overflow_signal());
            _sampler->configure(overflow{
                get_sampling_overflow_signal(),
                [](int _sig, pid_t, long, int64_t _idx) {
                    perf::get_instance(_idx)->set_ready_signal(_sig);
                    return true;
                },
                [](int, pid_t, long, int64_t _idx) {
                    return perf::get_instance(_idx)->start();
                },
                [](int, pid_t, long, int64_t _idx) {
                    if(!perf::get_instance(_idx) || !perf::get_instance(_idx)->is_open())
                        return true;
                    auto _stopped = perf::get_instance(_idx)->stop();
                    if(_stopped) perf::get_instance(_idx)->close();
                    return _stopped;
                },
                _tid, threading::get_sys_tid() });
        }

        if(get_use_tmp_files())
        {
            auto _file = get_offload_file();
            if(_file && *_file) _sampler->set_offload(&offload_buffer);
        }

        static_assert(tim::trait::buffer_size<sampling::sampler_t>::value > 0,
                      "Error! Zero buffer size");

        ROCPROFSYS_CONDITIONAL_THROW(
            _sampler->get_buffer_size() !=
                tim::trait::buffer_size<sampling::sampler_t>::value,
            "dynamic sampler has a buffer size different from static trait: %zu instead "
            "of %zu",
            _sampler->get_buffer_size(),
            tim::trait::buffer_size<sampling::sampler_t>::value);

        ROCPROFSYS_CONDITIONAL_THROW(
            _sampler->get_buffer_size() <= 0,
            "dynamic sampler requires a positive buffer size: %zu",
            _sampler->get_buffer_size());

        for(auto itr : *_signal_types)
        {
            if(itr == get_sampling_overflow_signal())
            {
                auto _freq = get_sampling_overflow_freq();
                auto _overflow_event =
                    get_setting_value<std::string>("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT")
                        .value_or("perf::PERF_COUNT_HW_CACHE_REFERENCES");
                ROCPROFSYS_VERBOSE(2,
                                   "[SIG%i] Sampler for thread %lu will be triggered "
                                   "every %.1f %s events...\n",
                                   itr, _tid, _freq, _overflow_event.c_str());
            }
            else
            {
                const char* _type =
                    (itr == get_sampling_realtime_signal()) ? "wall" : "CPU";
                const auto* _timer =
                    dynamic_cast<const timer*>(_sampler->get_trigger(itr));
                if(_timer)
                {
                    ROCPROFSYS_VERBOSE(
                        2,
                        "[SIG%i] Sampler for thread %lu will be triggered %.1fx per "
                        "second of %s-time (every %.3e milliseconds)...\n",
                        itr, _tid, _timer->get_frequency(units::sec), _type,
                        _timer->get_period(units::msec));
                }
            }
        }
        rocpd_initialize_sampling_category();
        rocpd_initialize_thread_info(_tid);
        rocpd_init_track(_tid);

        *_running = true;
        sampling::get_sampler_init(_tid)->sample();
        start_duration_thread();
        _sampler->start();
    }
    else if(!_setup && _sampler && _is_running)
    {
        ROCPROFSYS_DEBUG("Stopping sampler for thread %lu...\n", _tid);
        *_running = false;

        if(_tid == threading::get_id() && !_signal_types->empty())
        {
            sampling::block_signals(*_signal_types);
        }

        notify_duration_thread();

        if(_tid == 0)
        {
            // this propagates to all threads
            block_samples();
            _sampler->ignore(*_signal_types);
        }

        _sampler->stop();
        _sampler->reset();
        *_running = false;
        if(_perf_sampler) _perf_sampler->stop();

        if(_tid == 0)
        {
            for(int64_t i = 1; i < ROCPROFSYS_MAX_THREADS; ++i)
            {
                if(sampling::get_sampler(i)) sampling::get_sampler(i)->stop();
                if(perf::get_instance(i)) perf::get_instance(i)->stop();
            }

            for(int64_t i = 1; i < ROCPROFSYS_MAX_THREADS; ++i)
            {
                if(sampling::get_sampler(i))
                {
                    sampling::get_sampler(i)->reset();
                    *get_sampler_running(i) = false;
                }
            }

            // wait for the samples to finish
            for(auto& itr : get_sampler_allocators())
                if(itr) itr->flush();

            stop_duration_thread();
        }

        if(trait::runtime_enabled<backtrace_metrics>::get())
            backtrace_metrics::configure(_setup, _tid);

        ROCPROFSYS_DEBUG("Sampler destroyed for thread %lu\n", _tid);
    }

    return (_signal_types) ? *_signal_types : std::set<int>{};
}

struct timer_sampling_data
{
    int64_t                                   m_tid     = -1;
    uint64_t                                  m_beg     = 0;
    uint64_t                                  m_end     = 0;
    std::vector<tim::unwind::processed_entry> m_stack   = {};
    backtrace_metrics                         m_metrics = {};
};

struct overflow_sampling_data
{
    int64_t                                   m_tid   = -1;
    uint64_t                                  m_beg   = 0;
    uint64_t                                  m_end   = 0;
    std::vector<tim::unwind::processed_entry> m_stack = {};
};

std::vector<timer_sampling_data>
post_process_timer_data(int64_t, const bundle_t*, const std::vector<bundle_t*>&);

std::vector<overflow_sampling_data>
post_process_overflow_data(int64_t, const bundle_t*, const std::vector<bundle_t*>&);

void
post_process_perfetto(int64_t, const std::vector<timer_sampling_data>&,
                      const std::vector<overflow_sampling_data>&);

void
post_process_timemory(int64_t, const std::vector<timer_sampling_data>&,
                      const std::vector<overflow_sampling_data>&);

void
post_process_rocpd(int64_t _tid, const std::vector<timer_sampling_data>& _timer_data,
                   const std::vector<overflow_sampling_data>& _overflow_data);

auto static_strings = std::set<std::string>{};

}  // namespace

unique_ptr_t<std::set<int>>&
get_signal_types(int64_t _tid)
{
    return signal_type_instances::instance(construct_on_thread{ _tid },
                                           rocprofsys::get_sampling_signals(_tid));
}

std::set<int>
setup()
{
    if(!get_use_sampling()) return std::set<int>{};
    return configure(true);
}

std::set<int>
shutdown()
{
    if(is_child_process())
    {
        for(auto& itr : *sampler_instances::get())
            itr.release();
        return std::set<int>{};
    }

    auto _v = configure(false);
    if(utility::get_thread_index() == 0) stop_duration_thread();
    return _v;
}

void
block_samples()
{
    trait::runtime_enabled<sampler_t>::set(false);
}

void
unblock_samples()
{
    trait::runtime_enabled<sampler_t>::set(true);
}

void
block_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        ROCPROFSYS_VERBOSE(2, "No signals to block...\n");
        return;
    }

    ROCPROFSYS_DEBUG("Blocking signals [%s] on thread #%lu...\n",
                     get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_BLOCK, &_v, nullptr);
}

void
unblock_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        ROCPROFSYS_VERBOSE(2, "No signals to unblock...\n");
        return;
    }

    ROCPROFSYS_DEBUG("Unblocking signals [%s] on thread #%lu...\n",
                     get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_UNBLOCK, &_v, nullptr);
}

void
post_process()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    size_t _total_data       = 0;
    size_t _total_threads    = 0;
    auto   _external_samples = std::atomic<size_t>{ 0 };
    auto   _internal_samples = std::atomic<size_t>{ 0 };

    ROCPROFSYS_VERBOSE(2 || get_debug_sampling(), "Stopping sampling components...\n");

    rocprofsys::component::backtrace::stop();
    configure(false, 0);

    for(auto& itr : get_sampler_allocators())
        if(itr) itr->flush();

    for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
    {
        auto& _sampler = get_sampler(i);

        if(!_sampler)
        {
            // this should be relatively common
            ROCPROFSYS_CONDITIONAL_PRINT(
                get_debug() && get_verbose() >= 2,
                "Post-processing sampling entries for thread %lu skipped (no sampler)\n",
                i);
            continue;
        }

        auto* _init = get_sampler_init(i).get();

        if(!_init)
        {
            // this is not common
            ROCPROFSYS_PRINT("Post-processing sampling entries for thread %lu skipped "
                             "(not initialized)\n",
                             i);
            continue;
        }

        const auto& _thread_info = thread_info::get(i, SequentTID);

        ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                           "Getting sampler data for thread %lu...\n", i);

        auto _raw_data    = _sampler->get_data();
        auto _loaded_data = load_offload_buffer(i);
        for(auto line : _loaded_data)
        {
            while(!line.is_empty())
            {
                auto _v = sampler_bundle_t{};
                line.read(&_v);
                _raw_data.emplace_back(std::move(_v));
            }
            line.destroy();
        }

        ROCPROFSYS_VERBOSE(2 || get_debug_sampling(),
                           "Sampler data for thread %lu has %zu initial entries...\n", i,
                           _raw_data.size());

        ROCPROFSYS_CI_THROW(
            _sampler->get_sample_count() != _raw_data.size(),
            "Error! sampler recorded %zu samples but %zu samples were returned\n",
            _sampler->get_sample_count(), _raw_data.size());
        // single sample that is useless (backtrace to unblocking signals)
        if(_raw_data.size() == 1 && _raw_data.front().size() <= 1) _raw_data.clear();

        std::vector<sampling::bundle_t*> _data{};
        for(auto& itr : _raw_data)
        {
            auto* _bt = itr.get<backtrace>();
            auto* _cc = itr.get<callchain>();
            auto* _ts = itr.get<backtrace_timestamp>();
            if(_thread_info && ((_bt && !_bt->empty()) || (_cc && !_cc->empty())) &&
               _ts && _thread_info->is_valid_time(_ts->get_timestamp()))
            {
                _data.emplace_back(&itr);
            }
        }

        _total_data += _data.size();
        _total_threads += (!_data.empty()) ? 1 : 0;

        if(!_data.empty())
        {
            ROCPROFSYS_VERBOSE(2 || get_debug_sampling(),
                               "Sampler data for thread %lu has %zu valid entries...\n",
                               i, _data.size());

            auto _timer_data    = post_process_timer_data(i, _init, _data);
            auto _overflow_data = post_process_overflow_data(i, _init, _data);

            if(get_use_perfetto()) post_process_perfetto(i, _timer_data, _overflow_data);
            if(get_use_timemory()) post_process_timemory(i, _timer_data, _overflow_data);
            if(get_use_rocpd()) post_process_rocpd(i, _timer_data, _overflow_data);
        }
        else
        {
            ROCPROFSYS_VERBOSE(
                2 || get_debug_sampling(),
                "Sampler data for thread %lu has zero valid entries out of "
                "%zu... (skipped)\n",
                i, _raw_data.size());
        }
    }

    ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                       "Destroying samplers and allocators...\n");

    get_offload_file().reset();  // remove the temporary file

    for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
        get_sampler(i).reset();

    for(auto& itr : get_sampler_allocators())
    {
        if(itr) itr.reset();
    }

    if(get_use_tmp_files() && get_offload_file())
    {
        get_offload_file()->remove();
        get_offload_file().reset();
    }

    ROCPROFSYS_VERBOSE(1 || get_debug_sampling(),
                       "Collected %zu samples from %zu threads... %zu samples out of %zu "
                       "were taken while within instrumented routines\n",
                       _total_data, _total_threads, _internal_samples.load(),
                       (_internal_samples + _external_samples));
}

namespace
{
std::vector<timer_sampling_data>
post_process_timer_data(int64_t _tid, const bundle_t* _init,
                        const std::vector<bundle_t*>& _data)
{
    auto _results = std::vector<timer_sampling_data>{};

    const auto* _last = _init;
    for(const auto& itr : _data)
    {
        auto*       _bt_data      = itr->get<backtrace>();
        auto*       _bt_time      = itr->get<backtrace_timestamp>();
        auto*       _bt_metrics   = itr->get<backtrace_metrics>();
        const auto* _last_metrics = _last->get<backtrace_metrics>();

        if(!_bt_data || !_bt_time || _bt_data->empty() || _bt_time->get_tid() != _tid)
            continue;

        auto _ret    = timer_sampling_data{};
        _ret.m_tid   = _bt_time->get_tid();
        _ret.m_beg   = _last->get<backtrace_timestamp>()->get_timestamp();
        _ret.m_end   = _bt_time->get_timestamp();
        _ret.m_stack = backtrace::filter_and_patch(_bt_data->get());
        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            auto _hw_counters_enabled = [](const auto* _bt_v) {
                return (_bt_v != nullptr) &&
                       (*_bt_v)(type_list<backtrace_metrics::hw_counters>{}) &&
                       (*_bt_v)(category::thread_hardware_counter{});
            };

            if(_bt_metrics && _last_metrics && _hw_counters_enabled(_bt_metrics) &&
               _hw_counters_enabled(_last_metrics))
            {
                _ret.m_metrics = (*_bt_metrics) - (*_last_metrics);
            }
        }

        _results.emplace_back(std::move(_ret));
        _last = itr;
    }

    std::sort(_results.begin(), _results.end(),
              [](const auto& _lhs, const auto& _rhs) { return _lhs.m_beg < _rhs.m_beg; });

    return _results;
}

std::vector<overflow_sampling_data>
post_process_overflow_data(int64_t                       _tid, const bundle_t*,
                           const std::vector<bundle_t*>& _data)
{
    auto _results = std::vector<overflow_sampling_data>{};

    uint64_t _last_call_ts   = 0;
    uint64_t _perf_ts_offset = 0;
    for(const auto& itr : _data)
    {
        auto* _bt_call = itr->get<callchain>();
        auto* _bt_time = itr->get<backtrace_timestamp>();

        if(!_bt_call || !_bt_time || _bt_call->empty() || _bt_time->get_tid() != _tid)
            continue;

        for(const auto& pitr : callchain::filter_and_patch(_bt_call->get()))
        {
            if(_last_call_ts == 0)
            {
                _last_call_ts   = pitr.first;
                _perf_ts_offset = (_bt_time->get_timestamp() - pitr.first);
                continue;
            }

            auto _ret     = overflow_sampling_data{};
            _ret.m_tid    = _bt_time->get_tid();
            _ret.m_beg    = _last_call_ts + _perf_ts_offset;
            _ret.m_end    = pitr.first + _perf_ts_offset;
            _ret.m_stack  = pitr.second;
            _last_call_ts = pitr.first;
            _results.emplace_back(std::move(_ret));
        }
    }

    std::sort(_results.begin(), _results.end(),
              [](const auto& _lhs, const auto& _rhs) { return _lhs.m_beg < _rhs.m_beg; });

    return _results;
}

void
post_process_perfetto(int64_t _tid, const std::vector<timer_sampling_data>& _timer_data,
                      const std::vector<overflow_sampling_data>& _overflow_data)
{
    auto _valid_metrics = backtrace_metrics::valid_array_t{};

    for(const auto& itr : _timer_data)
    {
        _valid_metrics |= itr.m_metrics.get_valid();
    }

    if(trait::runtime_enabled<backtrace_metrics>::get())
    {
        ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                           "[%li] Post-processing metrics for perfetto...\n", _tid);
        backtrace_metrics::init_perfetto(_tid, _valid_metrics);
        for(const auto& itr : _timer_data)
            itr.m_metrics.post_process_perfetto(_tid, 0.5 * (itr.m_beg + itr.m_end));
        backtrace_metrics::fini_perfetto(_tid, _valid_metrics);
    }

    ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                       "[%li] Post-processing backtraces for perfetto...\n", _tid);

    const auto& _thread_info = thread_info::get(_tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", _tid);

    if(!_thread_info) return;

    auto _overflow_event =
        get_setting_value<std::string>("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT").value_or("");

    if(!_overflow_event.empty() && !_overflow_data.empty())
    {
        auto _beg_ns = std::max(_overflow_data.front().m_beg, _thread_info->get_start());
        auto _end_ns = std::min(_overflow_data.back().m_end, _thread_info->get_stop());

        const auto _overflow_prefix = std::string_view{ "PERF_COUNT_" };
        const auto _overflow_pos    = _overflow_event.find(_overflow_prefix);
        if(_overflow_pos != std::string::npos)
            _overflow_event =
                _overflow_event.substr(_overflow_pos + _overflow_prefix.length());

        const auto* _main_name =
            static_strings.emplace(join(" ", _overflow_event, "samples [rocprof-sys]"))
                .first->c_str();

        auto _track = tracing::get_perfetto_track(
            category::overflow_sampling{},
            [](auto _seq_id, auto _sys_id) {
                return TIMEMORY_JOIN(" ", "Thread", _seq_id, "Overflow", "(S)", _sys_id);
            },
            _thread_info->index_data->sequent_value,
            _thread_info->index_data->system_value);

        tracing::push_perfetto_track(category::overflow_sampling{}, _main_name, _track,
                                     _beg_ns, [&](::perfetto::EventContext ctx) {
                                         if(config::get_perfetto_annotations())
                                         {
                                             tracing::add_perfetto_annotation(
                                                 ctx, "begin_ns", _beg_ns);
                                         }
                                     });

        for(const auto& itr : _overflow_data)
        {
            auto _beg = itr.m_beg;
            auto _end = itr.m_end;

            if(!_thread_info->is_valid_lifetime({ _beg, _end })) continue;

            for(const auto& iitr : itr.m_stack)
            {
                const auto* _name =
                    static_strings.emplace(demangle(iitr.name)).first->c_str();
                tracing::push_perfetto_track(
                    category::overflow_sampling{}, _name, _track, _beg,
                    [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "file", iitr.location);
                            tracing::add_perfetto_annotation(ctx, "pc",
                                                             as_hex(iitr.address));
                            tracing::add_perfetto_annotation(ctx, "line_address",
                                                             as_hex(iitr.line_address));
                            if(iitr.lineinfo)
                            {
                                auto _lines = iitr.lineinfo.lines;
                                std::reverse(_lines.begin(), _lines.end());
                                size_t _n = 0;
                                for(const auto& line : _lines)
                                {
                                    auto _label = JOIN('-', "lineinfo", _n++);
                                    tracing::add_perfetto_annotation(
                                        ctx, _label.c_str(),
                                        JOIN('@', demangle(line.name),
                                             JOIN(':', line.location, line.line)));
                                }
                            }
                        }
                    });
                tracing::pop_perfetto_track(category::overflow_sampling{}, _name, _track,
                                            _end);
            }
        }

        tracing::pop_perfetto_track(category::overflow_sampling{}, _main_name, _track,
                                    _end_ns, [&](::perfetto::EventContext ctx) {
                                        if(config::get_perfetto_annotations())
                                        {
                                            tracing::add_perfetto_annotation(
                                                ctx, "end_ns", _end_ns);
                                        }
                                    });
    }

    if(!_timer_data.empty())
    {
        auto _beg_ns = std::max(_timer_data.front().m_beg, _thread_info->get_start());
        auto _end_ns = std::min(_timer_data.back().m_end, _thread_info->get_stop());

        auto _track = tracing::get_perfetto_track(
            category::timer_sampling{},
            [](auto _seq_id, auto _sys_id) {
                return TIMEMORY_JOIN(" ", "Thread", _seq_id, "(S)", _sys_id);
            },
            _thread_info->index_data->sequent_value,
            _thread_info->index_data->system_value);

        tracing::push_perfetto_track(category::timer_sampling{}, "samples [rocprof-sys]",
                                     _track, _beg_ns, [&](::perfetto::EventContext ctx) {
                                         if(config::get_perfetto_annotations())
                                         {
                                             tracing::add_perfetto_annotation(
                                                 ctx, "begin_ns", _beg_ns);
                                         }
                                     });

        auto _labels = backtrace_metrics::get_hw_counter_labels(_tid);
        for(const auto& itr : _timer_data)
        {
            size_t   _ncount = 0;
            uint64_t _beg    = itr.m_beg;
            uint64_t _end    = itr.m_end;
            if(!_thread_info->is_valid_lifetime({ _beg, _end })) continue;

            for(const auto& iitr : itr.m_stack)
            {
                auto _ncur = _ncount++;
                // the begin/end + HW counters will be same for entire call-stack so only
                // annotate the top and the bottom functons to keep the data consumption
                // low
                bool _include_common = (_ncur == 0 || _ncur + 1 == itr.m_stack.size());

                // Only annotate HW counters when first or last and HW counters are not
                // empty
                bool _include_hw =
                    _include_common && !itr.m_metrics.get_hw_counters().empty();

                // annotations common to both modes
                auto _common_annotate = [&](::perfetto::EventContext& ctx,
                                            bool                      _is_last) {
                    if(_include_common && _is_last)
                    {
                        tracing::add_perfetto_annotation(ctx, "begin_ns", _beg);
                        tracing::add_perfetto_annotation(ctx, "end_ns", _end);
                    }

                    if(_include_hw)
                    {
                        // current values when read
                        auto _hw_cnt_vals = itr.m_metrics.get_hw_counters();
                        for(size_t i = 0; i < _labels.size(); ++i)
                            tracing::add_perfetto_annotation(ctx, _labels.at(i),
                                                             _hw_cnt_vals.at(i));
                    }
                };

                if(get_sampling_include_inlines() && iitr.lineinfo)
                {
                    auto _lines = iitr.lineinfo.lines;
                    std::reverse(_lines.begin(), _lines.end());
                    size_t _n = 0;
                    for(const auto& line : _lines)
                    {
                        const auto* _name =
                            static_strings.emplace(demangle(line.name)).first->c_str();
                        auto _info = JOIN(':', line.location, line.line);
                        tracing::push_perfetto_track(
                            category::timer_sampling{}, _name, _track, _beg,
                            [&](::perfetto::EventContext ctx) {
                                if(config::get_perfetto_annotations())
                                {
                                    _common_annotate(ctx, (_n == 0 && _ncur == 0) ||
                                                              (_n + 1 == _lines.size()));
                                    tracing::add_perfetto_annotation(ctx, "file",
                                                                     iitr.location);
                                    tracing::add_perfetto_annotation(ctx, "lineinfo",
                                                                     _info);
                                    tracing::add_perfetto_annotation(ctx, "inlined",
                                                                     (_n++ > 0));
                                }
                            });
                        tracing::pop_perfetto_track(category::timer_sampling{}, _name,
                                                    _track, _end);
                    }
                }
                else
                {
                    const auto* _name = static_strings.emplace(iitr.name).first->c_str();
                    tracing::push_perfetto_track(
                        category::timer_sampling{}, _name, _track, _beg,
                        [&](::perfetto::EventContext ctx) {
                            if(config::get_perfetto_annotations())
                            {
                                _common_annotate(ctx, true);
                                tracing::add_perfetto_annotation(ctx, "file",
                                                                 iitr.location);
                                tracing::add_perfetto_annotation(ctx, "pc",
                                                                 as_hex(iitr.address));
                                tracing::add_perfetto_annotation(
                                    ctx, "line_address", as_hex(iitr.line_address));
                                if(iitr.lineinfo)
                                {
                                    auto _lines = iitr.lineinfo.lines;
                                    std::reverse(_lines.begin(), _lines.end());
                                    size_t _n = 0;
                                    for(const auto& line : _lines)
                                    {
                                        auto _label = JOIN('-', "lineinfo", _n++);
                                        tracing::add_perfetto_annotation(
                                            ctx, _label.c_str(),
                                            JOIN('@', demangle(line.name),
                                                 JOIN(':', line.location, line.line)));
                                    }
                                }
                            }
                        });

                    tracing::pop_perfetto_track(category::timer_sampling{}, _name, _track,
                                                _end);
                }
            }
        }

        tracing::pop_perfetto_track(category::timer_sampling{}, "samples [rocprof-sys]",
                                    _track, _end_ns, [&](::perfetto::EventContext ctx) {
                                        if(config::get_perfetto_annotations())
                                        {
                                            tracing::add_perfetto_annotation(
                                                ctx, "end_ns", _end_ns);
                                        }
                                    });
    }
}

void
post_process_timemory(int64_t _tid, const std::vector<timer_sampling_data>& _timer_data,
                      const std::vector<overflow_sampling_data>& _overflow_data)
{
    ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                       "[%li] Post-processing data for timemory...\n", _tid);

    // compute the total number of entries
    int64_t _sum = 0;
    for(const auto& itr : _overflow_data)
        _sum += itr.m_stack.size();
    for(const auto& itr : _timer_data)
        _sum += itr.m_stack.size();

    for(const auto& itr : _overflow_data)
    {
        using bundle_t = tim::lightweight_tuple<comp::trip_count, sampling_wall_clock>;

        auto _data = std::vector<bundle_t>{};
        _data.reserve(itr.m_stack.size());

        for(const auto& iitr : itr.m_stack)
        {
            _data.emplace_back(tim::string_view_t{ iitr.name });
            _data.back().push(itr.m_tid);
            _data.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _data.size(); ++i)
        {
            auto& iitr = _data.at(_data.size() - i - 1);
            iitr.stop();
            if constexpr(tim::trait::is_available<sampling_wall_clock>::value)
            {
                auto* _sc = iitr.get<sampling_wall_clock>();
                if(_sc)
                {
                    auto _value = static_cast<double>(itr.m_end - itr.m_beg) /
                                  sampling_wall_clock::get_unit();
                    _sc->set_value(_value);
                    _sc->set_accum(_value);
                }
            }
            iitr.pop();
        }
    }

    for(const auto& itr : _timer_data)
    {
        using bundle_t = tim::lightweight_tuple<comp::trip_count, sampling_wall_clock,
                                                sampling_cpu_clock, hw_counters>;

        double _elapsed_wc = (itr.m_end - itr.m_beg);

        auto _data = std::vector<bundle_t>{};
        _data.reserve(itr.m_stack.size());

        // generate the instances of the tuple of components and start them
        for(const auto& iitr : itr.m_stack)
        {
            _data.emplace_back(tim::string_view_t{ iitr.name });
            _data.back().push(itr.m_tid);
            _data.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _data.size(); ++i)
        {
            auto& iitr = _data.at(_data.size() - i - 1);
            iitr.stop();

            if constexpr(tim::trait::is_available<sampling_wall_clock>::value)
            {
                auto* _sc = iitr.get<sampling_wall_clock>();
                if(_sc)
                {
                    auto _value = _elapsed_wc / sampling_wall_clock::get_unit();
                    _sc->set_value(_value);
                    _sc->set_accum(_value);
                }
            }

            const auto& _metrics = itr.m_metrics;
            if constexpr(tim::trait::is_available<sampling_cpu_clock>::value)
            {
                auto* _cc = iitr.get<sampling_cpu_clock>();

                if(_cc && _metrics && _metrics(category::thread_cpu_time{}))
                {
                    double _elapsed_cc = _metrics.get_cpu_timestamp();

                    _cc->set_value(_elapsed_cc / sampling_cpu_clock::get_unit());
                    _cc->set_accum(_elapsed_cc / sampling_cpu_clock::get_unit());
                }
            }

            if constexpr(tim::trait::is_available<hw_counters>::value)
            {
                auto* _hw_counter = iitr.get<hw_counters>();

                if(_hw_counter && _metrics &&
                   _metrics(type_list<backtrace_metrics::hw_counters>{}) &&
                   _metrics(category::thread_hardware_counter{}))
                {
                    _hw_counter->set_value(_metrics.get_hw_counters());
                    _hw_counter->set_accum(_metrics.get_hw_counters());
                }
            }

            iitr.pop();
        }
    }

    for(auto&& itr : _overflow_data)
    {
        using bundle_t =
            tim::lightweight_tuple<sampling_percent, quirk::config<quirk::flat_scope>>;

        auto _data = std::vector<bundle_t>{};
        _data.reserve(itr.m_stack.size());

        // generate the instances of the tuple of components and start them
        for(const auto& iitr : itr.m_stack)
        {
            _data.emplace_back(tim::string_view_t{ iitr.name });
            _data.back().push(itr.m_tid);
            _data.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _data.size(); ++i)
        {
            auto&  iitr   = _data.at(_data.size() - i - 1);
            double _value = (1.0 / _sum) * 100.0;
            iitr.store(std::plus<double>{}, _value);
            iitr.stop();
            iitr.pop();
        }
    }

    for(auto&& itr : _timer_data)
    {
        using bundle_t =
            tim::lightweight_tuple<sampling_percent, quirk::config<quirk::flat_scope>>;

        auto _data = std::vector<bundle_t>{};
        _data.reserve(itr.m_stack.size());

        // generate the instances of the tuple of components and start them
        for(const auto& iitr : itr.m_stack)
        {
            _data.emplace_back(tim::string_view_t{ iitr.name });
            _data.back().push(itr.m_tid);
            _data.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _data.size(); ++i)
        {
            auto&  iitr   = _data.at(_data.size() - i - 1);
            double _value = (1.0 / _sum) * 100.0;
            iitr.store(std::plus<double>{}, _value);
            iitr.stop();
            iitr.pop();
        }
    }
}

void
rocpd_post_process_overflow_data(
    int64_t _tid, const std::vector<overflow_sampling_data>& _overflow_data)
{
    auto& data_processor = get_data_processor();

    const auto& _thread_info = thread_info::get(_tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    auto _overflow_event =
        get_setting_value<std::string>("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT").value_or("");

    if(!_overflow_event.empty() && !_overflow_data.empty())
    {
        auto _beg_ns = std::max(_overflow_data.front().m_beg, _thread_info->get_start());
        auto _end_ns = std::min(_overflow_data.back().m_end, _thread_info->get_stop());

        const auto _overflow_prefix = std::string_view{ "PERF_COUNT_" };
        const auto _overflow_pos    = _overflow_event.find(_overflow_prefix);
        if(_overflow_pos != std::string::npos)
            _overflow_event =
                _overflow_event.substr(_overflow_pos + _overflow_prefix.length());

        const auto* _main_name =
            static_strings.emplace(join(" ", _overflow_event, "samples [rocprof-sys]"))
                .first->c_str();
        auto main_name_id = data_processor.insert_string(_main_name);

        const auto& _thread_info = thread_info::get(_tid, SequentTID);
        ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", _tid);
        if(!_thread_info) return;
        size_t thread_id = _thread_info->index_data->system_value;

        auto thread_primary_key = data_processor.map_thread_id_to_primary_key(thread_id);
        const auto _track_name  = get_track_name(*_thread_info);

        rocpd_insert_region<category::overflow_sampling>(
            thread_primary_key, _beg_ns, _end_ns, main_name_id, _track_name.c_str());

        for(const auto& itr : _overflow_data)
        {
            auto _beg = itr.m_beg;
            auto _end = itr.m_end;

            if(!_thread_info->is_valid_lifetime({ _beg, _end })) continue;

            for(const auto& iitr : itr.m_stack)
            {
                const auto* _name =
                    static_strings.emplace(demangle(iitr.name)).first->c_str();
                auto name_id = data_processor.insert_string(_name);
                rocpd_insert_region<category::overflow_sampling>(
                    thread_primary_key, _beg, _end, name_id, _track_name.c_str(),
                    generate_call_stack_json(iitr).c_str(),
                    generate_line_info_json(iitr).c_str());
            }
        }
    }
}

void
rocpd_post_process_backtrace_metrics(int64_t                                 _tid,
                                     const std::vector<timer_sampling_data>& _timer_data)
{
    auto _valid_metrics = backtrace_metrics::valid_array_t{};

    for(const auto& itr : _timer_data)
    {
        _valid_metrics |= itr.m_metrics.get_valid();
    }

    if(trait::runtime_enabled<backtrace_metrics>::get() && get_use_rocpd())
    {
        ROCPROFSYS_VERBOSE(3 || get_debug_sampling(),
                           "[%li] Post-processing metrics for rocpd...\n", _tid);
        backtrace_metrics::init_rocpd(_tid, _valid_metrics);  // move to setup
        for(const auto& itr : _timer_data)
            itr.m_metrics.post_process_rocpd(_tid, 0.5 * (itr.m_beg + itr.m_end));
        backtrace_metrics::fini_rocpd(_tid, _valid_metrics);
    }
}

void
rocpd_post_process_timer_data(int64_t                                 _tid,
                              const std::vector<timer_sampling_data>& _timer_data)
{
    auto& data_processor = get_data_processor();

    const auto& _thread_info = thread_info::get(_tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    if(!_timer_data.empty())
    {
        rocpd_post_process_backtrace_metrics(_tid, _timer_data);

        auto _beg_ns = std::max(_timer_data.front().m_beg, _thread_info->get_start());
        auto _end_ns = std::min(_timer_data.back().m_end, _thread_info->get_stop());

        const auto _track_name = get_track_name(*_thread_info);

        auto thread_primary_key = data_processor.map_thread_id_to_primary_key(
            _thread_info->index_data->system_value);

        const auto main_name_id = data_processor.insert_string("samples [rocprof-sys]");
        rocpd_insert_region<category::timer_sampling>(
            thread_primary_key, _beg_ns, _end_ns, main_name_id, _track_name.c_str());

        auto _labels = backtrace_metrics::get_hw_counter_labels(_tid);
        for(const auto& itr : _timer_data)
        {
            size_t   _ncount = 0;
            uint64_t _beg    = itr.m_beg;
            uint64_t _end    = itr.m_end;
            if(!_thread_info->is_valid_lifetime({ _beg, _end })) continue;

            for(const auto& iitr : itr.m_stack)
            {
                auto _ncur = _ncount++;
                // the begin/end + HW counters will be same for entire call-stack so only
                // annotate the top and the bottom functons to keep the data consumption
                // low
                bool _include_common = (_ncur == 0 || _ncur + 1 == itr.m_stack.size());

                // Only annotate HW counters when first or last and HW counters are not
                // empty
                bool _include_hw =
                    _include_common && !itr.m_metrics.get_hw_counters().empty();

                std::string hw_counter_json = "{}";
                if(_include_hw)
                {
                    // current values when read
                    hw_counter_json = generate_hw_counter_json(_tid, itr.m_metrics);
                }

                if(get_sampling_include_inlines() && iitr.lineinfo)
                {
                    auto _lines = iitr.lineinfo.lines;
                    std::reverse(_lines.begin(), _lines.end());
                    size_t _n = 0;
                    for(const auto& line : _lines)
                    {
                        const auto* _name =
                            static_strings.emplace(demangle(line.name)).first->c_str();
                        auto inlined_name_id = data_processor.insert_string(_name);

                        auto inlined_call_stack = ::rocpd::json::create();
                        inlined_call_stack->set("name", std::string(demangle(line.name)));
                        inlined_call_stack->set("location", std::string(line.location));
                        inlined_call_stack->set("line", std::to_string(line.line));
                        inlined_call_stack->set("inlined", "true");

                        rocpd_insert_region<category::timer_sampling>(
                            thread_primary_key, _beg, _end, inlined_name_id,
                            _track_name.c_str(), inlined_call_stack->to_string().c_str(),
                            "{}",
                            // Only include HW counters for first inlined function
                            (_n == 0) ? hw_counter_json.c_str() : "{}");
                    }
                }
                else
                {
                    const auto* _name = static_strings.emplace(iitr.name).first->c_str();
                    const auto  name_id = data_processor.insert_string(_name);
                    rocpd_insert_region<category::timer_sampling>(
                        thread_primary_key, _beg, _end, name_id, _track_name.c_str(),
                        generate_call_stack_json(iitr).c_str(),
                        generate_line_info_json(iitr).c_str(), hw_counter_json.c_str());
                }
            }
        }
    }
}

void
post_process_rocpd(int64_t _tid, const std::vector<timer_sampling_data>& _timer_data,
                   const std::vector<overflow_sampling_data>& _overflow_data)
{
    rocpd_post_process_overflow_data(_tid, _overflow_data);
    rocpd_post_process_timer_data(_tid, _timer_data);
}

struct sampling_initialization
{
    static void preinit()
    {
        sampling_wall_clock::label()       = "sampling_wall_clock";
        sampling_wall_clock::description() = "Wall clock time (via sampling)";

        sampling_cpu_clock::label()       = "sampling_cpu_clock";
        sampling_cpu_clock::description() = "CPU clock time (via sampling)";

        sampling_percent::label()       = "sampling_percent";
        sampling_percent::description() = "Percentage of samples";
        sampling_percent::set_precision(3);

        sampling_gpu_busy_gfx::label()       = "sampling_gpu_busy_gfx_percent";
        sampling_gpu_busy_gfx::description() = "Utilization of GFX engines on GPU(s)";
        sampling_gpu_busy_gfx::set_precision(0);
        sampling_gpu_busy_gfx::set_format_flags(
            sampling_gpu_busy_gfx::get_format_flags() & std::ios_base::showpoint);

        sampling_gpu_busy_umc::label() = "sampling_gpu_busy_umc_percent";
        sampling_gpu_busy_umc::description() =
            "Utilization of memory controller on GPU(s)";
        sampling_gpu_busy_umc::set_precision(0);
        sampling_gpu_busy_umc::set_format_flags(
            sampling_gpu_busy_umc::get_format_flags() & std::ios_base::showpoint);

        sampling_gpu_busy_mm::label() = "sampling_gpu_busy_mm_percent";
        sampling_gpu_busy_mm::description() =
            "Utilization of multimedia engines on GPU(s)";
        sampling_gpu_busy_mm::set_precision(0);
        sampling_gpu_busy_mm::set_format_flags(sampling_gpu_busy_mm::get_format_flags() &
                                               std::ios_base::showpoint);

        sampling_gpu_memory::label()       = "sampling_gpu_memory_usage";
        sampling_gpu_memory::description() = "Memory usage of GPU(s)";

        sampling_gpu_power::label()        = "sampling_gpu_power";
        sampling_gpu_power::description()  = "Power usage of GPU(s)";
        sampling_gpu_power::unit()         = units::watt;
        sampling_gpu_power::display_unit() = "watts";
        sampling_gpu_power::set_precision(2);
        sampling_gpu_power::set_format_flags(sampling_gpu_power::get_format_flags());

        sampling_gpu_temp::label()        = "sampling_gpu_temperature";
        sampling_gpu_temp::description()  = "Temperature of GPU(s)";
        sampling_gpu_temp::unit()         = 1;
        sampling_gpu_temp::display_unit() = "degC";
        sampling_gpu_temp::set_precision(1);
        sampling_gpu_temp::set_format_flags(sampling_gpu_temp::get_format_flags());

        sampling_gpu_vcn::label()       = "sampling_gpu_vcn_percent";
        sampling_gpu_vcn::description() = "VCN instance(s) activity";
        sampling_gpu_vcn::set_precision(0);
        sampling_gpu_vcn::set_format_flags(sampling_gpu_vcn::get_format_flags() &
                                           std::ios_base::showpoint);

        sampling_gpu_jpeg::label()       = "sampling_gpu_jpeg_percent";
        sampling_gpu_jpeg::description() = "JPEG instance(s) activity";
        sampling_gpu_jpeg::set_precision(0);
        sampling_gpu_jpeg::set_format_flags(sampling_gpu_jpeg::get_format_flags() &
                                            std::ios_base::showpoint);
    }
};
}  // namespace
}  // namespace sampling
}  // namespace rocprofsys

TIMEMORY_INVOKE_PREINIT(rocprofsys::sampling::sampling_initialization)
