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

#include "library/thread_info.hpp"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/delay.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/process/threading.hpp>

#include <cstdint>

namespace rocprofsys
{
namespace
{
auto&
get_info_data()
{
    using thread_data_t = thread_data<std::optional<thread_info>, project::rocprofsys>;
    static auto& _v     = thread_data_t::instance(construct_on_init{});
    return _v;
}

auto&
get_index_data()
{
    using thread_data_t =
        thread_data<std::optional<thread_index_data>, project::rocprofsys>;
    static auto& _v = thread_data_t::instance(construct_on_init{});
    return _v;
}

auto&
get_info_data(int64_t _tid)
{
    return get_info_data()->at(_tid);
}

auto&
get_index_data(int64_t _tid)
{
    return get_index_data()->at(_tid);
}

auto
init_index_data(int64_t _tid, bool _offset = false)
{
    auto& itr = get_index_data(_tid);
    if(!itr)
    {
        threading::offset_this_id(_offset);
        itr = thread_index_data{};

        ROCPROFSYS_CONDITIONAL_THROW(itr->internal_value != _tid,
                                     "Error! thread_info::init_index_data was called for "
                                     "thread %zi on thread %zi\n",
                                     _tid, itr->internal_value);

        int _verb = 2;
        // if thread created using finalization, bump up the minimum verbosity level
        if(get_state() >= State::Finalized && _offset) _verb += 2;
        if(!config::settings_are_configured())
        {
            ROCPROFSYS_BASIC_VERBOSE_F(_verb,
                                       "Thread %li on PID %i (rank: %i) assigned "
                                       "rocprof-sys TID %li (internal: %li)\n",
                                       itr->system_value, process::get_id(), dmp::rank(),
                                       itr->sequent_value, itr->internal_value);
        }
        else
        {
            ROCPROFSYS_VERBOSE_F(
                _verb,
                "Thread %li on PID %i (rank: %i) assigned rocprof-sys TID "
                "%li (internal: %li)\n",
                itr->system_value, process::get_id(), dmp::rank(), itr->sequent_value,
                itr->internal_value);
        }
    }
    return itr;
}

thread_local int64_t offset_causal_count = 0;
const auto           unknown_thread      = std::optional<thread_info>{};
int64_t              peak_num_threads    = max_supported_threads;
}  // namespace

std::string
thread_index_data::as_string() const
{
    auto _ss = std::stringstream{};
    _ss << sequent_value << " [" << as_hex(system_value) << "] (#" << internal_value
        << ")";
    return _ss.str();
}

int64_t
grow_data(int64_t _tid)
{
    struct data_growth
    {};

    if(_tid >= peak_num_threads)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto_lock_t _lk{ type_mutex<data_growth>() };

        // check again after locking
        if(_tid >= peak_num_threads)
        {
            TIMEMORY_PRINTF_WARNING(
                stderr, "[%li] Growing thread data from %li to %li...\n", _tid,
                peak_num_threads, peak_num_threads + max_supported_threads);
            fflush(stderr);

            for(auto itr : grow_functors())
            {
                if(itr)
                {
                    int64_t _new_capacity = (*itr)(_tid + 1);
                    TIMEMORY_PRINTF_WARNING(stderr,
                                            "[%li] Grew thread data from %li to %li...\n",
                                            _tid, peak_num_threads, _new_capacity);
                }
            }
            peak_num_threads += max_supported_threads;
        }
    }

    return peak_num_threads;
}

bool
thread_info::exists()
{
    return (get_info_data() != nullptr);
}

size_t
thread_info::get_peak_num_threads()
{
    return peak_num_threads;
}

const std::optional<thread_info>&
thread_info::init(bool _offset)
{
    static thread_local bool _once      = false;
    auto&                    _info_data = get_info_data();
    auto                     _tid       = utility::get_thread_index();

    if(!_info_data)
    {
        static auto _dummy = std::optional<thread_info>{};
        return (_dummy.reset(), _dummy);  // always reset for safety
    }

    if(!_once && (_once = true))
    {
        grow_data(_tid);
        threading::offset_this_id(_offset);
        auto& _info           = _info_data->at(_tid);
        _info                 = thread_info{};
        _info->is_offset      = threading::offset_this_id();
        _info->index_data     = init_index_data(_tid, _info->is_offset);
        _info->lifetime.first = tim::get_clock_real_now<uint64_t, std::nano>();

        const auto _sequent_tid = _info->index_data->sequent_value;
        _info->causal_count     = (!_info->is_offset && _sequent_tid < peak_num_threads)
                                      ? &causal::delay::get_local(_sequent_tid)
                                      : &offset_causal_count;

        if(_info->is_offset) set_thread_state(ThreadState::Disabled);
    }

    return _info_data->at(_tid);
}

const std::optional<thread_info>&
thread_info::get()
{
    if(!exists())
    {
        static thread_local auto _v = std::optional<thread_info>{};
        return _v;
    }
    return get_info_data(utility::get_thread_index());
}

const std::optional<thread_info>&
thread_info::get(native_handle_t& _tid)
{
    return get(native_handle_t{ _tid });
}

const std::optional<thread_info>&
thread_info::get(native_handle_t&& _tid)
{
    const auto& _v = get_info_data();
    if(_v)
    {
        for(const auto& itr : *_v)
        {
            if(itr && itr->index_data &&
               pthread_equal(itr->index_data->pthread_value, _tid) == 0)
                return itr;
        }
    }

    ROCPROFSYS_CI_THROW(unknown_thread, "Unknown thread has been assigned a value");
    return unknown_thread;
}

const std::optional<thread_info>&
thread_info::get(std::thread::id _tid)
{
    const auto& _v = get_info_data();
    if(_v)
    {
        for(const auto& itr : *_v)
        {
            if(itr && itr->index_data && itr->index_data->stl_value == _tid) return itr;
        }
    }

    ROCPROFSYS_CI_THROW(unknown_thread, "Unknown thread has been assigned a value");
    return unknown_thread;
}

const std::optional<thread_info>&
thread_info::get(int64_t _tid, ThreadIdType _type)
{
    if(_type == ThreadIdType::InternalTID)
        return get_info_data(_tid);
    else if(_type == ThreadIdType::SystemTID)
    {
        const auto& _v = get_info_data();
        if(_v)
        {
            for(const auto& itr : *_v)
            {
                if(itr && itr->index_data && itr->index_data->system_value == _tid)
                    return itr;
            }
        }
    }
    else if(_type == ThreadIdType::SequentTID)
    {
        const auto& _v = get_info_data();
        if(_v)
        {
            for(const auto& itr : *_v)
            {
                if(itr && itr->index_data && itr->index_data->sequent_value == _tid)
                    return itr;
            }
        }
    }
    else if(_type == ThreadIdType::PthreadID)
    {
        ROCPROFSYS_THROW("rocprof-sys does not support thread_info::get(int64_t, "
                         "ThreadIdType) with ThreadIdType::PthreadID\n");
    }
    else if(_type == ThreadIdType::StlThreadID)
    {
        ROCPROFSYS_THROW("rocprof-sys does not support thread_info::get(int64_t, "
                         "ThreadIdType) with ThreadIdType::StlThreadID\n");
    }

    ROCPROFSYS_CI_THROW(unknown_thread, "Unknown thread has been assigned a value");
    return unknown_thread;
}

void
thread_info::set_start(uint64_t _ts, bool _force)
{
    auto& _v = get_info_data(utility::get_thread_index());
    if(!_v) init();
    if(_force || (_ts > 0 && (_v->lifetime.first == 0 || _ts < _v->lifetime.first)))
        _v->lifetime.first = _ts;
}

void
thread_info::set_stop(uint64_t _ts)
{
    auto  _tid = utility::get_thread_index();
    auto& _v   = get_info_data(_tid);
    if(_v)
    {
        _v->lifetime.second = _ts;
        // if the main thread, make sure all child threads have a end lifetime
        // less than or equal to the main thread end lifetime
        if(_tid == 0)
        {
            for(auto& itr : *get_info_data())
            {
                if(itr && itr->index_data && itr->index_data->internal_value != _tid)
                {
                    if(itr->lifetime.second > _v->lifetime.second)
                        itr->lifetime.second = _v->lifetime.second;
                    else if(itr->lifetime.second == 0)
                        itr->lifetime.second = _v->lifetime.second;
                }
            }
        }
    }
}

uint64_t
thread_info::get_start() const
{
    return lifetime.first;
}

uint64_t
thread_info::get_stop() const
{
    return lifetime.second;
}

bool
thread_info::is_valid_time(uint64_t _ts) const
{
    return (_ts >= lifetime.first && _ts <= lifetime.second);
}

bool
thread_info::is_valid_lifetime(uint64_t _beg, uint64_t _end) const
{
    return (is_valid_time(_beg) && is_valid_time(_end));
}

bool
thread_info::is_valid_lifetime(lifetime_data_t _v) const
{
    return (is_valid_time(_v.first) && is_valid_time(_v.second));
}

thread_info::lifetime_data_t
thread_info::get_valid_lifetime(lifetime_data_t _v) const
{
    if(!is_valid_time(_v.first)) _v.first = lifetime.first;
    if(!is_valid_time(_v.second)) _v.second = lifetime.second;
    return _v;
}

std::string
thread_info::as_string() const
{
    std::stringstream _ss{};
    _ss << std::boolalpha << "is_offset=" << is_offset;
    if(index_data)
    {
        _ss << ", index_data=(" << index_data->internal_value << ", "
            << index_data->system_value << ", " << index_data->sequent_value << ", "
            << index_data->pthread_value << ", " << index_data->stl_value << ")";
    }
    if(causal_count) _ss << ", causal count=" << *causal_count;
    _ss << ", lifetime=(" << lifetime.first << ":" << lifetime.second << ")";
    return _ss.str();
}
}  // namespace rocprofsys
