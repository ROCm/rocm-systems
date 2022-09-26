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

#include "library/thread_info.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>

namespace omnitrace
{
namespace
{
using thread_index_data_t =
    thread_data<std::optional<thread_index_data>, project::omnitrace>;
using thread_info_data_t = thread_data<std::optional<thread_info>, project::omnitrace>;

auto&
get_index_data(int64_t _tid)
{
    static auto& _v = thread_index_data_t::instances();
    return _v.at(_tid);
}

auto
init_index_data(int64_t _tid, bool _offset = false)
{
    auto& itr = get_index_data(_tid);
    if(!itr)
    {
        threading::offset_this_id(_offset);
        itr = thread_index_data{};
        if(!config::settings_are_configured())
        {
            OMNITRACE_BASIC_VERBOSE_F(
                2, "Thread %li on PID %i (rank: %i) assigned omnitrace TID %li\n",
                itr->system_value, process::get_id(), dmp::rank(), itr->sequent_value);
        }
        else
        {
            OMNITRACE_VERBOSE_F(
                2, "Thread %li on PID %i (rank: %i) assigned omnitrace TID %li\n",
                itr->system_value, process::get_id(), dmp::rank(), itr->sequent_value);
        }
    }
    return itr;
}

const auto unknown_thread = std::optional<thread_info>{};
}  // namespace

const std::optional<thread_info>&
thread_info::init(bool _offset)
{
    static thread_local bool _once      = false;
    auto&                    _instances = thread_info_data_t::instances();
    auto                     _tid       = utility::get_thread_index();

    if(!_once && (_once = true))
    {
        threading::offset_this_id(_offset);
        auto& _info           = _instances.at(_tid);
        _info                 = thread_info{};
        _info->is_offset      = threading::offset_this_id();
        _info->index_data     = init_index_data(_tid, _info->is_offset);
        _info->lifetime.first = tim::get_clock_real_now<uint64_t, std::nano>();
        if(_info->is_offset) set_thread_state(ThreadState::Disabled);
    }

    return _instances.at(_tid);
}

const std::optional<thread_info>&
thread_info::get()
{
    return thread_info_data_t::instances().at(utility::get_thread_index());
}

const std::optional<thread_info>&
thread_info::get(int64_t _tid, ThreadIdType _type)
{
    if(_type == ThreadIdType::InternalTID)
        return thread_info_data_t::instances().at(_tid);
    else if(_type == ThreadIdType::SystemTID)
    {
        const auto& _v = thread_info_data_t::instances();
        for(const auto& itr : _v)
        {
            if(itr && itr->index_data->system_value == _tid) return itr;
        }
    }
    else if(_type == ThreadIdType::SequentTID)
    {
        const auto& _v = thread_info_data_t::instances();
        for(const auto& itr : _v)
        {
            if(itr && itr->index_data->sequent_value == _tid) return itr;
        }
    }

    OMNITRACE_CI_THROW(unknown_thread, "Unknown thread has been assigned a value");
    return unknown_thread;
}

void
thread_info::set_start(uint64_t _ts, bool _force)
{
    auto& _v = thread_info_data_t::instances().at(utility::get_thread_index());
    if(!_v) init();
    if(_force || (_ts > 0 && (_v->lifetime.first == 0 || _ts < _v->lifetime.first)))
        _v->lifetime.first = _ts;
}

void
thread_info::set_stop(uint64_t _ts)
{
    auto  _tid = utility::get_thread_index();
    auto& _v   = thread_info_data_t::instances().at(_tid);
    if(_v)
    {
        _v->lifetime.second = _ts;
        // if the main thread, make sure all child threads have a end lifetime
        // less than or equal to the main thread end lifetime
        if(_tid == 0)
        {
            for(auto& itr : thread_info_data_t::instances())
            {
                if(itr && itr->index_data && itr->index_data->internal_value > _tid)
                {
                    if(itr->lifetime.second > _v->lifetime.second)
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
        _ss << ", index_data=(" << index_data->internal_value << ", "
            << index_data->system_value << ", " << index_data->sequent_value << ")";
    _ss << ", lifetime=(" << lifetime.first << ":" << lifetime.second << ")";
    return _ss.str();
}
}  // namespace omnitrace
