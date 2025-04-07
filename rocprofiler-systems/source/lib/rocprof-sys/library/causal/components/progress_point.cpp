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

#include "library/causal/components/progress_point.hpp"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/debug.hpp"
#include "core/timemory.hpp"
#include "library/causal/experiment.hpp"
#include "library/thread_data.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>

namespace rocprofsys
{
namespace causal
{
namespace component
{
namespace
{
using progress_allocator_t = tim::data::ring_buffer_allocator<progress_point>;
using progress_map_t       = std::unordered_map<tim::hash_value_t, progress_point*>;

auto&
get_progress_map()
{
    using thread_data_t = thread_data<identity<progress_map_t>>;
    static auto& _v     = thread_data_t::instance(construct_on_init{});
    return _v;
}

progress_map_t&
get_progress_map(int64_t _tid)
{
    return get_progress_map()->at(_tid);
}

auto&
get_progress_allocator(int64_t _tid)
{
    return thread_data<progress_allocator_t>::instance(construct_on_thread{ _tid });
}
}  // namespace

std::unordered_map<tim::hash_value_t, progress_point>
progress_point::get_progress_points()
{
    auto _data = std::unordered_map<tim::hash_value_t, progress_point>{};
    if(!get_progress_map()) return _data;
    for(const auto& titr : *get_progress_map())
    {
        for(const auto& itr : titr)
        {
            if(itr.second)
            {
                auto& ditr = _data[itr.first];
                ditr += *itr.second;
                ditr.set_hash(itr.second->get_hash());
                itr.second->set_value(0);
            }
        }
    }
    return _data;
}

std::string
progress_point::label()
{
    return "progress_point";
}

std::string
progress_point::description()
{
    return "Tracks progress point latency and throughput for causal profiling";
}

void
progress_point::start()
{
    ++m_arrival;
}

void
progress_point::stop()
{
    ++m_departure;
}

void
progress_point::mark()
{
    ++m_delta;
}

void
progress_point::set_value(int64_t _v)
{
    m_delta     = _v;
    m_arrival   = _v;
    m_departure = _v;
}

progress_point&
progress_point::operator+=(const progress_point& _v)
{
    if(this != &_v)
    {
        m_delta += _v.m_delta;
        m_arrival += _v.m_arrival;
        m_departure += _v.m_departure;
    }
    return *this;
}

progress_point&
progress_point::operator-=(const progress_point& _v)
{
    if(this != &_v)
    {
        m_delta -= _v.m_delta;
        m_arrival -= _v.m_arrival;
        m_departure -= _v.m_departure;
    }
    return *this;
}

bool
progress_point::is_throughput_point() const
{
    return (m_delta != 0);
}

bool
progress_point::is_latency_point() const
{
    return (m_arrival != 0 || m_departure != 0);
}

int64_t
progress_point::get_delta() const
{
    return m_delta;
}

int64_t
progress_point::get_arrival() const
{
    if(!is_latency_point()) return m_arrival;
    // when it is a latency point, we want the difference to be greater than zero
    return (m_arrival >= m_departure) ? (m_arrival + 1) : m_arrival;
}

int64_t
progress_point::get_departure() const
{
    // if(!is_latency_point()) return m_departure;
    // return (m_departure <= m_arrival) ? m_departure : (m_departure + 1);
    return m_departure;
}

int64_t
progress_point::get_latency_delta() const
{
    return (get_arrival() - get_departure());
}

int64_t
progress_point::get_laps() const
{
    return std::max(get_delta(), get_latency_delta());
}

void
progress_point::print(std::ostream& os) const
{
    os << tim::get_hash_identifier(m_hash) << " :: ";
    tim::operation::base_printer<progress_point>(os, *this);
}
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys

namespace tim
{
namespace operation
{
namespace causal = rocprofsys::causal;

void
push_node<causal::component::progress_point>::operator()(type&        _obj, scope::config,
                                                         hash_value_t _hash,
                                                         int64_t      _tid) const
{
    auto itr = causal::component::get_progress_map(_tid).emplace(_hash, nullptr);
    if(itr.second && !itr.first->second)
    {
        auto& _alloc = causal::component::get_progress_allocator(_tid);
        auto* _val   = _alloc->allocate(1);
        _alloc->construct(_val);
        _val->set_hash(_hash);
        itr.first->second = _val;
    }
    _obj.set_hash(_hash);
    _obj.set_iterator(itr.first->second);
}

void
pop_node<causal::component::progress_point>::operator()(type& _obj, int64_t) const
{
    auto* itr = _obj.get_iterator();
    if(itr && !(_obj.get_is_invalid() || _obj.get_is_running()))
    {
        *itr += _obj;
    }
}
}  // namespace operation
}  // namespace tim
