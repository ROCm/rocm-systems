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

#include "constraint.hpp"
#include "config.hpp"
#include "debug.hpp"
#include "state.hpp"
#include "utility.hpp"

#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>

#include <chrono>
#include <cstdint>
#include <ratio>
#include <string>
#include <thread>
#include <type_traits>

namespace rocprofsys
{
namespace constraint
{
namespace
{
namespace units = ::tim::units;

using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::nano>;

#define ROCPROFSYS_CLOCK_IDENTIFIER(VAL)                                                 \
    clock_identifier { #VAL, VAL }

auto
clock_name(std::string _v)
{
    constexpr auto _clock_prefix = std::string_view{ "clock_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    auto _pos = _v.find(_clock_prefix);
    if(_pos == 0) _v = _v.substr(_pos + _clock_prefix.length());
    if(_v == "process_cputime_id") _v = "cputime";
    return _v;
}

auto accepted_clock_ids =
    std::set<clock_identifier>{ ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_PROCESS_CPUTIME_ID),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_RAW),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME_COARSE),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_COARSE),
                                ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_BOOTTIME) };

template <typename Tp>
clock_identifier
find_clock_identifier(const Tp& _v)
{
    const char* _descript = "";
    if constexpr(std::is_integral<Tp>::value)
    {
        _descript = "value";
        for(const auto& itr : accepted_clock_ids)
        {
            if(itr.value == _v)
            {
                return itr;
            }
        }
    }
    else
    {
        _descript        = "name";
        auto _clock_name = clock_name(_v);
        for(const auto& itr : accepted_clock_ids)
        {
            if(itr.name == _clock_name || itr.raw_name == _v ||
               std::to_string(itr.value) == _v)
            {
                return itr;
            }
        }
    }

    ROCPROFSYS_THROW("Unknown clock id %s: %s. Valid choices: %s\n", _descript,
                     timemory::join::join("", _v).c_str(),
                     timemory::join::join("", accepted_clock_ids).c_str());
}

void
sleep(uint64_t _n)
{
    std::this_thread::sleep_for(std::chrono::nanoseconds{ _n });
}

timespec
get_timespec(clockid_t clock_id) noexcept
{
    struct timespec _ts;
    clock_gettime(clock_id, &_ts);
    return _ts;
}

template <typename Tp = uint64_t, typename Precision = std::nano>
Tp
get_clock_now(clockid_t clock_id) noexcept
{
    constexpr Tp factor = (Precision::den == std::nano::den)
                              ? 1
                              : (Precision::den / static_cast<Tp>(std::nano::den));
    auto         _ts    = get_timespec(clock_id);
    return (_ts.tv_sec * std::nano::den + _ts.tv_nsec) * factor;
}
}  // namespace

//--------------------------------------------------------------------------------------//
//
//  stages implementation
//
//--------------------------------------------------------------------------------------//

stages::stages()
: init{ [](const spec&) { return get_state() < State::Finalized; } }
, wait{ [](const spec& _spec) {
    sleep(std::min<uint64_t>(100 * units::msec, _spec.delay * units::sec));
    return get_state() < State::Finalized;
} }
, start{ [](const spec&) { return get_state() < State::Finalized; } }
, collect{ [](const spec& _spec) {
    sleep(std::min<uint64_t>(100 * units::msec, _spec.duration * units::sec));
    return get_state() < State::Finalized;
} }
, stop{ [](const spec&) { return get_state() < State::Finalized; } }
{}

//--------------------------------------------------------------------------------------//
//
//  clock identifier implementation
//
//--------------------------------------------------------------------------------------//

clock_identifier::clock_identifier(std::string_view _name, int _val)
: value{ _val }
, raw_name{ _name }
, name{ clock_name(std::string{ _name }) }
{}

bool
clock_identifier::operator<(const clock_identifier& _rhs) const
{
    return value < _rhs.value;
}

bool
clock_identifier::operator==(const clock_identifier& _rhs) const
{
    return std::tie(raw_name, value) == std::tie(_rhs.raw_name, _rhs.value);
}

bool
clock_identifier::operator==(int _rhs) const
{
    return (value == _rhs);
}

bool
clock_identifier::operator==(std::string _rhs) const
{
    return (raw_name == std::string_view{ _rhs }) ||
           (name == clock_name(std::move(_rhs)));
}

std::string
clock_identifier::as_string() const
{
    auto _name = name;
    for(auto& itr : _name)
        itr = tolower(itr);
    auto _ss = std::stringstream{};
    _ss << _name << "(id=" << raw_name << ", value=" << value << ")";
    return _ss.str();
}

//--------------------------------------------------------------------------------------//
//
//  spec implementation
//
//--------------------------------------------------------------------------------------//

spec::spec(clock_identifier _id, double _delay, double _dur, uint64_t _n, uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
, clock_id{ std::move(_id) }
{}

spec::spec(int _clock_id, double _delay, double _dur, uint64_t _n, uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
, clock_id{ find_clock_identifier(_clock_id) }
{}

spec::spec(const std::string& _clock_id, double _delay, double _dur, uint64_t _n,
           uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
, clock_id{ find_clock_identifier(_clock_id) }
{}

spec::spec(const std::string& _line)
: spec{ config::get_setting_value<std::string>("ROCPROFSYS_TRACE_PERIOD_CLOCK_ID")
            .value_or("CLOCK_REALTIME"),
        config::get_setting_value<double>("ROCPROFSYS_TRACE_DELAY").value_or(0.0),
        config::get_setting_value<double>("ROCPROFSYS_TRACE_DURATION").value_or(0.0) }
{
    auto _delim = tim::delimit(_line, ":");
    if(!_delim.empty()) delay = utility::convert<double>(_delim.at(0));
    if(_delim.size() > 1) duration = utility::convert<double>(_delim.at(1));
    if(_delim.size() > 2) repeat = utility::convert<uint64_t>(_delim.at(2));
    if(_delim.size() > 3) clock_id = find_clock_identifier(_delim.at(3));
}

void
spec::operator()(const stages& _stages) const
{
    auto _n = repeat;
    if(_n < 1) _n = std::numeric_limits<uint64_t>::max();

    while(get_state() < State::Active)
        sleep(1 * units::usec);

    for(uint64_t i = 0; i < _n; ++i)
    {
        auto _spec = spec{ clock_id, delay, duration, i, repeat };
        auto _wait = [_spec](const auto& _func, auto _dur) {
            auto _ret = true;
            auto _now = get_clock_now(_spec.clock_id.value);
            auto _del = (_dur * units::sec);
            auto _end = _now + _del;
            while(get_clock_now(_spec.clock_id.value) < _end && (_ret = _func(_spec)))
            {
            }
            return _ret;
        };

        ROCPROFSYS_VERBOSE(2,
                           "Executing constraint spec %lu of %lu :: delay: %6.3f, "
                           "duration: %6.3f, clock: %s\n",
                           i, _spec.repeat, _spec.delay, _spec.duration,
                           _spec.clock_id.as_string().c_str());

        if(_stages.init(_spec) && _wait(_stages.wait, _spec.delay) &&
           _stages.start(_spec) && _wait(_stages.collect, _spec.duration) &&
           _stages.stop(_spec))
        {
        }
        else
        {
            break;
        }
    }
}

//--------------------------------------------------------------------------------------//
//
//  global usage functions
//
//--------------------------------------------------------------------------------------//

const std::set<clock_identifier>&
get_valid_clock_ids()
{
    return accepted_clock_ids;
}

std::vector<spec>
get_trace_specs()
{
    auto _v = std::vector<constraint::spec>{};

    {
        auto _delay_v =
            config::get_setting_value<double>("ROCPROFSYS_TRACE_DELAY").value_or(0.0);
        auto _duration_v =
            config::get_setting_value<double>("ROCPROFSYS_TRACE_DURATION").value_or(0.0);
        auto _clock_v = find_clock_identifier(
            config::get_setting_value<std::string>("ROCPROFSYS_TRACE_PERIOD_CLOCK_ID")
                .value_or("CLOCK_REALTIME"));

        if(_delay_v > 0.0 || _duration_v > 0.0)
        {
            _v.emplace_back(_clock_v, _delay_v, _duration_v);
        }
    }

    {
        auto _periods_v =
            config::get_setting_value<std::string>("ROCPROFSYS_TRACE_PERIODS")
                .value_or("");
        if(!_periods_v.empty())
        {
            for(auto itr : tim::delimit(_periods_v, " ;\t\n"))
                _v.emplace_back(itr);
        }
    }

    return _v;
}

stages
get_trace_stages()
{
    auto _v = stages{};

    _v.init = [](const spec&) { return get_state() < State::Finalized; };
    _v.wait = [](const spec& _spec) {
        sleep(std::min<uint64_t>(100 * units::msec, _spec.delay * units::sec));
        return get_state() < State::Finalized;
    };
    _v.start   = [](const spec&) { return get_state() < State::Finalized; };
    _v.collect = [](const spec& _spec) {
        sleep(std::min<uint64_t>(100 * units::msec, _spec.duration * units::sec));
        return get_state() < State::Finalized;
    };
    _v.stop = [](const spec&) { return get_state() < State::Finalized; };

    return _v;
}
}  // namespace constraint
}  // namespace rocprofsys
