// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "constraint.hpp"
#include "common/delimit.hpp"
#include "config.hpp"
#include "utility.hpp"

#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <chrono>
#include <cstdint>
#include <ratio>
#include <string>
#include <type_traits>

namespace rocprofsys
{
namespace constraint
{
namespace
{
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

    auto _choices = std::vector<std::string>{};
    _choices.reserve(accepted_clock_ids.size());
    for(const auto& itr : accepted_clock_ids)
        _choices.emplace_back(itr.as_string());

    throw std::runtime_error(fmt::format("Unknown clock id {}: {}. Valid choices: {}",
                                         _descript, _v, fmt::join(_choices, "")));
}
}  // namespace

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

spec::spec(clock_identifier _id, double _delay, double _dur, std::uint64_t _n,
           std::uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
, clock_id{ std::move(_id) }
{}

spec::spec(int _clock_id, double _delay, double _dur, std::uint64_t _n,
           std::uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
, clock_id{ find_clock_identifier(_clock_id) }
{}

spec::spec(const std::string& _clock_id, double _delay, double _dur, std::uint64_t _n,
           std::uint64_t _rep)
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
    auto _delim = rocprofsys::common::delimit(_line, ":");
    if(!_delim.empty()) delay = utility::convert<double>(_delim.at(0));
    if(_delim.size() > 1) duration = utility::convert<double>(_delim.at(1));
    if(_delim.size() > 2) repeat = utility::convert<std::uint64_t>(_delim.at(2));
    if(_delim.size() > 3) clock_id = find_clock_identifier(_delim.at(3));
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
            for(auto itr : rocprofsys::common::delimit(_periods_v, " ;\t\n"))
                _v.emplace_back(itr);
        }
    }

    return _v;
}
}  // namespace constraint
}  // namespace rocprofsys
