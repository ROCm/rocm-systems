// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"

#include <unistd.h>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>

namespace rocprofiler
{
namespace common
{
namespace units
{
static constexpr int64_t nsec   = 1;
static constexpr int64_t usec   = 1000 * nsec;
static constexpr int64_t msec   = 1000 * usec;
static constexpr int64_t csec   = 10 * msec;
static constexpr int64_t dsec   = 10 * csec;
static constexpr int64_t sec    = 10 * dsec;
static constexpr int64_t minute = 60 * sec;
static constexpr int64_t hour   = 60 * minute;

static constexpr int64_t byte     = 1;
static constexpr int64_t kilobyte = 1000 * byte;
static constexpr int64_t megabyte = 1000 * kilobyte;
static constexpr int64_t gigabyte = 1000 * megabyte;
static constexpr int64_t terabyte = 1000 * gigabyte;
static constexpr int64_t petabyte = 1000 * terabyte;

static constexpr int64_t kibibyte = 1024 * byte;
static constexpr int64_t mebibyte = 1024 * kibibyte;
static constexpr int64_t gibibyte = 1024 * mebibyte;
static constexpr int64_t tebibyte = 1024 * gibibyte;
static constexpr int64_t pebibyte = 1024 * tebibyte;

static constexpr int64_t B  = 1;
static constexpr int64_t KB = 1000 * B;
static constexpr int64_t MB = 1000 * KB;
static constexpr int64_t GB = 1000 * MB;
static constexpr int64_t TB = 1000 * GB;
static constexpr int64_t PB = 1000 * TB;

static constexpr int64_t Bi  = 1;
static constexpr int64_t KiB = 1024 * Bi;
static constexpr int64_t MiB = 1024 * KiB;
static constexpr int64_t GiB = 1024 * MiB;
static constexpr int64_t TiB = 1024 * GiB;
static constexpr int64_t PiB = 1024 * TiB;

static constexpr int64_t nanowatt  = 1;
static constexpr int64_t microwatt = 1000 * nanowatt;
static constexpr int64_t milliwatt = 1000 * microwatt;
static constexpr int64_t watt      = 1000 * milliwatt;
static constexpr int64_t kilowatt  = 1000 * watt;
static constexpr int64_t megawatt  = 1000 * kilowatt;
static constexpr int64_t gigawatt  = 1000 * megawatt;

static constexpr int64_t hertz     = 1;
static constexpr int64_t kilohertz = 1000 * hertz;
static constexpr int64_t megahertz = 1000 * kilohertz;
static constexpr int64_t gigahertz = 1000 * megahertz;

static constexpr int64_t Hz  = 1;
static constexpr int64_t KHz = 1000 * Hz;
static constexpr int64_t MHz = 1000 * KHz;
static constexpr int64_t GHz = 1000 * MHz;

inline int64_t
get_page_size()
{
    static auto _pagesz = sysconf(_SC_PAGESIZE);
    return _pagesz;
}

const int64_t clocks_per_sec = sysconf(_SC_CLK_TCK);

//--------------------------------------------------------------------------------------//

inline std::string
time_repr(int64_t _unit)
{
    switch(_unit)
    {
        case nsec: return "nsec"; break;
        case usec: return "usec"; break;
        case msec: return "msec"; break;
        case csec: return "csec"; break;
        case dsec: return "dsec"; break;
        case sec: return "sec"; break;
        default: return "UNK"; break;
    }
    return std::string{};
}

//--------------------------------------------------------------------------------------//

inline std::string
mem_repr(int64_t _unit)
{
    switch(_unit)
    {
        case byte: return "B"; break;
        case kilobyte: return "KB"; break;
        case megabyte: return "MB"; break;
        case gigabyte: return "GB"; break;
        case terabyte: return "TB"; break;
        case petabyte: return "PB"; break;
        case kibibyte: return "KiB"; break;
        case mebibyte: return "MiB"; break;
        case gibibyte: return "GiB"; break;
        case tebibyte: return "TiB"; break;
        case pebibyte: return "PiB"; break;
        default: return "UNK"; break;
    }
    return std::string{};
}

//--------------------------------------------------------------------------------------//

inline std::string
freq_repr(int64_t _unit)
{
    switch(_unit)
    {
        case hertz: return "Hz"; break;
        case kilohertz: return "KHz"; break;
        case megahertz: return "MHz"; break;
        case gigahertz: return "GHz"; break;
        default: return "UNK"; break;
    }
    return std::string{};
}

//--------------------------------------------------------------------------------------//

inline std::string
power_repr(int64_t _unit)
{
    switch(_unit)
    {
        case nanowatt: return "nanowatts"; break;
        case microwatt: return "microwatts"; break;
        case milliwatt: return "milliwatts"; break;
        case watt: return "watts"; break;
        case kilowatt: return "kilowatts"; break;
        case megawatt: return "megawatts"; break;
        case gigawatt: return "gigawatts"; break;
        default: return "UNK"; break;
    }
    return std::string{};
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_memory_unit(std::string _unit)
{
    using string_t    = std::string;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, string_t, int64_t>;

    if(_unit.empty()) return return_type{"MB", units::megabyte};

    for(auto& itr : _unit)
        itr = tolower(itr);

    for(const auto& itr : {inner_t{"byte", "b", units::byte},
                           inner_t{"kilobyte", "kb", units::kilobyte},
                           inner_t{"megabyte", "mb", units::megabyte},
                           inner_t{"gigabyte", "gb", units::gigabyte},
                           inner_t{"terabyte", "tb", units::terabyte},
                           inner_t{"petabyte", "pb", units::petabyte},
                           inner_t{"kibibyte", "kib", units::KiB},
                           inner_t{"mebibyte", "mib", units::MiB},
                           inner_t{"gibibyte", "gib", units::GiB},
                           inner_t{"tebibyte", "tib", units::TiB},
                           inner_t{"pebibyte", "pib", units::PiB}})
    {
        if(_unit == std::get<0>(itr) || _unit == std::get<1>(itr))
        {
            if(std::get<2>(itr) == units::byte)
                return return_type{std::get<0>(itr), std::get<2>(itr)};
            return return_type{mem_repr(std::get<2>(itr)), std::get<2>(itr)};
        }
    }

    ROCP_WARNING << "Warning!! No memory unit matching \"" << _unit << "\". Using default...\n";

    return return_type{"MB", units::megabyte};
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_timing_unit(std::string _unit)
{
    using string_t    = std::string;
    using strset_t    = std::unordered_set<string_t>;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, strset_t, int64_t>;

    if(_unit.empty()) return return_type{"sec", units::sec};

    for(auto& itr : _unit)
        itr = tolower(itr);

    for(const auto& itr :
        {inner_t{"nsec", strset_t{"ns", "nanosecond", "nanoseconds"}, units::nsec},
         inner_t{"usec", strset_t{"us", "microsecond", "microseconds"}, units::usec},
         inner_t{"msec", strset_t{"ms", "millisecond", "milliseconds"}, units::msec},
         inner_t{"csec", strset_t{"cs", "centisecond", "centiseconds"}, units::csec},
         inner_t{"dsec", strset_t{"ds", "decisecond", "deciseconds"}, units::dsec},
         inner_t{"sec", strset_t{"s", "second", "seconds"}, units::sec},
         inner_t{"min", strset_t{"minute", "minutes"}, units::minute},
         inner_t{"hr", strset_t{"hr", "hour", "hours"}, units::hour}})
    {
        if(_unit == std::get<0>(itr) || std::get<1>(itr).find(_unit) != std::get<1>(itr).end())
        {
            return return_type{time_repr(std::get<2>(itr)), std::get<2>(itr)};
        }
    }

    ROCP_WARNING << "Warning!! No timing unit matching \"" << _unit << "\". Using default...\n";

    return return_type{"sec", units::sec};
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_frequncy_unit(std::string _unit)
{
    using string_t    = std::string;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, string_t, int64_t>;

    if(_unit.empty()) return return_type{"MHz", units::megahertz};

    for(auto& itr : _unit)
        itr = tolower(itr);

    for(const auto& itr : {inner_t{"hertz", "hz", units::hertz},
                           inner_t{"kilohertz", "khz", units::kilohertz},
                           inner_t{"megahertz", "mhz", units::megahertz},
                           inner_t{"gigahertz", "ghz", units::gigahertz}})
    {
        if(_unit == std::get<0>(itr) || _unit == std::get<1>(itr))
        {
            return return_type{freq_repr(std::get<2>(itr)), std::get<2>(itr)};
        }
    }

    ROCP_WARNING << "Warning!! No frequency unit matching \"" << _unit << "\". Using default...\n";

    return return_type{"MHz", units::megahertz};
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_power_unit(const std::string& _unit)
{
    using string_t    = std::string;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, string_t, int64_t>;

    if(_unit.empty()) return return_type{"watts", units::watt};

    auto _lunit = _unit;
    for(auto& itr : _lunit)
        itr = tolower(itr);

    for(const auto& itr : {inner_t{"nanowatt", "nW", units::nanowatt},
                           inner_t{"microwatt", "uW", units::microwatt},
                           inner_t{"milliwatt", "mW", units::milliwatt},
                           inner_t{"watt", "W", units::watt},
                           inner_t{"kilowatt", "KW", units::kilowatt},
                           inner_t{"megawatt", "MW", units::megawatt},
                           inner_t{"gigawatt", "GW", units::gigawatt}})
    {
        if(_lunit == std::get<0>(itr) || _lunit + "s" == std::get<0>(itr) ||
           _unit == std::get<1>(itr))
        {
            return return_type{power_repr(std::get<2>(itr)), std::get<2>(itr)};
        }
    }

    ROCP_WARNING << "Warning!! No power unit matching \"" << _unit << "\". Using default...\n";

    return return_type{"watts", units::watt};
}

//--------------------------------------------------------------------------------------//

namespace temperature
{
enum unit_system : int8_t
{
    Celsius = 0,
    Fahrenheit,
    Kelvin
};

template <typename Tp>
Tp
convert(Tp _v, unit_system _from, unit_system _to)
{
    switch(_from)
    {
        case Celsius:
        {
            switch(_to)
            {
                case Celsius: return _v;
                case Fahrenheit: return static_cast<Tp>((_v * 1.8) + 32);
                case Kelvin: return (_v - 273);
            }
        }
        case Fahrenheit:
        {
            switch(_to)
            {
                case Celsius: return static_cast<Tp>((_v - 32) / 1.8);
                case Fahrenheit: return _v;
                case Kelvin: return (_v - 273);
            }
        }
        case Kelvin:
        {
            switch(_to)
            {
                case Celsius: return (_v + 273);
                case Fahrenheit: return static_cast<Tp>(((_v + 273) * 1.8) + 32);
                case Kelvin: return _v;
            }
        }
    }
}
}  // namespace temperature
}  // namespace units
}  // namespace common
}  // namespace rocprofiler
