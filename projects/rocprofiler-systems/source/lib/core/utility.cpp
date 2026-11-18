// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "utility.hpp"

#include "common/delimit.hpp"
#include "common/string_utility.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace rocprofsys::utility
{
namespace
{
template <typename ContainerT, typename Arg>
concept has_emplace =
    requires(ContainerT& _targ, Arg&& _v) { _targ.emplace(std::forward<Arg>(_v)); };

template <typename ContainerT, typename Arg>
    requires has_emplace<ContainerT, Arg>
decltype(auto)
emplace(ContainerT& _targ, Arg&& _v)
{
    return _targ.emplace(std::forward<Arg>(_v));
}

template <typename ContainerT, typename Arg>
    requires(!has_emplace<ContainerT, Arg>)
decltype(auto)
emplace(ContainerT& _targ, Arg&& _v)
{
    return _targ.emplace_back(std::forward<Arg>(_v));
}
}  // namespace

template <typename Tp, typename ContainerT, typename Up>
ContainerT
parse_numeric_range(std::string _input_string, const std::string& _label, Up _incr)
{
    auto const _get_value = [&](const std::string& _inp) {
        auto const _parsed = utility::string::safe_from_chars<Tp>(_inp);
        if(_parsed.is_error())
        {
            throw std::invalid_argument(
                fmt::format("Invalid {} value '{}': {}", _label, _inp, _parsed.error()));
        }
        return _parsed.value();
    };

    _input_string = utility::string::to_lower(_input_string);
    auto result   = ContainerT{};
    for(auto _v : rocprofsys::delimit(_input_string, ",; \t\n\r"))
    {
        if(_v.find_first_not_of("0123456789-:") != std::string::npos)
        {
            LOG_WARNING(
                "Invalid {} specification. Only numerical values (e.g., 0), ranges "
                "(e.g., 0-7), and ranges with increments (e.g. 20-40:10) are permitted. "
                "Ignoring {}...",
                _label, _v);
            continue;
        }

        auto       _incr_v   = _incr;
        auto const _incr_pos = _v.find(':');
        if(_incr_pos != std::string::npos)
        {
            auto const _incr_str = _v.substr(_incr_pos + 1);
            if(!_incr_str.empty())
            {
                auto const _parsed =
                    utility::string::safe_from_chars<std::uint64_t>(_incr_str);
                if(_parsed.is_error())
                {
                    throw std::invalid_argument(
                        fmt::format("Invalid {} increment '{}': {}", _label, _incr_str,
                                    _parsed.error()));
                }
                _incr_v = static_cast<Up>(_parsed.value());
            }
            _v = _v.substr(0, _incr_pos);
        }

        if(_v.find('-') != std::string::npos)
        {
            // tim::delimit collapses consecutive '-' and drops empty fields, so
            // "5--7", "-1", and "5-" would otherwise sneak past the size check
            // below; reject leading/trailing/consecutive dashes explicitly.
            if(_v.front() == '-' || _v.back() == '-' ||
               _v.find("--") != std::string::npos)
            {
                LOG_WARNING("Invalid {} range specification: {}. Leading, trailing, or "
                            "consecutive '-' not permitted; required format N-M, "
                            "e.g. 0-4. Ignoring {}...",
                            _label, _v, _v);
                continue;
            }

            // split the string into two parts at the '-' character and check if the
            // result is valid
            auto _vv = rocprofsys::delimit(_v, "-");
            if(_vv.size() != 2)
            {
                LOG_WARNING("Invalid {} range specification: {}. Required format N-M, "
                            "e.g. 0-4. Ignoring {}...",
                            _label, _v, _v);
                continue;
            }

            Tp _vn = _get_value(_vv.at(0));
            Tp _vN = _get_value(_vv.at(1));
            if(_vn > _vN)
            {
                LOG_WARNING("Invalid {} range specification: {}. Start exceeds end; "
                            "required format N-M with N <= M, e.g. 0-4. Ignoring {}...",
                            _label, _v, _v);
                continue;
            }
            // A non-positive increment never reaches _vN and would loop forever.
            if(_incr_v <= 0)
            {
                throw std::invalid_argument(fmt::format(
                    "Invalid {} range specification: {}. Increment must be positive",
                    _label, _v));
            }
            do
            {
                emplace(result, _vn);
                _vn += _incr_v;
            } while(_vn <= _vN);
        }
        else
        {
            emplace(result, _get_value(_v));
        }
    }
    return result;
}

template std::set<std::int64_t>
parse_numeric_range<std::int64_t, std::set<std::int64_t>>(std::string, const std::string&,
                                                          long);
template std::vector<std::int64_t>
parse_numeric_range<std::int64_t, std::vector<std::int64_t>>(std::string,
                                                             const std::string&, long);
template std::unordered_set<std::int64_t>
parse_numeric_range<std::int64_t, std::unordered_set<std::int64_t>>(std::string,
                                                                    const std::string&,
                                                                    long);

}  // namespace rocprofsys::utility
