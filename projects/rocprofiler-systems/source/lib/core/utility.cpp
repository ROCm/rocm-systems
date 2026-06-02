// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "utility.hpp"
#include <cstdint>

#include "logger/debug.hpp"

namespace rocprofsys
{
namespace utility
{
namespace
{
template <typename ContainerT, typename Arg>
auto
emplace_impl(ContainerT& _targ, Arg&& _v,
             int) -> decltype(_targ.emplace(std::forward<Arg>(_v)))
{
    return _targ.emplace(std::forward<Arg>(_v));
}

template <typename ContainerT, typename Arg>
auto
emplace_impl(ContainerT& _targ, Arg&& _v,
             long) -> decltype(_targ.emplace_back(std::forward<Arg>(_v)))
{
    return _targ.emplace_back(std::forward<Arg>(_v));
}

template <typename ContainerT, typename Arg>
decltype(auto)
emplace(ContainerT& _targ, Arg&& _v)
{
    return emplace_impl(_targ, std::forward<Arg>(_v), 0);
}
}  // namespace

template <typename Tp, typename ContainerT, typename Up>
ContainerT
parse_numeric_range(std::string _input_string, const std::string& _label, Up _incr)
{
    auto _get_value = [](const std::string& _inp) {
        std::stringstream iss{ _inp };
        auto              var = Tp{};
        iss >> var;
        return var;
    };

    for(auto& itr : _input_string)
        itr = tolower(itr);
    auto _result = ContainerT{};
    for(auto _v : tim::delimit(_input_string, ",; \t\n\r"))
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

        auto _incr_v   = _incr;
        auto _incr_pos = _v.find(':');
        if(_incr_pos != std::string::npos)
        {
            auto _incr_str = _v.substr(_incr_pos + 1);
            if(!_incr_str.empty()) _incr_v = static_cast<Up>(std::stoull(_incr_str));
            _v = _v.substr(0, _incr_pos);
        }

        if(_v.find('-') != std::string::npos)
        {
            auto _vv = tim::delimit(_v, "-");
            if(_vv.size() != 2)
            {
                throw std::runtime_error(fmt::format(
                    "Invalid {} range specification: {}. Required format N-M, e.g. 0-4",
                    _label, _v));
            }

            Tp _vn = _get_value(_vv.at(0));
            Tp _vN = _get_value(_vv.at(1));
            do
            {
                emplace(_result, _vn);
                _vn += _incr_v;
            } while(_vn <= _vN);
        }
        else
        {
            emplace(_result, std::stoll(_v));
        }
    }
    return _result;
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

void
trim_str(std::string& str)
{
    const auto start = str.find_first_not_of(" \n\r\t\f\v");
    if(start == std::string::npos)
    {
        str.clear();
        return;
    }
    str.erase(0, start);
    const auto end = str.find_last_not_of(" \n\r\t\f\v");
    str.erase(end + 1);
}

}  // namespace utility
}  // namespace rocprofsys
