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

#include "utility.hpp"
#include "debug.hpp"

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
            ROCPROFSYS_BASIC_VERBOSE_F(
                0,
                "Invalid %s specification. Only numerical values (e.g., 0), ranges "
                "(e.g., 0-7), and ranges with increments (e.g. 20-40:10) are permitted. "
                "Ignoring %s...",
                _label.c_str(), _v.c_str());
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
            ROCPROFSYS_CONDITIONAL_THROW(
                _vv.size() != 2,
                "Invalid %s range specification: %s. Required format N-M, e.g. 0-4",
                _label.c_str(), _v.c_str());
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

template std::set<int64_t>
parse_numeric_range<int64_t, std::set<int64_t>>(std::string, const std::string&, long);
template std::vector<int64_t>
parse_numeric_range<int64_t, std::vector<int64_t>>(std::string, const std::string&, long);
template std::unordered_set<int64_t>
parse_numeric_range<int64_t, std::unordered_set<int64_t>>(std::string, const std::string&,
                                                          long);
}  // namespace utility
}  // namespace rocprofsys
