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

#include <rocprofiler-sdk/cxx/details/mpl.hpp>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace sdk
{
namespace parse
{
template <typename Tp>
inline Tp
from_string(const std::string& str)
{
    auto ss  = std::stringstream{str};
    auto val = Tp{};
    ss >> val;
    return val;
}

template <typename Tp>
inline Tp
from_string(const char* cstr)
{
    auto ss  = std::stringstream{cstr};
    auto val = Tp{};
    ss >> val;
    return val;
}

/// \brief tokenize a string into a set
///
template <typename ContainerT = std::vector<std::string>,
          typename ValueT     = typename ContainerT::value_type,
          typename PredicateT = std::function<ValueT(ValueT&&)>>
inline ContainerT
tokenize(
    std::string_view line,
    std::string_view delimiters = "\"',;: ",
    PredicateT&&     predicate  = [](ValueT&& s) -> ValueT { return s; })
{
    using value_type = ValueT;

    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    ContainerT _result = {};
    if(mpl::reserve(_result, 0))
    {
        size_t _nmax = 0;
        for(char itr : line)
        {
            if(delimiters.find(itr) != std::string::npos) ++_nmax;
        }
        mpl::reserve(_result, _nmax);
    }
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _delimp) that is not a delimiter
        _beginp = line.find_first_not_of(delimiters, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == std::string::npos) break;
        // starting at the position of the new string, find the next delimiter
        _delimp = line.find_first_of(delimiters, _beginp);

        auto _tmp = value_type{};
        // starting at the position of the new string, get the characters
        // between this position and the next delimiter
        if(_beginp < line.length()) _tmp = line.substr(_beginp, _delimp - _beginp);

        // don't add empty strings
        if(!_tmp.empty())
        {
            mpl::emplace(_result, std::forward<PredicateT>(predicate)(std::move(_tmp)));
        }
    }
    return _result;
}

/// \brief tokenize a string into a set
///
template <typename ContainerT = std::vector<std::string>,
          typename DelimT     = std::string_view,
          typename ValueT     = typename ContainerT::value_type,
          typename PredicateT = ValueT (*)(DelimT&&)>
inline ContainerT
tokenize(
    std::string_view           line,
    const std::vector<DelimT>& delimiters,
    PredicateT&&               predicate = [](DelimT&& s) -> ValueT { return ValueT{s}; })
{
    ContainerT _result = {};
    size_t     _start  = 0;
    size_t     _end    = std::string::npos;

    while(_start != std::string::npos)
    {
        _end = std::string::npos;

        // Find the earliest occurrence of any delimiter
        for(const auto& itr : delimiters)
        {
            size_t pos = line.find(itr, _start);
            if(pos != std::string::npos && (_end == std::string::npos || pos < _end))
            {
                _end = pos;
            }
        }

        // Extract token and update start position
        if(_end != std::string::npos)
        {
            mpl::emplace(_result,
                         std::forward<PredicateT>(predicate)(line.substr(_start, _end - _start)));
            _start = _end;

            // Move start past the delimiter
            for(const auto& delimiter : delimiters)
            {
                if(line.compare(_start, delimiter.size(), delimiter) == 0)
                {
                    _start += delimiter.size();
                    break;
                }
            }
        }
        else
        {
            // Last token after the final delimiter
            mpl::emplace(_result, std::forward<PredicateT>(predicate)(line.substr(_start)));
            break;
        }
    }

    return _result;
}

///  \brief apply a string transformation to substring in between a common delimiter.
///
template <typename PredicateT = std::function<std::string(const std::string&)>>
inline std::string
str_transform(std::string_view input,
              std::string_view _begin,
              std::string_view _end,
              PredicateT&&     predicate)
{
    size_t      _beg_pos = 0;  // position that is the beginning of the new string
    size_t      _end_pos = 0;  // position of the delimiter in the string
    std::string _result  = std::string{input};
    while(_beg_pos < _result.length() && _end_pos < _result.length())
    {
        // find the first sequence of characters after the end-position
        _beg_pos = _result.find(_begin, _end_pos);

        // if sequence wasn't found, we are done
        if(_beg_pos == std::string::npos) break;

        // starting after the position of the first delimiter, find the end sequence
        if(!_end.empty())
            _end_pos = _result.find(_end, _beg_pos + 1);
        else
            _end_pos = _beg_pos + _begin.length();

        // break if not found
        if(_end_pos == std::string::npos) break;

        // length of the substr being operated on
        auto _len = _end_pos - _beg_pos;

        // get the substring between the two delimiters (including first delimiter)
        auto _sub = _result.substr(_beg_pos, _len);

        // apply the transform
        auto _transformed = predicate(_sub);

        // only replace if necessary
        if(_sub != _transformed)
        {
            _result = _result.replace(_beg_pos, _len, _transformed);
            // move end to the end of transformed string
            _end_pos = _beg_pos + _transformed.length();
        }
    }
    return _result;
}

inline std::string
strip(std::string&& str, std::string_view characters)
{
    constexpr auto npos = std::string_view::npos;

    auto bpos = str.find_first_not_of(characters);
    auto epos = str.find_last_not_of(characters);

    if(bpos != npos && epos != npos)
        return str.substr(bpos, (epos - bpos + 1));
    else if(bpos != npos)
        return str.substr(bpos);
    else if(epos != npos)
        return str.substr(0, epos + 1);

    return str;
}
}  // namespace parse
}  // namespace sdk
}  // namespace rocprofiler
