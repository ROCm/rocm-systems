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

#pragma once

#include <string>
#include <vector>

namespace omnitrace
{
inline namespace common
{
namespace
{
template <typename ContainerT = std::vector<std::string>>
inline ContainerT
delimit(const std::string& line, const char* delimiters = "\"',;: ");

template <typename ContainerT>
inline ContainerT
delimit(const std::string& line, const char* delimiters)
{
    ContainerT _result{};
    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _delimp) that is not a delimiter
        _beginp = line.find_first_not_of(delimiters, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == std::string::npos) break;
        // starting at the position of the new string, find the next delimiter
        _delimp = line.find_first_of(delimiters, _beginp);
        std::string _tmp{};
        // starting at the position of the new string, get the characters
        // between this position and the next delimiter
        _tmp = line.substr(_beginp, _delimp - _beginp);
        // don't add empty strings
        if(!_tmp.empty()) _result.emplace(_result.end(), _tmp);
    }
    return _result;
}
}  // namespace
}  // namespace common
}  // namespace omnitrace
