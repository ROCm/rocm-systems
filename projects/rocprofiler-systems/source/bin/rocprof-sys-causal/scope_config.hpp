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

#pragma once

#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace causal
{

struct ScopeConfig
{
    std::string_view short_flag;
    std::string_view long_flag;
    std::string_view description;
    std::string_view dtype;
};

inline const std::vector<ScopeConfig> SCOPE_CONFIGS = {
    { "-B", "--binary-scope",
      "Restricts causal experiments to the binaries matching the list of "
      "regular expressions. Each space designates a group and multiple "
      "scopes can be grouped together with a semi-colon",
      "integers" },

    { "-S", "--source-scope",
      "Restricts causal experiments to the source files or source file + "
      "lineno pairs (i.e. <file> or <file>:<line>) matching the list of "
      "regular expressions. Each space designates a group and multiple "
      "scopes can be grouped together with a semi-colon",
      "integers" },

    { "-F", "--function-scope",
      "Restricts causal experiments to the functions matching the list of "
      "regular expressions. Each space designates a group and multiple "
      "scopes can be grouped together with a semi-colon",
      "regex-list" },

    { "-BE", "--binary-exclude",
      "Excludes causal experiments from being performed on the binaries matching "
      "the list of regular expressions. Each space designates a group and multiple "
      "excludes can be grouped together with a semi-colon",
      "integers" },

    { "-SE", "--source-exclude",
      "Excludes causal experiments from being performed on the code from the "
      "source files or source file + lineno pair (i.e. <file> or <file>:<line>) "
      "matching the list of regular expressions. Each space designates a group and "
      "multiple excludes can be grouped together with a semi-colon",
      "integers" },

    { "-FE", "--function-exclude",
      "Excludes causal experiments from being performed on the functions matching "
      "the list of regular expressions. Each space designates a group and multiple "
      "excludes can be grouped together with a semi-colon",
      "regex-list" }
};

}  
}  
