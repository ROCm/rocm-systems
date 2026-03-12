// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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

}  // namespace causal
}  // namespace rocprofsys
