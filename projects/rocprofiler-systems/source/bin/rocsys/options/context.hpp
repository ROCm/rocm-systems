// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::options
{

/// @brief SDK and build queries. Null functions are not called until a trace
///        catalog is used. `--traces` and `avail --traces` are the only callers.
struct domain_query
{
    std::vector<std::string> (*choices)()           = nullptr;
    std::vector<std::string> (*supported)()         = nullptr;
    bool (*runtime_compiled)(std::string_view name) = nullptr;
};

/// @brief Injected queries for option callbacks.
struct context
{
    domain_query domains = {};
};

}  // namespace rocprofsys::rocsys::options
