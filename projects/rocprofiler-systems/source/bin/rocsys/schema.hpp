// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/backend.hpp"
#include "parser/schema.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::rocsys
{
/// Every rocsys option, in help order. An option that is not given writes nothing, so
/// the tool keeps its own default.
/// @throws std::logic_error if two options share an id or a spelling.
[[nodiscard]] parser::schema
make_schema();

/// The backend --traces names. It lists each ROCm API name, and its alias, whose
/// ROCPROFSYS_ROCM_DOMAINS value is in @p supported_domains, the values the installed
/// rocprofiler-sdk takes, such as hip-api for hip_api. The runtime names are the row's
/// own choices.
[[nodiscard]] std::unique_ptr<parser::backend_interface>
make_trace_backend(std::vector<std::string> supported_domains);

/// Variables that --traces writes for @p names, the --traces values as typed. An alias
/// selects its row, so hip and hip-api both turn on hip_api. A listed name is on and
/// every other --traces name is off.
[[nodiscard]] parser::value_map_t
traces_environment(const std::vector<std::string>& names);
}  // namespace rocprofsys::rocsys
