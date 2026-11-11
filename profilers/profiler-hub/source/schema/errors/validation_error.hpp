// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace profiler_hub::schema
{

struct validation_error
{
    std::string object;
    std::string field;
    std::string reason;
    std::string documentation;
};

struct validation_result
{
    bool                          ok = true;
    std::string                   schema_version;
    std::string                   object;
    std::vector<validation_error> errors;

    [[nodiscard]] bool success() const noexcept { return ok; }

    [[nodiscard]] std::string format() const;
    void                      print(std::ostream& out) const;
};

// docs/schema/<major>.<minor>.<patch>_usage.md
[[nodiscard]] std::string
documentation_link(profiler_hub::version_t version);

}  // namespace profiler_hub::schema
