// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>

namespace rocprofsys
{
class output_file_registry;
}

namespace rocprofsys::output
{

struct run_metadata;

void
print_summary(std::ostream& os, const output_file_registry& registry);

void
print_summary(std::ostream& os, const output_file_registry& registry,
              const run_metadata& meta);

}  // namespace rocprofsys::output
