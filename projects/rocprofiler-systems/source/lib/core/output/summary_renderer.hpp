// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
class output_file_registry;
}

namespace rocprofsys::output
{

struct run_metadata;

// Convenience overload: passes a default-constructed run_metadata.
// Useful for tests and the registry-member forwarder.
void
print_summary(std::ostream& os, const output_file_registry& registry);

void
print_summary(std::ostream& os, const output_file_registry& registry,
              const run_metadata& meta);

// Exposed for unit testing. Word-boundary preferred (last
// whitespace at or before width); falls back to byte chunking with
// UTF-8 backoff so multi-byte code points are never split.
[[nodiscard]] std::vector<std::string>
wrap_to_width(std::string_view content, std::size_t width);

}  // namespace rocprofsys::output
