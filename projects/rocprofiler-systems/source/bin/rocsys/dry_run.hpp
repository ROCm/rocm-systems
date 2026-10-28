// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_op.hpp"

#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

namespace rocprofsys::rocsys
{

/// @brief Print @p verb, the forwarded @p argv, and the rocsys environment ops.
///        Inherited ops are labelled as overrides. Embedded ops are labelled as
///        defaults baked into a rewritten binary. This does not launch anything.
void
write_dry_run(std::ostream& out, std::string_view verb, std::span<const std::string> argv,
              const env_plan& ops);

}  // namespace rocprofsys::rocsys
