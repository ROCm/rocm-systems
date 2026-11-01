// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/option_spec.hpp"
#include "verb.hpp"

#include <iosfwd>
#include <span>
#include <string>

namespace rocprofsys::rocsys::help
{

/// @brief "Common options" for `rocsys --help`. An option is common when it
///        requires no verb capability.
void
write_common_options(std::ostream& out, std::span<const options::option_spec> table);

/// @brief "rocsys options for `<verb>`", limited to options the verb's
///        capabilities allow. Rewrite also prints how to profile the rewritten binary.
void
write_verb_options(std::ostream& out, const verb_spec& verb,
                   std::span<const options::option_spec> table);

/// @brief True when a token before `--` asks the legacy tool for help.
[[nodiscard]] bool
requests_tool_help(std::span<const std::string> passthrough);

}  // namespace rocprofsys::rocsys::help
