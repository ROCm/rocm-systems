// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_op.hpp"
#include "result.hpp"
#include "verb.hpp"

#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::rocsys
{

/// @brief What the launcher runs. Delivery builds this and does not touch the process.
struct launch_plan
{
    bool in_process = {};
    /// @brief argv0 for an in-process verb, or the sibling binary name for exec.
    ///        The launcher joins the sibling name with its own directory.
    std::string              program = {};
    std::vector<std::string> argv    = {};
    /// @brief Ops the launcher applies to the process. Embedded rewrite ops are
    ///        already tokens in @ref argv and are not repeated here.
    env_plan                               env  = {};
    std::optional<common_utils::tool_mode> mode = {};
};

/// @brief Build the argv and inherited env for @p verb.
///        @p argv is the full rocsys argv, including argv0. When @p strip_subcommand
///        is true, argv[1] is the verb and is omitted. Rewrite inserts `--env K=V`
///        right after argv0 and puts an implicit `-o` last before `--`.
[[nodiscard]] result<launch_plan>
plan(const verb_spec& verb, bool strip_subcommand, const std::vector<std::string>& argv,
     const env_plan& ops);

}  // namespace rocprofsys::rocsys
