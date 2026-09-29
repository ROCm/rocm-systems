// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/ompt/specification.hpp"

#include <array>
#include <string_view>

namespace rocprofsys::domains::callback::ompt::rules
{
enum class flag_decode_mode
{
    first_match,
    all_matches,
};

enum class no_match_behavior
{
    omit,
    emit_none,
};

/**
 * A single textual representation of a flag bit.
 *
 * Rules are processed in contiguous groups with the same `type`, `key`, and
 * `mode`:
 *
 * - `first_match`: emits the first matching rule in the group.
 * - `all_matches`: emits all matching values from the group, joined by ", ".
 *
 * The order of rules inside a `first_match` group defines precedence.
 */
template <typename FlagT>
struct flag_rule
{
    FlagT             mask;
    std::string_view  type;
    std::string_view  key;
    std::string_view  value;
    flag_decode_mode  mode;
    no_match_behavior on_no_match = no_match_behavior::omit;
};

using ompt_flag_rule = flag_rule<int>;

inline constexpr std::array k_parallel{
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_parallel_flag_t::ompt_parallel_invoker_program),
        .type  = "ompt_parallel_flag_t",
        .key   = "invoker",
        .value = "program",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_parallel_flag_t::ompt_parallel_invoker_runtime),
        .type  = "ompt_parallel_flag_t",
        .key   = "invoker",
        .value = "runtime",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_parallel_flag_t::ompt_parallel_league),
        .type  = "ompt_parallel_flag_t",
        .key   = "invoker_cause",
        .value = "teams_construct",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_parallel_flag_t::ompt_parallel_team),
        .type  = "ompt_parallel_flag_t",
        .key   = "invoker_cause",
        .value = "parallel_construct",
        .mode  = flag_decode_mode::first_match,
    },
};

inline constexpr std::array k_task_create{
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_initial),
        .type  = "ompt_task_flag_t",
        .key   = "classification",
        .value = "initial",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_implicit),
        .type  = "ompt_task_flag_t",
        .key   = "classification",
        .value = "implicit",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_explicit),
        .type  = "ompt_task_flag_t",
        .key   = "classification",
        .value = "explicit",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_target),
        .type  = "ompt_task_flag_t",
        .key   = "classification",
        .value = "target",
        .mode  = flag_decode_mode::first_match,
    },

    ompt_flag_rule{
        .mask        = static_cast<int>(ompt_task_flag_t::ompt_task_undeferred),
        .type        = "ompt_task_flag_t",
        .key         = "properties",
        .value       = "undeferred",
        .mode        = flag_decode_mode::all_matches,
        .on_no_match = no_match_behavior::emit_none,
    },
    ompt_flag_rule{
        .mask        = static_cast<int>(ompt_task_flag_t::ompt_task_untied),
        .type        = "ompt_task_flag_t",
        .key         = "properties",
        .value       = "untied",
        .mode        = flag_decode_mode::all_matches,
        .on_no_match = no_match_behavior::emit_none,
    },
    ompt_flag_rule{
        .mask        = static_cast<int>(ompt_task_flag_t::ompt_task_final),
        .type        = "ompt_task_flag_t",
        .key         = "properties",
        .value       = "final",
        .mode        = flag_decode_mode::all_matches,
        .on_no_match = no_match_behavior::emit_none,
    },
    ompt_flag_rule{
        .mask        = static_cast<int>(ompt_task_flag_t::ompt_task_mergeable),
        .type        = "ompt_task_flag_t",
        .key         = "properties",
        .value       = "mergeable",
        .mode        = flag_decode_mode::all_matches,
        .on_no_match = no_match_behavior::emit_none,
    },
    ompt_flag_rule{
        .mask        = static_cast<int>(ompt_task_flag_t::ompt_task_merged),
        .type        = "ompt_task_flag_t",
        .key         = "properties",
        .value       = "merged",
        .mode        = flag_decode_mode::all_matches,
        .on_no_match = no_match_behavior::emit_none,
    },
};

inline constexpr std::array k_implicit_task{
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_initial),
        .type  = "flags",
        .key   = "kind",
        .value = "initial",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_task_flag_t::ompt_task_implicit),
        .type  = "flags",
        .key   = "kind",
        .value = "implicit",
        .mode  = flag_decode_mode::first_match,
    },
};

inline constexpr std::array k_cancel{
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_parallel),
        .type  = "ompt_cancel_flag_t",
        .key   = "construct",
        .value = "parallel",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_sections),
        .type  = "ompt_cancel_flag_t",
        .key   = "construct",
        .value = "sections",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_loop),
        .type  = "ompt_cancel_flag_t",
        .key   = "construct",
        .value = "loop",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_taskgroup),
        .type  = "ompt_cancel_flag_t",
        .key   = "construct",
        .value = "taskgroup",
        .mode  = flag_decode_mode::first_match,
    },

    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_activated),
        .type  = "ompt_cancel_flag_t",
        .key   = "state",
        .value = "activated",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_detected),
        .type  = "ompt_cancel_flag_t",
        .key   = "state",
        .value = "detected",
        .mode  = flag_decode_mode::first_match,
    },
    ompt_flag_rule{
        .mask  = static_cast<int>(ompt_cancel_flag_t::ompt_cancel_discarded_task),
        .type  = "ompt_cancel_flag_t",
        .key   = "state",
        .value = "discarded_task",
        .mode  = flag_decode_mode::first_match,
    },
};
}  // namespace rocprofsys::domains::callback::ompt::rules
