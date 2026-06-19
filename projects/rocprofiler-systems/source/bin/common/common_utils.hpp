// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/preset_registry.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace common_utils
{
/**
 * Result of translating command-line arguments for the argument parser.
 * Owns any translated strings so their lifetime covers the parse_args call.
 */
struct translated_args
{
    std::vector<char*>       argv_ptrs;  // non-owning pointers for parser
    std::vector<std::string> command;    // args after "--"
    std::vector<std::string> owned;      // RAII ownership of translated strings
};

/**
 * Translate legacy preset flags (e.g., --balanced -> --preset=balanced)
 * and deprecated flag aliases (e.g., --cputime -> --sample-cputime)
 * and split argv into parser args and command args (separated by "--").
 *
 * Also maps old flag names to new canonical names.
 * Flags with '=' values (e.g. --freq=100) are handled by splitting on '=' before lookup.
 */
[[nodiscard]] translated_args
translate_arguments(
    int argc, char** argv, preset_registry& registry,
    const std::unordered_map<std::string, std::string>& deprecated_flags = {});

/**
 * Export configuration to JSON file or stdout.
 */
void
export_config(const std::vector<std::string>&        current_env,
              const std::unordered_set<std::string>& initial_envs,
              const std::string& preset_name, std::string_view tool_name,
              const std::string& output_file = "");

/**
 * Run the shared post-parse validation sequence.
 * Called by both run and sample after argument parsing.
 */
void
run_post_parse_validation(std::string_view tool_name, struct domain_flag_state& state,
                          int verbose_level);

using help_group_names = std::vector<std::string>;
using help_topic_map   = std::map<std::string, help_group_names>;

struct domain_help_entry
{
    std::string              description;
    std::vector<std::string> flag_patterns;
};

using domain_help_map = std::map<std::string, domain_help_entry>;

const help_topic_map&
get_help_topic_map();

const domain_help_map&
get_domain_help_map();

void
print_compact_help(std::string_view tool_name, std::ostream& out = std::cout);

[[nodiscard]] bool
print_help_for_topic(const std::string& captured_help, std::string_view topic,
                     std::string_view tool_name, std::ostream& out = std::cout);

[[nodiscard]] bool
print_help_for_domain(const std::string& captured_help, std::string_view domain,
                      std::string_view tool_name, std::ostream& out = std::cout);

using related_topics_map = std::map<std::string_view, std::vector<std::string_view>>;

/**
 * Curated topic -> related topics mapping used by print_see_also().
 * Exposed for unit-test validation (every referenced topic must exist
 * in get_help_topic_map() or get_domain_help_map()).
 */
const related_topics_map&
get_related_topics_map();

/**
 * Print a "See also" footer for a given help topic, listing related
 * topics the user may want to consult next. Helps cross-topic
 * discoverability without changing where each flag is physically
 * registered.
 *
 * No output is emitted when @p topic has no curated relations.
 */
void
print_see_also(std::string_view topic, std::ostream& out = std::cout);

/**
 * Build a NUL-terminated `char*` array suitable for execvpe() / argv-style APIs.
 *
 * Returned pointers are non-owning and reference the internal buffers of `src`
 * via `std::string::data()`. The caller MUST keep `src` alive for the whole
 * lifetime of the returned vector and MUST NOT mutate `src` (any push_back /
 * resize that reallocates invalidates every pointer). Non-const reference is
 * required because `data()` only returns a writable pointer on a non-const
 * string.
 */
[[nodiscard]] std::vector<char*>
to_c_argv(std::vector<std::string>& src);

void
print_command(const std::vector<std::string>& argv, std::string_view prefix = {});

namespace detail
{
void
print_environment_impl(const std::vector<std::string>&              env,
                       const std::function<bool(std::string_view)>& is_updated,
                       bool include_general_vars, std::string_view prefix);
}  // namespace detail

template <typename UpdatedEnvsT>
void
print_environment(const std::vector<std::string>& env, const UpdatedEnvsT& updated_envs,
                  bool include_general_vars = false, std::string_view prefix = {})
{
    detail::print_environment_impl(
        env,
        [&](std::string_view key) {
            // Both std::string and std::string_view sets accept string_view in count().
            return updated_envs.count(typename UpdatedEnvsT::key_type{ key }) > 0;
        },
        include_general_vars, prefix);
}

template <typename ParserT>
std::string
capture_help_text(ParserT& parser)
{
    std::ostringstream oss;
    auto*              old_stream = parser.set_ostream(&oss);
    parser.print_help();
    parser.set_ostream(old_stream);
    return oss.str();
}

/**
 * Shared help dispatch: handles --help (compact), --help=<topic>, --help=all.
 * Returns the exit code the caller should use to terminate the program.
 */
template <typename ParserT>
[[nodiscard]] int
dispatch_help(ParserT& parser, std::string_view tool_name, int exit_code)
{
    std::string topic;
    if(parser.exists("help")) topic = parser.template get<std::string>("help");

    if(topic.empty())
    {
        print_compact_help(tool_name);
    }
    else if(topic == "all")
    {
        parser.print_help();
    }
    else
    {
        auto captured = capture_help_text(parser);

        if(print_help_for_domain(captured, topic, tool_name) ||
           print_help_for_topic(captured, topic, tool_name))
        {
            print_see_also(topic);
        }
        else
        {
            std::cerr << "[rocprof-sys] Unknown help topic '" << topic << "'.\n\n"
                      << "Available topics (use --help=<topic>):\n";

            std::cerr << "\n  Group topics:\n";
            for(const auto& [name, groups] : get_help_topic_map())
                std::cerr << "    " << name << "\n";

            std::cerr << "\n  Domain topics:\n";
            for(const auto& [name, info] : get_domain_help_map())
                std::cerr << "    " << name << "  - " << info.description << "\n";

            std::cerr << "\n  --help=all  Show all options\n";
        }
    }
    return exit_code;
}

}  // namespace common_utils
}  // namespace rocprofsys
