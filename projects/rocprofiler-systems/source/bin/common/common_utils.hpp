// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/preset_registry.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace common_utils
{
/**
 * Thrown by argument actions that require immediate program termination
 * (e.g., --list-presets, --explain, --help). Caught at the parse_args
 * call site to exit gracefully with proper RAII cleanup.
 */
struct cli_done
{
    int exit_code;
    explicit cli_done(int code) noexcept
    : exit_code(code)
    {}
};

/**
 * Result of translating command-line arguments for the argument parser.
 * Owns any translated strings so their lifetime covers the parse_args call.
 */
struct translated_args
{
    std::vector<char*>       argv_ptrs;  // non-owning pointers for parser
    std::vector<char*>       command;    // args after "--"
    std::vector<std::string> owned;      // RAII ownership of translated strings
};

/**
 * Translate legacy preset flags (e.g., --balanced -> --preset=balanced)
 * and split argv into parser args and command args (separated by "--").
 */
[[nodiscard]] translated_args
translate_arguments(int argc, char** argv, preset_registry& registry);

/**
 * Export configuration to JSON file or stdout.
 */
void
export_config(const std::vector<char*>&              current_env,
              const std::unordered_set<std::string>& initial_envs,
              const std::string& preset_name, std::string_view tool_name,
              const std::string& output_file = "");

/**
 * Run the shared post-parse validation sequence.
 * Called by both run and sample after argument parsing.
 */
void
run_post_parse_validation(std::string_view tool_name, std::string_view preset_name,
                          bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                          bool parallel_enabled, int verbose_level,
                          preset_registry& registry);

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
print_compact_help(std::string_view tool_name, std::ostream& os = std::cout);

bool
print_help_for_topic(const std::string& captured_help, std::string_view topic,
                     std::string_view tool_name, std::ostream& os = std::cout);

bool
print_help_for_domain(const std::string& captured_help, std::string_view domain,
                      std::string_view tool_name, std::ostream& os = std::cout);

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
 * @throws cli_done after printing help output.
 */
template <typename ParserT>
void
dispatch_help(ParserT& parser, std::string_view tool_name, int exit_code)
{
    std::string topic;
    if(parser.exists("help"))
    {
        try
        {
            topic = parser.template get<std::string>("help");
        } catch(...)
        {
            // no value provided — bare --help
        }
    }

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

        if(!print_help_for_domain(captured, topic, tool_name) &&
           !print_help_for_topic(captured, topic, tool_name))
        {
            std::cerr << "[rocprof-sys] Unknown help topic '" << topic << "'.\n\n"
                      << "Available topics (use --help=<topic>):\n";

            std::cerr << "\n  Group topics:\n";
            for(const auto& [name, _] : get_help_topic_map())
                std::cerr << "    " << name << "\n";

            std::cerr << "\n  Domain topics:\n";
            for(const auto& [name, info] : get_domain_help_map())
                std::cerr << "    " << name << "  - " << info.description << "\n";

            std::cerr << "\n  --help=all  Show all options\n";
        }
    }
    throw cli_done{ exit_code };
}

inline void
warn_if_output_not_writable(std::string_view tool_name)
{
    auto output_dir = get_output_directory();
    if(!check_directory_writable(output_dir))
    {
        std::cerr << "[rocprof-sys][WARNING] Output directory '" << output_dir
                  << "' is not writable!\n";
        std::cerr << "  Try: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- <command>\n";
    }
}

inline void
validate_configuration(std::string_view tool_name)
{
    // Check for conflicting ENABLE/DISABLE categories (causes std::abort() at runtime)
    const char* enable_cats  = std::getenv("ROCPROFSYS_ENABLE_CATEGORIES");
    const char* disable_cats = std::getenv("ROCPROFSYS_DISABLE_CATEGORIES");
    if(enable_cats && std::strlen(enable_cats) > 0 && disable_cats &&
       std::strlen(disable_cats) > 0)
    {
        std::cerr << "[rocprof-sys][warning] Both ROCPROFSYS_ENABLE_CATEGORIES and "
                     "ROCPROFSYS_DISABLE_CATEGORIES are set.\n"
                  << "  This will cause an abort at runtime. Use only one.\n"
                  << "  ROCPROFSYS_ENABLE_CATEGORIES=" << enable_cats << "\n"
                  << "  ROCPROFSYS_DISABLE_CATEGORIES=" << disable_cats << "\n";
    }

    // Check ROCPROFSYS_TMPDIR writability
    const char* tmpdir     = std::getenv("ROCPROFSYS_TMPDIR");
    auto        tmpdir_str = std::string{ tmpdir ? tmpdir : "/tmp" };
    if(!check_directory_writable(tmpdir_str))
    {
        std::cerr << "[rocprof-sys][WARNING] Temp directory '" << tmpdir_str
                  << "' is not writable!\n"
                  << "  Try: export ROCPROFSYS_TMPDIR=/tmp\n";
    }

    (void) tool_name;
}

}  // namespace common_utils
}  // namespace rocprofsys
