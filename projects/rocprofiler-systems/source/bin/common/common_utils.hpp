// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/preset_registry.hpp"

#include <nlohmann/json.hpp>

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

// ============================================================================
// Output directory utilities
// ============================================================================

/**
 * Gets the output directory from environment or returns default.
 */
std::string
get_output_directory(const char* env_var = "ROCPROFSYS_OUTPUT_PATH");

/**
 * Checks if a directory is writable.
 */
bool
check_directory_writable(const std::string& dir);

// ============================================================================
// Pre-execution info
// ============================================================================

/**
 * Prints pre-execution information including preset details and output location.
 * All output goes to stderr to keep stdout clean for structured output (--export-config).
 * Uses the registry from domain_flag_state to avoid re-reading JSON files.
 */
void
print_pre_execution_info(std::string_view tool_name, std::string_view preset_mode,
                         preset_registry& registry);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Strips a leading "--" prefix from a preset name if present.
 * Provides backwards compatibility with old --preset-name style flags.
 */
inline std::string
strip_flag_prefix(std::string_view name)
{
    if(name.size() > 2 && name.compare(0, 2, "--") == 0)
        return std::string{ name.substr(2) };
    return std::string{ name };
}

// ============================================================================
// Validation functions
// ============================================================================

/**
 * Checks if ROCm is available on the system.
 */
bool
check_rocm_available();

/**
 * Warns if ROCm is not available.
 */
void
warn_if_rocm_unavailable();

/**
 * Warns if the output directory is not writable.
 */
void
warn_if_output_not_writable(std::string_view tool_name);

/**
 * Validates the configuration for common issues.
 */
void
validate_configuration();

/**
 * Validate domain flag combinations and print warnings for potential conflicts.
 * @param gpu_enabled Whether --gpu flag was used
 * @param rocm_enabled Whether --rocm flag was used
 * @param cpu_enabled Whether --cpu flag was used
 * @param parallel_enabled Whether --parallel flag was used
 * @param preset_name The active preset name (empty if none)
 */
void
validate_domain_flags(bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                      bool parallel_enabled, std::string_view preset_name);

// ============================================================================
// Shared functions used by both rocprof-sys-run and rocprof-sys-sample
// ============================================================================

/**
 * Collect resolved ROCPROFSYS_* settings by comparing current env against initial env.
 * Returns only variables that were added or changed.
 */
std::map<std::string, std::string>
collect_resolved_settings(const std::vector<char*>&              current_env,
                          const std::unordered_set<std::string>& initial_envs);

/**
 * Export configuration to JSON file or stdout.
 * @param tool_name The tool name (e.g., "run", "sample") for metadata description.
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

// ============================================================================
// Topic-based help system
// ============================================================================

using help_group_names = std::vector<std::string>;
using help_topic_map   = std::map<std::string, help_group_names>;

struct domain_help_entry
{
    std::string              description;
    std::vector<std::string> flag_patterns;  // e.g. "--gpu", "--gpus", "-G"
};

using domain_help_map = std::map<std::string, domain_help_entry>;

/**
 * Returns the topic-to-group-name map for group-based help filtering.
 */
const help_topic_map&
get_help_topic_map();

/**
 * Returns the domain-to-flags map for domain-based help filtering.
 */
const domain_help_map&
get_domain_help_map();

/**
 * Print the compact help summary for bare --help.
 */
void
print_compact_help(std::string_view tool_name, std::ostream& os = std::cout);

/**
 * Extract and print sections matching a group-based topic from captured help text.
 * @return true if the topic was valid (even if no sections matched this tool)
 */
bool
print_help_for_topic(const std::string& captured_help, std::string_view topic,
                     std::string_view tool_name, std::ostream& os = std::cout);

/**
 * Extract and print argument lines matching a domain topic from captured help text.
 * @return true if the domain was valid
 */
bool
print_help_for_domain(const std::string& captured_help, std::string_view domain,
                      std::string_view tool_name, std::ostream& os = std::cout);

/**
 * Capture full help text from the parser into a string.
 */
template <typename ParserT>
std::string
capture_help_text(ParserT& parser)
{
    std::ostringstream ss;
    auto*              old = parser.set_ostream(&ss);
    parser.print_help();
    parser.set_ostream(old);
    return ss.str();
}

/**
 * Shared help dispatch: handles --help (compact), --help=<topic>, --help=all.
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
            // no value provided -bare --help
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

        // Try domain-based first, then group-based
        if(!print_help_for_domain(captured, topic, tool_name) &&
           !print_help_for_topic(captured, topic, tool_name))
        {
            std::cerr << "[rocprof-sys] Unknown help topic '" << topic << "'.\n\n"
                      << "Available topics (use --help=<topic>):\n";

            std::cerr << "\n  Group topics:\n";
            for(const auto& [t, _] : get_help_topic_map())
                std::cerr << "    " << t << "\n";

            std::cerr << "\n  Domain topics:\n";
            for(const auto& [d, info] : get_domain_help_map())
                std::cerr << "    " << d << "  - " << info.description << "\n";

            std::cerr << "\n  --help=all  Show all options\n";
        }
    }
    exit(exit_code);
}

}  // namespace common_utils
}  // namespace rocprofsys
