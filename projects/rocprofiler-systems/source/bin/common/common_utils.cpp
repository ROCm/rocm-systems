// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/common_utils.hpp"

#include "common/env_vars.hpp"
#include "common/json_config.hpp"

#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace rocprofsys
{
namespace common_utils
{

static std::string
strip_flag_prefix(std::string_view name)
{
    if(name.size() > 2 && name.compare(0, 2, "--") == 0)
        return std::string{ name.substr(2) };
    return std::string{ name };
}

translated_args
translate_arguments(int argc, char** argv, preset_registry& registry)
{
    translated_args result;
    bool            past_separator = false;

    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == nullptr) continue;

        if(past_separator)
        {
            result.command.emplace_back(argv[i]);
        }
        else if(std::string_view{ argv[i] } == "--")
        {
            past_separator = true;
        }
        else
        {
            auto translated = registry.translate_legacy_flag(argv[i]);
            if(!translated.empty())
            {
                result.owned.push_back(std::move(translated));
                result.argv_ptrs.emplace_back(result.owned.back().data());
            }
            else
            {
                result.argv_ptrs.emplace_back(argv[i]);
            }
        }
    }
    return result;
}

std::string
get_output_directory(const char* env_var = nullptr)
{
    const char* output_path =
        std::getenv(env_var ? env_var : env_vars::OUTPUT_PATH.data());
    if(output_path && std::strlen(output_path) > 0) return std::string(output_path);

    return "rocprof-sys-output";
}

bool
check_directory_writable(const std::string& dir)
{
    struct stat st;
    if(stat(dir.c_str(), &st) == 0)
    {
        return (access(dir.c_str(), W_OK) == 0);
    }

    std::string parent = dir;
    size_t      pos    = parent.find_last_of('/');
    if(pos != std::string::npos)
    {
        parent = parent.substr(0, pos);
        if(parent.empty()) parent = ".";
    }
    else
    {
        parent = ".";
    }

    return (access(parent.c_str(), W_OK) == 0);
}

void
print_pre_execution_info(std::string_view tool_name, std::string_view preset_mode,
                         preset_registry& registry)
{
    auto output_dir = get_output_directory();

    bool tracing_on   = true;
    bool profiling_on = true;

    if(!preset_mode.empty() && !tool_name.empty())
    {
        auto normalized = strip_flag_prefix(preset_mode);
        tracing_on      = registry.is_section_enabled(normalized, "tracing", true);
        profiling_on    = registry.is_section_enabled(normalized, "profiling", true);

        constexpr size_t box_width       = 60;
        constexpr size_t box_inner_width = box_width - 2;
        // Build the box line from Unicode box-drawing character U+2550 (═)
        std::string box_line;
        box_line.reserve(box_width * 3);
        for(size_t i = 0; i < box_width; ++i)
            box_line += "\u2550";

        constexpr std::string_view prefix       = "ROCm Systems Profiler - ";
        const size_t               content_size = prefix.size() + tool_name.size();
        const size_t               padding =
            content_size < box_inner_width ? box_inner_width - content_size : 0;

        std::cerr << "\n"
                  << "\u2554" << box_line << "\u2557\n"
                  << "\u2551 " << prefix << tool_name << std::string(padding, ' ')
                  << " \u2551\n"
                  << "\u255a" << box_line << "\u255d\n"
                  << "\n";

        std::cerr << "Preset:        " << preset_mode << "\n";

        auto description = registry.describe(normalized);
        if(!description.empty())
        {
            std::cerr << "\n" << description << "\n";
        }
    }

    std::cerr << "\nOutput:        " << output_dir << "\n";

    if(!check_directory_writable(output_dir))
    {
        std::cerr << "\nWARNING: Output directory may not be writable!\n";
        std::cerr << "   Try: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- <command>\n\n";
    }

    std::cerr << "\nResults will be available in:\n";
    if(profiling_on)
    {
        std::cerr << "  \u2022 Text profile:  " << output_dir << "/wall_clock.txt\n"
                  << "  \u2022 JSON data:      " << output_dir << "/wall_clock.json\n";
    }
    if(tracing_on)
    {
        std::cerr << "  \u2022 Trace (visual): " << output_dir
                  << "/perfetto-trace.proto\n";
    }
    if(tracing_on)
    {
        std::cerr << "\nTo visualize trace:\n"
                  << "  Open " << output_dir
                  << "/perfetto-trace.proto in https://ui.perfetto.dev\n";
    }
    std::cerr << "\n";
}

void
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

void
validate_configuration()
{
    // Check for conflicting ENABLE/DISABLE categories (causes std::abort() at runtime)
    const char* enable_cats  = std::getenv(env_vars::ENABLE_CATEGORIES.data());
    const char* disable_cats = std::getenv(env_vars::DISABLE_CATEGORIES.data());
    if(enable_cats && std::strlen(enable_cats) > 0 && disable_cats &&
       std::strlen(disable_cats) > 0)
    {
        std::cerr << "[rocprof-sys][warning] Both " << env_vars::ENABLE_CATEGORIES
                  << " and " << env_vars::DISABLE_CATEGORIES << " are set.\n"
                  << "  This will cause an abort at runtime. Use only one.\n"
                  << "  " << env_vars::ENABLE_CATEGORIES << "=" << enable_cats << "\n"
                  << "  " << env_vars::DISABLE_CATEGORIES << "=" << disable_cats << "\n";
    }

    // Check ROCPROFSYS_TMPDIR writability
    const char* tmpdir     = std::getenv(env_vars::TMPDIR.data());
    auto        tmpdir_str = std::string{ tmpdir ? tmpdir : "/tmp" };
    if(!check_directory_writable(tmpdir_str))
    {
        std::cerr << "[rocprof-sys][WARNING] Temp directory '" << tmpdir_str
                  << "' is not writable!\n"
                  << "  Try: export " << env_vars::TMPDIR << "=/tmp\n";
    }
}

void
validate_domain_flags(bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                      bool parallel_enabled, std::string_view preset_name)
{
    if(cpu_enabled && !preset_name.empty())
    {
        static constexpr std::array<std::string_view, 4> no_sampling_presets = {
            "trace-gpu", "trace-openmp", "workload-trace", "trace-hpc"
        };
        for(const auto& preset : no_sampling_presets)
        {
            if(preset_name == preset)
            {
                std::cerr << "[rocprof-sys][note] --cpu flag used with '" << preset_name
                          << "' preset which disables CPU sampling.\n"
                          << "  The --cpu flag will override the preset's sampling "
                             "settings.\n";
                break;
            }
        }
    }

    if(rocm_enabled && !gpu_enabled)
    {
        std::cerr << "[rocprof-sys][note] --rocm enables ROCm API tracing. Consider "
                     "adding --gpu for GPU metrics.\n";
    }

    if(parallel_enabled && !rocm_enabled)
    {
        std::cerr << "[rocprof-sys][note] --parallel enables MPI/OpenMP profiling. "
                     "Consider adding --rocm for GPU collective tracing.\n";
    }

    int domain_count = (gpu_enabled ? 1 : 0) + (rocm_enabled ? 1 : 0) +
                       (cpu_enabled ? 1 : 0) + (parallel_enabled ? 1 : 0);
    if(domain_count >= 3 && preset_name.empty())
    {
        std::cerr << "[rocprof-sys][note] Multiple domain flags specified. Consider "
                     "using a preset like --preset=detailed for comprehensive "
                     "profiling.\n";
    }
}

std::map<std::string, std::string>
collect_resolved_settings(const std::vector<char*>&              current_env,
                          const std::unordered_set<std::string>& initial_envs)
{
    std::map<std::string, std::string> result;

    // Build a map of initial env vars for efficient lookup
    std::unordered_map<std::string, std::string> initial_map;
    for(const auto& env_str : initial_envs)
    {
        auto eq_pos = env_str.find('=');
        if(eq_pos != std::string::npos)
        {
            initial_map[env_str.substr(0, eq_pos)] = env_str.substr(eq_pos + 1);
        }
    }

    for(const auto* env_entry : current_env)
    {
        if(env_entry == nullptr) continue;

        std::string_view entry(env_entry);
        auto             eq_pos = entry.find('=');
        if(eq_pos == std::string_view::npos) continue;

        std::string key(entry.substr(0, eq_pos));
        std::string val(entry.substr(eq_pos + 1));

        if(key.find("ROCPROFSYS_") != 0) continue;

        auto it = initial_map.find(key);
        if(it == initial_map.end() || it->second != val)
        {
            result[key] = val;
        }
    }
    return result;
}

void
export_config(const std::vector<char*>&              current_env,
              const std::unordered_set<std::string>& initial_envs,
              const std::string& preset_name, std::string_view tool_name,
              const std::string& output_file)
{
    auto settings = collect_resolved_settings(current_env, initial_envs);
    auto json_str =
        rocprofsys::json_config::export_config_as_json(settings, preset_name, tool_name);

    if(output_file.empty())
    {
        std::cout << json_str << '\n';
    }
    else
    {
        std::ofstream ofs(output_file);
        if(ofs.is_open())
        {
            ofs << json_str << '\n';
            std::cerr << "[rocprof-sys] Configuration exported to: " << output_file
                      << '\n';
        }
        else
        {
            std::cerr << "[rocprof-sys] ERROR: Could not write to: " << output_file
                      << '\n';
        }
    }
}

void
run_post_parse_validation(std::string_view tool_name, std::string_view preset_name,
                          bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                          bool parallel_enabled, int verbose_level,
                          preset_registry& registry)
{
    if(!preset_name.empty() && verbose_level >= 1)
    {
        print_pre_execution_info(tool_name, preset_name, registry);
    }

    warn_if_output_not_writable(tool_name);
    validate_configuration();
    validate_domain_flags(gpu_enabled, rocm_enabled, cpu_enabled, parallel_enabled,
                          preset_name);
}

// ============================================================================
// Topic-based help system
// ============================================================================

namespace
{
std::string
strip_ansi(const std::string& s)
{
    std::string result;
    result.reserve(s.size());
    bool in_escape = false;
    for(char c : s)
    {
        if(in_escape)
        {
            if(c == 'm') in_escape = false;
            continue;
        }
        if(c == '\033')
        {
            in_escape = true;
            continue;
        }
        result += c;
    }
    return result;
}

bool
is_section_header(const std::string& line, std::string& bracket_name)
{
    auto stripped = strip_ansi(line);
    auto first    = stripped.find_first_not_of(" \t");
    if(first == std::string::npos) return false;
    stripped = stripped.substr(first);
    if(stripped.empty() || stripped.front() != '[') return false;
    // Find the closing bracket -the bracket name ends at the first ']'
    auto close = stripped.find(']');
    if(close == std::string::npos) return false;
    bracket_name = stripped.substr(0, close + 1);
    return true;
}

bool
line_contains_flag(const std::string& line, const std::string& flag)
{
    auto stripped = strip_ansi(line);
    // Flag lines have leading whitespace then the flag name
    auto pos = stripped.find(flag);
    if(pos == std::string::npos) return false;
    // Verify it's a word boundary (not a substring of a longer flag)
    auto end = pos + flag.size();
    if(end < stripped.size())
    {
        char next = stripped[end];
        if(next != ' ' && next != ',' && next != '=' && next != '\t' && next != '[')
            return false;
    }
    return true;
}
}  // namespace

const help_topic_map&
get_help_topic_map()
{
    static const help_topic_map map = {
        { "preset", { "[PRESET OPTIONS]", "[DOMAIN OPTIONS]", "[EXPORT OPTIONS]" } },
        { "general", { "[GENERAL OPTIONS]" } },
        { "tracing", { "[TRACING OPTIONS]" } },
        { "profiling", { "[PROFILE OPTIONS]" } },
        { "sampling",
          { "[GENERAL SAMPLING OPTIONS]", "[SAMPLING TIMER OPTIONS]",
            "[ADVANCED SAMPLING OPTIONS]" } },
        { "process", { "[HOST/DEVICE (PROCESS SAMPLING) OPTIONS]" } },
        { "counters", { "[HARDWARE COUNTER OPTIONS]" } },
        { "backend", { "[BACKEND OPTIONS]" } },
        { "debug", { "[DEBUG OPTIONS]" } },
        { "execution", { "[EXECUTION OPTIONS]" } },
        { "misc", { "[MISCELLANEOUS OPTIONS]" } },
    };
    return map;
}

const domain_help_map&
get_domain_help_map()
{
    static const domain_help_map map = {
        { "gpu",
          { "GPU metrics, device sampling, GPU counters",
            { "--gpu", "-D", "--device", "--gpus", "--process-freq", "--process-wait",
              "--process-duration", "-G", "--gpu-events", "--ai-nics" } } },
        { "cpu",
          { "CPU sampling, timers, CPU counters",
            { "--cpu", "-H", "--host", "-S", "--sample", "-f", "--freq",
              "--sampling-freq", "--sampling-wait", "--sampling-duration", "-t", "--tids",
              "--cputime", "--sample-cputime", "--realtime", "--sample-realtime",
              "--sample-overflow", "-C", "--cpu-events" } } },
        { "rocm",
          { "ROCm API tracing options",
            { "--rocm", "-T", "--trace", "--hsa-interrupt" } } },
        { "parallel",
          { "MPI, OpenMP, Kokkos, RCCL options",
            { "--parallel", "-I", "--include", "-E", "--exclude" } } },
    };
    return map;
}

void
print_compact_help(std::string_view tool_name, std::ostream& os)
{
    os << "Usage: rocprof-sys-" << tool_name << " [OPTIONS] -- <command> [args...]\n"
       << "\n"
       << "QUICK START\n"
       << "  --preset=NAME          Load a profiling preset\n"
       << "  --list-presets         Show all available presets\n"
       << "  --explain=NAME         Show detailed info about a preset\n"
       << "\n"
       << "DOMAIN FLAGS (composable with presets)\n"
       << "  --gpu[=metrics]        GPU metrics (temp, power, busy, mem_usage)\n"
       << "  --rocm[=domains]       ROCm API tracing (hip, kernel, memory, hsa)\n"
       << "  --cpu[=hz]             CPU call-stack sampling (default 100 Hz)\n"
       << "  --parallel[=runtimes]  Parallel runtimes (mpi, openmp, kokkos, rccl)\n"
       << "\n"
       << "COMMON OPTIONS\n"
       << "  -o, --output PATH      Output directory\n"
       << "  -T, --trace            Enable/disable Perfetto tracing\n"
       << "  -P, --profile          Enable/disable call-stack profiling\n";
    if(tool_name == "run") os << "  -S, --sample           Enable/disable sampling\n";
    os << "  --export-config[=FILE] Export resolved config as JSON\n"
       << "  -v, --verbose          Increase verbosity\n"
       << "\n"
       << "HELP TOPICS (use --help=<topic> for details)\n"
       << "\n"
       << "  Group topics:\n"
       << "    all          Full help output (all options)\n"
       << "    preset       Preset, domain, and export options\n"
       << "    general      General options (output, trace, profile)\n"
       << "    tracing      Tracing-specific options\n"
       << "    profiling    Profile output format options\n"
       << "    sampling     Sampling frequency and timer options\n"
       << "    process      Host/device process sampling options\n"
       << "    counters     Hardware counter options (CPU/GPU events)\n"
       << "    backend      Backend options (include/exclude)\n"
       << "    debug        Debug, logging, and verbosity options\n"
       << "    misc         Miscellaneous options\n"
       << "\n"
       << "  Domain topics:\n"
       << "    gpu          GPU metrics, device sampling, GPU counters\n"
       << "    cpu          CPU sampling, timers, CPU counters\n"
       << "    rocm         ROCm API tracing options\n"
       << "    parallel     MPI, OpenMP, Kokkos, RCCL options\n"
       << "\n"
       << "EXAMPLES\n"
       << "  rocprof-sys-" << tool_name << " --preset=balanced -- ./myapp\n"
       << "  rocprof-sys-" << tool_name << " --preset=trace-hpc --rocm -- ./hpc_app\n"
       << "  rocprof-sys-" << tool_name << " --gpu=temp,power --cpu=50 -- ./myapp\n";
}

bool
print_help_for_topic(const std::string& captured, std::string_view topic,
                     std::string_view tool_name, std::ostream& os)
{
    const auto& topic_map = get_help_topic_map();
    auto        it        = topic_map.find(std::string{ topic });
    if(it == topic_map.end()) return false;

    // Build set of target header strings
    std::set<std::string> target_headers(it->second.begin(), it->second.end());

    // Split captured text into lines
    std::istringstream       iss(captured);
    std::string              line;
    std::vector<std::string> lines;
    while(std::getline(iss, line))
        lines.push_back(line);

    // Find section boundaries
    struct Section
    {
        size_t      start;
        size_t      end;
        std::string header;
    };
    size_t               preamble_end = lines.size();
    std::vector<Section> sections;

    for(size_t i = 0; i < lines.size(); ++i)
    {
        std::string bracket_name;
        if(is_section_header(lines[i], bracket_name))
        {
            if(sections.empty())
                preamble_end = i;
            else
                sections.back().end = i;
            sections.push_back({ i, lines.size(), bracket_name });
        }
    }

    os << "rocprof-sys-" << tool_name << " --help=" << topic << "\n\n";

    // Print matching sections
    bool found = false;
    for(const auto& sec : sections)
    {
        if(target_headers.count(sec.header) > 0)
        {
            found = true;
            for(size_t i = sec.start; i < sec.end; ++i)
                os << lines[i] << '\n';
        }
    }

    if(!found)
    {
        os << "\n[rocprof-sys] No options found for topic '" << topic
           << "' in rocprof-sys-" << tool_name << ".\n";
    }
    return true;
}

bool
print_help_for_domain(const std::string& captured, std::string_view domain,
                      std::string_view tool_name, std::ostream& os)
{
    const auto& domain_map = get_domain_help_map();
    auto        it         = domain_map.find(std::string{ domain });
    if(it == domain_map.end()) return false;

    const auto& entry = it->second;

    // Split captured text into lines
    std::istringstream       iss(captured);
    std::string              line;
    std::vector<std::string> lines;
    while(std::getline(iss, line))
        lines.push_back(line);

    // Print header
    std::string upper_domain{ domain };
    for(auto& c : upper_domain)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    os << upper_domain << " OPTIONS (" << entry.description << ")\n\n";

    // Skip lines before "Options:" to avoid matching flags in the usage summary
    size_t options_start = 0;
    for(size_t i = 0; i < lines.size(); ++i)
    {
        auto stripped = strip_ansi(lines[i]);
        auto trimmed  = stripped.find_first_not_of(" \t");
        if(trimmed != std::string::npos && stripped.substr(trimmed).find("Options:") == 0)
        {
            options_start = i + 1;
            break;
        }
    }

    // Find argument lines matching any of the domain's flag patterns.
    // An argument may span multiple lines (description continuation),
    // so also include non-flag continuation lines that follow a match.
    bool found_any = false;
    bool in_match  = false;
    for(size_t idx = options_start; idx < lines.size(); ++idx)
    {
        const auto& current_line = lines[idx];
        auto        stripped     = strip_ansi(current_line);
        auto        first        = stripped.find_first_not_of(" \t");

        // Skip separators and empty lines at the top
        if(first == std::string::npos)
        {
            if(in_match) os << '\n';
            in_match = false;
            continue;
        }

        // Check if this is a section header -skip it
        if(stripped[first] == '[')
        {
            in_match = false;
            continue;
        }

        // Check if this line starts a new argument (has - prefix after indent)
        bool is_arg_line = (stripped[first] == '-');

        if(is_arg_line)
        {
            // Check if any flag pattern matches this line
            in_match = false;
            for(const auto& flag : entry.flag_patterns)
            {
                if(line_contains_flag(current_line, flag))
                {
                    in_match  = true;
                    found_any = true;
                    break;
                }
            }
        }
        // else: continuation line -keep in_match state

        if(in_match)
        {
            os << current_line << '\n';
        }
    }

    if(!found_any)
    {
        os << "  No " << domain << " options found in rocprof-sys-" << tool_name << ".\n";
    }

    return true;
}

}  // namespace common_utils
}  // namespace rocprofsys
