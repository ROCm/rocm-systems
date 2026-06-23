// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/common_utils.hpp"

#include "common/domain_flag_state.hpp"
#include "common/env_vars.hpp"
#include "common/json_config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace rocprofsys
{
namespace common_utils
{

namespace
{
constexpr std::string_view rocprofsys_prefix = "ROCPROFSYS";

bool
starts_with_rocprofsys(std::string_view entry) noexcept
{
    return entry.compare(0, rocprofsys_prefix.size(), rocprofsys_prefix) == 0;
}

[[nodiscard]] std::string_view
env_key(std::string_view entry) noexcept
{
    const auto eq_pos = entry.find('=');
    return (eq_pos == std::string_view::npos) ? std::string_view{}
                                              : entry.substr(0, eq_pos);
}
}  // namespace

std::vector<char*>
to_c_argv(std::vector<std::string>& src)
{
    std::vector<char*> out;
    out.reserve(src.size() + 1);
    for(auto& entry : src)
        out.emplace_back(entry.data());
    out.emplace_back(nullptr);
    return out;
}

void
print_command(const std::vector<std::string>& argv, std::string_view prefix)
{
    auto cmd = std::accumulate(argv.begin(), argv.end(), std::string{},
                               [](std::string acc, const std::string& arg) {
                                   if(!acc.empty()) acc += ' ';
                                   acc += arg;
                                   return acc;
                               });
    if(cmd.empty()) return;
    std::cerr << prefix << "Executing '" << cmd << "'...\n" << std::flush;
}

namespace detail
{
void
print_environment_impl(const std::vector<std::string>&              env,
                       const std::function<bool(std::string_view)>& is_updated_key,
                       bool include_general_vars, std::string_view prefix)
{
    std::vector<std::string_view> entries;
    entries.reserve(env.size());
    std::copy(env.begin(), env.end(), std::back_inserter(entries));
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());

    auto is_updated = [&](std::string_view entry) {
        return is_updated_key(env_key(entry));
    };
    auto is_general = [&](std::string_view entry) {
        return !is_updated(entry) && starts_with_rocprofsys(entry);
    };

    const bool has_updated = std::any_of(entries.begin(), entries.end(), is_updated);
    const bool has_general =
        include_general_vars && std::any_of(entries.begin(), entries.end(), is_general);
    if(!has_updated && !has_general) return;

    auto emit_matching = [&](auto pred) {
        for(const auto& entry : entries)
            if(pred(entry)) std::cerr << prefix << entry << '\n';
    };

    std::cerr << '\n';
    if(include_general_vars) emit_matching(is_general);
    emit_matching(is_updated);
    std::cerr << std::flush;
}
}  // namespace detail

static std::string
strip_flag_prefix(std::string_view name)
{
    if(name.size() > 2 && name.compare(0, 2, "--") == 0)
        return std::string{ name.substr(2) };
    return std::string{ name };
}

translated_args
translate_arguments(int argc, char** argv, preset_registry& registry,
                    const std::unordered_map<std::string, std::string>& deprecated_flags)
{
    translated_args result;
    bool            past_separator = false;

    for(int arg_idx = 0; arg_idx < argc; ++arg_idx)
    {
        if(argv[arg_idx] == nullptr) continue;

        if(past_separator)
        {
            result.command.emplace_back(argv[arg_idx]);
        }
        else if(std::string_view{ argv[arg_idx] } == "--")
        {
            past_separator = true;
        }
        else
        {
            // deprecated aliases checked before preset translation to avoid
            // double-mapping
            if(!deprecated_flags.empty())
            {
                auto        arg_sv = std::string_view{ argv[arg_idx] };
                std::string flag_name;
                std::string eq_suffix;

                auto eq_pos = arg_sv.find('=');
                if(eq_pos != std::string_view::npos)
                {
                    flag_name = std::string{ arg_sv.substr(0, eq_pos) };
                    eq_suffix = std::string{ arg_sv.substr(eq_pos) };
                }
                else
                {
                    flag_name = std::string{ arg_sv };
                }

                auto match = deprecated_flags.find(flag_name);
                if(match != deprecated_flags.end())
                {
                    std::cerr << "[rocprof-sys] WARNING: '" << flag_name
                              << "' is deprecated. Use '" << match->second
                              << "' instead.\n";
                    result.owned.push_back(match->second + eq_suffix);
                    result.argv_ptrs.emplace_back(result.owned.back().data());
                    continue;
                }
            }

            auto translated = registry.translate_legacy_flag(argv[arg_idx]);
            if(!translated.empty())
            {
                result.owned.push_back(std::move(translated));
                result.argv_ptrs.emplace_back(result.owned.back().data());
            }
            else
            {
                result.argv_ptrs.emplace_back(argv[arg_idx]);
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
    if(stat(dir.c_str(), &st) == 0) return access(dir.c_str(), W_OK) == 0;

    std::string candidate = dir;
    while(true)
    {
        const auto pos = candidate.find_last_of('/');
        if(pos == std::string::npos)
        {
            candidate = ".";
            break;
        }
        candidate = candidate.substr(0, pos);
        if(candidate.empty())
        {
            candidate = "/";
            break;
        }
        if(stat(candidate.c_str(), &st) == 0) break;
    }

    return access(candidate.c_str(), W_OK) == 0;
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
        for(size_t col = 0; col < box_width; ++col)
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
        std::cerr << "[rocprof-sys][WARNING] Both " << env_vars::ENABLE_CATEGORIES
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
collect_resolved_settings(const std::vector<std::string>&        current_env,
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

    for(const auto& env_entry : current_env)
    {
        std::string_view entry{ env_entry };
        auto             eq_pos = entry.find('=');
        if(eq_pos == std::string_view::npos) continue;

        std::string key(entry.substr(0, eq_pos));
        std::string val(entry.substr(eq_pos + 1));

        if(key.find("ROCPROFSYS_") != 0) continue;

        auto match = initial_map.find(key);
        if(match == initial_map.end() || match->second != val)
        {
            result[key] = val;
        }
    }
    return result;
}

void
export_config(const std::vector<std::string>&        current_env,
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
run_post_parse_validation(std::string_view tool_name, domain_flag_state& state,
                          int verbose_level)
{
    if(!state.active_preset_name.empty() && verbose_level >= 1)
    {
        print_pre_execution_info(tool_name, state.active_preset_name, state.registry);
    }

    warn_if_output_not_writable(tool_name);
    validate_configuration();
    validate_domain_flags(state.gpu_domain_enabled, state.rocm_domain_enabled,
                          state.cpu_domain_enabled, state.parallel_domain_enabled,
                          state.active_preset_name);
}

// ============================================================================
// Topic-based help system
// ============================================================================

namespace
{
std::string
strip_ansi(const std::string& text)
{
    std::string result;
    result.reserve(text.size());
    bool in_escape = false;
    for(char ch : text)
    {
        if(in_escape)
        {
            if(ch == 'm') in_escape = false;
            continue;
        }
        if(ch == '\033')
        {
            in_escape = true;
            continue;
        }
        result += ch;
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
        { "output", { "[OUTPUT FORMAT OPTIONS]" } },
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
              "--process-duration", "-G", "--gpu-events", "--ai-nics", "--use-amd-smi",
              "--amd-smi-metrics" } } },
        { "cpu",
          { "CPU sampling, timers, CPU counters",
            { "--cpu", "-H", "--host", "-S", "--sample", "--sampling-freq",
              "--sampling-wait", "--sampling-duration", "-t", "--tids",
              "--sample-cputime", "--sample-realtime", "--sample-overflow", "-C",
              "--cpu-events" } } },
        { "rocm",
          { "ROCm API tracing options",
            { "--rocm", "-T", "--trace", "--hsa-interrupt", "--selected-regions",
              "--use-amd-smi", "--gpus", "--ai-nics" } } },
        { "parallel",
          { "MPI, OpenMP, Kokkos, RCCL options",
            { "--parallel", "-I", "--include", "-E", "--exclude" } } },
    };
    return map;
}

// Hand-curated topic relations. Listing a topic here surfaces it under
// the "See also" footer of another topic. Keep the per-topic list short
// (≤4 entries) so the footer stays useful rather than noisy.
const related_topics_map&
get_related_topics_map()
{
    static const related_topics_map map = {
        { "tracing", { "rocm", "process", "backend", "output" } },
        { "profiling", { "sampling", "counters", "backend", "output" } },
        { "output", { "tracing", "profiling", "backend" } },
        { "sampling", { "process", "profiling", "counters", "cpu" } },
        { "process", { "gpu", "sampling" } },
        { "counters", { "gpu", "cpu", "sampling" } },
        { "backend", { "tracing", "profiling", "rocm" } },
        { "general", { "preset", "tracing", "profiling" } },
        { "preset", { "tracing", "profiling", "sampling" } },
        { "misc", { "debug", "general" } },
        { "gpu", { "rocm", "process", "counters" } },
        { "cpu", { "sampling", "process", "counters" } },
        { "rocm", { "gpu", "tracing", "parallel" } },
        { "parallel", { "rocm", "sampling" } },
    };
    return map;
}

void
print_see_also(std::string_view topic, std::ostream& out)
{
    const auto& relations = get_related_topics_map();
    auto        it        = relations.find(topic);
    if(it == relations.end() || it->second.empty()) return;

    out << "\n  See also (related topics):\n";
    for(const auto& related : it->second)
        out << "    --help=" << related << "\n";
}

void
print_compact_help(std::string_view tool_name, std::ostream& out)
{
    out << "Usage: rocprof-sys-" << tool_name << " [OPTIONS] -- <command> [args...]\n"
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
    if(tool_name == "run") out << "  -S, --sample           Enable/disable sampling\n";
    out << "  --export-config[=FILE] Export resolved config as JSON\n"
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
                     std::string_view tool_name, std::ostream& out)
{
    const auto& topic_map = get_help_topic_map();
    auto        match     = topic_map.find(std::string{ topic });
    if(match == topic_map.end()) return false;

    // Build set of target header strings
    std::set<std::string> target_headers(match->second.begin(), match->second.end());

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

    for(size_t line_idx = 0; line_idx < lines.size(); ++line_idx)
    {
        std::string bracket_name;
        if(is_section_header(lines[line_idx], bracket_name))
        {
            if(sections.empty())
                preamble_end = line_idx;
            else
                sections.back().end = line_idx;
            sections.push_back({ line_idx, lines.size(), bracket_name });
        }
    }

    out << "rocprof-sys-" << tool_name << " --help=" << topic << "\n\n";

    // Print matching sections
    bool found = false;
    for(const auto& sec : sections)
    {
        if(target_headers.count(sec.header) > 0)
        {
            found = true;
            for(size_t line_idx = sec.start; line_idx < sec.end; ++line_idx)
                out << lines[line_idx] << '\n';
        }
    }

    if(!found)
    {
        out << "\n[rocprof-sys] No options found for topic '" << topic
            << "' in rocprof-sys-" << tool_name << ".\n";
    }
    return true;
}

bool
print_help_for_domain(const std::string& captured, std::string_view domain,
                      std::string_view tool_name, std::ostream& out)
{
    const auto& domain_map = get_domain_help_map();
    auto        match      = domain_map.find(std::string{ domain });
    if(match == domain_map.end()) return false;

    const auto& entry = match->second;

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
    out << upper_domain << " OPTIONS (" << entry.description << ")\n\n";

    // Skip lines before "Options:" to avoid matching flags in the usage summary
    size_t options_start = 0;
    for(size_t line_idx = 0; line_idx < lines.size(); ++line_idx)
    {
        auto stripped = strip_ansi(lines[line_idx]);
        auto trimmed  = stripped.find_first_not_of(" \t");
        if(trimmed != std::string::npos && stripped.substr(trimmed).find("Options:") == 0)
        {
            options_start = line_idx + 1;
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
            if(in_match) out << '\n';
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
            out << current_line << '\n';
        }
    }

    if(!found_any)
    {
        out << "  No " << domain << " options found in rocprof-sys-" << tool_name
            << ".\n";
    }

    return true;
}

}  // namespace common_utils
}  // namespace rocprofsys
