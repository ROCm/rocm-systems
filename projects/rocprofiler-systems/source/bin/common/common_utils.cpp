// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/common_utils.hpp"

#include "common/json_config.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace rocprofsys
{
namespace common_utils
{

std::string
get_output_directory(const char* env_var)
{
    const char* output_path = std::getenv(env_var);
    if(output_path && strlen(output_path) > 0) return std::string(output_path);

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

std::string
generate_preset_description(std::string_view preset_mode)
{
    auto normalized = strip_flag_prefix(preset_mode);

    // Load the raw JSON once — used for both metadata and tree display
    auto preset_dir = rocprofsys::preset_loader::find_preset_directory();
    if(preset_dir.empty()) return "";

    auto          filepath = preset_dir + "/" + normalized + ".json";
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return "";

    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(ifs);
    } catch(const nlohmann::json::exception&)
    {
        return "";
    }

    // Extract description from metadata
    auto meta        = rocprofsys::json_config::get_config_metadata(j);
    auto description = meta ? meta->description : "";

    // Build tree lines from JSON sections
    std::vector<std::string> lines;

    // Tracing
    if(j.contains("tracing"))
    {
        const auto& t     = j["tracing"];
        bool        on    = t.value("enabled", false);
        std::string entry = std::string("Tracing:         ") + (on ? "ON" : "OFF");
        if(on && t.contains("buffer_size_kb"))
        {
            auto kb = t["buffer_size_kb"].value("value", 0);
            if(kb >= 1024000)
                entry += " (buffer: " + std::to_string(kb / 1024000) + " GB)";
            else if(kb > 0)
                entry += " (buffer: " + std::to_string(kb) + " KB)";
        }
        lines.push_back(entry);
    }

    // Profiling
    if(j.contains("profiling"))
    {
        const auto& p     = j["profiling"];
        bool        on    = p.value("enabled", false);
        std::string entry = std::string("Profiling:       ") + (on ? "ON" : "OFF");
        if(on && p.contains("flat_profile") && p["flat_profile"].value("enabled", false))
            entry += " (flat profile)";
        lines.push_back(entry);
    }

    // Sampling
    if(j.contains("sampling"))
    {
        const auto& s     = j["sampling"];
        bool        on    = s.value("enabled", false);
        std::string entry = std::string("CPU Sampling:    ") + (on ? "ON" : "OFF");
        if(on && s.contains("frequency_hz"))
        {
            auto freq = s["frequency_hz"].value("value", 0);
            if(freq > 0) entry += " @ " + std::to_string(freq) + " Hz";
        }
        if(s.contains("cpus") && s["cpus"].value("value", "") == "none")
        {
            entry = "CPU Sampling:    Disabled (none)";
        }
        lines.push_back(entry);
    }

    // Domains: GPU
    if(j.contains("domains") && j["domains"].contains("gpu"))
    {
        const auto& gpu = j["domains"]["gpu"];
        if(gpu.value("enabled", false))
        {
            std::string entry = "GPU Metrics:     ON";
            if(gpu.contains("metrics"))
            {
                std::vector<std::string> names;
                for(const auto& [name, m] : gpu["metrics"].items())
                {
                    if(m.value("enabled", false)) names.push_back(name);
                }
                if(!names.empty())
                {
                    entry += " (";
                    for(size_t i = 0; i < names.size(); ++i)
                    {
                        if(i > 0) entry += ", ";
                        entry += names[i];
                    }
                    entry += ")";
                }
            }
            lines.push_back(entry);
        }
    }

    // Domains: ROCm
    if(j.contains("domains") && j["domains"].contains("rocm"))
    {
        const auto& rocm = j["domains"]["rocm"];
        if(rocm.value("enabled", false) && rocm.contains("api_domains"))
        {
            std::vector<std::string> apis;
            for(const auto& [name, api] : rocm["api_domains"].items())
            {
                if(api.value("enabled", false)) apis.push_back(name);
            }
            if(!apis.empty())
            {
                std::string entry = "ROCm Domains:    ";
                for(size_t i = 0; i < apis.size(); ++i)
                {
                    if(i > 0) entry += ", ";
                    entry += apis[i];
                }
                lines.push_back(entry);
            }
        }
    }

    // Domains: Parallel runtimes
    if(j.contains("domains") && j["domains"].contains("parallel"))
    {
        const auto& par = j["domains"]["parallel"];
        if(par.contains("runtimes"))
        {
            std::vector<std::string> runtimes;
            for(const auto& [name, rt] : par["runtimes"].items())
            {
                if(rt.value("enabled", false)) runtimes.push_back(name);
            }
            if(!runtimes.empty())
            {
                std::string entry = "Parallel:        ";
                for(size_t i = 0; i < runtimes.size(); ++i)
                {
                    if(i > 0) entry += ", ";
                    entry += runtimes[i];
                }
                lines.push_back(entry);
            }
        }
    }

    // Hardware counters
    if(j.contains("hardware_counters") && j["hardware_counters"].value("enabled", false))
    {
        const auto& hw = j["hardware_counters"];
        if(hw.contains("papi_events"))
        {
            auto val =
                rocprofsys::json_config::json_value_to_string(hw["papi_events"]["value"]);
            lines.push_back("PAPI Events:     " + val);
        }
        if(hw.contains("rocm_events"))
        {
            auto val =
                rocprofsys::json_config::json_value_to_string(hw["rocm_events"]["value"]);
            lines.push_back("ROCm Events:     " + val);
        }
    }

    // Output: rocPD
    if(j.contains("output") && j["output"].contains("rocpd_output") &&
       j["output"]["rocpd_output"].value("enabled", false))
    {
        lines.emplace_back("rocPD Output:    ON");
    }

    if(lines.empty()) return description;

    // Format with tree characters
    std::ostringstream oss;
    oss << description << "\n";
    for(size_t i = 0; i < lines.size(); ++i)
    {
        bool is_last = (i + 1 == lines.size());
        oss << "  " << (is_last ? "\u2514\u2500 " : "\u251c\u2500 ") << lines[i];
        if(!is_last) oss << "\n";
    }
    return oss.str();
}

void
print_pre_execution_info(std::string_view tool_name, std::string_view preset_mode)
{
    auto output_dir = get_output_directory();

    if(!preset_mode.empty() && !tool_name.empty())
    {
        constexpr size_t           box_width       = 60;
        constexpr size_t           box_inner_width = box_width - 2;
        constexpr std::string_view box_line =
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
            "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550";
        constexpr std::string_view prefix       = "ROCm Systems Profiler - ";
        const size_t               content_size = prefix.size() + tool_name.size();
        const size_t               padding =
            content_size < box_inner_width ? box_inner_width - content_size : 0;

        std::cout << "\n"
                  << "\u2554" << box_line << "\u2557\n"
                  << "\u2551 " << prefix << tool_name << std::string(padding, ' ')
                  << " \u2551\n"
                  << "\u255a" << box_line << "\u255d\n"
                  << "\n";

        std::cout << "Preset:        " << preset_mode << "\n";

        auto description = generate_preset_description(preset_mode);
        if(!description.empty())
        {
            std::cout << "\n" << description << "\n";
        }
    }

    std::cout << "\nOutput:        " << output_dir << "\n";

    if(!check_directory_writable(output_dir))
    {
        std::cerr << "\nWARNING: Output directory may not be writable!\n";
        std::cerr << "   Try: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- <command>\n\n";
    }

    std::cout << "\nResults will be available in:\n"
              << "  \u2022 Text profile:  " << output_dir << "/wall_clock.txt\n"
              << "  \u2022 Trace (visual): " << output_dir << "/perfetto-trace.proto\n"
              << "  \u2022 JSON data:      " << output_dir << "/wall_clock.json\n"
              << "\nTo visualize trace:\n"
              << "  Open " << output_dir
              << "/perfetto-trace.proto in https://ui.perfetto.dev\n"
              << "\n";
}

bool
validate_preset_modes(const std::vector<std::string>& active_presets)
{
    if(active_presets.size() > 1)
    {
        std::cerr << "\nERROR: Multiple preset modes specified: ";
        for(const auto& active_preset : active_presets)
        {
            std::cerr << active_preset;
            if(active_preset != active_presets.back()) std::cerr << ", ";
        }
        std::cerr << "\n\n";

        std::cerr << "Only ONE preset mode can be used at a time.\n\n";
        std::cerr
            << "Available presets (use with --preset=<name>):\n"
            << "  General Purpose:\n"
            << "    balanced           Balanced profiling with moderate overhead\n"
            << "    profile-only       Profiling without tracing, minimal overhead\n"
            << "    detailed           Full trace + hardware counters\n"
            << "  Workload-Specific:\n"
            << "    trace-hpc          MPI/OpenMP/HPC applications\n"
            << "    workload-trace     General compute workloads (AI/ML, HPC, etc.)\n"
            << "    trace-gpu          GPU workload analysis\n"
            << "    trace-openmp       OpenMP offload workloads\n"
            << "    profile-mpi        MPI communication latency profiling\n"
            << "    trace-hw-counters  Hardware counter collection\n"
            << "  API Tracing:\n"
            << "    sys-trace          Comprehensive system API tracing\n"
            << "    runtime-trace      Runtime API tracing (no compiler/HSA)\n\n";

        std::cerr
            << "Choose one preset or use manual options for custom configuration.\n";
        std::cerr << "See --help for all options.\n\n";

        return false;
    }
    return true;
}

bool
check_rocm_available()
{
    // Check ROCM_PATH first, then fall back to default /opt/rocm
    const char* rocm_path = std::getenv("ROCM_PATH");
    if(rocm_path && std::strlen(rocm_path) > 0)
    {
        auto hipconfig = std::string(rocm_path) + "/bin/hipconfig";
        if(access(hipconfig.c_str(), X_OK) == 0) return true;
    }
    return (access("/opt/rocm/bin/hipconfig", X_OK) == 0);
}

void
warn_if_rocm_unavailable()
{
    if(!check_rocm_available())
    {
        std::cerr << "\nWARNING: GPU tracing requested but ROCm is not available\n\n";
        std::cerr << "GPU features will be disabled.\n\n";
    }
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

void
list_presets(std::string_view tool_name)
{
    auto presets = rocprofsys::preset_loader::load_all_presets();
    if(presets.empty())
    {
        std::cerr << "[rocprof-sys] No presets found. Check ROCPROFSYS_PRESET_DIR "
                     "or installation.\n";
        return;
    }

    std::cout << "\nAvailable Presets:\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Group presets by category
    std::map<std::string, std::vector<const rocprofsys::preset_loader::preset_info*>>
        by_category;
    for(const auto& [name, info] : presets)
    {
        auto cat = info.category.empty() ? "General" : info.category;
        by_category[cat].push_back(&info);
    }

    for(const auto& [category, preset_list] : by_category)
    {
        std::cout << category << ":\n";
        for(const auto* info : preset_list)
        {
            std::cout << "  " << info->name;
            if(!info->description.empty()) std::cout << " - " << info->description;
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Usage: rocprof-sys-" << tool_name << " --preset=<name> -- ./app\n";
    std::cout << "       rocprof-sys-" << tool_name
              << " --explain=<name>  # Show preset details\n";
}

bool
explain_preset(std::string_view preset_name, std::string_view tool_name)
{
    auto info =
        rocprofsys::preset_loader::load_preset_or_file(std::string{ preset_name });
    if(!info)
    {
        std::cerr << "[rocprof-sys] Preset '" << preset_name
                  << "' not found. Use --list-presets to see available presets.\n";
        return false;
    }

    std::cout << "\nPreset: " << info->name << "\n";
    std::cout << std::string(40, '-') << "\n";
    if(!info->description.empty())
        std::cout << "Description: " << info->description << "\n";
    if(!info->use_case.empty()) std::cout << "Use case:    " << info->use_case << "\n";
    if(!info->category.empty()) std::cout << "Category:    " << info->category << "\n";

    std::cout << "\nEnvironment Variables:\n";
    for(const auto& [key, val] : info->settings)
    {
        std::cout << "  " << key << " = " << val << "\n";
    }

    std::cout << "\nUsage: rocprof-sys-" << tool_name << " --preset=" << preset_name
              << " -- ./app\n";
    return true;
}

void
validate_domain_flags(bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                      bool parallel_enabled, std::string_view preset_name)
{
    // Warn if --cpu is used with a preset that disables sampling
    if(cpu_enabled && !preset_name.empty())
    {
        static const std::vector<std::string> no_sampling_presets = {
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

    // Warn if --rocm is used without --gpu (may want GPU metrics too)
    if(rocm_enabled && !gpu_enabled)
    {
        std::cerr << "[rocprof-sys][note] --rocm enables ROCm API tracing. Consider "
                     "adding --gpu for GPU metrics.\n";
    }

    // Warn if --parallel is used without ROCm tracing (may miss GPU collective ops)
    if(parallel_enabled && !rocm_enabled)
    {
        std::cerr << "[rocprof-sys][note] --parallel enables MPI/OpenMP profiling. "
                     "Consider adding --rocm for GPU collective tracing.\n";
    }

    // Warn if multiple domain flags are used without a preset
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
                          bool parallel_enabled, int verbose_level)
{
    // Check ROCm availability once for all relevant conditions
    bool rocm_needed = gpu_enabled || rocm_enabled;
    if(!rocm_needed && !preset_name.empty())
    {
        static const std::vector<std::string> gpu_presets = {
            "workload-trace", "trace-hpc",    "sys-trace",        "runtime-trace",
            "trace-gpu",      "trace-openmp", "trace-hw-counters"
        };
        for(const auto& preset : gpu_presets)
        {
            if(preset_name == preset)
            {
                rocm_needed = true;
                break;
            }
        }
    }
    if(rocm_needed) warn_if_rocm_unavailable();

    if(!preset_name.empty() && verbose_level >= 1)
    {
        print_pre_execution_info(tool_name, preset_name);
    }

    warn_if_output_not_writable(tool_name);
    validate_configuration(tool_name);
    validate_domain_flags(gpu_enabled, rocm_enabled, cpu_enabled, parallel_enabled,
                          preset_name);
}

}  // namespace common_utils
}  // namespace rocprofsys
