// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/preset_registry.hpp"

#include "common/env_vars.hpp"
#include "embedded_presets.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <sstream>

namespace rocprofsys
{

namespace
{
std::string
find_preset_directory()
{
    const auto* preset_dir_env = std::getenv(env_vars::PRESET_DIR.data());
    if(preset_dir_env && std::strlen(preset_dir_env) > 0)
    {
        auto dir = std::string{ preset_dir_env };
        if(common::path::exists(dir)) return dir;
    }

    auto root = common::path::get_rocprofsys_root();
    if(!root.empty())
    {
        auto candidate =
            common::join('/', root, "share", "rocprofiler-systems", "presets");
        if(common::path::exists(candidate)) return candidate;
    }

    const auto* rocm_path = std::getenv("ROCM_PATH");
    if(rocm_path && std::strlen(rocm_path) > 0)
    {
        auto candidate = common::join('/', std::string{ rocm_path }, "share",
                                      "rocprofiler-systems", "presets");
        if(common::path::exists(candidate)) return candidate;
    }

    return {};
}

/// Collect names of enabled items from a JSON object (items with "enabled": true).
std::string
collect_enabled_names(const nlohmann::json& obj)
{
    std::string result;
    for(const auto& [name, val] : obj.items())
    {
        if(val.value("enabled", false))
        {
            if(!result.empty()) result += ", ";
            result += name;
        }
    }
    return result;
}

preset_registry::preset_info
extract_metadata(const nlohmann::json& content)
{
    preset_registry::preset_info info;
    if(content.contains("metadata"))
    {
        const auto& meta = content["metadata"];
        if(meta.contains("name")) info.name = meta["name"].get<std::string>();
        if(meta.contains("cli_flag")) info.cli_flag = meta["cli_flag"].get<std::string>();
        if(meta.contains("description"))
            info.description = meta["description"].get<std::string>();
        if(meta.contains("use_case")) info.use_case = meta["use_case"].get<std::string>();
        if(meta.contains("category")) info.category = meta["category"].get<std::string>();
    }
    info.settings = json_config::resolve_config(content);
    return info;
}
}  // namespace

preset_registry::preset_registry()
: m_directory{ find_preset_directory() }
{
    load_embedded();
}

void
preset_registry::load_embedded()
{
    for(size_t i = 0; i < embedded_presets::num_presets; ++i)
    {
        const auto& entry = embedded_presets::presets[i];
        try
        {
            auto       j    = nlohmann::json::parse(entry.json);
            auto       info = extract_metadata(j);
            const auto name = std::string{ entry.name };

            m_json_cache[name] = std::move(j);
            m_presets[name]    = std::move(info);
        } catch(const nlohmann::json::exception& e)
        {
            std::cerr << "[rocprof-sys] WARNING: Failed to parse embedded preset '"
                      << entry.name << "': " << e.what() << '\n';
        }
    }
}

std::string
preset_registry::translate_legacy_flag(std::string_view arg) const
{
    // Must start with "--" and not contain "="
    if(arg.size() <= 2 || arg.compare(0, 2, "--") != 0 ||
       arg.find('=') != std::string_view::npos)
        return {};

    auto name = std::string{ arg.substr(2) };
    if(m_presets.count(name) == 0) return {};

    std::cerr << "[rocprof-sys] WARNING: '" << arg
              << "' is deprecated. Use '--preset=" << name << "' instead.\n";
    return "--preset=" + name;
}

std::optional<preset_registry::preset_info>
preset_registry::load_file(const std::string& filepath)
{
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return std::nullopt;

    try
    {
        auto j    = nlohmann::json::parse(ifs);
        auto info = extract_metadata(j);

        // Cache the raw JSON — move into primary key, copy to secondary if needed
        m_json_cache[filepath] = std::move(j);
        if(!info.name.empty() && info.name != filepath)
            m_json_cache[info.name] = m_json_cache[filepath];

        return info;
    } catch(const nlohmann::json::exception& e)
    {
        std::cerr << "[rocprof-sys] WARNING: Failed to parse preset '" << filepath
                  << "': " << e.what() << '\n';
        return std::nullopt;
    }
}

std::string
preset_registry::resolve_filepath(const std::string& name_or_path)
{
    // Reject any input containing ".." to prevent path traversal
    if(name_or_path.find("..") != std::string::npos)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset path '" << name_or_path
                  << "' contains '..'. Ignoring.\n";
        return {};
    }

    const bool is_path = name_or_path.find('/') != std::string::npos ||
                         (name_or_path.size() > 5 &&
                          name_or_path.compare(name_or_path.size() - 5, 5, ".json") == 0);

    if(is_path)
    {
        // Explicit file path — validate existence via realpath
        auto resolved = common::path::realpath(name_or_path);
        if(resolved.empty())
        {
            std::cerr << "[rocprof-sys] WARNING: Preset file '" << name_or_path
                      << "' not found.\n";
            return {};
        }
        return resolved;
    }

    // Bare preset name — resolve within the preset directory
    if(m_directory.empty()) return {};

    auto filepath  = common::join('/', m_directory, name_or_path + ".json");
    auto resolved  = common::path::realpath(filepath);
    auto canon_dir = common::path::realpath(m_directory);
    if(resolved.empty() || canon_dir.empty() ||
       resolved.compare(0, canon_dir.size(), canon_dir) != 0)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset path '" << filepath
                  << "' resolves outside preset directory. Ignoring.\n";
        return {};
    }

    return filepath;
}

std::optional<preset_registry::preset_info>
preset_registry::find(const std::string& name_or_path)
{
    auto it = m_presets.find(name_or_path);
    if(it != m_presets.end()) return it->second;

    auto filepath = resolve_filepath(name_or_path);
    if(filepath.empty()) return std::nullopt;

    auto info = load_file(filepath);
    if(!info) return std::nullopt;

    m_presets[name_or_path] = std::move(*info);
    return m_presets[name_or_path];
}

std::optional<env_settings>
preset_registry::get_settings(const std::string& name_or_path)
{
    auto info = find(name_or_path);
    if(!info) return std::nullopt;
    return info->settings;
}

void
preset_registry::ensure_all_loaded()
{
    if(m_all_loaded) return;
    m_all_loaded = true;

    if(m_directory.empty()) return;

    auto dir_closer = [](DIR* d) { closedir(d); };
    auto dir_guard  = std::unique_ptr<DIR, decltype(dir_closer)>(
        opendir(m_directory.c_str()), dir_closer);
    if(!dir_guard) return;

    errno = 0;
    while(auto* entry = readdir(dir_guard.get()))
    {
        std::string_view filename{ entry->d_name };

        constexpr std::string_view json_ext = ".json";
        if(filename.size() <= json_ext.size() ||
           filename.compare(filename.size() - json_ext.size(), json_ext.size(),
                            json_ext) != 0)
            continue;

        if(filename == "schema.json") continue;

        const auto preset_name =
            std::string{ filename.substr(0, filename.size() - json_ext.size()) };

        // Skip if preset is already cached (e.g. from embedded presets)
        if(m_presets.count(preset_name) > 0) continue;

        const auto filepath = common::join('/', m_directory, std::string{ filename });
        if(auto info = load_file(filepath)) m_presets[preset_name] = std::move(*info);

        errno = 0;
    }

    if(errno != 0)
    {
        std::cerr << "[rocprof-sys] WARNING: Error reading preset directory '"
                  << m_directory << "': " << std::strerror(errno) << '\n';
    }

    if(errno != 0)
    {
        std::cerr << "[rocprof-sys] WARNING: Error reading preset directory '"
                  << m_directory << "': " << std::strerror(errno) << "\n";
    }
}

const std::map<std::string, preset_registry::preset_info>&
preset_registry::all()
{
    ensure_all_loaded();
    return m_presets;
}

bool
preset_registry::is_section_enabled(std::string_view preset_name,
                                    std::string_view section, bool default_value) const
{
    auto it = m_json_cache.find(std::string{ preset_name });
    if(it == m_json_cache.end()) return default_value;

    const auto& json = it->second;
    if(!json.contains(section)) return default_value;
    return json[std::string{ section }].value("enabled", default_value);
}

void
preset_registry::list(std::string_view tool_name, std::ostream& os)
{
    const auto& presets = all();
    if(presets.empty())
    {
        std::cerr << "[rocprof-sys] No presets found. Check ROCPROFSYS_PRESET_DIR "
                     "or installation.\n";
        return;
    }

    os << "\nAvailable Presets:\n";
    os << std::string(60, '=') << "\n\n";

    std::map<std::string, std::vector<const preset_info*>> by_category;
    for(const auto& [name, info] : presets)
    {
        auto cat = info.category.empty() ? "General" : info.category;
        by_category[cat].push_back(&info);
    }

    for(const auto& [category, preset_list] : by_category)
    {
        os << category << ":\n";
        for(const auto* info : preset_list)
        {
            os << "  " << info->name;
            if(!info->description.empty()) os << " - " << info->description;
            os << "\n";
        }
        os << "\n";
    }

    os << "Usage: rocprof-sys-" << tool_name << " --preset=<name> -- ./app\n";
    os << "       rocprof-sys-" << tool_name
       << " --explain=<name>  # Show preset details\n";
}

bool
preset_registry::explain(std::string_view preset_name, std::string_view tool_name,
                         std::ostream& os)
{
    auto info = find(std::string{ preset_name });
    if(!info)
    {
        std::cerr << "[rocprof-sys] Preset '" << preset_name
                  << "' not found. Use --list-presets to see available presets.\n";
        return false;
    }

    os << "\nPreset: " << info->name << "\n";
    os << std::string(40, '-') << "\n";
    if(!info->description.empty()) os << "Description: " << info->description << "\n";
    if(!info->use_case.empty()) os << "Use case:    " << info->use_case << "\n";
    if(!info->category.empty()) os << "Category:    " << info->category << "\n";

    os << "\nEnvironment Variables:\n";
    for(const auto& [key, val] : info->settings)
    {
        os << "  " << key << " = " << val << "\n";
    }

    os << "\nUsage: rocprof-sys-" << tool_name << " --preset=" << preset_name
       << " -- ./app\n";
    return true;
}

std::string
preset_registry::describe(std::string_view preset_name)
{
    auto cache_it = m_json_cache.find(std::string{ preset_name });
    if(cache_it == m_json_cache.end())
    {
        // Trigger load
        if(!find(std::string{ preset_name }).has_value()) return "";
        cache_it = m_json_cache.find(std::string{ preset_name });
        if(cache_it == m_json_cache.end()) return "";
    }
    const auto& preset_json = cache_it->second;

    auto meta        = json_config::get_config_metadata(preset_json);
    auto description = meta ? meta->description : "";

    std::vector<std::string> lines;

    // Tracing
    if(preset_json.contains("tracing"))
    {
        const auto& tracing = preset_json["tracing"];
        bool        enabled = tracing.value("enabled", false);
        std::string entry   = std::string("Tracing:         ") + (enabled ? "ON" : "OFF");
        if(enabled && tracing.contains("buffer_size_kb"))
        {
            constexpr int KB_PER_GB = 1024 * 1024;
            auto          buffer_kb = tracing["buffer_size_kb"].value("value", 0);
            if(buffer_kb >= KB_PER_GB)
                entry += " (buffer: " + std::to_string(buffer_kb / KB_PER_GB) + " GB)";
            else if(buffer_kb > 0)
                entry += " (buffer: " + std::to_string(buffer_kb) + " KB)";
        }
        lines.push_back(entry);
    }

    // Profiling
    if(preset_json.contains("profiling"))
    {
        const auto& profiling = preset_json["profiling"];
        bool        enabled   = profiling.value("enabled", false);
        std::string entry = std::string("Profiling:       ") + (enabled ? "ON" : "OFF");
        if(enabled && profiling.contains("flat_profile") &&
           profiling["flat_profile"].value("enabled", false))
            entry += " (flat profile)";
        lines.push_back(entry);
    }

    // Sampling
    if(preset_json.contains("sampling"))
    {
        const auto& sampling = preset_json["sampling"];
        bool        enabled  = sampling.value("enabled", false);
        std::string entry = std::string("CPU Sampling:    ") + (enabled ? "ON" : "OFF");
        if(enabled && sampling.contains("frequency_hz"))
        {
            auto freq = sampling["frequency_hz"].value("value", 0);
            if(freq > 0) entry += " @ " + std::to_string(freq) + " Hz";
        }
        if(sampling.contains("cpus") && sampling["cpus"].value("value", "") == "none")
        {
            entry = "CPU Sampling:    Disabled (none)";
        }
        lines.push_back(entry);
    }

    // Domains: GPU
    if(preset_json.contains("domains") && preset_json["domains"].contains("gpu"))
    {
        const auto& gpu = preset_json["domains"]["gpu"];
        if(gpu.value("enabled", false))
        {
            std::string entry = "GPU Metrics:     ON";
            if(gpu.contains("metrics"))
            {
                auto names = collect_enabled_names(gpu["metrics"]);
                if(!names.empty()) entry += " (" + names + ")";
            }
            lines.push_back(entry);
        }
    }

    // Domains: ROCm
    if(preset_json.contains("domains") && preset_json["domains"].contains("rocm"))
    {
        const auto& rocm = preset_json["domains"]["rocm"];
        if(rocm.value("enabled", false) && rocm.contains("api_domains"))
        {
            auto apis = collect_enabled_names(rocm["api_domains"]);
            if(!apis.empty()) lines.push_back("ROCm Domains:    " + apis);
        }
    }

    // Domains: Parallel runtimes
    if(preset_json.contains("domains") && preset_json["domains"].contains("parallel"))
    {
        const auto& parallel = preset_json["domains"]["parallel"];
        if(parallel.contains("runtimes"))
        {
            auto runtime_names = collect_enabled_names(parallel["runtimes"]);
            if(!runtime_names.empty())
                lines.push_back("Parallel:        " + runtime_names);
        }
    }

    // Hardware counters
    if(preset_json.contains("hardware_counters") &&
       preset_json["hardware_counters"].value("enabled", false))
    {
        const auto& counters = preset_json["hardware_counters"];
        if(counters.contains("papi_events"))
        {
            auto events =
                json_config::json_value_to_string(counters["papi_events"]["value"]);
            lines.push_back("PAPI Events:     " + events);
        }
        if(counters.contains("rocm_events"))
        {
            auto events =
                json_config::json_value_to_string(counters["rocm_events"]["value"]);
            lines.push_back("ROCm Events:     " + events);
        }
    }

    // Output: rocPD
    if(preset_json.contains("output") && preset_json["output"].contains("rocpd_output") &&
       preset_json["output"]["rocpd_output"].value("enabled", false))
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

}  // namespace rocprofsys
