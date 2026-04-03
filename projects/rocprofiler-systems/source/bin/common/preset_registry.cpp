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
#include <sstream>

namespace rocprofsys
{

namespace
{
std::string
find_preset_directory()
{
    const auto* preset_dir_env = std::getenv(std::string{ env_vars::PRESET_DIR }.c_str());
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
            auto        j = nlohmann::json::parse(entry.json);
            preset_info info;

            if(j.contains("metadata"))
            {
                const auto& meta = j["metadata"];
                if(meta.contains("name")) info.name = meta["name"].get<std::string>();
                if(meta.contains("cli_flag"))
                    info.cli_flag = meta["cli_flag"].get<std::string>();
                if(meta.contains("description"))
                    info.description = meta["description"].get<std::string>();
                if(meta.contains("use_case"))
                    info.use_case = meta["use_case"].get<std::string>();
                if(meta.contains("category"))
                    info.category = meta["category"].get<std::string>();
            }

            info.settings = json_config::resolve_config(j);

            auto name          = std::string{ entry.name };
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
    if(arg.size() <= 2 || arg.substr(0, 2) != "--" ||
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
        auto        j = nlohmann::json::parse(ifs);
        preset_info info;

        if(j.contains("metadata"))
        {
            const auto& meta = j["metadata"];
            if(meta.contains("name")) info.name = meta["name"].get<std::string>();
            if(meta.contains("cli_flag"))
                info.cli_flag = meta["cli_flag"].get<std::string>();
            if(meta.contains("description"))
                info.description = meta["description"].get<std::string>();
            if(meta.contains("use_case"))
                info.use_case = meta["use_case"].get<std::string>();
            if(meta.contains("category"))
                info.category = meta["category"].get<std::string>();
        }

        info.settings = json_config::resolve_config(j);

        // Cache the raw JSON by preset name (if available) and by filepath
        if(!info.name.empty()) m_json_cache[info.name] = j;
        m_json_cache[filepath] = std::move(j);

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
    // If it looks like a path, return as-is
    if(name_or_path.find('/') != std::string::npos ||
       (name_or_path.size() > 5 &&
        name_or_path.compare(name_or_path.size() - 5, 5, ".json") == 0))
    {
        return name_or_path;
    }

    // Reject bare preset names containing ".."
    if(name_or_path.find("..") != std::string::npos)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset name '" << name_or_path
                  << "' contains '..'. Ignoring.\n";
        return {};
    }

    if(m_directory.empty()) return {};

    auto filepath = common::join('/', m_directory, name_or_path + ".json");

    auto resolved  = common::path::realpath(filepath);
    auto canon_dir = common::path::realpath(m_directory);
    if(resolved.empty() || canon_dir.empty() ||
       resolved.substr(0, canon_dir.size()) != canon_dir)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset path '" << filepath
                  << "' resolves outside preset directory. Ignoring.\n";
        return {};
    }

    return filepath;
}

const preset_registry::preset_info*
preset_registry::find(const std::string& name_or_path)
{
    // Check if is cached first
    auto it = m_presets.find(name_or_path);
    if(it != m_presets.end()) return &it->second;

    auto filepath = resolve_filepath(name_or_path);
    if(filepath.empty()) return nullptr;

    auto info = load_file(filepath);
    if(!info) return nullptr;

    m_presets[name_or_path] = std::move(*info);
    return &m_presets[name_or_path];
}

void
preset_registry::ensure_all_loaded()
{
    if(m_all_loaded) return;
    m_all_loaded = true;

    if(m_directory.empty()) return;

    auto* dir = opendir(m_directory.c_str());
    if(!dir) return;

    errno = 0;
    while(auto* entry = readdir(dir))
    {
        std::string_view filename{ entry->d_name };

        constexpr std::string_view json_ext = ".json";
        if(filename.size() <= json_ext.size() ||
           filename.substr(filename.size() - json_ext.size()) != json_ext)
            continue;

        if(filename == "schema.json") continue;

        auto preset_name =
            std::string{ filename.substr(0, filename.size() - json_ext.size()) };

        // Skip if preset is already cached (e.g. from embedded presets)
        if(m_presets.count(preset_name) > 0) continue;

        auto filepath = common::join('/', m_directory, std::string{ filename });
        if(auto info = load_file(filepath)) m_presets[preset_name] = std::move(*info);

        errno = 0;
    }

    if(errno != 0)
    {
        std::cerr << "[rocprof-sys] WARNING: Error reading preset directory '"
                  << m_directory << "': " << std::strerror(errno) << '\n';
    }

    closedir(dir);
}

const std::map<std::string, preset_registry::preset_info>&
preset_registry::all()
{
    ensure_all_loaded();
    return m_presets;
}

const nlohmann::json*
preset_registry::raw_json(const std::string& name)
{
    auto it = m_json_cache.find(name);
    if(it != m_json_cache.end()) return &it->second;

    // Trigger a load (which populates both m_presets and m_json_cache)
    if(find(name) != nullptr)
    {
        it = m_json_cache.find(name);
        if(it != m_json_cache.end()) return &it->second;
    }
    return nullptr;
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
    const auto* info = find(std::string{ preset_name });
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
    const auto* j = raw_json(std::string{ preset_name });
    if(!j) return "";

    auto meta        = json_config::get_config_metadata(*j);
    auto description = meta ? meta->description : "";

    std::vector<std::string> lines;

    // Tracing
    if(j->contains("tracing"))
    {
        const auto& t     = (*j)["tracing"];
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
    if(j->contains("profiling"))
    {
        const auto& p     = (*j)["profiling"];
        bool        on    = p.value("enabled", false);
        std::string entry = std::string("Profiling:       ") + (on ? "ON" : "OFF");
        if(on && p.contains("flat_profile") && p["flat_profile"].value("enabled", false))
            entry += " (flat profile)";
        lines.push_back(entry);
    }

    // Sampling
    if(j->contains("sampling"))
    {
        const auto& s     = (*j)["sampling"];
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
    if(j->contains("domains") && (*j)["domains"].contains("gpu"))
    {
        const auto& gpu = (*j)["domains"]["gpu"];
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
    if(j->contains("domains") && (*j)["domains"].contains("rocm"))
    {
        const auto& rocm = (*j)["domains"]["rocm"];
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
    if(j->contains("domains") && (*j)["domains"].contains("parallel"))
    {
        const auto& par = (*j)["domains"]["parallel"];
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
    if(j->contains("hardware_counters") &&
       (*j)["hardware_counters"].value("enabled", false))
    {
        const auto& hw = (*j)["hardware_counters"];
        if(hw.contains("papi_events"))
        {
            auto val = json_config::json_value_to_string(hw["papi_events"]["value"]);
            lines.push_back("PAPI Events:     " + val);
        }
        if(hw.contains("rocm_events"))
        {
            auto val = json_config::json_value_to_string(hw["rocm_events"]["value"]);
            lines.push_back("ROCm Events:     " + val);
        }
    }

    // Output: rocPD
    if(j->contains("output") && (*j)["output"].contains("rocpd_output") &&
       (*j)["output"]["rocpd_output"].value("enabled", false))
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
