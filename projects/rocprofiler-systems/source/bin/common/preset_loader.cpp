// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/preset_loader.hpp"

#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>

namespace rocprofsys
{
namespace preset_loader
{

std::string
find_preset_directory()
{
    const auto* preset_dir_env = std::getenv("ROCPROFSYS_PRESET_DIR");
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

std::optional<preset_info>
load_preset_file(const std::string& filepath)
{
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return std::nullopt;

    try
    {
        auto        j = nlohmann::json::parse(ifs);
        preset_info info;

        // Extract metadata from schema format
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

        // Resolve settings using json_config resolver
        info.settings = json_config::resolve_config(j);

        return info;
    } catch(const nlohmann::json::exception& e)
    {
        std::cerr << "[rocprof-sys] WARNING: Failed to parse preset '" << filepath
                  << "': " << e.what() << '\n';
        return std::nullopt;
    }
}

std::map<std::string, preset_info>
load_all_presets()
{
    std::map<std::string, preset_info> presets;

    auto preset_dir = find_preset_directory();
    if(preset_dir.empty()) return presets;

    auto* dir = opendir(preset_dir.c_str());
    if(!dir) return presets;

    while(auto* entry = readdir(dir))
    {
        std::string_view filename{ entry->d_name };

        // Skip non-.json files
        constexpr std::string_view json_ext = ".json";
        if(filename.size() <= json_ext.size() ||
           filename.substr(filename.size() - json_ext.size()) != json_ext)
            continue;

        // Skip schema.json
        if(filename == "schema.json") continue;

        auto preset_name =
            std::string{ filename.substr(0, filename.size() - json_ext.size()) };
        auto filepath = common::join('/', preset_dir, std::string{ filename });
        if(auto info = load_preset_file(filepath))
            presets[preset_name] = std::move(*info);
    }

    closedir(dir);
    return presets;
}

std::optional<preset_info>
load_preset(std::string_view name)
{
    auto preset_dir = find_preset_directory();
    if(preset_dir.empty()) return std::nullopt;

    auto filepath = common::join('/', preset_dir, std::string{ name } + ".json");

    // Verify the resolved path stays within the preset directory
    auto resolved  = common::path::realpath(filepath);
    auto canon_dir = common::path::realpath(preset_dir);
    if(resolved.empty() || canon_dir.empty() ||
       resolved.substr(0, canon_dir.size()) != canon_dir)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset path '" << filepath
                  << "' resolves outside preset directory. Ignoring.\n";
        return std::nullopt;
    }

    return load_preset_file(filepath);
}

std::optional<preset_info>
load_preset_or_file(const std::string& name_or_path)
{
    // If it looks like a path (contains '/' or ends with '.json'), load it directly as a
    // user-specified file. This intentionally allows loading arbitrary JSON config files
    // from any location — it is a feature for custom configurations. Path traversal
    // protection in load_preset() only applies to bare preset names resolved against the
    // preset directory.
    if(name_or_path.find('/') != std::string::npos ||
       (name_or_path.size() > 5 &&
        name_or_path.substr(name_or_path.size() - 5) == ".json"))
    {
        return load_preset_file(name_or_path);
    }

    // Reject bare preset names containing ".." as a defense-in-depth measure
    if(name_or_path.find("..") != std::string::npos)
    {
        std::cerr << "[rocprof-sys] WARNING: Preset name '" << name_or_path
                  << "' contains '..'. Ignoring.\n";
        return std::nullopt;
    }

    // Otherwise look up by preset name
    return load_preset(name_or_path);
}

void
apply_preset_to_environment(const preset_info& info, bool override_existing)
{
    for(const auto& [env_var, value] : info.settings)
    {
        if(!override_existing)
        {
            const auto* existing = std::getenv(env_var.c_str());
            if(existing && std::strlen(existing) > 0) continue;
        }
        setenv(env_var.c_str(), value.c_str(), 1);
    }
}

void
print_preset_summary(const preset_info& info, std::ostream& os)
{
    os << "[rocprof-sys] Applying preset: " << info.name << "\n";
    if(!info.description.empty()) os << "  Description: " << info.description << "\n";
    if(!info.use_case.empty()) os << "  Use case: " << info.use_case << "\n";
    os << "  Settings:\n";
    for(const auto& [env_var, value] : info.settings)
    {
        os << "    " << env_var << " = " << value << "\n";
    }
}

}  // namespace preset_loader
}  // namespace rocprofsys
