// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "schema_manifest.hpp"

#include "debug.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace profiler_hub::data_storage
{

namespace
{
const std::vector<std::string>&
known_schema_kinds()
{
    static const std::vector<std::string> kinds = {
        "rocpd_metadata", "rocpd_tables",     "rocpd_views",
        "rocpd_indexes",  "rocpd_data_views", "rocpd_summary_views",
    };
    return kinds;
}

// Parses a "<major>.<minor>.<patch>" string
profiler_hub::version_t
parse_schema_version(const std::string& version_str)
{
    profiler_hub::version_t version;
    std::istringstream      iss(version_str);
    std::string             part;

    if(std::getline(iss, part, '.'))
        version.major = static_cast<uint32_t>(std::stoul(part));
    if(std::getline(iss, part, '.'))
        version.minor = static_cast<uint32_t>(std::stoul(part));
    if(std::getline(iss, part, '.'))
        version.patch = static_cast<uint32_t>(std::stoul(part));

    return version;
}
}  // namespace

// Load versions.yml and build the schema-file map used during DB initialization.
void
load_schema_manifest(const std::string&       schema_dir,
                     version_file_map_t&      version_file_map,
                     profiler_hub::version_t& latest_version)
{
    const std::filesystem::path manifest_path =
        std::filesystem::path(schema_dir) / "versions.yml";

    LOG_TRACE("Loading schema manifest: '{}'", manifest_path.string());

    version_file_map.clear();
    latest_version = profiler_hub::version_t{};

    if(!std::filesystem::exists(manifest_path))
    {
        LOG_ERROR("Schema manifest not found: '{}'", manifest_path.string());
        throw std::runtime_error("Schema manifest not found: " + manifest_path.string());
    }

    auto ifs = std::ifstream{ manifest_path };
    if(!ifs.is_open())
    {
        LOG_ERROR("Failed to open schema manifest: '{}'", manifest_path.string());
        throw std::runtime_error("Failed to open schema manifest: " +
                                 manifest_path.string());
    }

    try
    {
        auto manifest_contents = std::stringstream{};
        manifest_contents << ifs.rdbuf();

        const auto yaml = YAML::Load(manifest_contents.str());

        // versions.yml is structured as:
        //
        //   rocprofiler-sdk-rocpd:
        //     rocpd_schemas:
        //       - version: "3.0.1"
        //         rocpd_tables: versions/3.0.1/rocpd_tables.sql
        //         ...
        for(const auto& entry : yaml["rocprofiler-sdk-rocpd"]["rocpd_schemas"])
        {
            const auto version_str = entry["version"].as<std::string>();

            kind_filename_map_t      kind_paths;
            std::vector<std::string> missing_kinds;

            for(const auto& kind : known_schema_kinds())
            {
                if(entry[kind])
                {
                    kind_paths[kind] = entry[kind].as<std::string>();
                }
                else
                {
                    missing_kinds.push_back(kind);
                }
            }

            // get_schema_query() unconditionally looks up every known kind via
            // kind_paths.at(...), so a partially-specified version entry must be
            // rejected here rather than surfacing later as an opaque
            // std::out_of_range at database-initialization time.
            if(!missing_kinds.empty())
            {
                std::string joined_missing;
                for(size_t i = 0; i < missing_kinds.size(); ++i)
                {
                    if(i != 0) joined_missing += ", ";
                    joined_missing += missing_kinds[i];
                }

                LOG_ERROR("Schema manifest entry for version '{}' in '{}' is missing "
                          "required schema kind(s): {}",
                          version_str,
                          manifest_path.string(),
                          joined_missing);
                throw std::runtime_error(
                    "Schema manifest entry for version '" + version_str +
                    "' is missing required schema kind(s): " + joined_missing);
            }

            version_file_map[version_str] = std::move(kind_paths);
            latest_version = std::max(latest_version, parse_schema_version(version_str));
        }
    } catch(const YAML::Exception& e)
    {
        LOG_ERROR(
            "Error loading schema manifest '{}': {}", manifest_path.string(), e.what());
        throw std::runtime_error("Failed to parse schema manifest: " +
                                 manifest_path.string() + ": " + e.what());
    } catch(const std::exception& e)
    {
        LOG_ERROR(
            "Error loading schema manifest '{}': {}", manifest_path.string(), e.what());
        throw std::runtime_error("Failed to load schema manifest: " +
                                 manifest_path.string() + ": " + e.what());
    }

    if(version_file_map.empty())
    {
        throw std::runtime_error("Schema manifest contains no versions: " +
                                 manifest_path.string());
    }
}

profiler_hub::version_t
resolve_schema_version(const version_file_map_t&      version_file_map,
                       const profiler_hub::version_t& latest_version,
                       const profiler_hub::version_t& requested)
{
    if(version_file_map.empty())
    {
        throw std::runtime_error(
            "Schema manifest is empty; cannot resolve schema version");
    }

    const profiler_hub::version_t resolved =
        requested.is_latest() ? latest_version : requested;

    if(version_file_map.find(resolved.to_string()) == version_file_map.end())
    {
        throw std::runtime_error("Unsupported rocpd schema version requested: " +
                                 requested.to_string());
    }

    return resolved;
}

}  // namespace profiler_hub::data_storage
