// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define _GNU_SOURCE 1

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"

#include "lib/rocprofiler-sdk-rocpd/details/format.hpp"

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <yaml-cpp/exceptions.h>
#include <yaml-cpp/node/convert.h>
#include <yaml-cpp/node/detail/impl.h>
#include <yaml-cpp/node/impl.h>
#include <yaml-cpp/node/iterator.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/parser.h>

#include <dlfcn.h>
#include <cstddef>
#include <initializer_list>
#include <unordered_map>

bool
operator==(rocpd_version_triplet_t lhs, rocpd_version_triplet_t rhs)
{
    return std::tie(lhs.major, lhs.minor, lhs.patch) == std::tie(rhs.major, rhs.minor, rhs.patch);
}

bool
operator<(rocpd_version_triplet_t lhs, rocpd_version_triplet_t rhs)
{
    return std::tie(lhs.major, lhs.minor, lhs.patch) < std::tie(rhs.major, rhs.minor, rhs.patch);
}

namespace rocpd
{
namespace sql
{
namespace
{
namespace common = ::rocprofiler::common;
namespace fs     = ::rocprofiler::common::filesystem;

std::string
get_install_path()
{
    auto* _rocpd_sql_load_schema_sym = dlsym(RTLD_DEFAULT, "rocpd_sql_load_schema");

    ROCP_CI_LOG_IF(WARNING, !_rocpd_sql_load_schema_sym)
        << "[rocprofiler-sdk-rocpd] dlsym(RTLD_DEFAULT, 'rocpd_sql_load_schema') failed "
           "(unexpectedly) from within the rocprofiler-sdk-rocpd library";

    if(!_rocpd_sql_load_schema_sym)
        _rocpd_sql_load_schema_sym = reinterpret_cast<void*>(&rocpd_sql_load_schema);

    if(Dl_info dl_info = {};
       dladdr(_rocpd_sql_load_schema_sym, &dl_info) != 0 && dl_info.dli_fname != nullptr)
    {
        auto _share_path =
            fs::path{dl_info.dli_fname}.lexically_normal().parent_path().parent_path() /
            std::string{"share/rocprofiler-sdk-rocpd"};
        ROCP_INFO << fmt::format("[rocprofiler-sdk-rocpd] resolved rocprofiler-sdk-rocpd SQL "
                                 "schema path as '{}' (dli_fname: {})",
                                 _share_path.string(),
                                 dl_info.dli_fname);
        return _share_path;
    }

    ROCP_CI_LOG(WARNING)
        << "Failed to locate the installation path of rocprofiler-sdk-rocpd via dladdr of the "
           "'rocpd_sql_load_schema' symbol (which should be in librocprofiler-sdk-rocpd.so)";

    return std::string{};
}

template <typename Tp>
auto
replace_all(std::string val, Tp from, std::string_view to)
{
    size_t pos = 0;
    while((pos = val.find(from, pos)) != std::string::npos)
    {
        if constexpr(std::is_same<common::mpl::unqualified_type_t<Tp>, char>::value)
        {
            val.replace(pos, 1, to);
            pos += to.length();
        }
        else
        {
            val.replace(pos, std::string_view{from}.length(), to);
            pos += to.length();
        }
    }
    return val;
}

rocpd_version_triplet_t
get_version_triplet(std::string_view version_str)
{
    auto parts = std::vector<std::string>{};
    parts.reserve(3);
    for(const auto& part : rocprofiler::sdk::parse::tokenize(version_str, "."))
        parts.emplace_back(part);

    auto version = rocpd_version_triplet_t{0, 0, 0};
    if(!parts.empty()) version.major = static_cast<uint32_t>(std::stoul(parts.at(0)));
    if(parts.size() > 1) version.minor = static_cast<uint32_t>(std::stoul(parts.at(1)));
    if(parts.size() > 2) version.patch = static_cast<uint32_t>(std::stoul(parts.at(2)));

    return version;
}

using kind_filename_map_t = std::unordered_map<rocpd_sql_schema_kind_t, std::string>;
using version_file_map_t  = std::unordered_map<std::string, kind_filename_map_t>;

const auto&
yaml_kind_keys()
{
    static const auto m = std::unordered_map<std::string_view, rocpd_sql_schema_kind_t>{
        {"rocpd_tables", ROCPD_SQL_SCHEMA_ROCPD_TABLES},
        {"rocpd_indexes", ROCPD_SQL_SCHEMA_ROCPD_INDEXES},
        {"rocpd_views", ROCPD_SQL_SCHEMA_ROCPD_VIEWS},
        {"rocpd_data_views", ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS},
        {"rocpd_summary_views", ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS},
        {"rocpd_metadata", ROCPD_SQL_SCHEMA_ROCPD_METADATA},
    };
    return m;
}

std::string
build_schema_paths_string(const char** schema_path_hints, uint64_t num_schema_path_hints)
{
    const auto _lib_schema_path = get_install_path();
    const auto _env_schema_path = rocprofiler::common::get_env("ROCPD_SCHEMA_PATH", "");
    const auto _usr_schema_path =
        (schema_path_hints)
            ? fmt::format(
                  "{}",
                  fmt::join(schema_path_hints, schema_path_hints + num_schema_path_hints, ":"))
            : std::string{};
    return fmt::format("{}:{}:{}", _usr_schema_path, _env_schema_path, _lib_schema_path);
}

void
load_version_file_map(const std::string&       _schema_paths,
                      version_file_map_t&      version_file_map,
                      rocpd_version_triplet_t& latest_version)
{
    auto _schema_versions_file = std::optional<std::string>{};
    for(const auto& itr : rocprofiler::sdk::parse::tokenize(_schema_paths, ":"))
    {
        auto _fpath = fs::path{itr} / "versions.yml";
        ROCP_TRACE << fmt::format("[rocprofiler-sdk-rocpd] Loading versions.yml: '{}'",
                                  _fpath.string());
        if(fs::exists(_fpath))
        {
            ROCP_INFO << fmt::format("[rocprofiler-sdk-rocpd] Found schema versions file: '{}'",
                                     _fpath.string());
            _schema_versions_file = _fpath;
            break;
        }
    }

    latest_version = rocpd_version_triplet_t{0, 0, 0};
    if(!_schema_versions_file) return;

    try
    {
        auto versioning_contents = std::stringstream{};
        ROCP_INFO << "Loading Schema Config: " << *_schema_versions_file;
        auto ifs = std::ifstream{*_schema_versions_file};
        versioning_contents << ifs.rdbuf();
        auto yaml = YAML::Load(versioning_contents.str());
        for(auto itr : yaml["rocprofiler-sdk-rocpd"]["rocpd_schemas"])
        {
            auto version = itr["version"].as<std::string>();
            for(const auto& fitr : yaml_kind_keys())
            {
                if(itr[fitr.first])
                {
                    version_file_map[version][fitr.second] = itr[fitr.first].as<std::string>();

                    latest_version = std::max(latest_version, get_version_triplet(version));
                }
            }
        }
    } catch(const YAML::Exception& e)
    {
        ROCP_ERROR << fmt::format(
            "[rocprofiler-sdk-rocpd] Error loading schema versions file: '{}' : {}",
            *_schema_versions_file,
            e.what());
        return;
    } catch(const std::exception& e)
    {
        ROCP_ERROR << fmt::format(
            "[rocprofiler-sdk-rocpd] Error loading schema versions file: '{}' : {}",
            *_schema_versions_file,
            e.what());
        return;
    }
}
}  // namespace
}  // namespace sql
}  // namespace rocpd

extern "C" {
rocpd_status_t
rocpd_sql_load_schema(rocpd_sql_engine_t                        engine,
                      rocpd_sql_schema_kind_t                   kind,
                      rocpd_sql_options_t                       options,
                      rocpd_version_triplet_t                   schema_version,
                      const rocpd_sql_schema_jinja_variables_t* variables,
                      rocpd_sql_load_schema_cb_t                callback,
                      const char**                              schema_path_hints,
                      uint64_t                                  num_schema_path_hints,
                      void*                                     user_data)
{
    namespace fs = ::rocpd::sql::fs;

    switch(engine)
    {
        case ROCPD_SQL_ENGINE_SQLITE3:
        {
            break;
        }
        case ROCPD_SQL_ENGINE_NONE:
        case ROCPD_SQL_ENGINE_LAST:
        {
            return ROCPD_STATUS_ERROR_SQL_INVALID_ENGINE;
        }
    }

    // Check the requested version is valid and catch invalid schema versions from older API
    if(ROCPROFILER_SDK_COMPUTE_VERSION(
           schema_version.major, schema_version.minor, schema_version.patch) >
       ROCPROFILER_SDK_COMPUTE_VERSION(99, 99, 99))
    {
        ROCP_ERROR << fmt::format("[rocprofiler-sdk-rocpd] Schema version is invalid: '{}.{}.{}'.",
                                  schema_version.major,
                                  schema_version.minor,
                                  schema_version.patch);
        ROCP_ERROR << fmt::format("[rocprofiler-sdk-rocpd] This is likely due to an older API "
                                  "being used. Please update the API to the latest version.");
        return ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION;
    }

    auto       version_file_map = rocpd::sql::version_file_map_t{};
    const auto _schema_paths =
        rocpd::sql::build_schema_paths_string(schema_path_hints, num_schema_path_hints);

    auto latest_version = rocpd_version_triplet_t{0, 0, 0};
    rocpd::sql::load_version_file_map(_schema_paths, version_file_map, latest_version);

    if(schema_version == rocpd_version_triplet_t{0, 0, 0}) schema_version = latest_version;

    const auto version_key = fmt::format("{}", schema_version);
    const auto version_itr = version_file_map.find(version_key);
    if(version_itr == version_file_map.end()) return ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION;

    const auto& kind_file_names = version_itr->second;

    if(kind_file_names.count(kind) == 0) return ROCPD_STATUS_ERROR_SQL_INVALID_SCHEMA_KIND;

    auto _schema_file = std::optional<std::string>{};
    for(const auto& itr : rocprofiler::sdk::parse::tokenize(_schema_paths, ":"))
    {
        auto _fpath = fs::path{itr} / kind_file_names.at(kind);
        ROCP_TRACE << fmt::format("[rocprofiler-sdk-rocpd] Searching for schema file: '{}'",
                                  _fpath.string());
        if(fs::exists(_fpath))
        {
            ROCP_INFO << fmt::format("[rocprofiler-sdk-rocpd] Found schema file: '{}'",
                                     _fpath.string());
            _schema_file = _fpath;
            break;
        }
    }

    if(!_schema_file) return ROCPD_STATUS_ERROR_SQL_SCHEMA_NOT_FOUND;

    auto read_file = [](const std::string& _file_path) -> std::string {
        auto _ifs = std::ifstream{_file_path, std::ios::in | std::ios::binary};
        if(!_ifs.is_open()) return std::string{};

        auto _buffer = std::stringstream{};
        _buffer << _ifs.rdbuf();
        return _buffer.str();
    };

    auto _contents = read_file(*_schema_file);

    if(_contents.empty()) return ROCPD_STATUS_ERROR_SQL_SCHEMA_PERMISSION_DENIED;

    if(engine == ROCPD_SQL_ENGINE_SQLITE3)
    {
        if((options & ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS) ==
           ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS)
            _contents = fmt::format("PRAGMA foreign_keys = ON;\n\n{}", _contents);
    }

    if(variables != nullptr)
    {
        using jinja_init_list_t = std::initializer_list<std::pair<std::string_view, const char*>>;

        if(variables->size == 0) return ROCPD_STATUS_ERROR_INVALID_ARGUMENT;

        // {{uuid}} is used in table names and require special handling
        if(const auto* value = variables->uuid; value != nullptr)
        {
            auto _value = std::string{value};

            // non-empty strings are prefixed with underscore for readability
            if(!_value.empty() && _value.find('_') != 0) _value = fmt::format("_{}", _value);

            // replace hyphens with underscores since these are used in table/view names
            if(_value.find('-') != std::string::npos)
                _value = rocpd::sql::replace_all(_value, "-", "_");

            // make substitutions
            _contents = rocpd::sql::replace_all(_contents, "{{uuid}}", _value);
        }

        const auto _schema_version = fmt::format("{}", schema_version);
        const auto _schema_major   = fmt::format("{}", schema_version.major);
        const auto _schema_minor   = fmt::format("{}", schema_version.minor);
        const auto _schema_patch   = fmt::format("{}", schema_version.patch);

        // make substitutions for remaining variables which do not require special handling like
        // {{uuid}}
        for(auto [key, value] : jinja_init_list_t{
                {"{{guid}}", variables->guid},
                {"{{schema_version}}", _schema_version.c_str()},
                {"{{schema_version_major}}", _schema_major.c_str()},
                {"{{schema_version_minor}}", _schema_minor.c_str()},
                {"{{schema_version_patch}}", _schema_patch.c_str()},
            })
        {
            if(value != nullptr)
            {
                _contents = rocpd::sql::replace_all(_contents, key, std::string_view{value});
            }
        }
    }

    const auto* cb_schema_path     = _schema_file->c_str();
    const auto* cb_schema_contents = _contents.c_str();

    if((options & ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS) ==
       ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS)
    {
        cb_schema_path     = rocprofiler::common::get_string_entry(cb_schema_path)->c_str();
        cb_schema_contents = rocprofiler::common::get_string_entry(cb_schema_contents)->c_str();
    }

    callback(engine,
             kind,
             options,
             schema_version,
             variables,
             cb_schema_path,
             cb_schema_contents,
             user_data);

    return ROCPD_STATUS_SUCCESS;
}

rocpd_status_t
rocpd_query_supported_schema_versions(rocpd_sql_engine_t engine,
                                      const char**       schema_path_hints,
                                      uint64_t           num_schema_path_hints,
                                      rocpd_query_supported_schema_versions_cb_t callback,
                                      void*                                      user_data)
{
    switch(engine)
    {
        case ROCPD_SQL_ENGINE_SQLITE3:
        {
            break;
        }
        case ROCPD_SQL_ENGINE_NONE:
        case ROCPD_SQL_ENGINE_LAST:
        {
            return ROCPD_STATUS_ERROR_SQL_INVALID_ENGINE;
        }
    }

    auto       version_file_map = rocpd::sql::version_file_map_t{};
    const auto _schema_paths =
        rocpd::sql::build_schema_paths_string(schema_path_hints, num_schema_path_hints);

    auto latest_version = rocpd_version_triplet_t{0, 0, 0};
    rocpd::sql::load_version_file_map(_schema_paths, version_file_map, latest_version);
    (void) latest_version;

    auto sorted_versions = std::vector<std::string>{};
    sorted_versions.reserve(version_file_map.size());
    for(const auto& kv : version_file_map)
        sorted_versions.emplace_back(kv.first);

    std::sort(sorted_versions.begin(),
              sorted_versions.end(),
              [](const std::string& a, const std::string& b) {
                  return rocpd::sql::get_version_triplet(a) < rocpd::sql::get_version_triplet(b);
              });

    auto versions = std::vector<rocpd_version_triplet_t>{};
    versions.reserve(sorted_versions.size());
    for(const auto& v : sorted_versions)
        versions.emplace_back(rocpd::sql::get_version_triplet(v));

    return callback(engine, versions.data(), static_cast<uint64_t>(versions.size()), user_data);
}
}
