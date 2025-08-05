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

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <dlfcn.h>
#include <initializer_list>
#include <unordered_map>

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
}  // namespace
}  // namespace sql
}  // namespace rocpd

extern "C" {
rocpd_status_t
rocpd_sql_load_schema(rocpd_sql_engine_t                        engine,
                      rocpd_sql_schema_kind_t                   kind,
                      rocpd_sql_options_t                       options,
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

    const auto kind_file_names = std::unordered_map<rocpd_sql_schema_kind_t, std::string_view>{
        {ROCPD_SQL_SCHEMA_ROCPD_TABLES, "rocpd_tables.sql"},
        {ROCPD_SQL_SCHEMA_ROCPD_INDEXES, "rocpd_indexes.sql"},
        {ROCPD_SQL_SCHEMA_ROCPD_VIEWS, "rocpd_views.sql"},
        {ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS, "data_views.sql"},
        {ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS, "summary_views.sql"},
        {ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS, "marker_views.sql"},
    };

    const auto _lib_schema_path = rocpd::sql::get_install_path();
    const auto _env_schema_path = rocprofiler::common::get_env("ROCPD_SCHEMA_PATH", "");
    const auto _usr_schema_path =
        (schema_path_hints)
            ? fmt::format(
                  "{}",
                  fmt::join(schema_path_hints, schema_path_hints + num_schema_path_hints, ":"))
            : std::string{};
    const auto _schema_paths =
        fmt::format("{}:{}:{}", _usr_schema_path, _env_schema_path, _lib_schema_path);

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

    auto _substitutions = std::vector<std::pair<std::string_view, std::string>>{};

    using jinja_init_list_t = std::initializer_list<std::pair<std::string_view, const char*>>;

    if(variables != nullptr)
    {
        if(variables->size == 0)
        {
            return ROCPD_STATUS_ERROR_SQL_INVALID_SCHEMA_KIND;
        }

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

        // make substitutions for remaining variables which do not require special handling like
        // {{uuid}}
        for(auto [key, value] : jinja_init_list_t{{"{{guid}}", variables->guid}})
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

    callback(engine, kind, options, variables, cb_schema_path, cb_schema_contents, user_data);

    return ROCPD_STATUS_SUCCESS;
}
}
