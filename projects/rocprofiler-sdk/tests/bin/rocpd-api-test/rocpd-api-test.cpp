// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file rocpd-api-test.cpp
 * @brief Small C++ test for rocprofiler-sdk-rocpd API: rocpd_get_version and
 *        rocpd_sql_load_schema. Run with ROCPD_SCHEMA_PATH set to the schema
 *        directory (e.g. build/share/rocprofiler-sdk-rocpd).
 */

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
auto show_full_schema_tables = false;
auto show_full_schema_views  = false;

/**
 * Extract schema_version from rocpd_metadata INSERT in tables schema (e.g. ("schema_version",
 * "3")). However, in new schema, the INSERT moved to the METADATA file.
 */
std::string
parse_schema_version(const char* content)
{
    if(content == nullptr)
    {
        return "";
    }

    auto       s   = std::string{content};
    const auto key = std::string{"\"schema_version\""};
    auto       pos = s.find(key);
    if(pos == std::string::npos)
    {
        return "";
    }

    pos += key.size();
    pos = s.find('"', pos);
    if(pos == std::string::npos)
    {
        return "";
    }

    ++pos;
    auto end = s.find('"', pos);
    if(end == std::string::npos)
    {
        return "";
    }
    return s.substr(pos, end - pos);
}

/** Parse object names from SQL content for a given keyword (e.g. "CREATE TABLE" or "CREATE VIEW").
 */
void
parse_sql_names(const char* content, const char* sql_keyword, std::vector<std::string>& out_names)
{
    if(content == nullptr)
    {
        return;
    }

    auto       s   = std::string{content};
    const auto key = std::string{sql_keyword};

    for(auto pos = size_t{0}; (pos = s.find(key, pos)) != std::string::npos; pos += key.size())
    {
        pos += key.size();
        /* Skip past "IF NOT EXISTS" and whitespace until we hit the quoted/backticked name */
        while(pos < s.size() && s[pos] != '"' && s[pos] != '`')
        {
            ++pos;
        }
        if(pos >= s.size())
        {
            break;
        }
        auto quote = s[pos];
        auto start = pos + 1;
        auto end   = s.find(quote, start);
        if(end == std::string::npos)
        {
            break;
        }
        out_names.push_back(s.substr(start, end - start));
    }
}

struct callback_data_t
{
    std::vector<std::string>* names;
    std::string*              schema_version;
    std::string*              schema_content;
};

void
tables_callback(rocpd_sql_engine_t /*engine*/,
                rocpd_sql_schema_kind_t /*kind*/,
                rocpd_sql_options_t /*options*/,
                rocpd_version_triplet_t /*schema_version_triplet*/,
                const rocpd_sql_schema_jinja_variables_t* /*variables*/,
                const char* /*schema_path*/,
                const char* schema_content,
                void*       user_data)
{
    auto* data = static_cast<callback_data_t*>(user_data);
    if(schema_content == nullptr || data == nullptr || data->names == nullptr)
    {
        return;
    }
    auto content = std::string{schema_content};
    if(content.find("CREATE TABLE") == std::string::npos ||
       content.find("rocpd_metadata") == std::string::npos)
    {
        return;
    }
    parse_sql_names(schema_content, "CREATE TABLE", *data->names);
    if(data->schema_version != nullptr)
    {
        *data->schema_version = parse_schema_version(schema_content);
    }
    if(data->schema_content != nullptr)
    {
        *data->schema_content = content;
    }
}

void
views_callback(rocpd_sql_engine_t /*engine*/,
               rocpd_sql_schema_kind_t kind,
               rocpd_sql_options_t /*options*/,
               rocpd_version_triplet_t /*schema_version_triplet*/,
               const rocpd_sql_schema_jinja_variables_t* /*variables*/,
               const char* /*schema_path*/,
               const char* schema_content,
               void*       user_data)
{
    auto* data = static_cast<callback_data_t*>(user_data);
    if(schema_content == nullptr || data == nullptr || data->names == nullptr)
    {
        return;
    }
    parse_sql_names(schema_content, "CREATE VIEW", *data->names);
    if(kind == ROCPD_SQL_SCHEMA_ROCPD_METADATA && data->schema_version != nullptr &&
       data->schema_version->empty())
    {
        *data->schema_version = parse_schema_version(schema_content);
    }
    if(data->schema_content != nullptr)
    {
        // append the schema content to the view content
        *data->schema_content += schema_content;
    }
}

int
load_schema(rocpd_version_triplet_t requested_version)
{
    auto status         = rocpd_status_t{ROCPD_STATUS_SUCCESS};
    auto table_names    = std::vector<std::string>{};
    auto schema_version = std::string{};
    auto schema_content = std::string{};
    auto view_content   = std::string{};
    auto tables_data    = callback_data_t{&table_names, &schema_version, &schema_content};

    auto variables = rocpd_sql_schema_jinja_variables_t{};
    variables.size = sizeof(rocpd_sql_schema_jinja_variables_t);
    variables.uuid = "";
    variables.guid = "";

    status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                   ROCPD_SQL_SCHEMA_ROCPD_TABLES,
                                   ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS,
                                   requested_version,
                                   &variables,
                                   tables_callback,
                                   nullptr,
                                   0,
                                   &tables_data);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "rocpd-api-test: rocpd_sql_load_schema(tables) failed: "
                  << rocpd_get_status_name(status) << " - "
                  << (rocpd_get_status_string(status) ? rocpd_get_status_string(status) : "unknown")
                  << "\n";
        return EXIT_FAILURE;
    }
    if(table_names.empty())
    {
        std::cerr << "rocpd-api-test: no tables found in schema\n";
        return EXIT_FAILURE;
    }

    auto           view_names = std::vector<std::string>{};
    constexpr auto view_kinds = std::array{
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_METADATA,
    };
    auto views_data = callback_data_t{&view_names, &schema_version, &view_content};
    for(auto kind : view_kinds)
    {
        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                       kind,
                                       ROCPD_SQL_OPTIONS_NONE,
                                       requested_version,
                                       &variables,
                                       views_callback,
                                       nullptr,
                                       0,
                                       &views_data);
        if(status != ROCPD_STATUS_SUCCESS)
        {
            std::cerr << "rocpd-api-test: rocpd_sql_load_schema(views) failed: "
                      << rocpd_get_status_name(status) << "\n";
            return EXIT_FAILURE;
        }
    }

    if(show_full_schema_tables || show_full_schema_views)
    {
        if(show_full_schema_tables)
        {
            std::cout << "  Tables content:\n";
            std::cout << schema_content << "\n";
        }
        if(show_full_schema_views)
        {
            std::cout << "  Views content:\n";
            std::cout << view_content << "\n";
        }
    }
    else
    {
        std::cout << "  rocpd-api-test: rocpd_sql_load_schema OK\n";
        std::cout << "  Schema version: " << (schema_version.empty() ? "unknown" : schema_version)
                  << "\n";
        std::cout << "  Number of tables: " << table_names.size() << "\n";
        std::cout << "  Tables:";
        for(const auto& name : table_names)
        {
            std::cout << " " << name;
        }
        std::cout << "\n\n";
        std::cout << "  Number of views: " << view_names.size() << "\n";
        std::cout << "  Views:";
        for(const auto& name : view_names)
        {
            std::cout << " " << name;
        }
        std::cout << "\n\n";
    }
    return EXIT_SUCCESS;
}

rocpd_status_t
query_supported_schema_versions_callback(rocpd_sql_engine_t /*engine*/,
                                         const rocpd_version_triplet_t* versions,
                                         uint64_t                       num_versions,
                                         void*                          user_data)
{
    auto* out = static_cast<std::vector<rocpd_version_triplet_t>*>(user_data);
    if(out == nullptr) return ROCPD_STATUS_ERROR_INVALID_ARGUMENT;
    out->assign(versions, versions + num_versions);
    return ROCPD_STATUS_SUCCESS;
}

std::vector<std::string>
version_splitter(const std::string& version)
{
    auto version_parts  = std::vector<std::string>{};
    auto stream_version = std::istringstream{version};
    auto part           = std::string{};
    while(std::getline(stream_version, part, '.'))
    {
        version_parts.push_back(part);
    }
    return version_parts;
}

}  // namespace

#define VERSION_STRING_TO_TRIPLET(version)                                                         \
    (rocpd_version_triplet_t{                                                                      \
        (uint32_t) std::stoi(version[0]),                                                          \
        (uint32_t) std::stoi(version[1]),                                                          \
        (uint32_t) std::stoi(version[2])})  // NOLINT(readability-magic-numbers)

int
main(int argc, char** argv)
{
    auto list_versions            = true;
    auto load_latest_schema       = true;
    auto load_all_schemas         = true;
    auto load_requested_schema    = false;
    auto requested_schema_version = rocpd_version_triplet_t{0, 0, 0};

    auto supported_commands = std::vector<std::string>{
        "--help",
        "-h",
        "--list",
        "-l",
        "--all",
        "-a",
        "--get-version",
        "-g",
        "--get-latest",
        "-L",
        "--show-full-tables",
        "-st",
        "--show-full-views",
        "-sv",
        "--show-full-schema",
        "-ss",
    };

    auto get_version_error = [](const char* name) {
        std::cerr << "rocpd-api-test: --get-version requires a rocpd schema version argument in "
                     "the format <major>.<minor>.<patch>\n";
        std::cout << "Use --list to list all supported schema versions, then example usage:\n";
        std::cout << "   " << name << " --get-version 3.0.0\n";
    };

    auto print_help = [](const char* name) {
        std::cout << "Usage: " << name
                  << " [--help | --all| --list | --get-latest | --get-version <version>] "
                     "[--show-full-schema | --show-full-tables | --show-full-views]\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help                  : display this help message\n";
        std::cout << "  -a, --all                   : get list of all schema and load all schema "
                     "versions\n";
        std::cout << "  -l, --list                  : list all supported schema versions\n";
        std::cout << "  -L, --get-latest            : load the latest schema version\n";
        std::cout << "  -g, --get-version <version> : load the specified schema version\n";
        std::cout << "\n";
        std::cout
            << "View schema content options (default lists table/view names and table counts):\n";
        std::cout << "  -ss, --show-full-schema     : show the full schema content (enables both "
                     "options below)\n";
        std::cout << "  -st, --show-full-tables     : just show the full table schema content\n";
        std::cout << "  -sv, --show-full-views      : just show the full view schema content\n\n";
        std::cout << "Example usage:\n";
        std::cout << "  " << name
                  << " --all                                  : lists all schemas, loads latest, "
                     "and iterate over all schemas\n";
        std::cout << "  " << name
                  << " --list                                 : lists all schemas\n";
        std::cout << "  " << name
                  << " --get-latest                           : loads the latest schema version\n";
        std::cout << "  " << name
                  << " --get-version 3.0.0                    : loads the schema version 3.0.0\n";
        std::cout << "  " << name
                  << " --get-version 3.0.0 --show-full-tables : loads the schema version 3.0.0 and "
                     "shows the full table schema content\n";
    };

    auto check_version_parts = [](const std::vector<std::string>& version_parts) {
        for(const auto& part : version_parts)
        {
            if(!std::all_of(part.begin(), part.end(), ::isdigit))
            {
                return false;
            }
        }
        return true;
    };

    if(argc <= 1)
    {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    for(auto i = 1; i < argc; ++i)
    {
        if(std::find(supported_commands.begin(), supported_commands.end(), argv[i]) ==
           supported_commands.end())
        {
            std::cerr << "rocpd-api-test: unsupported command: " << argv[i] << "\n";
            return EXIT_FAILURE;
        }

        if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_help(argv[0]);
            return EXIT_SUCCESS;
        }

        if(strcmp(argv[i], "--show-full-schema") == 0 || strcmp(argv[i], "-ss") == 0)
        {
            show_full_schema_tables = true;
            show_full_schema_views  = true;
        }
        if(strcmp(argv[i], "--show-full-tables") == 0 || strcmp(argv[i], "-st") == 0)
        {
            show_full_schema_tables = true;
        }
        if(strcmp(argv[i], "--show-full-views") == 0 || strcmp(argv[i], "-sv") == 0)
        {
            show_full_schema_views = true;
        }

        if(strcmp(argv[i], "--all") == 0 || strcmp(argv[i], "-a") == 0)
        {
            load_all_schemas      = true;
            list_versions         = false;
            load_latest_schema    = false;
            load_requested_schema = false;
        }

        if(strcmp(argv[i], "--list") == 0 || strcmp(argv[i], "-l") == 0)
        {
            load_all_schemas      = false;
            list_versions         = true;
            load_latest_schema    = false;
            load_requested_schema = false;
        }

        if(strcmp(argv[i], "--get-latest") == 0 || strcmp(argv[i], "-L") == 0)
        {
            load_all_schemas      = false;
            list_versions         = false;
            load_latest_schema    = true;
            load_requested_schema = false;
        }

        if(strcmp(argv[i], "--get-version") == 0 || strcmp(argv[i], "-g") == 0)
        {
            auto version = std::string{argv[i + 1]};
            // count the number of '.' in the version string
            if(std::count(version.begin(), version.end(), '.') != 2)
            {
                get_version_error(argv[0]);
                return EXIT_FAILURE;
            }
            auto version_parts = version_splitter(version);
            if(check_version_parts(version_parts) == false)
            {
                get_version_error(argv[0]);
                return EXIT_FAILURE;
            }
            requested_schema_version = VERSION_STRING_TO_TRIPLET(version_parts);
            load_all_schemas         = false;
            list_versions            = false;
            load_latest_schema       = false;
            load_requested_schema    = true;
            i++;
        }
    }

    // Get the rocpd module version
    auto major  = uint32_t{};
    auto minor  = uint32_t{};
    auto patch  = uint32_t{};
    auto status = rocpd_get_version(&major, &minor, &patch);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "rocpd-api-test: rocpd_get_version failed: " << rocpd_get_status_name(status)
                  << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "rocpd-api-test: rocpd_get_version OK (" << major << "." << minor << "." << patch
              << ")\n";

    auto local_list_of_schema_versions = std::vector<rocpd_version_triplet_t>{};

    if(list_versions || load_all_schemas)
    {
        status = rocpd_query_supported_schema_versions(ROCPD_SQL_ENGINE_SQLITE3,
                                                       nullptr,
                                                       0,
                                                       query_supported_schema_versions_callback,
                                                       &local_list_of_schema_versions);
        if(status != ROCPD_STATUS_SUCCESS)
        {
            std::cerr << "rocpd-api-test: rocpd_query_supported_schema_versions failed: "
                      << rocpd_get_status_name(status) << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "rocpd-api-test: rocpd_query_supported_schema_versions OK ("
                  << local_list_of_schema_versions.size() << " versions)\n";
        for(auto i = size_t{0}; i < local_list_of_schema_versions.size(); ++i)
        {
            std::cout << "  Version " << i << ": " << local_list_of_schema_versions[i].major << "."
                      << local_list_of_schema_versions[i].minor << "."
                      << local_list_of_schema_versions[i].patch << "\n";
        }
    }

    if(load_latest_schema)
    {
        std::cout << "\nLoading latest schema version (requesting 0.0.0)...\n";
        auto latest_schema_version = rocpd_version_triplet_t{0, 0, 0};
        if(load_schema(latest_schema_version) != EXIT_SUCCESS)
        {
            return EXIT_FAILURE;
        }
    }

    if(load_requested_schema)
    {
        std::cout << "\nLoading requested schema version: " << requested_schema_version.major << "."
                  << requested_schema_version.minor << "." << requested_schema_version.patch
                  << "\n";
        if(load_schema(requested_schema_version) != EXIT_SUCCESS)
        {
            std::cerr << "rocpd-api-test: load_schema failed. Use --list for a list of supported "
                         "schema versions.\n";
            return EXIT_FAILURE;
        }
    }

    if(load_all_schemas)
    {
        std::cout << "\nNow iterating over the entire list of schema versions:\n";
        for(const auto& version : local_list_of_schema_versions)
        {
            std::cout << "  For schema version: " << version.major << "." << version.minor << "."
                      << version.patch << ", load schema...\n";
            if(load_schema(version) != EXIT_SUCCESS)
            {
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}
